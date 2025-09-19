#!/usr/bin/env python3
"""
Local RAG chatbot over your PDF library (fully local: FAISS + Ollama).

Fixes applied:
- No unterminated f-strings.
- Long f-strings are split safely across lines.
- Lines kept reasonably short to avoid style warnings.
"""

import io
import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from pypdf import PdfReader
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# Local LLM via Ollama
import ollama

# -----------------------
# Config (adjust as needed)
# -----------------------
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # fast, good quality
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"  # set None to disable
DEFAULT_LLM = "qwen2.5:7b-instruct"  # or "llama3.1:8b-instruct"
CHUNK_SIZE = 1600  # characters per chunk
CHUNK_OVERLAP = 240  # characters overlap
TOPK_FAISS = 40  # initial recall
TOPK_FINAL = 8  # final context chunks
MAX_HISTORY_TURNS = 6  # keep recent turns
MAX_CONTEXT_CHARS = 18000  # truncate context to avoid huge prompts
OCR_CACHE_DIRNAME = ".ocr_cache"


# -----------------------
# Data structures
# -----------------------
@dataclass
class Chunk:
    doc_id: str
    title: str
    source: str
    page_start: int
    page_end: int
    text: str


# -----------------------
# Utilities
# -----------------------
def _find_pdf_start(path: Path, window: int = 8192) -> int:
    try:
        with path.open("rb") as f:
            head = f.read(window)
        return head.find(b"%PDF-")
    except Exception:
        return -1


def device_auto() -> str:
    # Lazy import torch to avoid import cost if not needed elsewhere.
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def read_pdf_text_per_page(path: Path) -> List[Tuple[int, str]]:
    # Returns list of (page_number_1_based, text), tolerant of preambles
    pages: List[Tuple[int, str]] = []

    start = _find_pdf_start(path)
    if start == -1:
        # Not a recognizable PDF (within the first 8 KB). Skip but log.
        print(f"[warn]Skipping file without %PDF- header in first 8KB: {path}")
        return pages

    # If there is a preamble, trim it into memory so parsers start at %PDF-
    data: Optional[bytes] = None
    if start > 0:
        try:
            with path.open("rb") as f:
                f.seek(start)
                data = f.read()
        except Exception as e:
            print(f"[warn] Failed to trim preamble for {path}: {e}")

    # Try pypdf first (fastest to set up)
    try:
        reader = PdfReader(io.BytesIO(data)) if data is not None else PdfReader(str(path))
        for i, page in enumerate(reader.pages):
            txt = ""
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            pages.append((i + 1, txt))
        # If pypdf yielded zero text across all pages, fall back to fitz
        if not any((t or "").strip() for _, t in pages):
            raise RuntimeError("pypdf extracted no text; falling back to PyMuPDF")
        return pages
    except Exception as e1:
        # Fallback: PyMuPDF (fitz) is more tolerant and often better at text extraction
        try:
            import fitz  # pymupdf
            pages = []
            if data is not None:
                with fitz.open(stream=data, filetype="pdf") as doc:
                    for i, pg in enumerate(doc):
                        pages.append((i + 1, pg.get_text("text") or ""))
            else:
                with fitz.open(str(path)) as doc:
                    for i, pg in enumerate(doc):
                        pages.append((i + 1, pg.get_text("text") or ""))
            return pages
        except Exception as e2:
            print(f"[warn] Failed to read {path} with pypdf and PyMuPDF: {e1} | {e2}")
            return []


def needs_ocr(
    pages_text: List[Tuple[int, str]], sample_pages: int = 6
) -> bool:
    if not pages_text:
        return False
    sample = pages_text[:sample_pages]
    empty = sum(1 for _, t in sample if len((t or "").strip()) < 20)
    empty_ratio = empty / max(1, len(sample))
    return empty_ratio > 0.5


def run_ocr(input_pdf: Path, output_pdf: Path) -> bool:
    try:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ocrmypdf",
            "--force-ocr",
            "--skip-text",
            "--sidecar",
            os.devnull,
            "--optimize",
            "0",
            str(input_pdf),
            str(output_pdf),
        ]
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if res.returncode != 0:
            err = res.stderr.decode("utf-8", errors="ignore")
            print(
                f"[warn] OCR failed for {input_pdf.name}: {err[:200]}"
            )
            return False
        return True
    except FileNotFoundError:
        print("[warn] ocrmypdf not found; skipping OCR.")
        return False


def normalize_whitespace(s: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", s).strip()


def consolidate_title(meta_title: Optional[str], fallback: str) -> str:
    t = (meta_title or "").strip()
    if not t:
        t = Path(fallback).stem
    return normalize_whitespace(t)[:200]


def combine_pages_into_chunks(
    pages: List[Tuple[int, str]],
    chunk_size: int,
    overlap: int,
) -> List[Tuple[int, int, str]]:
    # Concatenate per-page while tracking page spans; then chunk.
    buf: List[str] = []
    marks: List[Tuple[int, int]] = []
    for pno, txt in pages:
        if not txt:
            continue
        t = normalize_whitespace(txt)
        if not t:
            continue
        marks.append((len("".join(buf)), pno))
        buf.append(t + "\n")

    combined = "".join(buf)
    if not combined:
        return []

    step = chunk_size - overlap if chunk_size > overlap else chunk_size
    starts = list(range(0, len(combined), step))
    out: List[Tuple[int, int, str]] = []

    for s in starts:
        e = min(s + chunk_size, len(combined))
        ch = combined[s:e]

        # Find page_start/end by nearest markers
        if marks:
            ps = next((p for pos, p in marks if pos >= s), marks[-1][1])
            pe = next((p for pos, p in marks if pos >= e), marks[-1][1])
            if pe < ps:
                pe = ps
        else:
            ps = pe = 1

        out.append((ps, pe, ch))
        if e == len(combined):
            break

    return out


def save_faiss(index: faiss.Index, path: Path) -> None:
    faiss.write_index(index, str(path))


def load_faiss(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))


def fuzzy_dedupe(texts: List[str], threshold: int = 95) -> List[bool]:
    # Returns a mask of which items to keep (True = keep)
    keep = [True] * len(texts)
    seen: List[Tuple[int, str]] = []
    for i, t in enumerate(texts):
        for _, s in seen:
            if fuzz.ratio(t[:5000], s[:5000]) >= threshold:
                keep[i] = False
                break
        if keep[i]:
            seen.append((i, t))
    return keep


# -----------------------
# Embedding & Reranking
# -----------------------
class EmbeddingModel:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBED_MODEL,
        device: Optional[str] = None,
    ):
        self.device = device or device_auto()
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if isinstance(embs, list):
            embs = np.array(embs)
        return embs.astype("float32")


class Reranker:
    def __init__(
        self,
        model_name: Optional[str] = DEFAULT_RERANK_MODEL,
        device: Optional[str] = None,
    ):
        self.model = None
        self.tokenizer = None
        if model_name:
            self.device = device or device_auto()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = (
                AutoModelForSequenceClassification.from_pretrained(
                    model_name
                ).to(self.device)
            )
            self.model.eval()

    def score(self, query: str, passages: List[str]) -> List[float]:
        if self.model is None or len(passages) == 0:
            return [0.0] * len(passages)

        pairs = [(query, p) for p in passages]
        import torch

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.model.device)
            logits = self.model(**inputs).logits.squeeze(-1)
            scores = logits.detach().cpu().numpy().tolist()
        return scores


# -----------------------
# Indexer
# -----------------------
class RAGIndex:
    def __init__(
        self,
        index_dir: Path,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        rerank_model_name: Optional[str] = DEFAULT_RERANK_MODEL,
    ):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_path = self.index_dir / "index.faiss"
        self.meta_path = self.index_dir / "meta.jsonl"
        self.settings_path = self.index_dir / "settings.json"

        self.embed = EmbeddingModel(embed_model_name)
        self.rerank = Reranker(rerank_model_name)

        self.index: Optional[faiss.Index] = None
        self.meta: List[Dict[str, Any]] = []

    def build(self, books_dir: Path, ocr: bool = True) -> None:
        print(f"[info] Scanning PDFs in: {books_dir}")
        pdfs = sorted([p for p in books_dir.rglob("*.pdf") if p.is_file()])
        if not pdfs:
            print("[error] No PDFs found.")
            return

        ocr_cache = self.index_dir / OCR_CACHE_DIRNAME

        all_chunks: List[Chunk] = []
        for pdf in tqdm(pdfs, desc="Parsing PDFs"):
            use_path = pdf
            pages_text = read_pdf_text_per_page(pdf)

            if ocr and needs_ocr(pages_text):
                ocr_out = ocr_cache / pdf.relative_to(books_dir)
                ocr_out = ocr_out.with_suffix(".pdf")
                ocr_out.parent.mkdir(parents=True, exist_ok=True)
                if not ocr_out.exists():
                    ok = run_ocr(pdf, ocr_out)
                    if ok:
                        use_path = ocr_out
                        pages_text = read_pdf_text_per_page(use_path)
                else:
                    use_path = ocr_out
                    pages_text = read_pdf_text_per_page(use_path)

            title = consolidate_title(None, pdf.name)

            spans = combine_pages_into_chunks(
                pages_text, CHUNK_SIZE, CHUNK_OVERLAP
            )
            spans = [
                (ps, pe, t) for (ps, pe, t) in spans if t and len(t.strip()) > 50
            ]
            mask = fuzzy_dedupe([t for _, _, t in spans], threshold=97)
            spans = [sp for sp, keep in zip(spans, mask) if keep]

            for (ps, pe, t) in spans:
                doc_id = str(pdf)
                all_chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        title=title,
                        source=str(use_path),
                        page_start=ps,
                        page_end=pe,
                        text=t,
                    )
                )

        if not all_chunks:
            print("[error] No text extracted from PDFs.")
            return

        model_name = getattr(
            self.embed.model, "name_or_path", str(self.embed.model)
        )
        dev = self.embed.device
        num_chunks = len(all_chunks)
        msg = (
            f"[info] Embedding {num_chunks} chunks with {model_name} "
            f"on {dev}..."
        )
        print(msg)

        texts = [c.text for c in all_chunks]
        embeddings = self.embed.encode(texts, batch_size=64)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        print(f"[info] Saving FAISS index to {self.faiss_path}")
        save_faiss(index, self.faiss_path)

        print(f"[info] Writing metadata to {self.meta_path}")
        with self.meta_path.open("w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(
                    json.dumps(asdict(c), ensure_ascii=False) + "\n"
                )

        rerank_name = None
        if self.rerank and self.rerank.tokenizer:
            rerank_name = getattr(
                self.rerank.tokenizer, "name_or_path", None
            )

        settings = {
            "embed_model": model_name,
            "rerank_model": rerank_name,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "built_at": time.time(),
        }
        with self.settings_path.open("w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)

        self.index = index
        self.meta = [asdict(c) for c in all_chunks]
        print("[info] Index build complete.")

    def load(self) -> None:
        if not self.faiss_path.exists() or not self.meta_path.exists():
            raise RuntimeError(
                "Index files not found. Build the index first."
            )
        self.index = load_faiss(self.faiss_path)
        self.meta = []
        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                self.meta.append(json.loads(line))

    def search(
        self,
        query: str,
        topk_faiss: int = TOPK_FAISS,
        topk_final: int = TOPK_FINAL,
    ) -> List[Dict[str, Any]]:
        if not self.index or not self.meta:
            return []

        q_emb = self.embed.encode([query])
        k = min(topk_faiss, len(self.meta))
        _, I = self.index.search(q_emb, k)
        idxs = I[0].tolist()

        cands = [self.meta[i] for i in idxs if i >= 0]
        passages = [c["text"] for c in cands]

        if self.rerank and self.rerank.model:
            scores = self.rerank.score(query, passages)
            scored = list(zip(cands, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            cands = [c for c, _ in scored[:topk_final]]
        else:
            cands = cands[:topk_final]

        return cands


# -----------------------
# Chat
# -----------------------
SYSTEM_PROMPT = (
    "You are a helpful technical assistant. Answer using ONLY the provided "
    "context. Cite sources with bracketed numbers like [1], [2] that "
    "correspond to the context items. Each citation should include the exact "
    "bracket number(s). If the answer is not in the context, say you don't "
    "know.\n"
)


def format_context_for_prompt(
    chunks: List[Dict[str, Any]],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> Tuple[str, List[Dict[str, Any]]]:
    # Deduplicate by (doc_id, page span) while preserving order
    seen = set()
    unique = []
    for c in chunks:
        key = (c["doc_id"], c["page_start"], c["page_end"])
        if key not in seen:
            seen.add(key)
            unique.append(c)

    lines: List[str] = []
    budget = max_chars
    selected: List[Dict[str, Any]] = []

    for i, c in enumerate(unique, start=1):
        base_name = Path(c["doc_id"]).name
        ps = c["page_start"]
        pe = c["page_end"]
        header = f"[{i}] {base_name} (pages {ps}-{pe})"
        body = c["text"].strip()
        entry = f"{header}\n{body}\n"

        if len(entry) <= budget or not selected:
            lines.append(entry)
            selected.append({**c, "slot": i})
            budget -= len(entry)
        else:
            break

    return "\n".join(lines), selected


def generate_answer(
    model_name: str,
    messages: List[Dict[str, str]],
    stream: bool = True,
) -> str:
    out: List[str] = []
    if stream:
        for part in ollama.chat(
            model=model_name, messages=messages, stream=True
        ):
            content = part.get("message", {}).get("content", "")
            if content:
                sys.stdout.write(content)
                sys.stdout.flush()
                out.append(content)
        print()  # newline after stream
    else:
        resp = ollama.chat(model=model_name, messages=messages)
        content = resp.get("message", {}).get("content", "")
        print(content)
        out.append(content)
    return "".join(out)


def chat_loop(index: RAGIndex, model_name: str = DEFAULT_LLM) -> None:
    print(f"[info] Using LLM via Ollama: {model_name}")
    history: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    last_sources: List[Dict[str, Any]] = []

    print("RAG chat ready. Type your question.")
    print("Commands: /sources, /rebuild, /clear, /quit")

    while True:
        try:
            q = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue

        lower = q.lower()
        if lower in ("/quit", "/exit"):
            break
        if lower == "/clear":
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("[info] History cleared.")
            continue
        if lower == "/sources":
            if not last_sources:
                print("[info] No sources to show yet.")
            else:
                print("Sources for last answer:")
                for s in last_sources:
                    base = Path(s["doc_id"]).name
                    ps = s["page_start"]
                    pe = s["page_end"]
                    ttl = s.get("title", base)
                    print(
                        f"  [{s['slot']}] {ttl} — {base} (pages {ps}-{pe})"
                    )
            continue
        if lower == "/rebuild":
            print("[info] Rebuilding index...")
            print(
                "[warn] Rebuild from chat isn't supported. "
                "Rerun the script with --rebuild."
            )
            continue

        print("[info] Retrieving relevant passages...")
        hits = index.search(q, topk_faiss=TOPK_FAISS, topk_final=TOPK_FINAL)
        context_str, selected = format_context_for_prompt(hits)
        last_sources = selected

        history_limited = history[-(2 * MAX_HISTORY_TURNS + 1):]
        user_msg = (
            f"Question: {q}\n\n"
            f"Context (numbered):\n{context_str}\n\n"
            "Instructions:\n"
            "- Use citations like [1], [2] referencing the numbered "
            "context items.\n"
            "- If information is insufficient, say you don't know.\n"
            "- Be concise."
        )
        messages = history_limited + [{"role": "user", "content": user_msg}]

        print("Assistant>", end=" ")
        ans = generate_answer(model_name, messages, stream=True)

        history.extend(
            [
                {"role": "user", "content": q},
                {"role": "assistant", "content": ans},
            ]
        )


# -----------------------
# Main
# -----------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local RAG chatbot over your PDF library"
    )
    parser.add_argument(
        "--books_dir",
        type=str,
        required=True,
        help="Directory containing PDFs",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default="rag_index",
        help="Where to store the FAISS index",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild index",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default=DEFAULT_LLM,
        help="Ollama model name (e.g., qwen2.5:7b-instruct)",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranking",
    )
    args = parser.parse_args()

    books_dir = Path(args.books_dir).expanduser().resolve()
    index_dir = Path(args.index_dir).expanduser().resolve()

    rerank_model = None if args.no_rerank else DEFAULT_RERANK_MODEL
    index = RAGIndex(
        index_dir,
        embed_model_name=DEFAULT_EMBED_MODEL,
        rerank_model_name=rerank_model,
    )

    need_build = (
        args.rebuild
        or not index_dir.exists()
        or not (
            index.faiss_path.exists() and index.meta_path.exists()
        )
    )
    if need_build:
        index.build(books_dir, ocr=not args.no_ocr)
    else:
        index.load()

    settings = {
        "books_dir": str(books_dir),
        "index_dir": str(index_dir),
        "embed_model": DEFAULT_EMBED_MODEL,
        "rerank_model": rerank_model,
        "llm": args.llm,
    }
    with (index_dir / "run_settings.json").open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

    # Let torch fall back when MPS ops are missing (macOS).
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    chat_loop(index, model_name=args.llm)


if __name__ == "__main__":
    main()
