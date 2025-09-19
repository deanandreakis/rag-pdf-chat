# rag-pdf-chat

Local RAG (retrieval-augmented generation) chatbot for a library of PDFs.
Builds a FAISS index of PDF text chunks and queries a local LLM via Ollama
so answers are grounded in the supplied document context only.

Quickstart

- Install dev deps: `python -m pip install -e .[dev]`
- Build index & run CLI:
  - `python -m rag_cli --books_dir /path/to/pdfs --index_dir ./rag_index`
  - or `rag-cli --books_dir /path/to/pdfs --index_dir ./rag_index`

Development

- Lint: `ruff check .`
- Format: `black .` (line-length = 79, double quotes)
- Tests: this repo has no tests yet; run a single script with
  `python -m <module>`. If you add tests, run a single test with
  `pytest path/to/test_file.py::test_name -q`

Notes

- Text extraction: `pypdf` with PyMuPDF (`fitz`) fallback; optional OCR via
  `ocrmypdf` when PDFs are image-only.
- Do not commit generated FAISS indices, OCR caches, or other artifacts —
  these are listed in `.gitignore`.
- See `AGENTS.md` for guidelines aimed at automated agents and contributors.

License

See the `LICENSE` file in the repository root.
