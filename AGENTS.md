AGENTS.md — guidelines for agents operating in this repo

- Build / Run: use `python -m pip install -e .[dev]` to install dev deps.
- Start CLI: `rag-cli --books_dir PATH --index_dir ./rag_index` or `python -m rag_cli --books_dir PATH`.
- Lint / Format: `ruff check .` and `black .` (Black line-length = 79, double quotes).
- Tests: this repo has no tests; run a single script with `python -m <module>`.
- Single-test run: if tests are added, use `pytest path/to/test_file.py::test_name -q`.
- Style: follow Black/ruff settings in `pyproject.toml` (79 cols, double quotes).
- Imports: standard library first, then 3rd-party, then local; use absolute imports.
- Typing: add `typing` annotations for public functions and classes; prefer `Optional[...]`.
- Naming: use snake_case for functions/variables, PascalCase for classes, UPPER_SNAKE for constants.
- Errors: raise specific exceptions; use `print(..., file=sys.stderr)` for CLI warnings.
- Logging: prefer simple `print` in CLI scripts; use `logging` for libraries.
- Long lines: wrap f-strings across lines to avoid >79 columns.
- IO: prefer `pathlib.Path` for filesystem paths.
- Binary assets: do not commit generated indices or OCR outputs; they are in `.gitignore`.
- Cursor / Copilot: none found; if added, include rules from `.cursor/rules/` or `.github/copilot-instructions.md`.
- Commits: be minimal and focused; tests & linters should pass locally before pushing.
