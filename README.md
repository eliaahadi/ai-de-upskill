# AI + data engineering upskill workspace

Two small projects in one repo, tuned for a 2‑week daily plan. Local‑first, $0 stack on macOS.

- `de_pipeline/` DuckDB + Polars batch pipeline with Prefect and a Streamlit dashboard
- `ai_rag_app/` FastAPI RAG service with Chroma + sentence-transformers and optional Streamlit UI

## Quickstart

```
# 1) create venv and install
uv venv
uv sync --all-extras
pre-commit install

# 2) smoke test
uv run pytest -q

# 3) run empty dashboard (Day 1 check)
make run-de

# 4) run RAG API health (Day 8 check)
make run-rag
# visit http://127.0.0.1:8000/health
```

## Notes
Directories use underscores (e.g., `de_pipeline`) so you can run modules with `-m` cleanly.
Fill in each file per the day-by-day plan you have. All scripts are stubbed and safe to run now.
