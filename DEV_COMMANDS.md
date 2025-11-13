Developer command shortcuts

The project now includes a top-level `Makefile` with convenient targets that run the same developer commands
without depending on the `uv` pyproject integration. Prefer these `make` targets for local development.

Common targets:

- Run tests:
  make test

- Run the Streamlit app for the data engineering app:
  make de-app

- Run the DAG/flow locally:
  make flow

- Run the RAG service (uvicorn):
  make rag-serve

- Run the RAG UI (Streamlit):
  make rag-ui

- Rebuild the RAG index:
  make rag-index

- Lint check:
  make lint

- Format + autofix:
  make fmt

Developer environment helpers:

- Install the package in editable mode (recommended for tests & imports):
  make install-edit

- Clean artifacts:
  make clean

If you previously used `uv` scripts, these `make` targets are equivalent and avoid parsing ambiguities with `pyproject.toml`.

If you prefer (or need) the raw commands, they're listed below for reference:

```sh
pytest -q
streamlit run de_pipeline/app.py
python -m de_pipeline.flows.flow
uvicorn ai_rag_app.src.service:app --reload
streamlit run ai_rag_app/ui_app.py
python -m ai_rag_app.src.index_docs
ruff check . && black --check .
ruff check . --fix && black .
```

Want a reminder of available make targets? Run:

```
make help
```
