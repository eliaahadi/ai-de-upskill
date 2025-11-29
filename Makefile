
SHELL := /bin/zsh
PY := python
PIP := pip

TEST := $(PY) -m pytest -q
DE_APP := streamlit run de_pipeline/app.py
FLOW := $(PY) -m de_pipeline.flows.flow
RAG_SERVE := uvicorn ai_rag_app.src.service:app --reload
RAG_UI := streamlit run ai_rag_app/ui_app.py
RAG_INDEX := $(PY) -m ai_rag_app.src.index_docs
LINT := ruff check . && black --check .
FMT := ruff check . --fix && black .

.PHONY: test de-app flow rag-serve rag-ui rag-index lint fmt install-edit help clean reset docs run-de run-rag run-flow index-rag

test:
	$(TEST)

de-app:
	$(DE_APP)

flow:
	$(FLOW)

rag-serve:
	$(RAG_SERVE)

rag-ui:
	$(RAG_UI)

rag-index:
	$(RAG_INDEX)

lint:
	$(LINT)

fmt:
	$(FMT)

install-edit:
	$(PIP) install -e .

ci: lint test
	@echo "CI: lint and tests completed"

clean:
	@echo "Cleaning local artifacts"
	@rm -rf de_pipeline/duckdb/*.duckdb de_pipeline/data/staged/*.parquet ai_rag_app/vectorstore logs runs || true

reset: clean
	$(FLOW)

docs:
	@echo "Docs live in ./docs. Render Mermaid in VS Code with a Mermaid plugin, or keep as markdown."

help:
	@echo "Makefile targets:"
	@echo "  make test        - run pytest"
	@echo "  make de-app      - run Streamlit data-engineering app"
	@echo "  make flow        - run the DAG/flow locally"
	@echo "  make rag-serve   - run the RAG uvicorn server"
	@echo "  make rag-ui      - run the RAG Streamlit UI"
	@echo "  make rag-index   - rebuild the RAG index"
	@echo "  make lint        - run ruff + black checks"
	@echo "  make fmt         - run ruff autofix + black format"
	@echo "  make install-edit - install package in editable mode"

# Backwards-compatible aliases (old Makefile used these names)
run-de: de-app

run-rag: rag-serve

run-flow: flow

index-rag: rag-index

run-rag-ui:
	uv run streamlit run ai_rag_app/ui/app.py

eval-rag:
	uv run python -m ai_rag_app.src.eval_runner

mlflow-ui:
	uv run mlflow ui --backend-store-uri $$MLFLOW_TRACKING_URI --port 5001
