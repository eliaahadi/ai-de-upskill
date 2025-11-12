.PHONY: setup test fmt run-de run-rag index-rag flow

setup:
	uv venv && uv sync --all-extras && pre-commit install

test:
	uv run pytest -q

fmt:
	uv run ruff check . --fix && uv run black .

run-de:
	uv run streamlit run de_pipeline/app.py

flow:
	uv run python -m de_pipeline.flows.flow

run-rag:
	uv run uvicorn ai_rag_app.src.service:app --reload

index-rag:
	uv run python -m ai_rag_app.src.index_docs


---

## add make targets

Append these to your root `Makefile` (keep existing ones).

```make
clean:
	@echo "Cleaning local artifacts"
	rm -rf de_pipeline/duckdb/*.duckdb de_pipeline/data/staged/*.parquet ai_rag_app/vectorstore logs runs || true

reset: clean
	uv run python -m de_pipeline.flows.flow

docs:
	@echo "Docs live in ./docs. Render Mermaid in VS Code with a Mermaid plugin, or keep as markdown."
