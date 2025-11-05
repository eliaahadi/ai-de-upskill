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
