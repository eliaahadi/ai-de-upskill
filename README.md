# 2-week ai + data engineering upskill plan

[![CI](https://github.com/eliaahadi/ai-de-upskill/actions/workflows/ci.yml/badge.svg)](https://github.com/eliaahadi/ai-de-upskill/actions)

Two small projects in one repo, tuned for a 2‑week daily plan. Local‑first, $0 stack on macOS.

Focus
- Build a small data engineering pipeline with DuckDB, Polars, validation, orchestration, and a Streamlit dashboard.
- Build a minimal RAG microservice with FastAPI, Chroma, sentence-transformers, and an optional Streamlit UI.
- Local-first, $0, macOS friendly.

Repo
- `ai-de-upskill/`
  - `de_pipeline/`
  - `ai_rag_app/`

---
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

# project overview

You have two small projects in one workspace. Everything runs locally with uv.

- `de_pipeline` batch pipeline using DuckDB and Polars with a Streamlit dashboard
- `ai_rag_app` minimal RAG service with FastAPI and Chroma (wired on days 8–12)

### data engineering flow
graph LR
A[Raw CSV in data/raw] --> B[Polars ingest]
B --> C[Staged parquet in data/staged]
C --> D[DuckDB view stg_jobs]
D --> E[dim_company • dim_location • dim_job_title]
D --> F[fact_job_postings]
F --> G[Streamlit dashboard]

### rag microservice
graph LR
D[Docs in data/docs] --> CH[Chunker]
CH --> EM[Embeddings]
EM --> VS[Chroma vector store]
Q[User question] --> R[Retriever]
R --> CTX[Top-k context]
CTX --> LLM[Generator]
LLM --> A[Answer + sources]
## quick start

```bash
uv venv
uv sync --all-extras
pre-commit install

# build warehouse and run tests
uv run python -m de_pipeline.flows.flow
uv run pytest -q

# run dashboard
uv run streamlit run de_pipeline/app.py

## day 1 – environment and scaffolding

**Goal** have the workspace running tests and UIs booting.

**Tasks**
- Create `ai-de-upskill/` with `de_pipeline/` and `ai_rag_app/`.
- Set up `uv` (or your preferred tool), install dependencies, enable `pre-commit`.
- Run the provided smoke tests and Streamlit apps.
- Update root `README.md` with your own goals and notes.

**Outcome**
- `uv run pytest -q` passes.
- `make run-de` and `make run-rag` both start successfully.

---

## day 2 – ingest with duckdb and polars

**Goal** load raw data into a structured staged layer.

**Tasks**
- Choose a public csv dataset (e.g. trips, sales, health).
- Place files into `de_pipeline/data/raw`.
- Implement `ingest_raw_to_stage` to:
  - Read csv with Polars.
  - Clean basic types.
  - Write staged parquet/csv or DuckDB tables.

**Outcome**
- `make flow` (or running the flow module) logs successful ingest to `data/staged`.

---

## day 3 – transform and data quality

**Goal** create modeled tables and basic validations.

**Tasks**
- Design at least one `dim_` and one `fact_` table (e.g. `dim_date`, `fact_trips`).
- Implement `build_models` to create those tables in DuckDB.
- Add `pandera` (or similar) checks in tests for row counts, nulls, and expected ranges.

**Outcome**
- Tests validate your models.
- You can explain your dim/fact choices.

---

## day 4 – orchestration with prefect

**Goal** run ingest → transform → validate as one flow.

**Tasks**
- Use Prefect to define a `de_pipeline_local_flow`.
- Add tasks for ingest and transform.
- Parameterize input/output paths.

**Outcome**
- Single command flow run (e.g. `python -m de_pipeline.flows.flow`) completes end to end.

---

## day 5 – streamlit dashboard

**Goal** visualize key metrics from DuckDB.

**Tasks**
- Connect Streamlit to your DuckDB warehouse.
- Show:
  - 1–2 headline metrics.
  - At least one time series chart.
  - One filter (date, category, etc.).

**Outcome**
- Live dashboard backed by your pipeline outputs.

---

## day 6 – packaging and docs

**Goal** make this feel like a real, reusable mini-project.

**Tasks**
- Tighten `pyproject.toml` and scripts.
- Ensure `make setup`, `make test`, `make run-de` work on a fresh clone.
- Add diagrams:
  - Simple flow from raw → stage → models → dashboard.
- Update `README.md` with quickstart, stack, and assumptions.

**Outcome**
- Someone else could run your pipeline in under 5 minutes using the docs.

---

## day 7 – performance and reliability pass

**Goal** tune one thing and show you think like an engineer.

**Tasks**
- Benchmark one transform before/after using Polars or better filters/joins.
- Add minimal logging of durations for each step.
- Document one improvement and one future enhancement.

**Outcome**
- Micro “case study” section in your README.

---

## day 8 – rag project setup

**Goal** scaffold the RAG microservice.

**Tasks**
- In `ai_rag_app`, ensure FastAPI app with `/health` is running.
- Confirm project structure:
  - `src/service.py`
  - `src/index_docs.py`
  - `src/rag_chain.py`
  - `data/docs/`
- Drop a few sample pdf/md files into `data/docs`.

**Outcome**
- `make run-rag` returns `{"status": "ok"}` on `/health`.

---

## day 9 – chunking and embeddings

**Goal** build the vector store.

**Tasks**
- Implement `build_index` to:
  - Read files from `data/docs`.
  - Chunk text (e.g. 400–800 tokens with overlap).
  - Embed with `sentence-transformers` (e.g. `all-MiniLM-L6-v2`).
  - Store in Chroma at `vectorstore/`.

**Outcome**
- Script prints number of docs/chunks.
- Vector store directory populated.

---

## day 10 – retrieval and answer generation

**Goal** wire up `/ask` with retrieval and generation.

**Tasks**
- Implement `rag_chain.answer(question)`:
  - Retrieve top-k chunks.
  - Build a prompt including context.
  - Call your chosen model (local or API, depending on your rules).
- Update `/ask` endpoint to:
  - Return answer text.
  - Include sources (chunk metadata).
  - Include simple token/length info if available.

**Outcome**
- You can ask domain questions and get grounded answers with sources.

---

## day 11 – evaluation and guardrails

**Goal** make RAG less hand-wavy.

**Tasks**
- Add simple evaluation:
  - E.g. overlap between answer and retrieved context.
  - Flag empty or hallucinated answers.
- Add tests to assert:
  - Non-empty sources for known queries.
  - No crash when store missing or empty.
- Add basic error messages for bad input.

**Outcome**
- `pytest` includes RAG tests and passes.

---

## day 12 – streamlit ui for rag

**Goal** make it demo-ready.

**Tasks**
- Update `ui_app.py` to:
  - Let user ask a question.
  - Call FastAPI `/ask`.
  - Show answer, sources, and any eval metrics.
- Add a second panel or expander for debug info.

**Outcome**
- Clean, minimal interface for showing off your RAG system.

---

## day 13 – docker and developer experience

**Goal** run both apps predictably anywhere.

**Tasks**
- Add `Dockerfile` for:
  - `de_pipeline` dashboard.
  - `ai_rag_app` API.
- Add Make targets:
  - `make de-docker`
  - `make rag-docker` (or similar).
- Verify containers start and endpoints work locally.

**Outcome**
- Containerized versions that match your docs.

---

## day 14 – polish and storytelling

**Goal** convert work into portfolio signal.

**Tasks**
- Record 2–3 minute screen capture per project:
  - What it does.
  - How it is built.
  - How you would extend it.
- Finalize READMEs:
  - Problem, approach, stack, key decisions.
  - Links to diagrams and videos.
- Write a short summary you can paste into LinkedIn or your resume.

**Outcome**
- Two polished projects ready to share with recruiters, hiring managers, or clients.

---

## daily checklist

Use this each day.

- [ ] two-line plan for today
- [ ] run tests and fix red
- [ ] one commit with a clear message
- [ ] one measurable outcome written down
- [ ] one note for tomorrow
