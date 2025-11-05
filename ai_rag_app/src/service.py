from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="AI RAG Service", version="0.1.0")

class AskRequest(BaseModel):
    question: str

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskRequest) -> dict:
    # Day 10: wire to retrieval + generation pipeline.
    # For Day 1 health check, we raise until index exists.
    raise HTTPException(status_code=503, detail="Vector store not built yet. Run `uv run rag-index` on Day 9–10.")
