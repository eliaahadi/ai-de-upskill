from __future__ import annotations

from functools import lru_cache
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import chromadb

from .config import VSTORE_DIR, COLLECTION_NAME
from .rag_chain import answer as rag_answer
from .retriever import get_collection

MAX_QUESTION_CHARS = 1500
MAX_K = 10

app = FastAPI(title="AI RAG Service", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def _client() -> chromadb.PersistentClient:
    VSTORE_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(VSTORE_DIR))


@lru_cache
def _collection():
    return _client().get_or_create_collection(COLLECTION_NAME)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict:
    try:
        count = _collection().count()
    except Exception:
        count = 0
    return {"collection": COLLECTION_NAME, "documents": count, "path": str(VSTORE_DIR)}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    k: int = 5
    mode: str = "extractive"
    eval: bool = False

    @field_validator("question")
    @classmethod
    def _cap_len(cls, v: str) -> str:
        v = v.strip()
        if len(v) > MAX_QUESTION_CHARS:
            raise ValueError(f"question too long (>{MAX_QUESTION_CHARS} chars)")
        return v

    @field_validator("k")
    @classmethod
    def _cap_k(cls, v: int) -> int:
        return max(1, min(int(v), MAX_K))


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if get_collection().count() == 0:
        raise HTTPException(
            status_code=503, detail="Vector store is empty. Add docs and run the indexer."
        )
    result = rag_answer(req.question, k=req.k, mode=req.mode, with_eval=req.eval)
    return result
