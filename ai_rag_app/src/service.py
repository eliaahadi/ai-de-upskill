from __future__ import annotations

from functools import lru_cache
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import chromadb

from .config import VSTORE_DIR, COLLECTION_NAME
from .rag_chain import answer as rag_answer
from .retriever import get_collection


app = FastAPI(title="AI RAG Service", version="0.2.0")

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


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    # ensure there is an index
    if get_collection().count() == 0:
        raise HTTPException(
            status_code=503, detail="Vector store is empty. Add docs and run the indexer."
        )
    result = rag_answer(req.question, k=req.k, mode=req.mode)
    return result
