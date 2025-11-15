from __future__ import annotations

from functools import lru_cache
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import chromadb

from .config import VSTORE_DIR, COLLECTION_NAME

app = FastAPI(title="AI RAG Service", version="0.1.0")

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
