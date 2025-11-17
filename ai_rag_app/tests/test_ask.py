from __future__ import annotations
import shutil

from fastapi.testclient import TestClient

from ai_rag_app.src.service import app
from ai_rag_app.src.config import DOCS_DIR, VSTORE_DIR
from ai_rag_app.src.index_docs import build_index


def setup_module(module=None):
    # reset store and add a tiny doc
    if VSTORE_DIR.exists():
        shutil.rmtree(VSTORE_DIR)
    VSTORE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "sample.md").write_text(
        "# retrieval\n\nVector databases like Chroma store document chunks and enable similarity search for RAG.",
        encoding="utf-8",
    )
    build_index(DOCS_DIR, VSTORE_DIR)


def test_ask_returns_answer_and_sources():
    c = TestClient(app)
    r = c.post("/ask", json={"question": "What is a vector store used for?", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body.get("answer"), str) and len(body["answer"]) > 0
    assert isinstance(body.get("sources"), list) and len(body["sources"]) > 0
