from __future__ import annotations

# from pathlib import Path
import shutil

from fastapi.testclient import TestClient

from ai_rag_app.src.service import app, MAX_QUESTION_CHARS
from ai_rag_app.src.config import DOCS_DIR, VSTORE_DIR
from ai_rag_app.src.index_docs import build_index


def setup_module(module=None):
    if VSTORE_DIR.exists():
        shutil.rmtree(VSTORE_DIR)
    VSTORE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "sample.md").write_text(
        "# retrieval\n\nRetrieval augmented generation uses top-k document chunks "
        "from a vector store to ground answers and provide citations.",
        encoding="utf-8",
    )
    build_index(DOCS_DIR, VSTORE_DIR)


def test_eval_metrics_present() -> None:
    c = TestClient(app)
    r = c.post(
        "/ask", json={"question": "what is retrieval augmented generation", "k": 3, "eval": True}
    )
    assert r.status_code == 200
    body = r.json()
    assert "eval" in body and "q_ctx_cosine" in body["eval"] and "support_rate" in body["eval"]
    assert 0.0 <= body["eval"]["support_rate"] <= 1.0
    assert isinstance(body.get("sources"), list) and len(body["sources"]) > 0


def test_guardrails_question_too_long() -> None:
    c = TestClient(app)
    long_q = "x" * (MAX_QUESTION_CHARS + 5)
    r = c.post("/ask", json={"question": long_q})
    assert r.status_code in (400, 422)
