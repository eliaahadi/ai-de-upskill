from __future__ import annotations
from fastapi.testclient import TestClient
from ai_rag_app.src.service import app


def test_health() -> None:
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_stats() -> None:
    c = TestClient(app)
    r = c.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert "collection" in body and "documents" in body
