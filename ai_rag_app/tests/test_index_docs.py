from __future__ import annotations
import shutil

import pytest

from ai_rag_app.src.index_docs import build_index
from ai_rag_app.src.config import DOCS_DIR, VSTORE_DIR
from ai_rag_app.src.retriever import get_collection


@pytest.fixture(autouse=True, scope="module")
def clean_store():
    # isolate this test run (optional)
    if VSTORE_DIR.exists():
        shutil.rmtree(VSTORE_DIR)
    VSTORE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    # create a tiny sample doc
    (DOCS_DIR / "sample.md").write_text(
        "# sample\n\nThis is a tiny test document about vector stores and retrieval.\n\nRAG improves answers with sources.",
        encoding="utf-8",
    )
    yield
    # do not delete to allow manual inspection after tests


def test_build_index_and_stats() -> None:
    build_index(DOCS_DIR, VSTORE_DIR)
    col = get_collection()
    assert col.count() > 0
