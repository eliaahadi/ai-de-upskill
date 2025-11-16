from __future__ import annotations
from typing import List, Tuple
import chromadb
from .config import VSTORE_DIR, COLLECTION_NAME


def get_collection():
    client = chromadb.PersistentClient(path=str(VSTORE_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def retrieve(query: str, k: int = 5) -> List[Tuple[str, dict]]:
    col = get_collection()
    res = col.query(
        query_texts=[query], n_results=k, include=["documents", "metadatas", "distances", "ids"]
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0]
    out = []
    for i in range(len(docs)):
        m = metas[i] or {}
        m.update({"id": ids[i], "distance": dists[i]})
        out.append((docs[i], m))
    return out
