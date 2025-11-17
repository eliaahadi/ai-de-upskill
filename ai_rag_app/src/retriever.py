from __future__ import annotations
from typing import List, Tuple, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer

from .config import VSTORE_DIR, COLLECTION_NAME, DEFAULT_EMBED_MODEL


def get_collection():
    client = chromadb.PersistentClient(path=str(VSTORE_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def retrieve(query: str, k: int = 5) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Embed the query locally and search by vector. Returns [(doc_text, meta), ...]
    meta contains: source, chunk_index, id, distance, tokens_est (if present), etc.
    """
    col = get_collection()
    if col.count() == 0:
        return []

    model = SentenceTransformer(DEFAULT_EMBED_MODEL)
    q_emb = model.encode([query], normalize_embeddings=True).tolist()

    res = col.query(
        query_embeddings=q_emb,
        n_results=k,
        include=["documents", "metadatas", "distances", "ids"],
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0]

    out = []
    for i in range(len(docs)):
        m = (metas[i] or {}).copy()
        m.update({"id": ids[i], "distance": dists[i]})
        out.append((docs[i], m))
    return out
