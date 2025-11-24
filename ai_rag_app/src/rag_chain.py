from __future__ import annotations
from typing import List, Dict, Any, Tuple
import re
import numpy as np
from sentence_transformers import SentenceTransformer

from .retriever import retrieve
from .config import DEFAULT_EMBED_MODEL
from .eval import estimate_tokens, score_relevance, score_support

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and len(s.strip()) > 2]
    return [s for s in sents if len(s) >= 20]


def _extractive_answer(question: str, contexts: List[str], top_sentences: int = 6) -> str:
    model = SentenceTransformer(DEFAULT_EMBED_MODEL)
    indexed: List[Tuple[int, int, str]] = []
    for ci, ctx in enumerate(contexts):
        for si, s in enumerate(_split_sentences(ctx)):
            indexed.append((ci, si, s))
    if not indexed:
        return "I couldn't find enough grounded context to answer."

    q_vec = model.encode([question], normalize_embeddings=True)[0]
    s_vecs = model.encode([s for (_, _, s) in indexed], normalize_embeddings=True)
    scores = [float(q_vec @ s_vec) for s_vec in s_vecs]
    top_idx = np.argsort(scores)[-top_sentences:][::-1]
    picked = [indexed[i] + (scores[i],) for i in top_idx]
    picked.sort(key=lambda x: (x[0], x[1]))
    lines = [p[2] for p in picked]

    dedup: List[str] = []
    seen = set()
    for line in lines:
        key = line.lower()
        if key not in seen:
            dedup.append(line)
            seen.add(key)
    return " ".join(dedup)


def answer(
    question: str, k: int = 5, mode: str = "extractive", with_eval: bool = False
) -> Dict[str, Any]:
    hits = retrieve(question, k=k)
    if not hits:
        return {
            "answer": "Index is empty or nothing relevant was found. Try adding docs and re-indexing.",
            "sources": [],
            "retrieved": 0,
            "mode": mode,
        }

    contexts = [doc for (doc, _meta) in hits]
    ans = (
        _extractive_answer(question, contexts) if mode == "extractive" else "Mode not implemented."
    )

    sources = []
    for _doc, meta in hits:
        sources.append(
            {
                "source": meta.get("source"),
                "chunk_index": meta.get("chunk_index"),
                "id": meta.get("id"),
                "distance": meta.get("distance"),
                "tokens_est": meta.get("tokens_est"),
            }
        )

    payload: Dict[str, Any] = {
        "answer": ans,
        "sources": sources,
        "retrieved": len(hits),
        "mode": mode,
        "context_chars": sum(len(c) for c in contexts),
        "answer_tokens_est": estimate_tokens(ans),
        "question_tokens_est": estimate_tokens(question),
    }

    if with_eval:
        rel = score_relevance(question, contexts)  # {"q_ctx_cosine": ...}
        sup = score_support(ans, contexts, threshold=0.6)  # {"support_rate": ...}
        payload["eval"] = {**rel, **sup}
        # simple flags
        payload["flags"] = {
            "low_support": sup["support_rate"] < 0.5,
            "low_relevance": rel["q_ctx_cosine"] < 0.4,
        }

    return payload
