from __future__ import annotations
from typing import List, Dict, Any, Tuple
import re
import numpy as np
from sentence_transformers import SentenceTransformer

from .retriever import retrieve
from .config import DEFAULT_EMBED_MODEL


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and len(s.strip()) > 2]
    # keep reasonable-length sentences
    return [s for s in sents if len(s) >= 20]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _extractive_answer(question: str, contexts: List[str], top_sentences: int = 6) -> str:
    """
    Rank sentences by cosine similarity to the question and return the top few,
    re-sorted by their original order for readability.
    """
    model = SentenceTransformer(DEFAULT_EMBED_MODEL)

    # collect sentences with (chunk_idx, sent_idx, text)
    indexed: List[Tuple[int, int, str]] = []
    for ci, ctx in enumerate(contexts):
        for si, s in enumerate(_split_sentences(ctx)):
            indexed.append((ci, si, s))

    if not indexed:
        return "I couldn't find enough grounded context to answer."

    q_vec = model.encode([question], normalize_embeddings=True)[0]
    s_vecs = model.encode([s for (_, _, s) in indexed], normalize_embeddings=True)

    scores = [float(q_vec @ s_vec) for s_vec in s_vecs]
    # pick top N by score
    top_idx = np.argsort(scores)[-top_sentences:][::-1]
    picked = [indexed[i] + (scores[i],) for i in top_idx]  # (ci, si, s, score)

    # sort back by (chunk, sentence) so it reads well
    picked.sort(key=lambda x: (x[0], x[1]))
    lines = [p[2] for p in picked]

    # small de-dup
    dedup: List[str] = []
    seen = set()
    for line in lines:
        key = line.lower()
        if key not in seen:
            dedup.append(line)
            seen.add(key)

    return " ".join(dedup)


def answer(question: str, k: int = 5, mode: str = "extractive") -> Dict[str, Any]:
    """
    Retrieve top-k and produce an answer.
    mode: "extractive" (default). You can add "openai"/"hf" later if desired.
    """
    hits = retrieve(question, k=k)
    if not hits:
        return {
            "answer": "Index is empty or nothing relevant was found. Try adding docs and re-indexing.",
            "sources": [],
            "retrieved": 0,
            "mode": mode,
        }

    contexts = [doc for (doc, _meta) in hits]
    answer_text = (
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

    return {
        "answer": answer_text,
        "sources": sources,
        "retrieved": len(hits),
        "mode": mode,
        "context_chars": sum(len(c) for c in contexts),
    }
