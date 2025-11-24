from __future__ import annotations
from typing import Dict, List
import re
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import DEFAULT_EMBED_MODEL

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def estimate_tokens(text: str) -> int:
    # rough ~4 chars per token heuristic
    return max(1, round(len(text) / 4))


def split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and len(s.strip()) > 2]
    return [s for s in sents if len(s) >= 20]


def get_model() -> SentenceTransformer:
    # loads once and reuses process-wide
    from functools import lru_cache

    @lru_cache
    def _load() -> SentenceTransformer:
        return SentenceTransformer(DEFAULT_EMBED_MODEL)

    return _load()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def score_relevance(question: str, contexts: List[str]) -> Dict[str, float]:
    """Cosine between question and mean context embedding."""
    if not contexts:
        return {"q_ctx_cosine": 0.0}
    m = get_model()
    q = m.encode([question], normalize_embeddings=True)[0]
    ctx = m.encode(contexts, normalize_embeddings=True)
    ctx_mean = ctx.mean(axis=0)
    return {"q_ctx_cosine": float(q @ ctx_mean)}


def score_support(answer: str, contexts: List[str], threshold: float = 0.6) -> Dict[str, float]:
    """Fraction of answer sentences supported by any retrieved sentence above threshold."""
    if not answer or not contexts:
        return {"support_rate": 0.0}
    m = get_model()
    ans_sents = split_sentences(answer)
    if not ans_sents:
        return {"support_rate": 0.0}

    # candidate sentences from contexts
    ctx_sents: List[str] = []
    for c in contexts:
        ctx_sents.extend(split_sentences(c))
    if not ctx_sents:
        return {"support_rate": 0.0}

    a_vecs = m.encode(ans_sents, normalize_embeddings=True)
    c_vecs = m.encode(ctx_sents, normalize_embeddings=True).T  # (dim, n_ctx)

    supported = 0
    for i in range(a_vecs.shape[0]):
        # max cosine against any retrieved sentence
        max_sim = float(a_vecs[i] @ c_vecs).max()
        if max_sim >= threshold:
            supported += 1

    rate = supported / len(ans_sents)
    return {"support_rate": round(rate, 3)}
