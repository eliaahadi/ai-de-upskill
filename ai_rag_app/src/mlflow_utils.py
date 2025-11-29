from __future__ import annotations
import os
from pathlib import Path
import mlflow


def init_mlflow(experiment: str = "rag_eval") -> str:
    """Configure tracking URI and experiment. Returns the resolved URI."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"file:{Path.home()}/.cache/ai_rag_app/mlruns")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return uri


def log_eval_params(
    *, embed_model: str, chunk_size: int, chunk_overlap: int, k: int, docs_dir: str, vstore_dir: str
) -> None:
    mlflow.log_params(
        {
            "embed_model": embed_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "k": k,
            "docs_dir": docs_dir,
            "vstore_dir": vstore_dir,
        }
    )


def log_eval_metrics(
    *,
    retrieved: int,
    context_chars: int,
    support_rate: float | None,
    q_ctx_cosine: float | None,
    answer_tokens_est: int,
    question_tokens_est: int,
) -> None:
    metrics = {
        "retrieved": retrieved,
        "context_chars": context_chars,
        "answer_tokens_est": answer_tokens_est,
        "question_tokens_est": question_tokens_est,
    }
    if support_rate is not None:
        metrics["support_rate"] = float(support_rate)
    if q_ctx_cosine is not None:
        metrics["q_ctx_cosine"] = float(q_ctx_cosine)
    mlflow.log_metrics(metrics)
