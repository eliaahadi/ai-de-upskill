from __future__ import annotations
from pathlib import Path
from itertools import product
from datetime import datetime
import csv
import mlflow

from ai_rag_app.src.config import DOCS_DIR, VSTORE_DIR
from ai_rag_app.src.index_docs import build_index_with_params
from ai_rag_app.src.rag_chain import answer
from ai_rag_app.src.mlflow_utils import init_mlflow, log_eval_params, log_eval_metrics

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SPACE = {
    "embed_model": [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
    ],
    "chunk_size": [600, 900],
    "chunk_overlap": [100, 150],
    "k": [3, 5],
}

QUESTIONS = [
    "What is a vector store used for?",
    "How does retrieval augmented generation work?",
]


def score_row(support_rate, q_ctx_cosine):
    # composite score (support more important)
    sr = support_rate or 0.0
    qc = q_ctx_cosine or 0.0
    return 0.7 * sr + 0.3 * max(0.0, min(1.0, qc))


def main() -> None:
    uri = init_mlflow("rag_sweep")
    print(f"[sweep] MLflow at {uri}")

    rows = []
    sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    with mlflow.start_run(run_name=f"sweep_{sweep_id}"):
        for embed_model, chunk_size, chunk_overlap, k in product(
            SPACE["embed_model"], SPACE["chunk_size"], SPACE["chunk_overlap"], SPACE["k"]
        ):
            # re-index with params
            build_index_with_params(
                DOCS_DIR,
                VSTORE_DIR,
                embed_model=embed_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            # one parent run per config
            cfg_name = f"m={embed_model.split('/')[-1]}_cs={chunk_size}_co={chunk_overlap}_k={k}"
            with mlflow.start_run(run_name=cfg_name, nested=True):
                log_eval_params(
                    embed_model=embed_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    k=k,
                    docs_dir=str(DOCS_DIR),
                    vstore_dir=str(VSTORE_DIR),
                )

                # average metrics over questions
                agg = {"retrieved": 0, "context_chars": 0, "support_rate": 0.0, "q_ctx_cosine": 0.0}
                per_q = []
                for q in QUESTIONS:
                    res = answer(q, k=k, mode="extractive", with_eval=True)
                    sr = (res.get("eval") or {}).get("support_rate")
                    qc = (res.get("eval") or {}).get("q_ctx_cosine")
                    per_q.append((q, sr, qc))
                    log_eval_metrics(
                        retrieved=res.get("retrieved", 0),
                        context_chars=res.get("context_chars", 0),
                        support_rate=sr,
                        q_ctx_cosine=qc,
                        answer_tokens_est=res.get("answer_tokens_est", 0),
                        question_tokens_est=res.get("question_tokens_est", 0),
                    )

                    agg["retrieved"] += res.get("retrieved", 0)
                    agg["context_chars"] += res.get("context_chars", 0)
                    agg["support_rate"] += sr or 0.0
                    agg["q_ctx_cosine"] += qc or 0.0

                n = max(1, len(QUESTIONS))
                agg = {
                    k2: (
                        v / n
                        if k2 in ("support_rate", "q_ctx_cosine", "retrieved", "context_chars")
                        else v
                    )
                    for k2, v in agg.items()
                }
                comp = score_row(agg["support_rate"], agg["q_ctx_cosine"])
                mlflow.log_metric("avg_support_rate", agg["support_rate"])
                mlflow.log_metric("avg_q_ctx_cosine", agg["q_ctx_cosine"])
                mlflow.log_metric("composite_score", comp)

                rows.append(
                    {
                        "embed_model": embed_model,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "k": k,
                        "avg_support_rate": round(agg["support_rate"], 3),
                        "avg_q_ctx_cosine": round(agg["q_ctx_cosine"], 3),
                        "composite_score": round(comp, 3),
                    }
                )

        # write CSV summary
        out = REPORTS_DIR / f"sweep_{sweep_id}.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: r["composite_score"], reverse=True))
        print(f"[sweep] wrote {out}")
        mlflow.log_artifact(str(out))


if __name__ == "__main__":
    main()
