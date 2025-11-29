from __future__ import annotations
from pathlib import Path
from datetime import datetime
import csv
import yaml
import mlflow

from ai_rag_app.src.config import (
    DOCS_DIR,
    VSTORE_DIR,
    DEFAULT_EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from ai_rag_app.src.index_docs import build_index
from ai_rag_app.src.rag_chain import answer
from ai_rag_app.src.mlflow_utils import init_mlflow, log_eval_params, log_eval_metrics

QA_FILE = Path(__file__).resolve().parents[1] / "data" / "qa" / "qa.yml"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_qs() -> list[str]:
    if not QA_FILE.exists():
        return ["What do these docs say about vector stores?"]
    data = yaml.safe_load(QA_FILE.read_text(encoding="utf-8"))
    out: list[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict) and "q" in item:
                out.append(str(item["q"]))
    return out


def main() -> None:
    # ensure index
    build_index(DOCS_DIR, VSTORE_DIR)

    # initialize mlflow
    uri = init_mlflow("rag_eval")
    print(f"[eval] MLflow tracking at {uri}")

    questions = load_qs()
    rows = []

    run_name = f"batch_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        # shared params
        log_eval_params(
            embed_model=DEFAULT_EMBED_MODEL,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            k=5,
            docs_dir=str(DOCS_DIR),
            vstore_dir=str(VSTORE_DIR),
        )

        for i, q in enumerate(questions):
            with mlflow.start_run(run_name=f"q{i+1}", nested=True):
                res = answer(q, k=5, mode="extractive", with_eval=True)

                rows.append(
                    {
                        "question": q,
                        "answer": res.get("answer", ""),
                        "retrieved": res.get("retrieved", 0),
                        "context_chars": res.get("context_chars", 0),
                        "support_rate": (res.get("eval") or {}).get("support_rate"),
                        "q_ctx_cosine": (res.get("eval") or {}).get("q_ctx_cosine"),
                    }
                )

                log_eval_metrics(
                    retrieved=res.get("retrieved", 0),
                    context_chars=res.get("context_chars", 0),
                    support_rate=(res.get("eval") or {}).get("support_rate"),
                    q_ctx_cosine=(res.get("eval") or {}).get("q_ctx_cosine"),
                    answer_tokens_est=res.get("answer_tokens_est", 0),
                    question_tokens_est=res.get("question_tokens_est", 0),
                )
                mlflow.log_param("question", q)
                mlflow.log_metric("answer_len_chars", len(res.get("answer", "")))

        # write and log report
        out = REPORTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[eval] wrote {out}")
        mlflow.log_artifact(str(out))


if __name__ == "__main__":
    main()
