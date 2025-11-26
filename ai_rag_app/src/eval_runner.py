from __future__ import annotations
from pathlib import Path
from datetime import datetime
import csv
import yaml

from ai_rag_app.src.config import DOCS_DIR, VSTORE_DIR
from ai_rag_app.src.index_docs import build_index
from ai_rag_app.src.rag_chain import answer

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
    # Ensure index exists / updated
    build_index(DOCS_DIR, VSTORE_DIR)

    questions = load_qs()
    rows = []
    for q in questions:
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

    out = REPORTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
