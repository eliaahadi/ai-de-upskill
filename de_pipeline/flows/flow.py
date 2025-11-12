from __future__ import annotations

from pathlib import Path

from prefect import flow, task

from de_pipeline.src.ingest import ingest_raw_to_stage
from de_pipeline.src.transform import build_models

from datetime import datetime
import json

BASE_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
RAW_DIR = BASE_DIR / "data" / "raw"
STAGED_DIR = BASE_DIR / "data" / "staged"
WAREHOUSE_DIR = BASE_DIR / "duckdb"


@task(name="ingest_raw", log_prints=True)
def t_ingest_raw(raw_dir: Path, staged_dir: Path) -> None:
    ingest_raw_to_stage(raw_dir, staged_dir)


@task(name="build_models", log_prints=True)
def t_build_models(warehouse_dir: Path) -> None:
    build_models(warehouse_dir)


@flow(name="de_pipeline_local_flow", log_prints=True)
def run_flow(
    raw_dir: str | Path = RAW_DIR,
    staged_dir: str | Path = STAGED_DIR,
    warehouse_dir: str | Path = WAREHOUSE_DIR,
) -> None:
    """
    Day 4 orchestration

    Steps:
    1. Ingest raw CSVs → staged parquet
    2. Build dim/fact tables in DuckDB warehouse
    """
    raw_dir = Path(raw_dir)
    staged_dir = Path(staged_dir)
    warehouse_dir = Path(warehouse_dir)

    print(f"[flow] raw_dir={raw_dir.resolve()}")
    print(f"[flow] staged_dir={staged_dir.resolve()}")
    print(f"[flow] warehouse_dir={warehouse_dir.resolve()}")

    t_ingest_raw(raw_dir, staged_dir)
    t_build_models(warehouse_dir)

    # write a small run log
    run_log = {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "raw_dir": str(raw_dir),
        "staged_dir": str(staged_dir),
        "warehouse_dir": str(warehouse_dir),
    }
    (LOGS_DIR / f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json").write_text(
        json.dumps(run_log, indent=2)
    )

    print("[flow] Pipeline completed successfully.")


def main() -> None:
    run_flow()


if __name__ == "__main__":
    main()
