from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from prefect import flow, task

from de_pipeline.src.ingest import ingest_raw_to_stage
from de_pipeline.src.transform import build_models
from de_pipeline.src.metrics import write_metric


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
STAGED_DIR = BASE_DIR / "data" / "staged"
WAREHOUSE_DIR = BASE_DIR / "duckdb"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)


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
    print(f"[flow] raw={Path(raw_dir).resolve()}")
    print(f"[flow] staged={Path(staged_dir).resolve()}")
    print(f"[flow] warehouse={Path(warehouse_dir).resolve()}")

    t0 = time.perf_counter()
    t_ingest_raw(Path(raw_dir), Path(staged_dir))
    t_build_models(Path(warehouse_dir))
    elapsed = time.perf_counter() - t0

    run_log = {
        # timezone-aware UTC timestamp
        "run_at": (lambda dt: dt.replace("+00:00", "Z") if dt.endswith("+00:00") else dt)(
            datetime.now(timezone.utc).isoformat()
        ),
        "raw_dir": str(raw_dir),
        "staged_dir": str(staged_dir),
        "warehouse_dir": str(warehouse_dir),
        "elapsed_s": round(elapsed, 3),
    }
    (LOGS_DIR / f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json").write_text(
        json.dumps(run_log, indent=2)
    )
    write_metric({"step": "flow", "elapsed_s": round(elapsed, 3)})
    print("[flow] completed")


def main() -> None:
    run_flow()


if __name__ == "__main__":
    main()
