from __future__ import annotations
from prefect import flow, task
from pathlib import Path
from de_pipeline.src.ingest import ingest_raw_to_stage
from de_pipeline.src.transform import build_models

@task
def t_ingest(raw: Path, staged: Path) -> None:
    ingest_raw_to_stage(raw, staged)

@task
def t_transform(warehouse: Path) -> None:
    build_models(warehouse)

@flow(name="de_pipeline_local_flow")
def run_flow(raw_dir: str | Path = "de_pipeline/data/raw",
             staged_dir: str | Path = "de_pipeline/data/staged",
             warehouse_dir: str | Path = "de_pipeline/duckdb") -> None:
    t_ingest(Path(raw_dir), Path(staged_dir))
    t_transform(Path(warehouse_dir))

if __name__ == "__main__":
    run_flow()
