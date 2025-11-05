from __future__ import annotations
import pathlib

def ingest_raw_to_stage(raw_dir: str | pathlib.Path, staged_dir: str | pathlib.Path) -> None:
    raw_dir = pathlib.Path(raw_dir)
    staged_dir = pathlib.Path(staged_dir)
    staged_dir.mkdir(parents=True, exist_ok=True)
    # Day 2: read CSVs with Polars and write staged files; for now, just a stub.
    print(f"[ingest] Looking for raw files in {raw_dir.resolve()}")
    print(f"[ingest] Writing staged outputs to {staged_dir.resolve()}")
    print("[ingest] Stub complete (add Polars + DuckDB work on Day 2).")

def main() -> None:
    here = pathlib.Path(__file__).resolve().parents[1]
    ingest_raw_to_stage(here / "data" / "raw", here / "data" / "staged")

if __name__ == "__main__":
    main()
