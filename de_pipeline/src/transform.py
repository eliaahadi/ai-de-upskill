from __future__ import annotations
import pathlib

def build_models(warehouse_path: str | pathlib.Path) -> None:
    warehouse_path = pathlib.Path(warehouse_path)
    warehouse_path.mkdir(parents=True, exist_ok=True)
    # Day 3: create dim_* and fact_* in DuckDB; add pandera schema checks.
    print(f"[transform] Using warehouse at {warehouse_path.resolve()}")
    print("[transform] Stub complete (add DuckDB DDL/DML and pandera checks on Day 3).")

def main() -> None:
    here = pathlib.Path(__file__).resolve().parents[1]
    build_models(here / "duckdb")

if __name__ == "__main__":
    main()
