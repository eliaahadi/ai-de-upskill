from __future__ import annotations

import time
import pathlib
from typing import Set

import duckdb
import polars as pl
from de_pipeline.src.metrics import write_metric


def _get_staged_file(staged_dir: pathlib.Path) -> pathlib.Path:
    preferred = staged_dir / "stg_ai_job_market.parquet"
    if preferred.exists():
        return preferred
    files = list(staged_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No staged parquet files in {staged_dir.resolve()}")
    return files[0]


def _load_staged_df(staged_path: pathlib.Path) -> pl.DataFrame:
    df = pl.read_parquet(staged_path)
    print(f"[transform] Loaded {staged_path.name} ({df.height} rows, {df.width} cols)")
    return df


def _create_dim_tables(con: duckdb.DuckDBPyConnection, cols: Set[str]) -> None:
    if "company_name" in cols:
        con.execute(
            """
            CREATE OR REPLACE TABLE dim_company AS
            SELECT ROW_NUMBER() OVER () AS company_id, company_name
            FROM (SELECT DISTINCT company_name FROM stg_jobs
                  WHERE company_name IS NOT NULL AND company_name <> '')
            """
        )
        print("[transform] dim_company")

    if "location" in cols:
        con.execute(
            """
            CREATE OR REPLACE TABLE dim_location AS
            SELECT ROW_NUMBER() OVER () AS location_id, location
            FROM (SELECT DISTINCT location FROM stg_jobs
                  WHERE location IS NOT NULL AND location <> '')
            """
        )
        print("[transform] dim_location")

    if "job_title" in cols and "experience_level" in cols:
        con.execute(
            """
            CREATE OR REPLACE TABLE dim_job_title AS
            SELECT ROW_NUMBER() OVER () AS job_title_id, job_title, experience_level
            FROM (SELECT DISTINCT job_title, experience_level
                  FROM stg_jobs WHERE job_title IS NOT NULL AND job_title <> '')
            """
        )
        print("[transform] dim_job_title (with experience_level)")
    elif "job_title" in cols:
        con.execute(
            """
            CREATE OR REPLACE TABLE dim_job_title AS
            SELECT ROW_NUMBER() OVER () AS job_title_id, job_title
            FROM (SELECT DISTINCT job_title
                  FROM stg_jobs WHERE job_title IS NOT NULL AND job_title <> '')
            """
        )
        print("[transform] dim_job_title")


def _create_fact_table(con: duckdb.DuckDBPyConnection, cols: Set[str]) -> None:
    select_parts = ["j.*"]
    joins = []

    if "company_name" in cols:
        select_parts.append("dc.company_id")
        joins.append("LEFT JOIN dim_company dc ON j.company_name = dc.company_name")

    if "location" in cols:
        select_parts.append("dl.location_id")
        joins.append("LEFT JOIN dim_location dl ON j.location = dl.location")

    if "job_title" in cols:
        select_parts.append("djt.job_title_id")
        joins.append("LEFT JOIN dim_job_title djt ON j.job_title = djt.job_title")

    sql = f"""
        CREATE OR REPLACE TABLE fact_job_postings AS
        SELECT {", ".join(select_parts)} FROM stg_jobs j {' '.join(joins)}
    """
    con.execute(sql)
    print("[transform] fact_job_postings")


def build_models(warehouse_dir: str | pathlib.Path) -> None:
    here = pathlib.Path(__file__).resolve().parents[1]
    staged_dir = here / "data" / "staged"
    staged_path = _get_staged_file(staged_dir)
    df = _load_staged_df(staged_path)
    cols = set(df.columns)

    warehouse_dir = pathlib.Path(warehouse_dir)
    warehouse_dir.mkdir(parents=True, exist_ok=True)
    db_path = warehouse_dir / "warehouse.duckdb"

    t0 = time.perf_counter()
    con = duckdb.connect(str(db_path))
    con.execute("PRAGMA threads = 4")  # tweak if you want, DuckDB auto-scales too
    con.register("stg_jobs", df.to_arrow())

    _create_dim_tables(con, cols)
    _create_fact_table(con, cols)

    fact_rows = con.execute("SELECT COUNT(*) FROM fact_job_postings").fetchone()[0]
    con.close()
    elapsed = time.perf_counter() - t0

    print(f"[transform] rows={fact_rows} elapsed={elapsed:.2f}s")
    write_metric(
        {
            "step": "transform",
            "db": str(db_path),
            "rows_fact_job_postings": int(fact_rows),
            "elapsed_s": round(elapsed, 3),
        }
    )


def main() -> None:
    here = pathlib.Path(__file__).resolve().parents[1]
    build_models(here / "duckdb")


if __name__ == "__main__":
    main()
