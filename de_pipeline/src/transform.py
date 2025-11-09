from __future__ import annotations

import pathlib
from typing import Set

import duckdb
import polars as pl


def _get_staged_file(staged_dir: pathlib.Path) -> pathlib.Path:
    """Pick the main staged parquet file."""
    ai_job_market = staged_dir / "stg_ai_job_market.parquet"
    if ai_job_market.exists():
        return ai_job_market

    parquet_files = list(staged_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No staged parquet files found in {staged_dir.resolve()}. "
            "Run the ingest step first."
        )

    # Fallback to first parquet
    return parquet_files[0]


def _load_staged_df(staged_path: pathlib.Path) -> pl.DataFrame:
    df = pl.read_parquet(staged_path)
    print(
        f"[transform] Loaded staged data from {staged_path.name} "
        f"({df.height} rows, {df.width} cols)"
    )
    return df


def _create_dim_tables(con: duckdb.DuckDBPyConnection, cols: Set[str]) -> None:
    # company dimension
    if "company" in cols:
        con.execute(
            """
            CREATE OR REPLACE TABLE dim_company AS
            SELECT
                ROW_NUMBER() OVER () AS company_id,
                company AS company_name
            FROM (
                SELECT DISTINCT company
                FROM stg_jobs
                WHERE company IS NOT NULL AND company <> ''
            )
            """
        )
        print("[transform] Created dim_company")

    # location dimension (simple single column; you can refine later)
    if "location" in cols:
        con.execute(
            """
            CREATE OR REPLACE TABLE dim_location AS
            SELECT
                ROW_NUMBER() OVER () AS location_id,
                location
            FROM (
                SELECT DISTINCT location
                FROM stg_jobs
                WHERE location IS NOT NULL AND location <> ''
            )
            """
        )
        print("[transform] Created dim_location")

    # job title dimension (optionally includes experience_level if present)
    if "job_title" in cols:
        if "experience_level" in cols:
            con.execute(
                """
                CREATE OR REPLACE TABLE dim_job_title AS
                SELECT
                    ROW_NUMBER() OVER () AS job_title_id,
                    job_title,
                    experience_level
                FROM (
                    SELECT DISTINCT job_title, experience_level
                    FROM stg_jobs
                    WHERE job_title IS NOT NULL AND job_title <> ''
                )
                """
            )
        else:
            con.execute(
                """
                CREATE OR REPLACE TABLE dim_job_title AS
                SELECT
                    ROW_NUMBER() OVER () AS job_title_id,
                    job_title
                FROM (
                    SELECT DISTINCT job_title
                    FROM stg_jobs
                    WHERE job_title IS NOT NULL AND job_title <> ''
                )
                """
            )
        print("[transform] Created dim_job_title")


def _create_fact_table(con: duckdb.DuckDBPyConnection, cols: Set[str]) -> None:
    """
    Build fact_job_postings by joining stg_jobs to dims where available.

    Keeps it defensive:
    - Always selects all stg_jobs columns
    - Adds surrogate keys when corresponding dims exist
    """
    select_parts = ["j.*"]
    joins = []

    # company fk
    if {"company"}.issubset(cols):
        select_parts.append("dc.company_id")
        joins.append("LEFT JOIN dim_company dc ON j.company = dc.company_name")

    # location fk
    if {"location"}.issubset(cols):
        select_parts.append("dl.location_id")
        joins.append("LEFT JOIN dim_location dl ON j.location = dl.location")

    # job title fk
    if {"job_title"}.issubset(cols):
        select_parts.append("djt.job_title_id")
        joins.append("LEFT JOIN dim_job_title djt ON j.job_title = djt.job_title")

    fact_sql = f"""
    CREATE OR REPLACE TABLE fact_job_postings AS
    SELECT {", ".join(select_parts)}
    FROM stg_jobs j
    {' '.join(joins)}
    """

    con.execute(fact_sql)
    print("[transform] Created fact_job_postings")


def build_models(warehouse_dir: str | pathlib.Path) -> None:
    """
    Day 3
    - Load staged parquet
    - Register as stg_jobs
    - Create dim_* tables (company, location, job_title) when columns exist
    - Create fact_job_postings joined to dims where possible
    """
    here = pathlib.Path(__file__).resolve().parents[1]
    staged_dir = here / "data" / "staged"

    staged_path = _get_staged_file(staged_dir)
    df = _load_staged_df(staged_path)
    cols = set(df.columns)

    warehouse_dir = pathlib.Path(warehouse_dir)
    warehouse_dir.mkdir(parents=True, exist_ok=True)
    db_path = warehouse_dir / "warehouse.duckdb"

    con = duckdb.connect(str(db_path))
    print(f"[transform] Using warehouse at {db_path.resolve()}")

    # register staged as a DuckDB view
    con.register("stg_jobs", df.to_arrow())

    # build dims
    _create_dim_tables(con, cols)

    # build fact
    _create_fact_table(con, cols)

    # simple row counts
    fact_count = con.execute("SELECT COUNT(*) FROM fact_job_postings").fetchone()[0]
    print(f"[transform] fact_job_postings rows {fact_count}")

    if fact_count == 0:
        print(
            "[transform][warn] fact_job_postings is empty. "
            "Check your staged data or join conditions."
        )

    con.close()
    print("[transform] Day 3 modeling completed successfully.")


def main() -> None:
    here = pathlib.Path(__file__).resolve().parents[1]
    warehouse_dir = here / "duckdb"
    build_models(warehouse_dir)


if __name__ == "__main__":
    main()
