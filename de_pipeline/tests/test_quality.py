from __future__ import annotations

import pathlib

import duckdb
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[2]
WAREHOUSE = ROOT / "de_pipeline" / "duckdb" / "warehouse.duckdb"


@pytest.mark.skipif(
    not WAREHOUSE.exists(), reason="warehouse.duckdb not found. Run transform step first."
)
def test_fact_job_postings_not_empty() -> None:
    con = duckdb.connect(str(WAREHOUSE))
    # table must exist and have rows
    tables = {name for (name,) in con.execute("SHOW TABLES").fetchall()}
    assert "fact_job_postings" in tables, "fact_job_postings table is missing"

    count = con.execute("SELECT COUNT(*) FROM fact_job_postings").fetchone()[0]
    con.close()

    assert count > 0, "fact_job_postings should not be empty"


@pytest.mark.skipif(
    not WAREHOUSE.exists(), reason="warehouse.duckdb not found. Run transform step first."
)
def test_dims_if_exist_have_rows() -> None:
    con = duckdb.connect(str(WAREHOUSE))
    tables = {name for (name,) in con.execute("SHOW TABLES").fetchall()}

    for dim_table in ("dim_company", "dim_location", "dim_job_title"):
        if dim_table in tables:
            rows = con.execute(f"SELECT COUNT(*) FROM {dim_table}").fetchone()[0]
            assert rows > 0, f"{dim_table} exists but has no rows"

    con.close()
