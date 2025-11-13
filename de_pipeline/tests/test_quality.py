from __future__ import annotations

import pathlib
import duckdb
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
WAREHOUSE = ROOT / "de_pipeline" / "duckdb" / "warehouse.duckdb"

requires_wh = pytest.mark.skipif(
    not WAREHOUSE.exists(), reason="warehouse.duckdb not found. Run the flow first."
)


@requires_wh
def test_fact_exists_and_has_rows() -> None:
    con = duckdb.connect(str(WAREHOUSE))
    tables = {name for (name,) in con.execute("SHOW TABLES").fetchall()}
    assert "fact_job_postings" in tables
    rows = con.execute("SELECT COUNT(*) FROM fact_job_postings").fetchone()[0]
    assert rows > 0
    con.close()


@requires_wh
def test_optional_columns_if_present_are_not_all_null() -> None:
    con = duckdb.connect(str(WAREHOUSE))
    cols = [
        c[0] for c in con.execute("DESCRIBE SELECT * FROM fact_job_postings LIMIT 0").fetchall()
    ]
    for col in ("posted_date", "location", "job_title", "company_name"):
        if col in cols:
            non_null = con.execute(
                f"SELECT COUNT({col}) FROM fact_job_postings WHERE {col} IS NOT NULL"
            ).fetchone()[0]
            assert non_null > 0, f"{col} exists but all values are NULL"
    con.close()
