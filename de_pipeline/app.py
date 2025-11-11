from __future__ import annotations

import pathlib
from typing import Any, Dict

import duckdb
import polars as pl
import streamlit as st


BASE_DIR = pathlib.Path(__file__).resolve().parent
WAREHOUSE_PATH = BASE_DIR / "duckdb" / "warehouse.duckdb"


@st.cache_resource(show_spinner=False)
def get_connection() -> duckdb.DuckDBPyConnection:
    if not WAREHOUSE_PATH.exists():
        raise FileNotFoundError(
            f"warehouse.duckdb not found at {WAREHOUSE_PATH}. "
            "Run the Day 4 flow to build the warehouse first."
        )
    return duckdb.connect(str(WAREHOUSE_PATH))


def run_query(sql: str, params: Dict[str, Any] | None = None) -> pl.DataFrame:
    con = get_connection()
    if params:
        return pl.from_arrow(con.execute(sql, params).arrow())
    return pl.from_arrow(con.execute(sql).arrow())


def main() -> None:
    st.set_page_config(page_title="DE pipeline dashboard", layout="wide")
    st.title("AI job market analytics")

    # sanity check
    try:
        fact_preview = run_query("SELECT * FROM fact_job_postings LIMIT 5")
    except Exception as e:  # noqa: BLE001
        st.error(
            "Could not read from fact_job_postings. "
            "Make sure you have run the Day 4 flow successfully."
        )
        st.exception(e)
        return

    st.caption(
        "Data source: staged AI job market CSV → DuckDB warehouse (dim/fact model). "
        "Simple production-style analytics view."
    )

    # detect columns
    has_posted_date = "posted_date" in fact_preview.columns
    has_location = "location" in fact_preview.columns

    # handle company vs company_name
    company_col = None
    if "company" in fact_preview.columns:
        company_col = "company"
    elif "company_name" in fact_preview.columns:
        company_col = "company_name"

    has_company = company_col is not None
    has_job_title = "job_title" in fact_preview.columns

    # sidebar filters
    st.sidebar.header("Filters")

    date_filter = None
    if has_posted_date:
        date_min, date_max = run_query(
            "SELECT MIN(posted_date), MAX(posted_date) FROM fact_job_postings"
        ).row(0)
        if date_min is not None and date_max is not None:
            start, end = st.sidebar.date_input(
                "Posted date range",
                value=(date_min, date_max),
                min_value=date_min,
                max_value=date_max,
            )
            # handle edge cases
            if not isinstance(start, type(date_min)):
                start = date_min
            if not isinstance(end, type(date_max)):
                end = date_max
            date_filter = (start, end)

    location_filter = None
    if has_location:
        locations = (
            run_query(
                """
                SELECT DISTINCT location
                FROM fact_job_postings
                WHERE location IS NOT NULL AND location <> ''
                ORDER BY location
                """
            )
            .get_column("location")
            .to_list()
        )
        if locations:
            location_filter = st.sidebar.multiselect("Location", locations, default=[])

    company_filter = None
    if has_company:
        companies = (
            run_query(
                f"""
                SELECT DISTINCT {company_col} AS company
                FROM fact_job_postings
                WHERE {company_col} IS NOT NULL AND {company_col} <> ''
                ORDER BY {company_col}
                """
            )
            .get_column("company")
            .to_list()
        )
        if companies:
            company_filter = st.sidebar.multiselect("Company", companies, default=[])

    # build WHERE clause
    where_clauses = []
    params: Dict[str, Any] = {}

    if date_filter and has_posted_date:
        where_clauses.append("posted_date BETWEEN $start AND $end")
        params["start"], params["end"] = date_filter

    if location_filter and has_location:
        where_clauses.append("location IN $locations")
        params["locations"] = location_filter

    if company_filter and has_company:
        where_clauses.append(f"{company_col} IN $companies")
        params["companies"] = company_filter

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    # metrics
    metrics_df = run_query(
        f"""
        SELECT
            COUNT(*) AS total_postings,
            {f"COUNT(DISTINCT {company_col}) AS companies," if has_company else "0 AS companies,"}
            { "COUNT(DISTINCT location) AS locations," if has_location else "0 AS locations,"}
            { "COUNT(DISTINCT job_title) AS job_titles" if has_job_title else "0 AS job_titles"}
        FROM fact_job_postings
        {where_sql}
        """,
        params,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total postings", int(metrics_df["total_postings"][0]))
    c2.metric("Companies", int(metrics_df["companies"][0]))
    c3.metric("Locations", int(metrics_df["locations"][0]))
    c4.metric("Job titles", int(metrics_df["job_titles"][0]))

    st.markdown("---")

    # postings over time
    if has_posted_date:
        trend_df = run_query(
            f"""
            SELECT
                posted_date,
                COUNT(*) AS postings
            FROM fact_job_postings
            {where_sql}
            GROUP BY posted_date
            ORDER BY posted_date
            """,
            params,
        )
        if trend_df.height > 0:
            st.subheader("Postings over time")
            st.line_chart(trend_df.to_pandas().set_index("posted_date")["postings"])
        else:
            st.info("No postings for the selected filters.")

    # top locations
    if has_location:
        st.subheader("Top locations by postings")
        loc_df = run_query(
            f"""
            SELECT
                location,
                COUNT(*) AS postings
            FROM fact_job_postings
            {where_sql}
            GROUP BY location
            ORDER BY postings DESC
            LIMIT 10
            """,
            params,
        )
        st.dataframe(loc_df)

    # top job titles
    if has_job_title:
        st.subheader("Top job titles by postings")
        jt_df = run_query(
            f"""
            SELECT
                job_title,
                COUNT(*) AS postings
            FROM fact_job_postings
            {where_sql}
            GROUP BY job_title
            ORDER BY postings DESC
            LIMIT 10
            """,
            params,
        )
        st.dataframe(jt_df)

    with st.expander("Preview fact_job_postings"):
        st.dataframe(fact_preview)


if __name__ == "__main__":
    main()
