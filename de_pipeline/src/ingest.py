from __future__ import annotations

import pathlib
from typing import Iterable

import polars as pl


def _read_csv_files(raw_dir: pathlib.Path) -> list[pl.DataFrame]:
    """Load all CSVs from raw_dir into Polars DataFrames."""
    csv_files: Iterable[pathlib.Path] = raw_dir.glob("*.csv")
    dataframes: list[pl.DataFrame] = []

    for file in csv_files:
        if not file.is_file():
            continue

        df = pl.read_csv(file)

        # Light, generic cleanup
        # Strip whitespace in string columns
        str_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Utf8]
        if str_cols:
            df = df.with_columns([pl.col(str_cols).str.strip()])

        # If a posted_date column exists, parse it to Date
        if "posted_date" in df.columns:
            df = df.with_columns(
                pl.col("posted_date").str.strip().str.strptime(pl.Date, strict=False)
            )

        dataframes.append(df)

        print(f"[ingest] Loaded {file.name} with {df.height} rows, {df.width} columns")

    if not dataframes:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir.resolve()}. "
            "Drop your raw CSVs into this folder and rerun."
        )

    return dataframes


def ingest_raw_to_stage(raw_dir: str | pathlib.Path, staged_dir: str | pathlib.Path) -> None:
    """
    Read raw CSVs → write staged Parquet with minimal cleanup.

    For Day 2 we:
    - Ingest all `*.csv` in `data/raw`
    - Normalize whitespace
    - Parse posted_date if present
    - Write a single combined staged file per source CSV
    """
    raw_dir = pathlib.Path(raw_dir)
    staged_dir = pathlib.Path(staged_dir)
    staged_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ingest] Reading from: {raw_dir.resolve()}")
    print(f"[ingest] Writing staged outputs to: {staged_dir.resolve()}")

    dataframes = _read_csv_files(raw_dir)

    for i, df in enumerate(dataframes, start=1):
        # Name staged file based on source index or job_market pattern
        if "job_id" in df.columns and "job_title" in df.columns:
            staged_name = "stg_ai_job_market.parquet"
        else:
            staged_name = f"stg_table_{i}.parquet"

        out_path = staged_dir / staged_name
        df.write_parquet(out_path)
        print(
            f"[ingest] Wrote {df.height} rows x {df.width} cols "
            f"to {out_path.name}"
        )

    print("[ingest] Completed Day 2 staging successfully.")


def main() -> None:
    here = pathlib.Path(__file__).resolve().parents[1]
    raw_dir = here / "data" / "raw"
    staged_dir = here / "data" / "staged"
    ingest_raw_to_stage(raw_dir, staged_dir)


if __name__ == "__main__":
    main()