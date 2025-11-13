from __future__ import annotations

import time
import pathlib
from typing import Iterable

import polars as pl
from de_pipeline.src.metrics import write_metric


def _read_one_csv(path: pathlib.Path) -> pl.DataFrame:
    # Faster inference with a reasonable sample; collect eager
    try:
        df = pl.read_csv(path, infer_schema_length=2000)
    except Exception as exc:  # pragma: no cover - defensive
        # Re-raise with filename context to make debugging easier when used
        # inside tasks or test runners.
        raise RuntimeError(f"Failed to read CSV '{path}': {exc}") from exc
    # Normalize
    str_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Utf8]
    if str_cols:
        # Use regex-based trim for compatibility across Polars versions
        df = df.with_columns([pl.col(str_cols).str.replace_all(r"^\s+|\s+$", "")])

    # Common date column
    if "posted_date" in df.columns:
        # Trim then parse; use replace_all for broad compatibility. We keep
        # strict=False so parsing returns nulls for unparseable values
        # instead of raising.
        df = df.with_columns(
            pl.col("posted_date")
            .str.replace_all(r"^\s+|\s+$", "")
            .str.strptime(pl.Date, strict=False)
        )
    return df


def _staged_name_for(df: pl.DataFrame, src: pathlib.Path) -> str:
    if {"job_id", "job_title"}.issubset(df.columns) and (
        "company_name" in df.columns or "company" in df.columns
    ):
        return "stg_ai_job_market.parquet"
    # fallback: source-based
    return f"stg_{src.stem}.parquet"


def _should_skip(raw_file: pathlib.Path, staged_file: pathlib.Path) -> bool:
    """Skip if staged exists and is newer than raw."""
    return staged_file.exists() and staged_file.stat().st_mtime >= raw_file.stat().st_mtime


def ingest_raw_to_stage(raw_dir: str | pathlib.Path, staged_dir: str | pathlib.Path) -> None:
    """
    Day 7: performance + reliability
    - Reads all *.csv in raw_dir with Polars
    - Light cleanup + date parsing
    - Writes staged Parquet
    - Skips files that are already up-to-date
    - Emits timing metrics to logs/metrics.jsonl
    """
    raw_dir = pathlib.Path(raw_dir)
    staged_dir = pathlib.Path(staged_dir)
    staged_dir.mkdir(parents=True, exist_ok=True)

    csv_files: Iterable[pathlib.Path] = sorted(raw_dir.glob("*.csv"))
    if not any(f.is_file() for f in csv_files):
        raise FileNotFoundError(f"No CSV files found in {raw_dir.resolve()}")

    for file in csv_files:
        if not file.is_file():
            continue

        # Temp read just header to decide staged filename, then full read
        t0 = time.perf_counter()
        df = _read_one_csv(file)
        staged_name = _staged_name_for(df, file)
        out_path = staged_dir / staged_name

        if _should_skip(file, out_path):
            print(f"[ingest] Skip (up-to-date) {file.name} → {out_path.name}")
            write_metric(
                {
                    "step": "ingest",
                    "action": "skip",
                    "src": str(file),
                    "dst": str(out_path),
                    "rows": None,
                    "cols": None,
                    "elapsed_s": 0.0,
                }
            )
            continue

        df.write_parquet(out_path)
        elapsed = time.perf_counter() - t0
        print(
            f"[ingest] {file.name} → {out_path.name}  "
            f"{df.height} rows × {df.width} cols in {elapsed:.2f}s"
        )
        write_metric(
            {
                "step": "ingest",
                "action": "write",
                "src": str(file),
                "dst": str(out_path),
                "rows": int(df.height),
                "cols": int(df.width),
                "elapsed_s": round(elapsed, 3),
            }
        )


def main() -> None:
    here = pathlib.Path(__file__).resolve().parents[1]
    ingest_raw_to_stage(here / "data" / "raw", here / "data" / "staged")


if __name__ == "__main__":
    main()
