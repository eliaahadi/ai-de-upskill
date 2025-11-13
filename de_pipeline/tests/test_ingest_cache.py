from __future__ import annotations
import pathlib
import time

import pytest

from de_pipeline.src.ingest import ingest_raw_to_stage

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "de_pipeline" / "data" / "raw"
STAGED = ROOT / "de_pipeline" / "data" / "staged"


@pytest.mark.skipif(not any(RAW.glob("*.csv")), reason="no raw csvs present")
def test_ingest_skip_when_up_to_date(tmp_path: pathlib.Path) -> None:
    # stage once
    ingest_raw_to_stage(RAW, STAGED)
    # capture mtimes
    staged_files = list(STAGED.glob("*.parquet"))
    assert staged_files, "no staged files after ingest"
    before = {p.name: p.stat().st_mtime for p in staged_files}

    # re-run quickly; should skip (mtime unchanged)
    time.sleep(0.5)
    ingest_raw_to_stage(RAW, STAGED)
    after = {p.name: p.stat().st_mtime for p in staged_files}

    assert before == after, "staged files should be unchanged when raw is older"
