from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def write_metric(event: Dict[str, Any]) -> None:
    """Append one JSON line to logs/metrics.jsonl."""
    # Use timezone-aware UTC timestamps to avoid DeprecationWarning from
    # datetime.utcnow(). Prefer ISO with trailing 'Z' for compatibility.
    ts = datetime.now(timezone.utc).isoformat()
    if ts.endswith("+00:00"):
        ts = ts.replace("+00:00", "Z")
    event = {"ts": ts, **event}
    with (LOGS_DIR / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
