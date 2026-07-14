#!/usr/bin/env python3
"""Privacy-conscious logging for low-confidence ("unknown") turns.

Implements the stance in ``docs/privacy-unknown-data.md`` (Review-F5
Appendix A #10, risk RK6), approved for enforcement (ND-5):

- **Default = aggregate counters only.** Per-day, per-confidence-bucket
  counts in ``data/unknown_counters.csv``. No raw text, no precise
  timestamps, no identifiers.
- **Raw text is opt-in only.** Verbatim logging to ``data/unknown_data.csv``
  requires the explicit ``NLU_COLLECT_RAW_UNKNOWN`` env flag (consent proxy
  for the CLI/dev path; the app layer must map this to real user consent).

Dependency-light on purpose (stdlib only) so it is importable and testable
without model artifacts or heavy runtime deps.
"""

from __future__ import annotations

import csv
import os
from datetime import date, datetime
from pathlib import Path

RAW_OPT_IN_ENV = "NLU_COLLECT_RAW_UNKNOWN"

_BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_COUNTERS_PATH = _BASE_DIR / "data" / "unknown_counters.csv"
DEFAULT_RAW_PATH = _BASE_DIR / "data" / "unknown_data.csv"

_COUNTER_FIELDS = ["date", "confidence_bucket", "count"]


def raw_collection_enabled() -> bool:
    """True only when raw-utterance collection is explicitly opted in."""
    return os.environ.get(RAW_OPT_IN_ENV, "").lower() in ("1", "true", "yes")


def confidence_bucket(confidence: float, width: float = 0.1) -> str:
    """Coarse bucket label like ``0.6-0.7`` (clamped to [0, 1))."""
    c = min(max(confidence, 0.0), 0.9999)
    lo = int(c / width) * width
    return f"{lo:.1f}-{lo + width:.1f}"


def _bump_counter(counters_path: Path, confidence: float) -> None:
    today = date.today().isoformat()
    bucket = confidence_bucket(confidence)
    rows: dict[tuple[str, str], int] = {}
    if counters_path.exists():
        with open(counters_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[(row["date"], row["confidence_bucket"])] = int(row["count"])
    rows[(today, bucket)] = rows.get((today, bucket), 0) + 1
    counters_path.parent.mkdir(parents=True, exist_ok=True)
    with open(counters_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_COUNTER_FIELDS)
        for (d, b), n in sorted(rows.items()):
            w.writerow([d, b, n])


def _append_raw(raw_path: Path, text: str, confidence: float) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not raw_path.exists()
    with open(raw_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["text", "confidence", "timestamp"])
        w.writerow([text, f"{confidence:.6f}", datetime.now().isoformat()])


def record_unknown(
    text: str,
    confidence: float,
    counters_path: Path = DEFAULT_COUNTERS_PATH,
    raw_path: Path = DEFAULT_RAW_PATH,
) -> str:
    """Record a low-confidence turn per the privacy stance.

    Returns ``"raw"`` or ``"counter"`` indicating what was written. Raw text
    is written ONLY when :func:`raw_collection_enabled`; otherwise only an
    aggregate counter is updated and ``text`` never touches disk.
    """
    if raw_collection_enabled():
        _append_raw(raw_path, text, confidence)
        return "raw"
    _bump_counter(counters_path, confidence)
    return "counter"
