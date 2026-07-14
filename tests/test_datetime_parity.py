#!/usr/bin/env python3
"""
Golden-fixture parity tests for the lexicon-driven multilingual datetime parser.

Each row of tests/datetime_parity/nlu_datetime_parity_<lang>.csv is parsed by
EntityExtractor(language=<lang>) and checked against the expected day-offset and
clock time. The IDENTICAL fixtures are consumed by the Swift XCTest
(testExtractDateTimeMultilingual) so both platforms must agree on every row —
this is the Phase 2 parity gate.

Reference "now" is pinned to Tuesday 2026-06-30T00:00 UTC. A midnight reference
keeps every "today" time-of-day in the future (so it is not rolled to tomorrow)
and a UTC zone keeps local == UTC so asserted wall-clock hours are unchanged.
The Swift test uses the same reference instant.

Run: pytest tests/test_datetime_parity.py   (or: python tests/test_datetime_parity.py)
"""

import csv
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    import pytest
except ImportError:  # allow running as a plain script without pytest installed
    pytest = None

BASE_DIR = Path(__file__).parent.parent
# Load entities.py directly (it has no heavy deps) rather than via the nlu
# package, whose __init__ pulls in numpy/torch that need not be installed here.
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "nlu_entities", BASE_DIR / "packages" / "runtime" / "nlu_engine" / "entities.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
EntityExtractor = _mod.EntityExtractor

FIXTURE_DIR = Path(__file__).parent / "datetime_parity"
LOC_DIR = BASE_DIR / "content" / "localization"
NOW = datetime(2026, 6, 30, 0, 0, 0, tzinfo=timezone.utc)  # Tuesday 00:00 UTC

_WD = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
       "friday": 4, "saturday": 5, "sunday": 6}
_MON = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12}

# One extractor per language, reused across rows (lexicon tables built once).
_EXTRACTORS = {
    lang: EntityExtractor(entities_path=LOC_DIR / f"nlu_entities.{lang}.json", language=lang)
    for lang in ("fr", "de", "da")
}


def _expected_date_matches(token: str, dt: datetime) -> bool:
    if token == "today":
        return dt.date() == NOW.date()
    if token == "+1d":
        return dt.date() == (NOW + timedelta(days=1)).date()
    if token.startswith("next_"):
        wd = _WD[token[5:]]
        ahead = (wd - NOW.weekday()) % 7 or 7
        return dt.date() == (NOW + timedelta(days=ahead)).date()
    if token.endswith("min"):
        n = int(token[1:-3])
        return abs((dt - (NOW + timedelta(minutes=n))).total_seconds()) < 90
    if "_" in token and token.split("_")[0] in _MON:
        mon, day = token.split("_")
        return dt.month == _MON[mon] and dt.day == int(day)
    raise AssertionError(f"unknown expected_date token: {token}")


def _load_rows():
    rows = []
    for lang in ("fr", "de", "da"):
        with open(FIXTURE_DIR / f"nlu_datetime_parity_{lang}.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append((r["language"], r["utterance"], r["expected_date"], r["expected_time"]))
    return rows


def _maybe_parametrize(func):
    if pytest is not None:
        return pytest.mark.parametrize("lang,utterance,exp_date,exp_time", _load_rows())(func)
    return func


@_maybe_parametrize
def test_datetime_parity(lang, utterance, exp_date, exp_time):
    iso, _span, conf, _time_explicit, _explicit_day = \
        _EXTRACTORS[lang].extract_datetime(utterance, now=NOW)
    assert iso is not None and conf > 0, f"{lang!r} {utterance!r} produced no datetime"
    dt = datetime.fromisoformat(iso)
    assert _expected_date_matches(exp_date, dt), \
        f"{lang!r} {utterance!r}: date {dt.date()} != expected {exp_date}"
    if exp_time != "-":
        assert dt.strftime("%H:%M") == exp_time, \
            f"{lang!r} {utterance!r}: time {dt.strftime('%H:%M')} != expected {exp_time}"


if __name__ == "__main__":
    rows = _load_rows()
    failures = 0
    for lang, utt, exp_d, exp_t in rows:
        try:
            test_datetime_parity(lang, utt, exp_d, exp_t)
            print(f"[OK ] {lang} {utt!r}")
        except AssertionError as e:
            failures += 1
            print(f"[FAIL] {e}")
    print(f"\n{len(rows) - failures}/{len(rows)} passed")
    sys.exit(1 if failures else 0)
