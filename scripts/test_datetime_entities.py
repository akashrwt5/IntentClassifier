#!/usr/bin/env python3
"""
Golden-file regression tests for the datetime entity parser.

Covers ambiguous cases (at 6/7/12, noon/midnight, relative durations)
and verifies the parser returns the expected hour or raises a CONFIRM
signal for genuinely ambiguous inputs.

Usage:
    python scripts/test_datetime_entities.py

Exit code 1 if any golden case fails.
"""

import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from nlu.entities import EntityExtractor


extractor = EntityExtractor()

# Reference "now": Tuesday 2026-06-14 14:30 (2:30 PM)
NOW = datetime(2026, 6, 14, 14, 30, 0)


def _hour(iso: str) -> int:
    return datetime.fromisoformat(iso).hour


GOLDEN = [
    # (utterance, expected_hour, description)
    ("remind me in 10 minutes",    None,  "relative — expect ISO, not hour check"),
    ("in an hour",                 None,  "relative"),
    ("in half an hour",            None,  "relative"),
    ("remind me tomorrow at 9am",   9,    "explicit AM"),
    ("tomorrow at 3pm",            15,    "explicit PM"),
    ("at noon",                    12,    "noon → 12"),
    ("at midnight",                 0,    "midnight → 0"),
    ("at 9 am",                     9,    "explicit 9 AM"),
    ("at 9pm",                     21,    "explicit 9 PM"),
    ("at 6",                       18,    "ambiguous 6 — expect PM (1-6 rule)"),
    ("at 3",                       15,    "ambiguous 3 — expect PM (1-6 rule)"),
    ("at 9",                       21,    "ambiguous 9 — 9 AM is past at 14:30, flip to PM (21)"),
    ("at 11",                      23,    "ambiguous 11 — 11 AM is past at 14:30, flip to PM (23)"),
    ("at 8",                       20,    "ambiguous 8 — 8 AM is past at 14:30, flip to PM (20)"),
]


def run():
    passed = failed = 0
    for utt, expected_hour, desc in GOLDEN:
        result, span, conf = extractor.extract_datetime(utt, now=NOW)
        if result is None:
            status = "FAIL" if expected_hour is not None else "PASS"
            if expected_hour is None:
                # relative — we expect a non-None result; None means parse failure
                status = "FAIL"
                print(f"FAIL  {utt!r}: got None  [{desc}]")
                failed += 1
                continue
            print(f"FAIL  {utt!r}: no parse  [{desc}]")
            failed += 1
            continue

        if expected_hour is None:
            # Just check it parsed (relative durations)
            print(f"PASS  {utt!r} → {result}  [{desc}]")
            passed += 1
            continue

        actual_hour = _hour(result)
        # Special cases: "at 11" when now=14:30 — 11 AM is past, so should flip to 23 PM
        # "at 8" same pattern
        ok = actual_hour == expected_hour
        if ok:
            print(f"PASS  {utt!r} → {result} (hour={actual_hour})  [{desc}]")
            passed += 1
        else:
            print(f"FAIL  {utt!r} → {result} (hour={actual_hour}, expected={expected_hour})  [{desc}]")
            failed += 1

    print(f"\n{'='*50}")
    print(f"DateTime golden tests: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    run()
