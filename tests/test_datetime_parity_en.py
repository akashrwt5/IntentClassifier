#!/usr/bin/env python3
"""
English datetime golden-corpus regression test.

English historically had NO datetime parity fixture — `tests/datetime_parity/`
covers fr, de and da only, because English was the hardcoded special case rather
than a lexicon-driven language. This test locks the baseline captured in
`nlu_datetime_parity_en_golden.json` so that the A7 eviction of English datetime
vocabulary into Language Pack tables must reproduce the exact same
`(iso, time_explicit, explicit_day)` for every case.

If a case here fails during A7, the refactor changed behaviour. Fix the code,
never the fixture (charter STOP rule 3).

The corpus is deterministic by construction: the capture script admits only
cases resolved before the optional `dateparser` fallback, so these assertions
hold whether or not that package is installed. `test_corpus_is_dateparser_
independent` keeps that property honest.

Run: pytest tests/test_datetime_parity_en.py
"""

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pytest

_BASE = Path(__file__).parent.parent
_GOLDEN_PATH = _BASE / "tests" / "datetime_parity" / "nlu_datetime_parity_en_golden.json"

# Load entities.py directly — it has no heavy dependencies, unlike the
# nlu_engine package __init__ which pulls in numpy. Mirrors how
# tests/test_datetime_parity.py loads the same module.
_spec = importlib.util.spec_from_file_location(
    "nlu_entities", _BASE / "packages" / "runtime" / "nlu_engine" / "entities.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
EntityExtractor = _mod.EntityExtractor

_GOLDEN = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
_NOW = datetime.fromisoformat(_GOLDEN["now_iso"])
_CASES = _GOLDEN["cases"]

_EXTRACTOR = EntityExtractor()


@pytest.mark.parametrize("case", _CASES, ids=[c["utterance"] for c in _CASES])
def test_en_datetime_golden(case):
    """Every captured utterance must resolve identically to the pristine parser."""
    iso, _span, conf, time_explicit, explicit_day = _EXTRACTOR.extract_datetime(
        case["utterance"], now=_NOW)
    assert [iso, time_explicit, explicit_day] == [
        case["iso"], case["time_explicit"], case["explicit_day"]
    ], (f"{case['utterance']!r} (branch {case['branch']}): got "
        f"({iso}, {time_explicit}, {explicit_day}) expected "
        f"({case['iso']}, {case['time_explicit']}, {case['explicit_day']})")
    assert conf == case["conf"]


def test_corpus_is_large_enough():
    """Charter A1 requires >= 120 cases — enough to cover every parser branch."""
    assert len(_CASES) >= 120, f"golden corpus shrank to {len(_CASES)} cases"


def test_corpus_is_dateparser_independent():
    """No case may depend on the optional `dateparser` fallback (§8).

    A §8 resolution reports conf 0.85 and is computed against real wall-clock
    time rather than the pinned `now`, so admitting one would make this suite
    flaky and environment-dependent.
    """
    offenders = [c["utterance"] for c in _CASES if c["conf"] not in (1.0, 0.0)]
    assert not offenders, f"cases resolved via the dateparser fallback: {offenders}"


@pytest.mark.parametrize("case", _GOLDEN.get("strip_cases", []),
                         ids=[c["text"] for c in _GOLDEN.get("strip_cases", [])])
def test_en_strip_datetime_golden(case):
    """`strip_datetime` removes date/time fragments, leaving the reminder topic.

    It is cosmetic — never changes a resolved time — but it had NO coverage at
    all before A7, which is how the whole function came to be refactored
    unguarded. These expectations were verified equal to the pre-refactor
    implementation (git HEAD before the eviction) across all 27 cases before
    being captured, so the corpus pins the ORIGINAL behaviour, not the new
    implementation's opinion of it.
    """
    assert _EXTRACTOR.strip_datetime(case["text"]) == case["topic"]


def test_known_gaps_are_still_gaps():
    """Recorded defects must stay broken until deliberately fixed.

    If one starts resolving, `entities.py` behaviour changed and the recorded
    gap list is stale — that should fail loudly rather than drift.
    """
    for gap in _GOLDEN.get("known_gaps", []):
        iso, *_ = _EXTRACTOR.extract_datetime(gap["utterance"], now=_NOW)
        assert iso is None, (
            f"{gap['utterance']!r} now resolves to {iso}. If that fix was "
            f"deliberate, remove it from KNOWN_GAPS in "
            f"scripts/ci/capture_en_datetime_golden.py and recapture."
        )
