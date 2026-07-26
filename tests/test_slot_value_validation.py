#!/usr/bin/env python3
"""Expectation-driven slot-value validation — the recogniser must not FABRICATE.

Two production bugs shared one root cause: while a slot was being awaited, the
turn was mined for a value in lenient mode and *any* non-null hit was accepted,
so a non-answer was silently turned into a wrong slot value.

  * A closed enum ("what is the name of the memory?") fuzzy-matched the stopword
    "the" to the memory "three" (edit distance 2, 0.60), so an off-topic
    sentence — "who is the prime minister of india" — changed the program.
  * A date-time slot ("when should I remind you?") handed "no" to the permissive
    dateparser fallback, which read it as the month November and created a
    reminder for a date the user never gave.

The fix is at the recogniser: a value is only produced when the input is
genuinely a valid value for that slot's type. Fuzzy correction is for content
words, never function words; the dateparser fallback fires only on text that
carries a digit. These are the regression fixtures. Fix the code, never the
fixture — a change here means behaviour drifted.

This module loads entities.py directly (no heavy deps), mirroring
tests/test_datetime_parity_en.py.

Run: pytest tests/test_slot_value_validation.py
"""

import importlib.util
from pathlib import Path

import pytest

_BASE = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "nlu_entities", _BASE / "packages" / "runtime" / "nlu_engine" / "entities.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
EntityExtractor = _mod.EntityExtractor


@pytest.fixture(scope="module")
def ex():
    return EntityExtractor()


# --------------------------- enum: no fabrication --------------------------

@pytest.mark.parametrize("text", [
    "who is the prime minister of india",  # the classic: "the" -> "three"
    "the",
    "of the",
    "what time is it",
])
def test_stopword_never_resolves_to_an_enum_value(ex, text):
    value, _span, conf = ex.extract_enum("memory", text)
    assert value is None, (
        f"{text!r} fabricated memory={value!r} at {conf} — a function word must "
        f"never be treated as a typo of an enum name")


# ------------------- enum: genuine values still resolve --------------------

@pytest.mark.parametrize("text,expected", [
    ("mute", "Mute"),               # exact
    ("the mute", "Mute"),           # value embedded after a stopword
    ("crowd please", "Crowd"),      # value plus filler
    ("restauran", "Restaurant"),    # genuine 1-edit typo (content word)
    ("restraunt", "Restaurant"),    # genuine 2-edit typo (content word), 0.70
])
def test_genuine_enum_values_and_typos_still_resolve(ex, text, expected):
    value, _span, _conf = ex.extract_enum("memory", text)
    assert value == expected, f"{text!r} should resolve to {expected!r}, got {value!r}"


# ---------------------- date-time: no fabrication --------------------------

@pytest.mark.parametrize("text", [
    "no",       # dateparser read this as November
    "nope",
    "yes",
    "maybe",
    "sure",
])
def test_non_temporal_word_never_resolves_to_a_datetime(ex, text):
    iso, _span, conf, _te, _ed = ex.extract_datetime(text)
    assert iso is None, (
        f"{text!r} fabricated a datetime ({iso}, conf={conf}) — text with no "
        f"temporal signal must not reach the permissive parser")


# ----------------- date-time: legitimate inputs still parse ----------------

@pytest.mark.parametrize("text", [
    "tomorrow",        # grammar, day only -> engine prompts for time
    "friday",          # grammar, weekday only
    "saturday at 6",   # grammar, day + time
    "in 10 minutes",   # grammar, relative
    "9am",             # grammar, clock
    "june 5",          # absolute date via fallback (has a digit)
    "the 25th",        # absolute date via fallback (has a digit)
])
def test_legitimate_datetimes_still_resolve(ex, text):
    iso, _span, _conf, _te, _ed = ex.extract_datetime(text)
    assert iso is not None, f"{text!r} is a valid date/time and must still resolve"
