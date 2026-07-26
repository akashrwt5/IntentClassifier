"""
Train/eval leakage detection (charter A5).

The English holdout was 99.9% training data (Review-F5 blocker B9) while the
training guard reported clean, because it compared RAW strings: any pair
differing only by punctuation or spacing slipped through. These tests pin the
normalised comparison that closes that hole.
"""

import importlib
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_BUILDTIME = str(_ROOT / "packages" / "buildtime")
if _BUILDTIME not in sys.path:
    sys.path.insert(0, _BUILDTIME)

_L = importlib.import_module("nlu_training.leakage")


# --------------------------------------------------------------------------- #
# The regression: near-miss duplicates the old raw compare let through
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("train,evalset", [
    ("turn up the volume",  "turn up the volume?"),   # trailing punctuation
    ("set a reminder",      "Set a reminder."),       # case + full stop
    ("set  a   reminder",   "set a reminder"),        # collapsed whitespace
    ("what's my battery",   "whats my battery"),      # apostrophe
    ("mute it, please",     "mute it please"),        # interior comma
])
def test_near_miss_duplicates_are_caught(train, evalset):
    assert _L.find_leaks([train], [evalset]) == [train], (
        f"{train!r} vs {evalset!r} is the same utterance to a TF-IDF model; "
        f"the guard must treat it as a leak"
    )


def test_raw_comparison_would_have_missed_these():
    """Pin the contrast explicitly — this is the bug that shipped.

    A raw set-intersection finds nothing here; the normalised guard finds it.
    """
    train, evalset = ["turn up the volume"], ["turn up the volume?"]
    assert not (set(train) & set(evalset)), "precondition: raw compare sees no overlap"
    assert _L.find_leaks(train, evalset), "normalised compare must see the overlap"


# --------------------------------------------------------------------------- #
# It must not over-report
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("train,evalset", [
    ("turn up the volume",  "turn down the volume"),
    ("start translation",   "stop translation"),
    ("où est mon téléphone", "ou est mon telephone"),  # accents are NOT folded
])
def test_distinct_utterances_are_not_flagged(train, evalset):
    assert _L.find_leaks([train], [evalset]) == []


def test_accents_are_preserved():
    """'ou' (or) and 'où' (where) are different French words.

    Folding accents would create false leaks that mask real ones.
    """
    assert _L.normalize_text("où") != _L.normalize_text("ou")


# --------------------------------------------------------------------------- #
# Shape
# --------------------------------------------------------------------------- #

def test_returns_original_strings_not_normalised_ones():
    """A human has to find the offending row, so report what they will recognise."""
    assert _L.find_leaks(["Turn Up The Volume?"], ["turn up the volume"]) == \
        ["Turn Up The Volume?"]


def test_each_leak_reported_once():
    leaks = _L.find_leaks(["mute it", "mute it?", "MUTE IT."], ["mute it"])
    assert len(leaks) == 1


def test_empty_inputs_are_safe():
    assert _L.find_leaks([], ["a"]) == []
    assert _L.find_leaks(["a"], []) == []
    # Blank rows must never count as a universal match.
    assert _L.find_leaks(["", "   "], ["", "  "]) == []


def test_report_is_actionable():
    msg = _L.leak_report(["turn up the volume"], 100, source="permanent holdout")
    assert "1 training utterance" in msg and "turn up the volume" in msg
    assert "normalised" in msg
    assert "0 leaks" in _L.leak_report([], 100)
