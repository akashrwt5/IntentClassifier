"""Regression tests for the polarity-guard contradiction bug.

Bug: ``_apply_polarity_guards`` only inspected rules whose ``blocked`` intent
equalled the model's prediction, so when an utterance carried BOTH an up-cue and
a down-cue (e.g. "lower how LOUD it is"), the single matching mirror rule flipped
the model's CORRECT answer to the opposite (dangerous) volume action.

Fix: the guard abstains whenever the opposite cue is also present, trusting the
model — which already resolves these correctly. These tests pin both halves:
  A) both-cue utterances: the guard must NOT override the model.
  B) genuine single-cue contradictions: the guard must STILL correct them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "runtime"))

pytest.importorskip("onnxruntime")
_MODEL = Path(__file__).resolve().parents[1] / "models" / "intent_model.onnx"
pytestmark = pytest.mark.skipif(not _MODEL.exists(), reason="English ONNX model not present")


@pytest.fixture(scope="module")
def engine():
    from nlu_engine import NLUEngine

    return NLUEngine()


# A) Natural phrasings that mention the current state — BOTH cues present.
#    The model predicts these correctly; the guard must leave them alone.
BOTH_CUE = [
    ("lower how loud it is", "Cmd.VolumeDecrease"),
    ("turn it up its too quiet", "Cmd.VolumeIncrease"),
    ("its too loud turn it down", "Cmd.VolumeDecrease"),
    ("make it quieter its too loud", "Cmd.VolumeDecrease"),
]


@pytest.mark.parametrize("text,expected", BOTH_CUE)
def test_guard_does_not_flip_correct_model_prediction(engine, text, expected):
    r = engine.handle(f"sess-{text}", text)
    assert r.intent == expected, f"{text!r}: guard overrode the correct model answer"


# B) English volume/mute is now MODEL-ONLY. ALL polarity guards were removed for
#    English (owner directive, 2026-07-24) because holdout evidence showed they
#    only ever flipped CORRECT model answers. Assert the model handles direction
#    (and un/mute) unaided — no rule involved.
@pytest.mark.parametrize("text,expected", [
    ("please lower it", "Cmd.VolumeDecrease"),
    ("make it louder", "Cmd.VolumeIncrease"),
    ("make it less loud", "Cmd.VolumeDecrease"),
    ("cancel the mute please", "Cmd.VolumeUnmute"),
])
def test_model_handles_volume_direction_without_a_guard(engine, text, expected):
    r = engine.handle(f"sess-{text}", text)
    assert r.intent == expected


def test_polarity_guards_removed_for_english(engine):
    # English carries NO polarity guards at all — the raw model prediction stands.
    assert engine._polarity_guards == []


def test_english_intent_is_raw_model_output(engine):
    # No polarity rule may change the predicted label: the engine's intent must
    # equal the classifier's raw prediction for these volume/mute utterances.
    for text in ("lower how loud it is", "make it less loud", "cancel the mute please",
                 "turn it down", "unmute"):
        raw, _ = engine.classifier.classify(text)
        r = engine.handle(f"raw-{text}", text)
        assert r.intent == raw, f"{text!r}: engine changed the model output {raw} -> {r.intent}"
