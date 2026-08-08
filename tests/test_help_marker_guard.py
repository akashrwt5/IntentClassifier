"""ND-14 help-marker guard: asking HOW to use a feature must never TRIGGER it.

A confident state-changing action prediction is redirected to its read-only
help.* sibling when the utterance carries explicit help/question markers.
Read-only queries (activity/battery) are intentionally NOT paired, so genuine
"how many steps" style queries are never diverted.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "runtime"))

MODEL = REPO_ROOT / "models" / "intent" / "en" / "model.onnx"


@pytest.fixture(scope="module")
def engine():
    if not MODEL.exists():
        pytest.skip("trained artifacts not present (make train-multilingual)")
    pytest.importorskip("onnxruntime")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nlu_engine import NLUEngine

        return NLUEngine(model_name="en", language="en", semantic_enabled=False)


def test_help_question_redirects_action_to_help(engine):
    # "how do I ..." must land on a help.* intent, never fire a device action.
    engine.reset("hg1")
    r = engine.handle("hg1", "how do i turn up the loudness on my hearing aids?")
    assert (r.intent or "").startswith("Help_"), (r.intent, r.type)
    # The redirect target is a read-only help intent; its action (if any) must
    # itself be a help action, never a state-changing device/streaming command.
    assert not (r.action or "").startswith(
        ("volume.", "streaming.", "transcription.", "translation.", "memory.",
         "reminders.")), f"help question fired a state-changing action: {r.action}"


def test_guard_unit_redirects_only_with_marker(engine):
    # Direct unit check of the guard: marker present -> redirect; absent -> keep.
    assert engine._apply_help_guard(
        "how do i use transcription", "Cmd.TranscribeStart"
    ) == "Help_Transcribe"
    assert engine._apply_help_guard(
        "start transcribing now", "Cmd.TranscribeStart"
    ) == "Cmd.TranscribeStart"


def test_plain_command_still_fires(engine):
    # No help marker -> the guard is inert, the command executes normally.
    engine.reset("hg2")
    r = engine.handle("hg2", "mute")
    assert r.type == "FULFILL" and r.intent == "Cmd.VolumeMute"


def test_readonly_query_never_diverted(engine):
    # "how many steps" is a legit query, NOT paired -> must not become help.*.
    assert engine._apply_help_guard(
        "how many steps did i take today", "Cmd.ActivityStep"
    ) == "Cmd.ActivityStep"


def test_unpaired_action_is_untouched(engine):
    # Cmd.FindMyPhone has no help pair -> guard leaves it alone even with a marker.
    assert engine._apply_help_guard(
        "how do i find my phone", "Cmd.FindMyPhone"
    ) == "Cmd.FindMyPhone"
