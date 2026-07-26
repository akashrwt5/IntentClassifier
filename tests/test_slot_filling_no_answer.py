#!/usr/bin/env python3
"""What happens when a slot-filling turn is NOT a valid answer.

Once the recogniser stops fabricating values (see test_slot_value_validation),
a turn that does not answer the awaited slot reaches one of three outcomes, and
the engine must pick the right one:

  * a pure cancellation ("no", "never mind") -> ABANDON the flow, do not store
    "no" as a reminder name and do not drag the user through dead reprompts;
  * a genuine off-topic / unparseable answer -> RE-PROMPT, and after
    MAX_SLOT_ATTEMPTS fall back gracefully rather than committing garbage;
  * a correction that carries a value ("no, tomorrow at 5") -> fill the slot.

These require the trained English model, so they skip when it is absent, exactly
like tests/test_confirm_gate.py.

Run: pytest tests/test_slot_filling_no_answer.py
"""

import json
import sys
import warnings
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in ("packages/buildtime", "packages/runtime"):
    if str(_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(_ROOT / _p))

_MODEL = _ROOT / "models" / "intent" / "en" / "model.onnx"
_SCHEMA = json.loads((_ROOT / "content" / "nlu_schema.json").read_text(encoding="utf-8"))
_CANCEL_MSG = _SCHEMA.get("uncertain_confirm", {}).get("cancel_message", "Okay, I won't.")


@pytest.fixture(scope="module")
def eng():
    pytest.importorskip("onnxruntime")
    from nlu_engine import NLUEngine
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return NLUEngine(model_name="en", language="en", semantic_enabled=False)


needs_model = pytest.mark.skipif(not _MODEL.exists(),
                                 reason="trained English artifacts absent")


# --------------------------- the reported bugs -----------------------------

@needs_model
def test_off_topic_reply_does_not_fabricate_a_memory(eng):
    """'who is the prime minister of india' used to set memory=three via 'the'."""
    first = eng.handle("mem-offtopic", "change memory")
    assert first.type == "PROMPT"
    r = eng.handle("mem-offtopic", "who is the prime minister of india")
    assert r.type != "FULFILL" or r.action != "memory.change", (
        f"off-topic reply fabricated a memory change: {r.parameters}")
    # It should re-ask for the memory, not commit anything.
    assert r.type in ("PROMPT", "FALLBACK")


@needs_model
def test_no_during_reminder_time_cancels_instead_of_creating(eng):
    """'no' to 'when should I remind you?' used to create a November reminder."""
    eng.handle("rem-no", "set me up for a concert")
    r = eng.handle("rem-no", "no")
    assert not (r.type == "FULFILL" and r.action == "reminders.task.create"), (
        f"'no' created a reminder anyway: {r.parameters}")
    assert r.message == _CANCEL_MSG


# ------------------------- the outcomes generalise -------------------------

@needs_model
@pytest.mark.parametrize("cancel_word", ["no", "nope", "cancel", "never mind", "forget it"])
def test_pure_cancellation_abandons_the_flow(eng, cancel_word):
    """PURE meta-words with no intent of their own cancel the flow.

    Deliberately excludes "stop": it classifies as streaming.session.stop, a
    real device command, so it is handled by the interruption path (which runs
    first) rather than as a cancel. That precedence is correct — a genuine
    command wins over the meta layer; only words with no intent fall through to
    cancellation. See test_a_real_command_word_interrupts_not_cancels.
    """
    eng.handle(f"cancel-{cancel_word}", "set me up for a concert")
    r = eng.handle(f"cancel-{cancel_word}", cancel_word)
    assert r.message == _CANCEL_MSG and r.action is None


@needs_model
def test_a_real_command_word_interrupts_not_cancels(eng):
    """A word that IS a device command interrupts the flow instead of cancelling.

    "stop" -> streaming.session.stop. The interruption path runs before the
    cancel layer, so a real command takes precedence over meta-cancellation.
    """
    eng.handle("stop-cmd", "set me up for a concert")
    r = eng.handle("stop-cmd", "stop")
    assert r.message != _CANCEL_MSG
    assert r.interrupted_intent == "reminders.task.create"


@needs_model
def test_correction_carrying_a_value_still_fills(eng):
    """A refusal that ALSO gives a value is a correction, not a cancel."""
    eng.handle("rem-correct", "remind me to call mom")
    r = eng.handle("rem-correct", "no, tomorrow at 5")
    # The datetime is extracted; the flow advances (fulfils or confirms), it is
    # NOT cancelled.
    assert r.message != _CANCEL_MSG


@needs_model
def test_repeated_unparseable_answers_fall_back_not_fabricate(eng):
    """After MAX_SLOT_ATTEMPTS of non-answers the flow exits gracefully."""
    eng.handle("mem-stuck", "change memory")
    last = None
    for _ in range(4):
        last = eng.handle("mem-stuck", "who is the prime minister of india")
    assert last.type == "FALLBACK", (
        "an unanswerable slot should route out after the attempt budget, "
        f"got {last.type}")
