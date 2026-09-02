#!/usr/bin/env python3
"""VIK-038 — which awaited slots may have their answer re-classified.

While a slot is being filled, the engine probes the classifier to see whether the
user changed the subject. The probe is trained on COMMANDS, so a slot ANSWER is
out-of-distribution input for it, and a confidence score on OOD input is not a
quantity that can be thresholded. Measured on this pack's own weights, answering
the reminder's "What do you want to be reminded?" with

    "walk the dog"           -> Cmd.ActivityWalk      1.000
    "clean my hearing aids"  -> Help_CleanCare        1.000
    "start my workout"       -> Cmd.ActivityExercise  0.995

abandoned the reminder and did the other thing instead. Raising the threshold
cannot fix it: "start my workout" (a legitimate reminder) outscores "start
transcribing" (a real command, 0.962).

So the probe is gated on what the awaited slot can REFUSE, never on the intent's
name — this engine does not interpret intent labels:

    closed enum      a value is in the list or it is not; a miss is a fact  PROBE
    open free-text   every utterance is a legal value                      NO PROBE
    date-time        the parser decides, not the classifier                NO PROBE

These assert the gate itself rather than a whole conversation, so they need no
trained model and run everywhere. The equivalent iOS assertions live in
VoiceAIKit `OpenSlotNameDerivationTests`; the two must not diverge.

Run: pytest tests/test_slot_interruption_gate.py
"""

import json
from pathlib import Path

import pytest

# conftest.py puts packages/runtime on sys.path. The engine module imports numpy
# and joblib but no ML runtime at import time, so this needs no trained model.
pytest.importorskip("numpy")
pytest.importorskip("joblib")

from nlu_engine.engine import NLUEngine          # noqa: E402
from nlu_engine.entities import EntityExtractor  # noqa: E402

_BASE = Path(__file__).resolve().parent.parent
_SCHEMA = json.loads(
    (_BASE / "language_packs" / "en" / "nlu_schema.json").read_text(encoding="utf-8"))


class _Session:
    """Only the field the gate reads."""

    def __init__(self, awaiting):
        self.awaiting_slot = awaiting


class _Stub:
    """Carries the one collaborator the gate touches, so no model is needed.

    The real `NLUEngine` functions are bound to this below rather than copied, so
    a change to the engine's logic fails these tests instead of passing quietly.
    """

    def __init__(self):
        self.entities = EntityExtractor()

    _slot_def = staticmethod(NLUEngine._slot_def)
    _slot_can_refuse_the_answer = NLUEngine._slot_can_refuse_the_answer


def _gate_for(intent_name, awaiting):
    return _Stub()._slot_can_refuse_the_answer(
        _Session(awaiting), _SCHEMA["intents"][intent_name])


# --- the premise these tests rest on -------------------------------------

def test_the_reminder_flow_has_the_slot_kinds_these_tests_assume():
    """Fails loudly if the schema is retyped underneath the assertions below."""
    ents = EntityExtractor()
    slots = {s["name"]: s["entity"] for s in _SCHEMA["intents"]["reminders.add"]["slots"]}
    assert ents.is_open(slots["name"]), "the reminder's name slot must be OPEN"
    assert ents.is_date_time(slots["date-time"]), "the time slot must be a date-time"
    memory = {s["name"]: s["entity"] for s in _SCHEMA["intents"]["Cmd.MemoryChange"]["slots"]}
    assert not ents.is_open(memory["MemoryName"]), "the memory slot must be CLOSED"


# --- the gate ------------------------------------------------------------

def test_an_open_free_text_slot_is_never_probed():
    """The reminder's name accepts anything, so there is nothing to be right about."""
    assert _gate_for("reminders.add", "name") is False


def test_a_date_time_slot_is_never_probed():
    """The parser owns the answer to "when should I remind you?", not the classifier."""
    assert _gate_for("reminders.add", "date-time") is False


def test_a_closed_enum_slot_is_still_probed():
    """A memory name is in the list or it is not, so a genuine switch still fires.

    This is the case the gate must NOT break: "change my memory" then "increase
    volume" has to keep switching intent.
    """
    assert _gate_for("Cmd.MemoryChange", "MemoryName") is True


def test_no_awaited_slot_leaves_the_previous_behaviour_alone():
    assert _gate_for("reminders.add", None) is True


# --- the entity seam the gate asks through -------------------------------

@pytest.mark.parametrize("spelling", ["sys.date-time", "sys.date_time"])
def test_both_spellings_of_the_datetime_entity_are_recognised(spelling):
    """VIK-018: this schema hyphenates it and the v3 bundle surface does not.

    iOS compared the hyphenated literal, so under a pack the comparison was always
    false and every date slot silently failed to fill. Asking the question in one
    place is the fix; this pins it.
    """
    assert EntityExtractor().is_date_time(spelling) is True


def test_an_ordinary_entity_is_not_a_datetime():
    assert EntityExtractor().is_date_time("memory") is False


# --- VIK-041: politeness prefixes are carriers ---------------------------

@pytest.mark.parametrize("said,expected", [
    ("remind me to go for a walk",            "go for a walk"),
    ("Can you remind me to go for a walk",    "go for a walk"),
    ("Could you please remind me to buy milk", "buy milk"),
    ("Would you remind me to call mom",       "call mom"),
    ("I want you to remind me to call mom",   "call mom"),
])
def test_a_politeness_prefix_does_not_survive_into_the_reminder_name(said, expected):
    """VIK-041. Every carrier is `^`-anchored, so "Can you" in front of "remind me"
    stopped carrier 2 matching and the whole sentence became the reminder's name.

    ORDER IS LOAD-BEARING and the next test pins it.
    """
    carriers = NLUEngine._build_carrier_patterns("en")
    import re
    t = said.strip()
    for pat in carriers:
        t = re.sub(pat, "", t, count=1, flags=re.I)
    assert t.strip(" .,") == expected


def test_the_politeness_carriers_come_before_the_remind_me_carrier():
    """`_derive_topic` makes ONE pass in list order and every pattern is `^`-anchored,
    so a pattern is only tested against the string as it stands when its turn comes.

    With "can you" AFTER "remind me", "^remind me" is tried against "can you remind
    me…", misses, and is never retried once "^can you" has stripped — leaving
    "remind me to go for a walk" as the name. This is why `^please` is index 0.
    """
    carriers = NLUEngine._build_carrier_patterns("en")
    politeness = next(i for i, c in enumerate(carriers) if "can|could|would|will" in c)
    remind_me = next(i for i, c in enumerate(carriers) if "remind|tell|alert|notify" in c)
    assert politeness < remind_me, (
        "politeness prefixes must precede the 'remind me' carrier — see the "
        "ordering note in _DEFAULT_CARRIERS")
