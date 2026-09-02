#!/usr/bin/env python3
"""What ends up as the reminder's NAME, for every shape of request.

The open `remind` slot is filled with whatever `_derive_topic` leaves after the
request wrapper is stripped. Nothing asserted that text until now: the
conversation fixtures in `tests/conversations/` check the intent and the turn
type, and `tests/datetime_parity/` checks the resolved time — so a reminder could
be created, reported as success, and read back to the user with the wrong name
without a single test going red. Five separate defects lived in that gap.

The cases below are grouped by the defect they pin. Group 1 is the regression
guard: it passed before any of these fixes and must keep passing.

These need no trained model — `_derive_topic` is pure string work over the pack's
own carrier and grammar tables. The iOS equivalents live in VoiceAIKit
`TopicDerivationParityTests` / `OpenSlotNameDerivationTests`; the two runtimes
must agree case for case, so a change here without the matching Swift change is
the divergence VIK-025 was about.

Run: pytest tests/test_reminder_name_derivation.py
"""

import pytest

pytest.importorskip("numpy")
pytest.importorskip("joblib")

from nlu_engine.engine import NLUEngine          # noqa: E402
from nlu_engine.entities import EntityExtractor  # noqa: E402


class _Topic:
    """`_derive_topic` and its two helpers, without the model or ONNX runtime.

    The real functions are bound here rather than copied, so a change to the
    engine's logic fails these tests instead of passing quietly.
    """

    def __init__(self):
        self.entities = EntityExtractor()
        self._carrier = NLUEngine._build_carrier_patterns("en")
        self._leading_connector = NLUEngine._build_leading_connector("en")

    _derive_topic = NLUEngine._derive_topic


@pytest.fixture(scope="module")
def topic():
    t = _Topic()
    return lambda text: t._derive_topic(text)


# --- 1. the regression guard -------------------------------------------------
# Every one of these passed before the fixes below and must keep passing. If a
# case here breaks, the fix that broke it is wrong, not this file.

@pytest.mark.parametrize("said,name", [
    ("remind me to buy milk",                              "buy milk"),
    ("remind me to go for a walk",                         "go for a walk"),
    ("set a reminder to call the plumber",                 "call the plumber"),
    ("can you remind me to buy milk",                      "buy milk"),
    ("don't let me forget to take my pills",               "take my pills"),
    ("make sure i call mom tomorrow",                      "call mom"),
    ("i need to pick up the prescription on friday",       "pick up the prescription"),
    ("remind me to call mom tomorrow",                     "call mom"),
    ("remind me to take my pills at 9am",                  "take my pills"),
    ("notify me to drink water at 5",                      "drink water"),
    ("alert me to leave for the airport at 6:30",          "leave for the airport"),
    ("set a reminder to call the plumber tomorrow morning", "call the plumber"),
    ("remind me to water the plants in 10 minutes",        "water the plants"),
    ("remind me at 9 to call mom",                         "call mom"),
    ("remind me on friday to pay rent",                    "pay rent"),
    ("remind me in 10 minutes to check the oven",          "check the oven"),
    ("remind me to call the audiologist tomorrow at 3pm",  "call the audiologist"),
    ("remind me to take my medication tomorrow at 8am",    "take my medication"),
    ("remind me about the gas bill",                       "the gas bill"),
    # A payload whose words merely LOOK like date words must survive intact.
    ("remind me to buy sun cream",                         "buy sun cream"),
    ("remind me to drink the green tea",                   "drink the green tea"),
])
def test_the_ordinary_shapes_keep_working(topic, said, name):
    assert topic(said) == name


# --- 2. a leading time expression ---------------------------------------------

@pytest.mark.parametrize("said,name", [
    ("tomorrow morning remind me to water the plants", "water the plants"),
    ("at 5pm i have to call the doctor",               "call the doctor"),
    ("on monday remind me to pay rent",                "pay rent"),
    ("in 10 minutes remind me to check the oven",      "check the oven"),
])
def test_a_time_in_front_does_not_hide_the_carrier(topic, said, name):
    """Every carrier is `^`-anchored, so a leading time pushed it out of reach and
    the whole sentence became the name. `_derive_topic` strips the date/time first
    for exactly this reason — see the ordering note on that method."""
    assert topic(said) == name


# --- 3. the connector must not bite into the next word ------------------------

@pytest.mark.parametrize("said,name", [
    ("remind me tomorrow to water the plants", "water the plants"),
    ("remind me toothbrush replacement",       "toothbrush replacement"),
    ("remind me office party on friday",       "office party"),
])
def test_the_optional_connector_stops_at_a_word_boundary(topic, said, name):
    """`(?:to|that|about|of)?` had no `\\b`, so it ate the start of the next word:
    "tomorrow" -> "morrow", "toothbrush" -> "othbrush", "office" -> "fice"."""
    assert topic(said) == name


# --- 4. the carrier list is a shape, not a list of verbs ----------------------

@pytest.mark.parametrize("said,name", [
    ("nudge me to stretch",                "stretch"),
    ("ping me to leave for the airport",   "leave for the airport"),
    ("make a note to call the plumber",    "call the plumber"),
    ("add a reminder to buy milk",         "buy milk"),
    ("i mustn't forget the dry cleaning",  "the dry cleaning"),
    ("i can't forget to call the bank",    "call the bank"),
])
def test_a_verb_outside_the_enumerated_list_still_strips(topic, said, name):
    """The list named its verbs (`remind|tell|alert|notify`), so anything else kept
    its wrapper: "nudge me to stretch" was stored verbatim. The set of verbs people
    reach for is open; the sentence shape is not."""
    assert topic(said) == name


# --- 5. a spelled-out clock time is still a clock time ------------------------

@pytest.mark.parametrize("said,name", [
    ("remind me to call Mukesh at 9",    "call Mukesh"),
    ("remind me to call Mukesh at nine", "call Mukesh"),
    ("remind me to take pills at ten",   "take pills"),
])
def test_a_spelled_out_time_leaves_the_name_just_like_a_digit_one(topic, said, name):
    """VIK-040, on the Python side. The parser normalises "nine" to 9 before it
    matches; the strip patterns were written in `\\d` only, so the word stayed in
    the name while the time resolved correctly."""
    assert topic(said) == name


# --- 6. a recurrence must not eat a content word ------------------------------

@pytest.mark.parametrize("said,name", [
    ("every friday remind me to take the bins out",  "take the bins out"),
    ("remind me to take the bins out every friday",  "take the bins out"),
    ("every day remind me to take my pills",         "take my pills"),
    ("each saturday remind me to water the plants",  "water the plants"),
])
def test_a_recurrence_does_not_swallow_the_next_word(topic, said, name):
    """The recurrence pattern consumes its cue plus one following word (normally the
    unit, "every day"). Running it after the weekday strip meant the weekday was
    already gone, so it swallowed whatever real word had moved up behind it —
    "every friday take the bins out" lost "take"."""
    assert topic(said) == name


def test_the_carrier_order_still_puts_prefixes_first():
    """`_derive_topic` makes ONE pass in list order and every pattern is `^`-anchored.
    A prefix-stripper that runs late never gets its turn back: "please remind me to X"
    needs `^please` to strip BEFORE the carrier that follows it, or the carrier is
    tested against a string that still starts with "please" and misses."""
    c = NLUEngine._build_carrier_patterns("en")
    please = next(i for i, p in enumerate(c) if "please" in p and "can|could" not in p)
    politeness = next(i for i, p in enumerate(c) if "can|could|would|will" in p)
    generic = next(i for i, p in enumerate(c) if r"\w+\s+me\s+" in p)
    specific = next(i for i, p in enumerate(c) if "remind|tell|alert|notify" in p)
    assert politeness < please < generic < specific, (
        "order must be: prefix-strippers, then generic shapes, then specific carriers")
