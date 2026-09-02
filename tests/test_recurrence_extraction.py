#!/usr/bin/env python3
"""A one-off reminder must not come back marked as repeating.

`recurrence` is an optional slot on `reminders.add`, and the host renders it under
the label "Repeat" (VoiceAIKit `SlotFormatting.displayName`). So any value in it is
a promise to the user that the reminder recurs.

Nine values in the entity broke that promise, because their matchable text was a
bare DATE rather than a repetition:

  * the seven singular weekdays (`Friday` … `Sunday`), whose only synonym was
    "friday" — so "remind me to pay rent on friday", a one-off, came back as
    `recurrence: Friday` and displayed "Repeat: Friday";
  * `3 Months` / `6 Months`, whose CANONICAL NAME is itself matchable text
    (`entities.py` seeds the lookup with `table[value.lower()] = value`), so
    "renew the insurance in 3 months" matched even after the bare synonym was
    removed. The name had to carry the marker too, hence "Every 3 Months".

Nothing was lost. Every recurring phrasing those values covered is already on the
plural / marked value: "every friday", "each friday" and "fridays" all resolve to
`Fridays`.

The invariant these tests defend is in `test_every_matchable_form_carries_a_marker`:
every string the runtime can match on — synonym AND canonical — has to say that it
repeats. That is the rule; the cases below are what it looks like in use.

Run: pytest tests/test_recurrence_extraction.py
"""

import re

import pytest

pytest.importorskip("numpy")

from nlu_engine.entities import EntityExtractor  # noqa: E402


@pytest.fixture(scope="module")
def entities():
    return EntityExtractor()


# --- one-off reminders carry no recurrence -----------------------------------

@pytest.mark.parametrize("said", [
    "remind me to pay rent on friday",
    "remind me to buy milk on monday",
    "remind me to call the dentist this tuesday",
    "remind me to submit the form next wednesday",
    "remind me to renew the insurance in 3 months",
    "remind me in 6 months to book a checkup",
    "remind me to call mom tomorrow",
    "remind me to water the plants tomorrow morning",
    "remind me to take pills at 9am",
])
def test_a_one_off_reminder_has_no_recurrence(entities, said):
    value, _, _ = entities.extract("recurrence", said)
    assert value is None, f"{said!r} would display 'Repeat: {value}' for a one-off"


# --- genuine recurrences still resolve ---------------------------------------

@pytest.mark.parametrize("said,value", [
    ("remind me to take the bins out every friday",  "Fridays"),
    ("every friday remind me to take the bins out",  "Fridays"),
    ("remind me to back up fridays",                 "Fridays"),
    ("remind me to take my pills every day",         "Daily"),
    ("remind me to stretch every morning",           "Daily"),
    ("remind me to pay rent every month",            "Monthly"),
    ("remind me to weigh in weekly",                 "Weekly"),
    ("remind me to renew the policy every 6 months", "Every 6 Months"),
])
def test_a_real_recurrence_still_resolves(entities, said, value):
    got, _, _ = entities.extract("recurrence", said)
    assert got == value


# --- the rule itself ----------------------------------------------------------

# A repetition marker: an explicit cue ("every", "each"), an adverbial form
# ("daily", "weekly", "biweekly"), or a plural ("fridays"). Anything without one
# is a date or a duration, and a date is not a recurrence.
_MARKERS = re.compile(r"^(?:every|each)\b|(?:ly|s)$|^everyday$", re.I)


def test_every_matchable_form_carries_a_marker(entities):
    """Both halves matter: the runtime matches on canonical names as well as
    synonyms, and it was a canonical ("3 Months") that survived the first fix."""
    values = entities.entities["recurrence"]["values"]
    matchable = sorted({k for k in values} | {s for v in values.values() for s in v})
    unmarked = [m for m in matchable if not _MARKERS.search(m)]
    assert not unmarked, (
        "these recurrence forms do not say they repeat, so a one-off mentioning "
        f"them is reported as recurring: {unmarked}")


def test_dropping_the_singular_weekdays_lost_no_recurring_phrasing(entities):
    """The seven singular values were removed outright. Guard the premise that
    made that safe: their plural sibling still covers every recurring form."""
    values = entities.entities["recurrence"]["values"]
    for day in ("monday", "tuesday", "wednesday", "thursday",
                "friday", "saturday", "sunday"):
        plural = day.capitalize() + "s"
        assert plural in values, f"{plural} must exist to carry {day!r}"
        forms = {s.lower() for s in values[plural]}
        assert {day + "s", "every " + day, "each " + day} <= forms, \
            f"{plural} lost a recurring phrasing: {sorted(forms)}"
        assert day not in values, f"the bare value {day.capitalize()!r} is back"
