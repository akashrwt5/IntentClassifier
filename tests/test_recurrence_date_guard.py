#!/usr/bin/env python3
"""The build refuses entity values that are dates rather than repetitions.

`recurrence` is rendered to the user as "Repeat" (VoiceAIKit
`SlotFormatting.displayName`), so every value in it promises the reminder recurs.
Seven of its values were bare weekday names, which are DATES — "remind me to pay
rent on friday", a one-off, came back as `recurrence: Friday` and displayed
"Repeat: Friday". Nothing threw; the user was simply told the wrong thing, and it
shipped.

The guard is in `compile_entities`, so bad content cannot be built into a pack at
all. Three properties matter and each has a test here:

  * it is LANGUAGE-NEUTRAL — the date terms come from that language's own
    `datetime.json`, never from a list of English words in the compiler;
  * it is OPT-IN — a date-like value is fine in `memory` or `remind`, so the
    content declares where the rule applies rather than the compiler hardcoding
    an entity name;
  * it checks CANONICALS as well as synonyms — the runtime seeds its lookup with
    `table[value.lower()] = value`, so a canonical is matchable text in its own
    right. "3 Months" kept matching "in 3 months" after its bare synonym was
    removed, because the NAME still read as a duration.

Run: pytest tests/test_recurrence_date_guard.py
"""

import copy
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "packages" / "buildtime") not in sys.path:
    sys.path.insert(0, str(_ROOT / "packages" / "buildtime"))

from nlu_compiler.content_bundle import (  # noqa: E402
    _check_values_are_not_dates,
    _date_terms,
)

_ENTITIES = _ROOT / "language_packs" / "en" / "nlu_entities.json"


@pytest.fixture
def src():
    return json.loads(_ENTITIES.read_text(encoding="utf-8"))


# --- the shipped content passes -----------------------------------------------

def test_the_shipped_english_content_builds(src):
    _check_values_are_not_dates("en", src)  # must not raise


def test_recurrence_declares_the_rule(src):
    """If this flag is ever dropped the guard goes quiet, so pin it."""
    assert src["recurrence"].get("values_are_not_dates") is True


# --- it catches the defect it exists for --------------------------------------

def test_a_bare_weekday_fails_the_build(src):
    """This is the shape that shipped: the canonical IS the bare date."""
    bad = copy.deepcopy(src)
    bad["recurrence"]["values"]["Friday"] = ["friday"]
    with pytest.raises(SystemExit) as e:
        _check_values_are_not_dates("en", bad)
    assert "Friday" in str(e.value) and "friday" in str(e.value)


def test_a_bare_weekday_hidden_in_a_synonym_list_fails_too(src):
    bad = copy.deepcopy(src)
    bad["recurrence"]["values"]["Weekly"] = ["each week", "every week", "weekly", "tuesday"]
    with pytest.raises(SystemExit) as e:
        _check_values_are_not_dates("en", bad)
    assert "tuesday" in str(e.value)


def test_a_month_or_day_anchor_is_caught_as_well_as_a_weekday(src):
    """The rule is "a date", not "a weekday" — anchors and months count."""
    for form in ("tomorrow", "june"):
        bad = copy.deepcopy(src)
        bad["recurrence"]["values"][form.capitalize()] = [form]
        with pytest.raises(SystemExit) as e:
            _check_values_are_not_dates("en", bad)
        assert form in str(e.value)


# --- and does not fire where it should not ------------------------------------

def test_a_marked_recurring_form_is_allowed(src):
    """"fridays" and "every friday" are not bare dates and must keep building."""
    ok = copy.deepcopy(src)
    ok["recurrence"]["values"]["Fortnightly"] = ["every fortnight", "fortnightly"]
    _check_values_are_not_dates("en", ok)  # must not raise


def test_an_entity_that_does_not_declare_the_rule_is_untouched(src):
    """A memory or reminder named "Sunday" is legitimate; only a Repeat slot is not."""
    other = copy.deepcopy(src)
    other["memory"]["values"]["Sunday"] = ["sunday"]
    _check_values_are_not_dates("en", other)  # must not raise

    # ...and the same value under the marked entity does fail, so the difference
    # really is the declaration and not something incidental.
    marked = copy.deepcopy(src)
    marked["recurrence"]["values"]["Sunday"] = ["sunday"]
    with pytest.raises(SystemExit):
        _check_values_are_not_dates("en", marked)


# --- language neutrality ------------------------------------------------------

def test_the_date_terms_come_from_the_pack_not_from_the_compiler():
    """Nothing about English is compiled in: the terms are this language's own."""
    grammar = json.loads(
        (_ROOT / "language_packs" / "en" / "datetime.json").read_text(encoding="utf-8"))
    expected = {s.lower()
                for table in ("weekdays", "months", "day_anchors")
                for syns in grammar[table].values()
                for s in syns}
    assert _date_terms("en") == expected


def test_a_language_with_no_grammar_stands_down_instead_of_passing_quietly(src, capsys):
    """A guard that silently passes because it could not read its input is worse
    than no guard, so the unreadable case warns and is visible in the build log."""
    assert _date_terms("zz") is None
    _check_values_are_not_dates("zz", src)  # must not raise
    assert "cannot run for this language" in capsys.readouterr().out
