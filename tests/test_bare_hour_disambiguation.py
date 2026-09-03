#!/usr/bin/env python3
"""Which half of the clock a bare hour means — "at 6" is 06:00 or 18:00.

THE RULE: take the EARLIEST reading that is still ahead of us. One rule
everywhere — today, tomorrow, a named weekday. No daytime preference, no range
of hours that "usually mean PM".

What it replaced guessed the half BEFORE looking at the clock (1-6 -> PM,
7-12 -> AM) and consulted `now` only to rescue a guess that had landed in the
past. At 05:41 "wake me at 6" therefore scheduled 18:00 — 06:00 was nineteen
minutes away and in the future, but the guess had not landed in the past, so
nothing rescued it.

This is deliberately literal rather than clever. The engine cannot know whether
"at 6" is a wake-up or a dinner, so it does not pretend to: the user said six,
six is what they get, and "6 pm" is there for anyone who means the evening.

The accepted cost is on a named day, where both readings are ahead of us and the
earliest is therefore always the morning one: "meeting tomorrow at 5" resolves to
05:00. That is the user's half of the contract. The alternative was a range of
hours assumed to mean PM, which is an English-speaking assumption no language
pack could disagree with — VIK-042 exists because it was in the engine at all,
and narrowing it (1-5? 1-4?) would only have retuned the magic number.

WHY THIS FILE EXISTS SEPARATELY FROM THE GOLDEN CORPUS.
`tests/datetime_parity/` is captured at ONE clock, 14:30. Every case this rule
changes falls between 00:00 and 05:59, so the corpus cannot see any of them —
the whole change ran green against it. A rule about the current time needs to be
tested at more than one current time.

Run: pytest tests/test_bare_hour_disambiguation.py
"""

from datetime import datetime, timedelta

import pytest

pytest.importorskip("numpy")

from nlu_engine.entities import EntityExtractor  # noqa: E402

_DAY = datetime(2026, 9, 3)          # a Thursday


@pytest.fixture(scope="module")
def ex():
    return EntityExtractor()


def at(ex, text, now):
    iso, _span, _conf, _te, _ed = ex.extract_datetime(text, now=now)
    assert iso is not None, f"{text!r} did not resolve at {now:%H:%M}"
    return datetime.fromisoformat(iso).replace(tzinfo=None)


def clock(h, m=0):
    return _DAY.replace(hour=h, minute=m)


# --- today first ---------------------------------------------------------

@pytest.mark.parametrize("now_h,now_m,said,expect_h,expect_day", [
    (5, 41, 6, 6, 0),     # the reported bug: 06:00 is 19 minutes away
    (4,  0, 6, 6, 0),
    (4,  0, 5, 5, 0),
    (2, 30, 3, 3, 0),     # awake at 2:30, "at 3" means 3am
    (0, 17, 1, 1, 0),     # awake past midnight, "at 1" means 1am
    (9,  0, 11, 11, 0),   # morning still ahead
    (19, 0, 11, 23, 0),   # evening: 23:00 is the reading still ahead today
    (15, 0, 5, 17, 0),    # 05:00 gone, 17:00 still ahead today
])
def test_a_reading_still_ahead_today_wins(ex, now_h, now_m, said, expect_h, expect_day):
    got = at(ex, f"remind me at {said}", clock(now_h, now_m))
    assert got == _DAY.replace(hour=expect_h) + timedelta(days=expect_day)


# --- today spent: the daytime reading ------------------------------------

@pytest.mark.parametrize("now_h,said,expect_h", [
    (15, 3, 3),
    (16, 2, 2),
    (18, 1, 1),
    (22, 5, 5),
    (23, 6, 6),
    (12, 12, 0),
])
def test_when_both_readings_are_behind_us_the_earlier_one_returns_tomorrow(
        ex, now_h, said, expect_h):
    """Same rule, one day on — not a different rule for tomorrow."""
    got = at(ex, f"remind me at {said}", clock(now_h))
    assert got == _DAY.replace(hour=expect_h) + timedelta(days=1)


# --- a named day has no "today" to prefer --------------------------------

@pytest.mark.parametrize("said,expect_h", [
    ("remind me to take medicine tomorrow at 6", 6),
    ("remind me tomorrow at 3", 3),
    ("remind me tomorrow at 11", 11),
    ("remind me tomorrow at 5 pm", 17),      # say pm and you get pm
    ("remind me tomorrow morning at 6", 6),
])
def test_a_named_day_takes_the_earliest_reading_too(ex, said, expect_h):
    """The rule does not change because the day was named. Both readings are ahead
    of us there, so the earliest is the morning one — "take medicine tomorrow at 6"
    is 06:00, and "meeting tomorrow at 5" is 05:00 unless the user says pm."""
    got = at(ex, said, clock(6, 0))
    assert got == _DAY.replace(hour=expect_h) + timedelta(days=1)


@pytest.mark.parametrize("said,expect_h,expect_day", [
    ("remind me at noon", 12, 0),
    ("remind me tomorrow at noon", 12, 1),
    ("remind me at midnight", 0, 1),
])
def test_a_named_hour_is_not_ambiguous(ex, said, expect_h, expect_day):
    """Noon is 12:00 and says so; it must not be read as a bare 12 and offered
    00:00 as its other half. The rule this replaced kept noon by leaving 7-12
    alone — luck, not handling."""
    got = at(ex, said, clock(10, 30))
    assert got == _DAY.replace(hour=expect_h) + timedelta(days=expect_day)


# --- an explicit am/pm is never touched ----------------------------------

@pytest.mark.parametrize("said,expect_h,expect_day", [
    ("remind me at 1 am", 1, 1),          # 01:00 has passed, so tomorrow
    ("remind me at 5 am", 5, 1),
    ("set a reminder to go to the airport at 1 am", 1, 1),
    ("remind me at 6 am", 6, 0),          # still ahead today
    ("remind me at 1 pm", 13, 0),
    ("remind me at 11 pm", 23, 0),
])
def test_an_explicit_period_is_taken_at_its_word(ex, said, expect_h, expect_day):
    """1 AM is a legitimate time for a flight. Disambiguation applies only where
    there is something to disambiguate."""
    got = at(ex, said, clock(5, 41))
    assert got == _DAY.replace(hour=expect_h) + timedelta(days=expect_day)


# --- the invariant --------------------------------------------------------

def test_a_bare_hour_never_resolves_into_the_past(ex):
    """A reminder must never be created for a moment that has already gone."""
    late = []
    for now_h in range(24):
        for now_m in (0, 30, 59):
            now = clock(now_h, now_m)
            for said in range(1, 13):
                for form in ("remind me at {h}", "remind me at {h} am",
                             "remind me at {h} pm", "remind me tomorrow at {h}"):
                    got = at(ex, form.format(h=said), now)
                    if got <= now:
                        late.append((form.format(h=said), now, got))
    assert not late, late[:5]
