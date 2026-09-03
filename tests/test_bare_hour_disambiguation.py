#!/usr/bin/env python3
"""Which half of the clock a bare hour means — "at 6" is 06:00 or 18:00.

THE RULE: if either reading is still ahead of us TODAY, take the nearer one.
Only when today has run out does a daytime preference apply.

What it replaced guessed the half BEFORE looking at the clock (1-6 -> PM,
7-12 -> AM) and consulted `now` only to rescue a guess that had landed in the
past. At 05:41 "wake me at 6" therefore scheduled 18:00 — 06:00 was nineteen
minutes away and in the future, but the guess had not landed in the past, so
nothing rescued it.

Plain "next future occurrence" is not the answer either, which is why these
tests pin the daytime half as carefully as the today-first half. Say "at 2" at
16:00 and both readings fall tomorrow — 02:00 in ten hours, 14:00 in twenty-two
— so nearest picks 02:00. That is arithmetic, not evidence: the small hours won
only because morning comes first. Measured over 288 (now, hour) pairs, nearest
breaks 51 cases and fixes 33; today-first fixes 21 and breaks none.

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
])
def test_a_reading_still_ahead_today_wins(ex, now_h, now_m, said, expect_h, expect_day):
    got = at(ex, f"remind me at {said}", clock(now_h, now_m))
    assert got == _DAY.replace(hour=expect_h) + timedelta(days=expect_day)


# --- today spent: the daytime reading ------------------------------------

@pytest.mark.parametrize("now_h,said,expect_h", [
    (15, 3, 15),    # NOT 03:00 tomorrow, which "nearest" would pick
    (16, 2, 14),
    (18, 1, 13),
    (22, 5, 17),
    (23, 6, 18),
    (12, 12, 12),   # noon, not midnight
])
def test_when_today_is_spent_the_daytime_reading_wins_tomorrow(ex, now_h, said, expect_h):
    got = at(ex, f"remind me at {said}", clock(now_h))
    assert got == _DAY.replace(hour=expect_h) + timedelta(days=1)


# --- a named day has no "today" to prefer --------------------------------

@pytest.mark.parametrize("said,expect_h", [
    ("remind me tomorrow at 3", 15),
    ("remind me tomorrow at 6", 18),
    ("remind me tomorrow at 11", 11),
])
def test_a_named_day_keeps_the_daytime_reading(ex, said, expect_h):
    """Both readings are ahead of us on a named day, so "earliest ahead" would
    silently mean 03:00. There is no today to prefer — the daytime reading applies."""
    got = at(ex, said, clock(6, 0))
    assert got == _DAY.replace(hour=expect_h) + timedelta(days=1)


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
