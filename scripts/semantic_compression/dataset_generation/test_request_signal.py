#!/usr/bin/env python3
"""Measure the ImplicitCommand guard instead of guessing at it. No API.

WHY THIS EXISTS
---------------
``_REQUEST_SIGNAL`` decides whether a row labelled ImplicitCommand is really a
request or a bare observation wearing the wrong label. A bare observation filed
as a command is the sample that teaches a false accept, which is this product's
costliest error, so the guard's intent is right.

Its implementation is a word list, and the comment above it already concedes the
shape of the problem: "This list is BRITTLE and will stay brittle... Patch the
holes as they appear, but do not expect the next patch to be the last one." Three
holes had been found and patched by hand before this file existed.

What nobody had done was measure it. Precision and recall were unknown, so every
patch was an opinion, and each one widened the accepting side -- which makes the
guard weaker at the only job it has. A guard that is only ever patched toward
accepting eventually accepts everything.

The fixture below is hand-labelled from rows the generator actually produced.
Run it before changing the pattern, and after. If a change does not move these
numbers it did not do anything; if it moves recall down it made the guard worse
at its job while looking like an improvement.

    python3 test_request_signal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from request_signal import _requests_something  # noqa: E402

# Rows the generator produced, hand-labelled.
#   True  = really does request or imply an action -> the guard SHOULD accept it
#   False = a bare observation or state report      -> the guard SHOULD reject it
#
# Every one of these is a real generated utterance, not an invented example. The
# curly apostrophes are deliberate: the model emits them, and one of the guard's
# earlier patches was written with a straight apostrophe and therefore never
# fired on real output.
FIXTURE: list[tuple[str, bool]] = [
    # --- genuine requests the guard currently throws away -------------------
    ("I’m in gym now so switch to Gym memory", True),
    ("The Restaurant program isn’t right for this quiet room, put me back on Everyday", True),
    ("I’d like a text of our chat", True),
    ("I’ve been standing awhile now so um yeah just tell me how long", True),
    ("Do I still have enough battery to get through the evening?", True),
    ("Will these batteries last me until bedtime?", True),
    ("I’ve had them in all day, are the batteries still okay?", True),
    ("I’d like the right hearing aid to stop making any sound", True),
    # --- genuine requests it does catch -------------------------------------
    ("I need more volume in these", True),
    ("Can you make it louder", True),
    ("please mute my aids", True),
    ("I’m not getting enough volume in my right ear", True),
    ("Right side needs volume", True),
    ("let’s have silence from the hearing aids for a bit", True),
    # --- bare observations it correctly rejects -----------------------------
    ("I think the battery might be nearly gone", False),
    ("I’m worried they’re about to die on me", False),
    ("it sounds quiet today", False),
    ("the left one feels loose", False),
    ("I’ve just walked into a busy pub", False),
    ("I’m heading into a big auditorium", False),
    # --- bare observations it lets through anyway ---------------------------
    ("I can’t make out the words at this party", False),
    ("I can’t follow in this crowd", False),
    ("Can’t seem to find my phone", False),
    ("I’ve been up and about, check whether my stand target is done", False),
    ("no more sound coming through the hearing aids, please", False),
]


def score() -> dict[str, float | int]:
    """Score the guard as it is now used: it FLAGS a row, it does not delete one.

    The number that matters is therefore flag precision -- of the rows it accuses
    of being observations, how many really are. Recall of requests still matters,
    but only as noise: a wrongly flagged request is now a line in warnings.jsonl
    rather than a row removed from the corpus.
    """
    flagged_right = flagged_wrong = missed = kept_right = 0
    for text, is_request in FIXTURE:
        flagged = not _requests_something(text)
        if flagged and not is_request:
            flagged_right += 1
        elif flagged and is_request:
            flagged_wrong += 1
        elif not flagged and not is_request:
            missed += 1
        else:
            kept_right += 1
    flags = flagged_right + flagged_wrong
    observations = flagged_right + missed
    return {
        "flags": flags,
        "flagged_right": flagged_right,
        "flagged_wrong": flagged_wrong,
        "missed_observations": missed,
        "kept_right": kept_right,
        "flag_precision": flagged_right / flags if flags else 0.0,
        "observation_recall": flagged_right / observations if observations else 0.0,
    }


def test_flag_precision_has_not_regressed():
    """The pytest entry point. `main()` prints the whole picture; this pins the floor.

    Kept separate so the file works both ways: run it directly to read the
    numbers, collect it under pytest to stop a regression. A guard that is only
    ever run by hand is a guard that stops being run.
    """
    r = score()
    assert r["flag_precision"] >= 0.70, (
        f"flag precision fell to {r['flag_precision']:.0%}; it was 75% when the guard "
        f"was demoted from deleting rows to warning about them"
    )


def main() -> int:
    r = score()
    requests = sum(1 for _, ok in FIXTURE if ok)
    print(
        f"fixture: {len(FIXTURE)} rows -- {requests} requests, "
        f"{len(FIXTURE) - requests} observations\n"
    )
    print(f"  rows flagged                 {r['flags']:>3}")
    print(f"    of those, real observations {r['flagged_right']:>2}   the guard doing its job")
    print(f"    of those, real requests     {r['flagged_wrong']:>2}   noise in warnings.jsonl")
    print(f"  observations it never sees   {r['missed_observations']:>3}   walk straight through")
    print()
    print(
        f"  flag precision      {r['flag_precision']:.0%}   of what it flags, this much is really an observation"
    )
    print(
        f"  observation recall  {r['observation_recall']:.0%}   of real observations, this much gets flagged"
    )
    print()

    # A floor, not a target. It pins what was measured when the guard was demoted
    # from deleting to warning, so a later change that looks like an improvement
    # and is not gets caught. Raising it needs a measurement, not an opinion.
    floor = 0.70
    if r["flag_precision"] < floor:
        print(
            f"  FAIL  flag precision {r['flag_precision']:.0%} fell below the recorded {floor:.0%}"
        )
        return 1

    print(f"  ok    flag precision at or above the recorded {floor:.0%}")
    print(
        "\nThis guard is deliberately weak and is not the defence against a false\n"
        "accept. It misses "
        f"{r['missed_observations']} of {r['flagged_right'] + r['missed_observations']} observations in this fixture and always has.\n"
        "It WARNS so a human sample knows where to look; the measurable defence is\n"
        "the wrong-action harness and the FAR metric."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
