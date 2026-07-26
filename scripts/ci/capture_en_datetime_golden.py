#!/usr/bin/env python3
"""
Capture the English datetime GOLDEN corpus from the pristine extractor.

WHY THIS EXISTS
---------------
`tests/datetime_parity/` ships parity fixtures for fr, de and da — but not for
English, because English was the hardcoded special case rather than a
lexicon-driven language. The ~250 lines of inline English regex in
`entities.py::extract_datetime` are therefore completely unguarded.

Charter step A1 (`docs/Review-F5/ENGLISH-PRODUCTION-ROUTINE.md`) captures that
behaviour BEFORE the pack eviction in A7, so the refactor has a regression
oracle instead of enshrining whatever it breaks. Run this against the pristine
extractor once; after that the JSON is the oracle and this script only reruns
when a behaviour change is deliberate.

ADMISSIBILITY — why some utterances are refused
-----------------------------------------------
The parser has a final `dateparser` fallback (§8) that is NOT deterministic for
a golden corpus:

  - it resolves against real wall-clock time, not the pinned `now`;
  - it only exists when the optional `dateparser` package is installed, so the
    same utterance yields a timestamp on one machine and `None` on another.

A case that reaches §8 would therefore make the suite flaky and
environment-dependent. This script admits a case ONLY when it is provably
resolved before §8:

  - `conf == 1.0`  -> resolved by the deterministic English path; or
  - an explicitly declared no-match whose rejection happens at §2
    (the `yesterday` guard), which returns before §8.

Everything else is REFUSED and reported, so a flaky case cannot be captured by
accident. This is deliberate: the corpus is a regression oracle, and an oracle
that depends on an optional dependency is worse than no oracle.

USAGE
    python scripts/ci/capture_en_datetime_golden.py            # write the golden
    python scripts/ci/capture_en_datetime_golden.py --dry-run  # report only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "tests" / "datetime_parity" / "nlu_datetime_parity_en_golden.json"
REF_GRAMMAR_HINT = (
    "git show origin/claude/claude-setup-architecture-ebqobs-Temperaturescaling-fixes"
    ":packs/en/datetime/grammar.json"
)

# Pinned reference instant: Sunday 2026-06-14 14:30 UTC. Mid-afternoon on
# purpose — it exercises the roll-forward paths in `_pick_future_hour` that a
# midnight reference would hide, and an aware UTC value is used by the parser
# as-is (only a naive `now` gets localised), so the corpus is independent of the
# machine's timezone.
NOW = datetime(2026, 6, 14, 14, 30, tzinfo=timezone.utc)


def _load_extractor():
    """Load entities.py directly. It has no heavy dependencies, unlike the
    nlu_engine package __init__, which pulls in numpy."""
    spec = importlib.util.spec_from_file_location(
        "nlu_entities", REPO / "packages" / "runtime" / "nlu_engine" / "entities.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# The corpus — enumerated from the branches of entities.py::extract_datetime,
# section by section, so coverage tracks the source rather than intuition.
# --------------------------------------------------------------------------- #

# (utterance, branch-label) — the label records WHICH parser branch the case is
# there to pin, so a future reader can tell what a regression actually broke.
CORPUS: list[tuple[str, str]] = []


def _add(branch: str, *utterances: str) -> None:
    CORPUS.extend((u, branch) for u in utterances)


# §1 relative durations — digit form
_add("rel.digit",
     "in 10 minutes", "in 5 min", "in 1 minute", "in 45 mins", "in 1 hour",
     "in 2 hours", "in 3 hr", "in 2 days", "in 1 week", "in 3 weeks",
     "remind me in 30 minutes", "in 90 minutes")
# §1 "for N units" — same semantics as "in N units"
_add("rel.for",
     "for 20 minutes", "for 5 min", "for 2 hours", "for 1 hour",
     "for 3 days", "for 1 week")
# §1 "in an hour" / "in a minute"
_add("rel.article",
     "in an hour", "in a minute", "in a day", "in a week", "in an hr", "in a min")
# §1 "a few" / "a couple of"
_add("rel.quantifier",
     "in a few minutes", "in a few hours", "in a couple of minutes",
     "in a couple minutes", "in a couple of hours", "in a few min")
# §1 half an hour
_add("rel.half_hour", "in half an hour", "in half a hour")

# §2 explicit past rejection — returns before §8, so deterministic
_add("reject.yesterday", "yesterday", "remind me yesterday", "yesterday at 9am")

# §3 word-number normalisation. Note these all exercise CLOCK positions, which
# are read in §6 — after the normaliser. See KNOWN_GAPS for the duration case.
_add("wordnum",
     "nine pm", "nine thirty", "ten am", "eleven pm", "at eleven",
     "twelve pm", "seven in the evening")

# --------------------------------------------------------------------------- #
# KNOWN GAPS — real defects found while enumerating the corpus. Recorded, not
# fixed here: A1's acceptance gate requires `entities.py` to stay untouched, and
# changing them is a behaviour change that needs its own step and sign-off.
# --------------------------------------------------------------------------- #
KNOWN_GAPS: list[tuple[str, str]] = [
    ("in five minutes",
     "word-number RELATIVE DURATIONS are unsupported: §1 matches only `\\d+`, and "
     "it runs BEFORE the §3 word-number normaliser, so 'five' never becomes '5'. "
     "`_normalise_word_numbers('in five minutes')` does return 'in 5 minutes', and "
     "'in 5 minutes' parses correctly — only the ordering is wrong. Word-number "
     "CLOCK times ('nine pm') work because §6 runs after the normaliser. "
     "Environment-dependent to boot: with `dateparser` installed the §8 fallback "
     "likely absorbs these, so the feature appears to work in a dev environment "
     "and silently fails in a lean container."),
    ("in twenty minutes", "same root cause as 'in five minutes'"),
]

# §4 day anchors (priority: day-after-tomorrow before tomorrow)
_add("anchor",
     "day after tomorrow", "tomorrow", "today", "tonight", "next week")
_add("anchor.with_time",
     "tomorrow at 9am", "tomorrow at 3pm", "today at 5pm", "next week at noon",
     "day after tomorrow at 10am", "tomorrow at 9:30")

# §4 weekdays
_add("weekday",
     "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_add("weekday.with_time",
     "monday at 3pm", "friday at 9am", "sunday at noon")

# §4 period hints
_add("period",
     "in the morning", "this afternoon", "in the evening", "at night",
     "at noon", "at midnight")
_add("period.with_anchor",
     "tomorrow morning", "tomorrow afternoon", "tomorrow evening",
     "tomorrow night", "monday morning")

# §6 clock idioms
_add("idiom.half_past", "half past 9", "half past 11", "half past 6")
_add("idiom.quarter_past", "quarter past 9", "quarter past 2")
_add("idiom.quarter_to", "quarter to 3", "quarter to 1", "quarter to 12")
_add("idiom.n_past", "20 past 9", "10 past 2", "5 past 11")
_add("idiom.n_to", "10 to 3", "5 to 1", "20 to 12")

# §6 am/pm
_add("ampm",
     "9am", "9 am", "9:30pm", "12am", "12pm", "7pm", "6 am",
     "9 a.m.", "9 p m", "10:15am", "11:45 pm")

# §6 colon clock, no am/pm
_add("colon", "9:30", "14:45", "00:30", "23:59", "7:05")

# §6 continental + decimal notations (shared with fr/de written clock)
_add("continental", "15h30", "9h", "18h45")
_add("decimal", "15.30", "9.05")

# §6 "at N M" / "at N" / bare number
_add("at_hm", "at 9 30", "at 11 15")
_add("at_n", "at 9", "at 11", "remind me at 5", "at 3")
_add("bare", "9", "11", "7")

# §6 space-separated H MM
_add("space_hm", "9 30", "11 45")

# §6 digit paired with a period word
_add("digit_period",
     "9 in the morning", "9 tonight", "7 in the evening", "3 in the afternoon")

# §7 combinations exercising roll-forward against the 14:30 reference
_add("rollforward", "at 9", "at 2", "9am", "2pm", "at 10")

# Carrier-phrase forms the engine strips before slot filling
_add("carrier",
     "remind me tomorrow at 9am", "set a reminder for 3pm",
     "remind me to call mum at 6pm")


# Topic-strip corpus. `strip_datetime` removes date/time fragments so what
# remains is the reminder topic. It is cosmetic — it never changes a resolved
# time — but it had NO test coverage at all before A7, which is how a whole
# function came to be refactored unguarded.
STRIP_CORPUS: list[str] = [
    "remind me to call mum at 6pm", "call the doctor tomorrow",
    "take pills every day", "remind me at 9pm for dinner", "buy milk on monday",
    "dentist at 9:30", "water plants each morning", "meeting in the morning",
    "call dad tonight", "pick up parcel by 5", "gym this evening",
    "standup next monday at 9 am", "review notes in 10 minutes", "yoga at noon",
    "pay rent on the 1st", "text sarah tomorrow morning", "walk dog at 7:15 am",
    "laundry today", "book flight for 2 hours", "check oven in 5 min",
    "nothing to strip here", "remind me", "call mum at 6 p.m. tomorrow",
    "stretch every each morning", "midnight snack", "call at 12:00 midnight",
    "every tuesday standup",
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args(argv)

    mod = _load_extractor()
    if mod._HAS_DATEPARSER:
        print("NOTE: `dateparser` is installed here, so §8 can fire. Cases that "
              "reach it are refused below — the corpus stays dateparser-independent.")
    ex = mod.EntityExtractor()

    cases, refused = [], []
    seen: set[str] = set()
    for utterance, branch in CORPUS:
        if utterance in seen:
            continue  # the same utterance under two branch labels is one case
        seen.add(utterance)

        iso, _span, conf, time_explicit, explicit_day = ex.extract_datetime(
            utterance, now=NOW)

        deterministic_hit = conf == 1.0 and iso is not None
        declared_reject = branch.startswith("reject.") and iso is None

        if not (deterministic_hit or declared_reject):
            refused.append((utterance, branch, iso, conf))
            continue

        cases.append({
            "utterance": utterance,
            "branch": branch,
            "iso": iso,
            "conf": conf,
            "time_explicit": time_explicit,
            "explicit_day": explicit_day,
        })

    print(f"admitted: {len(cases)}   refused: {len(refused)}")
    if refused:
        print("\nREFUSED (would reach the dateparser fallback — not deterministic):")
        for utterance, branch, iso, conf in refused:
            print(f"  [{branch}] {utterance!r} -> iso={iso} conf={conf}")

    # Known gaps must STAY broken until deliberately fixed. If one starts
    # resolving, entities.py changed and the gap list is stale — fail loudly
    # rather than let the record drift out of date.
    healed = [u for u, _ in KNOWN_GAPS
              if ex.extract_datetime(u, now=NOW)[0] is not None]
    if healed:
        print(f"\nFAIL: KNOWN_GAPS entries now resolve: {healed}. "
              f"entities.py behaviour changed — update KNOWN_GAPS and say why.")
        return 1
    print(f"known gaps (recorded, still unfixed): {len(KNOWN_GAPS)}")

    branches = sorted({c["branch"] for c in cases})
    print(f"\nbranches covered: {len(branches)}")

    if len(cases) < 120:
        print(f"\nFAIL: only {len(cases)} admissible cases; the charter requires >= 120.")
        return 1

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    payload = {
        "_note": "English datetime GOLDEN corpus — captured from the pristine "
                 "extractor at a fixed `now`. Regression oracle for the A7 pack "
                 "eviction. Do NOT edit by hand and do NOT regenerate to make a "
                 "failing test pass; regenerate only when a behaviour change is "
                 "deliberate, and say why in the commit message.",
        "_admissibility": "Every case resolves before the optional `dateparser` "
                          "fallback (§8), so the corpus is deterministic and "
                          "independent of whether that package is installed. "
                          "Cases reaching §8 are refused at capture time.",
        "captured_by": "scripts/ci/capture_en_datetime_golden.py",
        "source": "packages/runtime/nlu_engine/entities.py::extract_datetime",
        "reference_grammar_crosscheck": REF_GRAMMAR_HINT,
        "now_iso": NOW.isoformat(),
        "dateparser_present_at_capture": mod._HAS_DATEPARSER,
        "branches": branches,
        "case_count": len(cases),
        "known_gaps": [{"utterance": u, "why": w} for u, w in KNOWN_GAPS],
        "strip_cases": [{"text": t, "topic": ex.strip_datetime(t)}
                        for t in STRIP_CORPUS],
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)} ({len(cases)} cases, {len(branches)} branches)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
