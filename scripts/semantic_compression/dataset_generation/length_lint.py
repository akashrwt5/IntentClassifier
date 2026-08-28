#!/usr/bin/env python3
"""Check generated utterance length against deployed speech, per intent. No API.

WHY
---
The quota report already says whether a batch met its short floor. It cannot say
whether the floor was the right number, and for a long time it was not: one value,
``min_short: 0.28``, governed 33 Help intents whose deployed short share runs from
26% to 97%. Every batch could be "compliant" while the corpus drifted steadily away
from how people actually speak.

This compares the generated distribution against DEPLOYED speech for the same
intent -- the same calibration idea as boundary_lint.py, applied to length. Two
independent things can go wrong, and they are tested separately because they have
different causes and different fixes:

  SHORT SHARE   too few rows at or under the intent's cap. The generator reaches
                for extra words when asked for a hard row.
  LONG TAIL     too many rows above what deployed speech for this intent ever
                produces. A different failure: the short rows can be present in
                the right proportion while a handful of monsters sit behind them.

WHY LENGTH IS WORTH A GUARD OF ITS OWN
--------------------------------------
Scoring the shipped encoder on dev_hard by utterance length:

    1-4 words     296 rows    79.1% accuracy
    5-7 words     381 rows    85.3%
    8-12 words    133 rows    88.7%

Short utterances are where this product's classifier fails. A corpus that drifts
long trains the model on the part it already handles and starves the part it does
not -- and the drift is invisible in every other metric. Diversity, near-duplicate
rate and type mix all look healthy while it happens.

    python3 length_lint.py                      # generated vs deployed
    python3 length_lint.py --markdown lint.md
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CHECKPOINTS = HERE / ".checkpoints" / "stage1"
TARGETS = HERE / "length_targets.yaml"

# --- Outside this directory ---------------------------------------------
DEPLOYED = HERE.parents[2] / "language_packs" / "en" / "train.csv"
# ------------------------------------------------------------------------

# A share alone must not fail an intent. On a 25-row batch one row is 4 points, so
# any fixed percentage gate sits BELOW the sampling noise it is supposed to sit
# above. This file used to gate on `short share >= 0.70 * target` and it could not
# say whether a miss was the generator or the batch size -- which is exactly the
# mistake boundary_lint.py had fixed, and the one the compression plan records
# making at Rev 4. Both gates below are now one-sided exact binomial tests: how
# likely is a result this bad, or worse, if the generator were behaving? Exact
# rather than normal because at n=25 the approximation is worse than the thing it
# approximates.
ALPHA = 0.05

# Below this many rows of discrepancy nothing is decided regardless of the p-value.
# Against a very low baseline two rows out of twenty-five can reach significance,
# and that is a batch-size artefact, not a finding. Same floor, same reason, as
# boundary_lint.MIN_FLAGS_TO_FAIL.
MIN_ROW_EXCESS = 3

# Floors are approximately honoured, never met: generation lands below the ask, so
# testing against the raw target would fail every intent forever and measure the
# generator's global tilt rather than this intent's behaviour. The null is
# therefore `target * DELIVERY_ALLOWANCE`.
#
# 0.75 is set below every well-behaved intent yet observed and well above the ones
# that are actually broken. Measured on the twelve-intent pilot, got/ask by intent:
#
#     median all 0.90    median target<=45% 0.95    median target>45% 0.87
#     well-behaved range 0.77 - 1.73
#     Help_FindMyHearingAids 0.33      Help_Pairing 0.36
#
# The handover records a median of 0.85 with a low-regime figure of 0.74 from an
# earlier run; that 0.74 is NOT reproduced on this pilot. The gap is unexplained
# and 0.75 is deliberately the conservative end of it. Re-set this once a full run
# gives 60 intents rather than 12 -- and re-set it against the run's own median,
# which the report prints for that purpose.
DELIVERY_ALLOWANCE = 0.75

# "Longer than deployed speech for this intent ever really gets." Taken per intent
# from deployed data rather than fixed in words, because 12 words is unremarkable
# for Help_Pairing and absurd for Cmd.BatteryLevel.
#
# This replaces a mean comparison (`generated mean > deployed mean + 3 words`). A
# mean cannot tell "uniformly a little longer" from "mostly fine, five monsters",
# and the two need different fixes. On the same pilot the mean rule flagged three
# intents, missed Cmd.MemoryChange (8 long rows against 1.1 expected) and
# Help_Tinnitus (4 against 0.3) entirely, and mislabelled Cmd.ActivityStand and
# Cmd.BatteryLevel as short-share problems when their short share was on target.
LONG_TAIL_PERCENTILE = 0.90


def load_deployed() -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    with DEPLOYED.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["intent"].strip()].append(len(row["text"].split()))
    return out


def load_generated(only=None, root=None) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    root = root or CHECKPOINTS
    if not root.exists():
        return out
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            intent = str(row.get("intent", "")).strip()
            if only and intent not in only:
                continue
            out[intent].append(len(str(row.get("utterance", "")).split()))
    return out


def load_targets() -> dict[str, dict]:
    if not TARGETS.exists():
        raise SystemExit(f"{TARGETS.name} not found -- run derive_length_targets.py first")
    return (yaml.safe_load(TARGETS.read_text(encoding="utf-8")) or {}).get("intents") or {}


def share(words, cap) -> float:
    return sum(1 for w in words if w <= cap) / len(words) if words else 0.0


def percentile(values: list[int], q: float) -> float:
    """Linear-interpolated percentile. No numpy: this file has no other need for it."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def p_at_most(hits: int, rows: int, rate: float) -> float:
    """P(hits or fewer) under ``rate``. One-sided exact binomial, deficit direction."""
    if rows <= 0 or not 0.0 < rate < 1.0:
        return 1.0
    return sum(comb(rows, i) * rate**i * (1.0 - rate) ** (rows - i) for i in range(0, hits + 1))


def p_at_least(hits: int, rows: int, rate: float) -> float:
    """P(hits or more) under ``rate``. One-sided exact binomial, excess direction."""
    if rows <= 0 or not 0.0 < rate < 1.0:
        return 1.0
    return sum(comb(rows, i) * rate**i * (1.0 - rate) ** (rows - i) for i in range(hits, rows + 1))


def assess(intent: str, gen_words: list[int], dep_words: list[int], rec: dict) -> dict:
    """Both length tests for one intent. Pure; every number in the report comes from here."""
    cap = int(rec["short_max_words"])
    target = float(rec["target_short_share"])
    rows = len(gen_words)

    # A target derived from a family median is a stand-in, not a measurement: the
    # intent it describes has no deployed rows. Missing it is worth seeing and is
    # not the same evidence as missing a target taken from real speech.
    inferred = not rec.get("deployed_rows")

    short_hits = sum(1 for w in gen_words if w <= cap)
    short_null = target * DELIVERY_ALLOWANCE
    short_expected = short_null * rows
    short_p = p_at_most(short_hits, rows, short_null) if target > 0 else 1.0
    short_bad = target > 0 and short_p < ALPHA and (short_expected - short_hits) >= MIN_ROW_EXCESS

    # The long-tail test is calibration against deployed data, exactly like
    # boundary_lint. With no deployed rows there is nothing to calibrate against,
    # and failing an intent against an assumed rate is an assumption wearing a
    # measurement's clothes. Report and hand to a human.
    if dep_words:
        cut = percentile(dep_words, LONG_TAIL_PERCENTILE)
        long_hits = sum(1 for w in gen_words if w > cut)
        long_null = sum(1 for w in dep_words if w > cut) / len(dep_words)
        long_expected = long_null * rows
        long_p = p_at_least(long_hits, rows, long_null)
        long_bad = (
            long_p < ALPHA
            and long_hits >= MIN_ROW_EXCESS
            and (long_hits - long_expected) >= MIN_ROW_EXCESS
        )
    else:
        cut = float("nan")
        long_hits = 0
        long_null = float("nan")
        long_expected = float("nan")
        long_p = 1.0
        long_bad = False

    return {
        "intent": intent,
        "rows": rows,
        "cap": cap,
        "target": target,
        "inferred": inferred,
        "calibrated": bool(dep_words),
        "short_hits": short_hits,
        "short_share": short_hits / rows if rows else 0.0,
        "short_null": short_null,
        "short_expected": short_expected,
        "short_p": short_p,
        "short_bad": short_bad,
        "long_cut": cut,
        "long_hits": long_hits,
        "long_null": long_null,
        "long_expected": long_expected,
        "long_p": long_p,
        "long_bad": long_bad,
        "gen_mean": statistics.mean(gen_words) if gen_words else float("nan"),
        "dep_mean": statistics.mean(dep_words) if dep_words else float("nan"),
        "got_over_ask": (short_hits / rows) / target if rows and target > 0 else float("nan"),
    }


def build(only=None, root=None) -> tuple[str, int]:
    dep, gen, targets = load_deployed(), load_generated(only, root), load_targets()
    if not gen:
        return f"no generated rows under {root or CHECKPOINTS}\n", 0

    results = []
    untargeted = []
    for intent in sorted(gen):
        rec = targets.get(intent)
        if not rec:
            untargeted.append(intent)
            continue
        results.append(assess(intent, gen[intent], dep.get(intent, []), rec))

    ratios = [r["got_over_ask"] for r in results if r["got_over_ask"] == r["got_over_ask"]]
    run_median = statistics.median(ratios) if ratios else float("nan")

    lines = [
        "# Utterance length lint",
        "",
        "`target` is the intent's deployed short share (`length_targets.yaml`).",
        "Two independent gates, each a one-sided exact binomial test:",
        "",
        f"- **short** — too few rows at or under the cap. Null is `target x "
        f"{DELIVERY_ALLOWANCE:.2f}`, because generation lands below the ask and "
        "testing against the raw ask would measure the generator's global tilt "
        "rather than this intent.",
        "- **long** — too many rows above this intent's deployed "
        f"{LONG_TAIL_PERCENTILE:.0%} word count. Null is the rate deployed speech "
        "itself produces above that cut.",
        "",
        f"An intent fails a gate when p < {ALPHA:.2f} **and** the discrepancy is at "
        f"least {MIN_ROW_EXCESS} rows. On a 25-row batch one row is 4 points, so a "
        "fixed percentage gate sits below the noise it is meant to sit above.",
        "",
        "| Intent | Rows | Cap | Target | Short | Exp | p | Long cut | Long | Exp | p | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|",
    ]

    failures = 0
    review: list[str] = []
    for r in results:
        bad = (r["short_bad"] or r["long_bad"]) and not r["inferred"]
        failures += bad
        if r["inferred"] and (r["short_bad"] or r["long_bad"]):
            review.append(r["intent"])
        verdict = "**FAIL**" if bad else ("*review*" if r["intent"] in review else "ok")
        long_cols = (
            f"{r['long_cut']:.0f} | {r['long_hits']} | {r['long_expected']:.1f} | "
            f"{r['long_p']:.3f}"
            if r["calibrated"]
            else "— | — | — | —"
        )
        lines.append(
            f"| `{r['intent']}` | {r['rows']} | {r['cap']} | {r['target']:.0%} | "
            f"{r['short_hits']} ({r['short_share']:.0%}) | {r['short_expected']:.1f} | "
            f"{r['short_p']:.3f} | {long_cols} | {verdict} |"
        )
        reasons = []
        if r["short_bad"]:
            reasons.append(
                f"short: {r['short_hits']} rows at or under {r['cap']} words against "
                f"{r['short_expected']:.1f} expected (p = {r['short_p']:.3f})"
            )
        if r["long_bad"]:
            reasons.append(
                f"long: {r['long_hits']} rows over {r['long_cut']:.0f} words against "
                f"{r['long_expected']:.1f} expected (p = {r['long_p']:.3f})"
            )
        if reasons and (bad or r["intent"] in review):
            lines.append(f"| | | | | | | | | | | | {'; '.join(reasons)} |")

    for intent in untargeted:
        lines.append(
            f"| `{intent}` | {len(gen[intent])} | — | — | — | — | — | — | — | — | — | *no target* |"
        )

    lines += ["", f"**{failures} intent(s) failed.**", ""]

    # The delivery allowance is the one number here that is set rather than
    # measured, so the run's own median is printed beside it every time. If these
    # drift apart the allowance is stale, not the generator.
    if ratios:
        lines += [
            f"Run median got/ask: **{run_median:.2f}** across {len(ratios)} intent(s), "
            f"against the assumed allowance of {DELIVERY_ALLOWANCE:.2f}. These are "
            "different questions -- the median is where the generator sits, the "
            "allowance is how far below the ask a single intent may sit before it "
            "is called broken.",
            "",
            "Read the median against the allowance only for rows generated under the "
            "CURRENT targets. Rows produced before `length_targets.yaml` existed are "
            "being scored against a number they were never asked for, and their median "
            "says the corpus is stale, not that the allowance is. Check the checkpoint "
            "dates before concluding anything from a gap here.",
            "",
        ]

    if review:
        lines += [
            f"**{len(review)} intent(s) marked *review*:** "
            + ", ".join(f"`{x}`" for x in review)
            + ". Their target is a family median standing in for an intent with no "
            "deployed rows, so a miss here may be the target's fault rather than the "
            "generator's. Judge it by hand.",
            "",
        ]
    if failures:
        lines += [
            "A short failure and a long failure are different defects. Short means the",
            "generator would not produce terse rows; long means it produced rows longer",
            "than this intent has ever really seen, which the short share alone cannot",
            "detect. Check the difficulty section of the prompt first: tying Hard to",
            "length is what caused this once already.",
            "",
        ]
    return "\n".join(lines) + "\n", failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--checkpoints",
        type=Path,
        default=None,
        help=(
            "Checkpoint directory to read (default .checkpoints/stage1). Point this at\n"
            ".checkpoints-pilot/stage1 to score a pilot run."
        ),
    )
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args(argv)
    text, failures = build(set(args.only) if args.only else None, args.checkpoints)
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
        print(f"wrote {args.markdown}")
    else:
        print(text)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
