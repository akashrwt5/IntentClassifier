#!/usr/bin/env python3
"""Check generated utterances against the Help-versus-Command boundary. Free, no API.

WHY
---
Precedence rule 4 in ``prompt.txt`` tells the model how to place a question-shaped
utterance: by the action the assistant would have to perform, not by the opening
words. Nothing measured whether it obeyed. The existing report covers length,
type, diversity, near-duplicates and quotas - all real, none of them able to see
a Help intent quietly filling up with commands.

That gap matters more than it looks. This is the taxonomy's costliest error
class: an information question misread as a command physically changes a hearing
aid. If the generated corpus blurs the boundary, the classifier trained on it
inherits the blur, and the first place anyone notices is a user's ear.

WHAT IT MEASURES
----------------
Every utterance gets a surface form from patterns measured on the deployed data:

  COMMAND-SHAPED   imperative; "can/could/will you ..."; "please ..."; a stated
                   want or need for an end state; a bare complaint about the
                   current state
  HELP-SHAPED      "how do I", "how to", "where", "what", "why", "can I"
  EXPLAIN-REQUEST  any of the above whose verb is help/explain/tell me/show me
                   -- the requested action IS explaining, so the form does not
                   decide (deployed: "can you + help-verb" is Help 78% of the
                   time against 10% for "can you + anything else")
  NEUTRAL          none of the above

A row is FLAGGED when a Help intent carries a command-shaped utterance, or a Cmd
intent carries a help-shaped one, and it is not an explain-request.

CALIBRATION IS THE POINT
------------------------
A flag is not automatically an error - deployed data carries some too, and
"can you walk me through pairing" is properly Help however it opens. So the
threshold is not zero: the same linter runs over ``train.csv`` and the generated
rows are compared against THAT rate, per intent. The question is never "are
there flags" but "more than real users produce".

    python3 boundary_lint.py --baseline          # deployed rates only
    python3 boundary_lint.py                     # generated vs deployed
    python3 boundary_lint.py --markdown lint.md  # write the report
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKPOINTS = HERE / ".checkpoints" / "stage1"

# --- Outside this directory ---------------------------------------------
DEPLOYED = HERE.parents[2] / "language_packs" / "en" / "train.csv"
# ------------------------------------------------------------------------

# A rate alone must not fail an intent. On a 25-row batch one row is 4%, so a fixed
# five-point tolerance sits BELOW the sampling noise it is supposed to sit above --
# and it failed two intents on a pilot where an exact test cleared one of them:
#
#   Cmd.BatteryLevel    8 flags, 5.0 expected from deployed    p = 0.109   noise
#   Cmd.ActivityStand   7 flags, 3.3 expected                  p = 0.037   real
#
# This is the same mistake the compression plan records making at Rev 4, thresholds
# set below what the instrument can resolve, made again in a different tool. The gate
# is now a one-sided exact binomial test: how likely is this many flags, or more, if
# the generator matched deployed speech exactly? Exact rather than normal because at
# n=25 the approximation is worse than the thing it approximates.
ALPHA = 0.05

# Below this many flags nothing is decided regardless of the p-value: two rows out of
# twenty-five can reach significance against a very low baseline, and that is a batch
# size artefact, not a finding.
MIN_FLAGS_TO_FAIL = 3

HELP_VERB = r"(help|explain|tell me|show me|teach|walk me|guide me)"

COMMAND_SHAPED = [
    (
        "imperative",
        r"^(turn|make|set|raise|lower|increase|decrease|mute|unmute|put|switch|start|stop|play|send|read|change|pair|connect|find)\b",
    ),
    ("can-you", r"^(can|could|will|would) you\b"),
    ("please", r"^please\b"),
    ("stated-need", r"^i (want|need|would like|d like|wanna)\b"),
    (
        "state-complaint",
        r"^(it s|its|the|my|this|that|sound|volume|audio|everything)\b.*\b(too (quiet|loud|soft|low|high)|not loud enough|barely)\b",
    ),
]
HELP_SHAPED = [
    ("how-do-i", r"^how (do|can|would|should|might) i\b"),
    ("how-to", r"^how (to|about)\b"),
    ("wh-question", r"^(where|what|why|which|when)\b"),
    ("can-i", r"^can i\b"),
    ("is-there", r"^is there\b"),
]
EXPLAIN = re.compile(rf"\b{HELP_VERB}\b")

# An utterance can open command-shaped and resolve into a question:
# "turn off tinnitus noise, how?" opens on an imperative and ends as a how-question;
# "my masker sounds too loud, where do I go to reduce it?" opens as a state complaint
# and does the same. Both are correct Help rows, and both were flagged by an earlier
# version of this file that only ever read the first few words.
#
# That failure got worse, not better, as the corpus improved: the short-utterance work
# pushed the generator toward terse inverted forms -- "tinnitus settings in the app,
# where?" -- which is precisely the shape an opening-only classifier misreads. A
# trailing or embedded wh-question therefore decides the reading before the opening
# does.
TRAILING_QUESTION = re.compile(
    r"(\b(how|where|what|why|which)\b[^?]*\?\s*$)|(,\s*(how|where|what|why|which)\b)"
)


def normalise(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).split())


def surface_form(text: str) -> tuple[str, str]:
    """(class, which pattern matched). Explain-requests win over everything."""
    t = normalise(text)
    if EXPLAIN.search(t):
        return "explain-request", "help-verb"
    if TRAILING_QUESTION.search(str(text).lower()):
        return "help-shaped", "trailing-wh"
    for name, pat in COMMAND_SHAPED:
        if re.search(pat, t):
            return "command-shaped", name
    for name, pat in HELP_SHAPED:
        if re.search(pat, t):
            return "help-shaped", name
    return "neutral", "-"


def p_at_least(flags: int, rows: int, rate: float) -> float:
    """P(flags or more) under the deployed rate. One-sided exact binomial."""
    if rows <= 0 or not 0.0 < rate < 1.0:
        return 1.0
    return sum(comb(rows, i) * rate**i * (1.0 - rate) ** (rows - i) for i in range(flags, rows + 1))


def family(intent: str) -> str:
    if intent.startswith("Help"):
        return "Help"
    if intent.startswith("Cmd."):
        return "Cmd"
    return "other"


def flagged(intent: str, form: str) -> bool:
    """A surface form that contradicts the intent's family."""
    fam = family(intent)
    if fam == "Help":
        return form == "command-shaped"
    if fam == "Cmd":
        return form == "help-shaped"
    return False  # Fallback and reminders.* have no such boundary


def scan(pairs) -> dict:
    per = defaultdict(lambda: {"n": 0, "flags": 0, "why": Counter()})
    for text, intent in pairs:
        form, which = surface_form(text)
        rec = per[intent]
        rec["n"] += 1
        if flagged(intent, form):
            rec["flags"] += 1
            rec["why"][which] += 1
    return per


def load_deployed() -> list[tuple[str, str]]:
    if not DEPLOYED.exists():
        raise SystemExit(f"deployed data not found: {DEPLOYED}")
    with DEPLOYED.open(encoding="utf-8-sig", newline="") as fh:
        return [(r["text"], r["intent"].strip()) for r in csv.DictReader(fh)]


def load_generated(only=None, root=None) -> list[tuple[str, str]]:
    root = root or CHECKPOINTS
    if not root.exists():
        return []
    out = []
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            intent = str(row.get("intent", "")).strip()
            if only and intent not in only:
                continue
            out.append((row.get("utterance", ""), intent))
    return out


def report(base, gen, tolerance: float) -> tuple[str, int]:
    lines = [
        "# Help / Command boundary lint",
        "",
        "Surface form measured against the intent's family. `deployed` is the same",
        "check run over `language_packs/en/train.csv` — the rate real users produce.",
        "A generated rate at or below deployed is the pass condition; the target is",
        "never zero, because an explain-request is properly Help however it opens.",
        "",
        "An intent fails when a one-sided exact binomial test puts this many flags or more",
        f"below {ALPHA:.0%} -- that is, they would rarely happen by chance if the generator",
        f"matched deployed speech -- and at least {MIN_FLAGS_TO_FAIL} rows are flagged. A fixed",
        "percentage tolerance was tried first and sat BELOW the sampling noise: on 25 rows one",
        "row is 4 points, so it failed intents an exact test clears.",
        f"and an intent fails only when at least {MIN_FLAGS_TO_FAIL} rows are flagged -- on a",
        "small batch one row is several points, and noise should not decide a gate.",
        "",
        "| Intent | Rows | Flagged | Deployed | Expected | p | Verdict | Cause |",
        "|---|---:|---:|---:|---:|---:|:-:|---|",
    ]
    failures = 0
    review: list[str] = []
    for intent in sorted(gen):
        g = gen[intent]
        b = base.get(intent, {"n": 0, "flags": 0})
        g_rate = g["flags"] / g["n"] if g["n"] else 0.0
        b_rate = b["flags"] / b["n"] if b["n"] else 0.0
        # An intent with no deployed rows has no baseline, and this whole check is
        # calibration against a baseline. Failing it against an assumed 0% is not a
        # measurement, it is an assumption wearing a measurement's clothes -- and it
        # fired on the first intent it met: Help_Activity's spec deliberately owns
        # bare imperative configuration phrases ("set up walking goal"), so every
        # correctly generated row looked like a violation of a rule that does not
        # apply to it. Report and hand to a human; do not pretend to judge.
        uncalibrated = not b["n"]
        pvalue = 1.0 if uncalibrated else p_at_least(g["flags"], g["n"], b_rate)
        bad = not uncalibrated and pvalue < ALPHA and g["flags"] >= MIN_FLAGS_TO_FAIL
        failures += bad
        if uncalibrated and g["flags"]:
            review.append(intent)
        expected_txt = "—" if uncalibrated else f"{b_rate * g['n']:.1f}"
        p_txt = "—" if uncalibrated else f"{pvalue:.3f}"
        cause = ", ".join(f"{k} {v}" for k, v in g["why"].most_common(2)) or "-"
        base_txt = f"{b_rate:.1%}" if b["n"] else "no baseline"
        lines.append(
            f"| `{intent}` | {g['n']} | {g['flags']} ({g_rate:.1%}) | {base_txt} | "
            f"{expected_txt} | {p_txt} | "
            f"{'**FAIL**' if bad else ('*review*' if uncalibrated and g['flags'] else 'ok')} "
            f"| {cause} |"
        )
    lines += ["", f"**{failures} intent(s) failed.**", ""]
    if review:
        lines += [
            f"**{len(review)} intent(s) marked *review*:** "
            + ", ".join(f"`{x}`" for x in review)
            + ". These have no deployed rows, so there is nothing to calibrate against "
            "and this check cannot judge them. Read the flagged rows against the intent's "
            "own spec -- an intent that legitimately owns imperative phrasing will look "
            "like a violation here and is not one.",
            "",
        ]
    if failures:
        lines += [
            "A FAIL means the generator placed more form-contradicting utterances in that",
            "intent than deployed speech contains. Read a sample before changing anything:",
            "the fix is usually one sentence in that intent's boundary_cases, not a prompt",
            "rewrite — and a prompt rewrite invalidates every measured run.",
            "",
        ]
    return "\n".join(lines) + "\n", failures


def baseline_only(base) -> str:
    lines = [
        "# Deployed boundary rates (baseline)",
        "",
        "| Intent | Rows | Form-contradicting | Main cause |",
        "|---|---:|---:|---|",
    ]
    rows = [(i, r) for i, r in base.items() if family(i) in ("Help", "Cmd") and r["n"]]
    for intent, r in sorted(rows, key=lambda x: -(x[1]["flags"] / x[1]["n"])):
        cause = ", ".join(f"{k} {v}" for k, v in r["why"].most_common(2)) or "-"
        lines.append(
            f"| `{intent}` | {r['n']} | {r['flags']} ({r['flags']/r['n']:.1%}) | {cause} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", action="store_true", help="print deployed rates and stop")
    ap.add_argument(
        "--checkpoints",
        type=Path,
        default=None,
        help="Checkpoint directory to read (default .checkpoints/stage1). Point this at\n.checkpoints-pilot/stage1 to score a pilot run.",
    )
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these intents")
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args(argv)

    base = scan(load_deployed())
    if args.baseline:
        text = baseline_only(base)
        print(text) if not args.markdown else args.markdown.write_text(text, encoding="utf-8")
        return 0

    rows = load_generated(set(args.only) if args.only else None, args.checkpoints)
    if not rows:
        print(f"no generated rows found under {args.checkpoints or CHECKPOINTS}")
        return 0
    gen = scan(rows)
    text, failures = report(base, gen, args.tolerance)
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
        print(f"wrote {args.markdown}")
    else:
        print(text)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
