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

import yaml

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


def p_at_most(flags: int, rows: int, rate: float) -> float:
    """P(flags or fewer) under the deployed rate. The other tail.

    This half exists because the test above could not see the failure it was
    most needed for. Five Help specs carry a CARVE-OUT: the assistant cannot
    perform the action, so explaining IS the action and a direct request
    belongs to the Help intent. Each states a measured command-shaped rate and
    says in terms that "a generated rate near zero is a defect, not a success".

    The 2026-08-30 pilot generated 0 of 26 for Help_FindMyHearingAids against a
    deployed 40.6%, and 0 of 25 for Help_Pairing against 29.5%. This file marked
    both `ok`, because a one-sided "at or below deployed" test cannot fail an
    intent for producing NONE of the thing it was asked for. Five rounds of
    spec work were invisible to the only instrument that could check them.
    """
    if rows <= 0 or not 0.0 < rate < 1.0:
        return 1.0
    return sum(comb(rows, i) * rate**i * (1.0 - rate) ** (rows - i) for i in range(0, flags + 1))


def carve_out_intents(path=None) -> set:
    """Intents whose spec asks generation to REPRODUCE a command-shaped rate.

    Read from the specs rather than listed here, so adding a sixth carve-out
    turns the two-sided test on for it without touching this file.
    """
    path = path or (HERE / "intent_specs.yaml")
    if not path.exists():
        return set()
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # Detect on the STATED RATE, not on a catchphrase. The first version of this
    # matched "near zero is a defect" and silently missed Help_HearShare, whose
    # carve-out says "should not drop it to zero" instead -- the one intent of
    # the five whose rate already fitted the old quota, and so the one nobody
    # would have noticed was unguarded. A mutation test found it.
    rate = re.compile(r"\d+\.?\d*% command-shaped")
    return {
        s["name"]
        for s in doc.get("intents") or []
        if s["name"].startswith("Help")
        and any(rate.search(x) for x in (s.get("boundary_cases") or []))
    }


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


# Section 12.C of the Help-vs-Command verification spec. Help data that leans on
# a handful of question openers is TEMPLATE-DRIVEN, and a model trained on it
# learns the opener rather than the intent. Deployed speech does not lean that
# way: 45.9% of all 2,946 deployed Help rows carry none of these markers, and the
# per-intent spread runs 19.3% (Help_Customize) to 78.1% (Help_Pairing). A global
# threshold would be wrong for both ends, so the baseline is per intent, like
# every other number in this file.
#
# THE MARKER LIST IS EXACTLY THE ONE THAT DOCUMENT NAMES and is not extended
# here. Adding markers shrinks the marker-free set and quietly makes the test
# stricter than the property it claims to measure.
#
# CONTAINS, not starts-with. HELP_SHAPED's patterns are anchored because they
# classify SURFACE FORM. This asks a different question -- does the row lean on a
# question marker anywhere. "tell me how to pair them" is an explain-request by
# surface form and marker-carrying by this one; both readings are right for their
# own purpose, and the two metrics are close to independent in the data: of
# Help_Tinnitus's 105 marker-free deployed rows, 0 are command-shaped.
QUESTION_MARKERS = re.compile(r"\b(how|what|where|can i|is there)\b")


def marker_free(text: str) -> bool:
    """True when no classic Help question marker appears anywhere."""
    return not QUESTION_MARKERS.search(normalise(text))


def carries_command(text: str) -> bool:
    """Does the utterance carry a command pattern ANYWHERE, short-circuit aside?

    `surface_form` returns explain-request the moment it sees a help-verb, and
    that is deliberate -- an explain-request is properly Help however it opens.
    But it means "can you help me find my hearing aids" is counted as an
    explain-request and NOT as command-shaped, so a batch full of direct
    requests phrased that way reports as 0% command-shaped.

    The 2026-08-30 pilots made that misleading rather than merely subtle:

        Help_FindMyHearingAids generated   4.0% command-shaped
                                          32.0% carrying a command pattern
                              deployed    40.6% / 46.4%

    Reading 4.0% alone says generation produced almost no direct requests. It
    produced them at about seven tenths of the deployed rate and phrased them
    with a help-verb. Those are different defects with different fixes, and the
    headline number cannot tell them apart. This is REPORTED, never scored: the
    pass condition stays on the same measure for both sides.
    """
    t = normalise(text)
    return any(re.search(pat, t) for _, pat in COMMAND_SHAPED)


def scan(pairs) -> dict:
    per = defaultdict(lambda: {"n": 0, "flags": 0, "cmd": 0, "free": 0, "why": Counter()})
    for text, intent in pairs:
        form, which = surface_form(text)
        rec = per[intent]
        rec["n"] += 1
        if family(intent) == "Help" and carries_command(text):
            rec["cmd"] += 1
        if family(intent) == "Help" and marker_free(text):
            rec["free"] += 1
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
    carve = carve_out_intents()
    lines = [
        "# Help / Command boundary lint",
        "",
        "Surface form measured against the intent's family. `deployed` is the same",
        "check run over `language_packs/en/train.csv` — the rate real users produce.",
        "A generated rate at or below deployed is the pass condition for most intents.",
        "For the CARVE-OUT intents -- those whose spec states a measured",
        "command-shaped rate and asks generation to reproduce it -- the test is",
        "TWO-SIDED, because for them a rate near zero is the defect their own spec",
        "names. Carve-outs found in the specs: "
        + (", ".join(f"`{x}`" for x in sorted(carve)) or "none") + ".",
        "The target is",
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
        "`Cmd-patterned` is DIAGNOSTIC, not scored: the share of rows carrying a",
        "command pattern anywhere, generated against deployed. A row reading \"can you",
        "help me find my hearing aids\" is a direct request that `Flagged` cannot see,",
        "because a help-verb makes it an explain-request first. When `Flagged` is near",
        "zero and this column is not, the defect is PHRASING, not a missing shape.",
        "",
        "| Intent | Rows | Flagged | Deployed | Cmd-patterned | Expected | p | Verdict | Cause |",
        "|---|---:|---:|---:|---:|---:|---:|:-:|---|",
    ]
    failures = 0
    review: list[str] = []
    for intent in sorted(gen):
        g = gen[intent]
        b = base.get(intent, {"n": 0, "flags": 0, "cmd": 0})
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
        # The other tail, for carve-out intents only. Everywhere else a low rate
        # is the desired outcome and testing for it would fail correct work.
        # Guarded the same way as the upper tail: only judge when deployed speech
        # predicts at least MIN_FLAGS_TO_FAIL rows, so noise does not decide.
        under = False
        if not uncalibrated and intent in carve:
            expected_rows = b_rate * g["n"]
            if expected_rows >= MIN_FLAGS_TO_FAIL:
                p_low = p_at_most(g["flags"], g["n"], b_rate)
                under = p_low < ALPHA
                if under:
                    pvalue = p_low
        failures += bad or under
        if uncalibrated and g["flags"]:
            review.append(intent)
        expected_txt = "—" if uncalibrated else f"{b_rate * g['n']:.1f}"
        p_txt = "—" if uncalibrated else f"{pvalue:.3f}"
        cause = ", ".join(f"{k} {v}" for k, v in g["why"].most_common(2)) or "-"
        # Diagnostic only. See carries_command: it separates "produced no direct
        # requests" from "produced them with a help-verb in front".
        if family(intent) == "Help" and (g.get("cmd") or b.get("cmd")):
            gc = g.get("cmd", 0) / g["n"] if g["n"] else 0.0
            bc = b.get("cmd", 0) / b["n"] if b.get("n") else 0.0
            patt = f"{gc:.1%} vs {bc:.1%}" if b.get("n") else f"{gc:.1%}"
        else:
            patt = "—"
        base_txt = f"{b_rate:.1%}" if b["n"] else "no baseline"
        lines.append(
            f"| `{intent}` | {g['n']} | {g['flags']} ({g_rate:.1%}) | {base_txt} | "
            f"{patt} | {expected_txt} | {p_txt} | "
            f"{'**FAIL**' if bad else ('**FAIL (UNDER)**' if under else ('*review*' if uncalibrated and g['flags'] else 'ok'))} "
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
            "A **FAIL** means the generator placed MORE form-contradicting utterances in",
            "that intent than deployed speech contains. Read a sample before changing",
            "anything: the fix is usually one sentence in that intent's boundary_cases, not",
            "a prompt rewrite — and a prompt rewrite invalidates every measured run.",
            "",
            "A **FAIL (UNDER)** is the opposite and has a different cause. The intent is a",
            "carve-out, its spec asked generation to reproduce a measured command-shaped",
            "rate, and generation produced far fewer. Check the QUOTAS before the spec: a",
            "type floor in generator_config.yaml can make the spec's rate arithmetically",
            "impossible, which is exactly what the 2026-08-30 pilot found. Prose loses to a",
            "number in the same prompt.",
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


def report_markers(base, gen) -> tuple[str, int]:
    """Help rows carrying no classic question marker, generated against deployed.

    Scored, one-sided, LOWER tail only. Producing more marker-free rows than
    deployed is not this check's business -- a Help row that reads as a command
    is boundary_lint's other half, and one that reads as a fragment is what
    length_lint measures. Producing FEWER is the template-driven failure this
    exists to catch, and nothing else in the pipeline can see it: Help_Tinnitus
    scores 0.0% command-shaped on both sides and passes the section above while
    generating 8.0% marker-free against a deployed 55.0%.
    """
    # Computed, never written down. A hardcoded corpus statistic in a generated
    # report is the exact defect this review has fixed five times elsewhere.
    _hn = sum(v["n"] for k, v in base.items() if family(k) == "Help")
    _hf = sum(v.get("free", 0) for k, v in base.items() if family(k) == "Help")
    overall = f"{_hf / _hn:.1%}" if _hn else "n/a"
    lines = [
        "",
        "## Marker-free Help",
        "",
        "Help rows containing none of `how`, `what`, `where`, `can I`, `is there` --"
        " anywhere,",
        f"not only at the start. Deployed speech is {overall} marker-free across all"
        f" {_hn} Help rows,",
        "so a generator that always opens with a question word is not reproducing it;"
        " it is",
        "teaching the opener instead of the intent.",
        "",
        "One-sided on the LOWER tail, per-intent baseline, and judged only when"
        " deployed",
        f"speech predicts at least {MIN_FLAGS_TO_FAIL} marker-free rows -- so a short"
        " batch cannot fail on noise.",
        "",
        "| Intent | Rows | Marker-free | Deployed | Expected | p | Verdict |",
        "|---|---:|---:|---:|---:|---:|:-:|",
    ]
    failures = 0
    judged = 0
    for intent in sorted(gen):
        if family(intent) != "Help":
            continue
        g = gen[intent]
        b = base.get(intent, {"n": 0, "free": 0})
        if not g["n"]:
            continue
        g_rate = g["free"] / g["n"]
        if not b.get("n"):
            lines.append(
                f"| `{intent}` | {g['n']} | {g['free']} ({g_rate:.1%}) | no baseline "
                f"| — | — | *review* |"
            )
            continue
        b_rate = b["free"] / b["n"]
        expected = b_rate * g["n"]
        if expected < MIN_FLAGS_TO_FAIL:
            verdict, p_txt = "—", "—"
        else:
            judged += 1
            pvalue = p_at_most(g["free"], g["n"], b_rate)
            bad = pvalue < ALPHA
            failures += bad
            verdict, p_txt = ("**FAIL (TEMPLATE)**" if bad else "ok"), f"{pvalue:.3f}"
        lines.append(
            f"| `{intent}` | {g['n']} | {g['free']} ({g_rate:.1%}) | {b_rate:.1%} "
            f"| {expected:.1f} | {p_txt} | {verdict} |"
        )
    lines += ["", f"**{failures} of {judged} judged intent(s) failed.**", ""]
    if judged:
        lines += [
            f"Read that against two limits of the test itself. At alpha {ALPHA} over"
            f" {judged} intents,",
            f"roughly {ALPHA * judged:.1f} failure(s) are expected BY CHANCE even from a"
            " perfect generator,",
            "so one or two FAILs on a full run is not evidence of a defect -- a pattern"
            " across",
            "related intents is. And the exact binomial assumes rows are independent,"
            " which",
            "they are not: a batch is one LLM call under composition quotas and an"
            " avoid-list,",
            "so its rows are negatively correlated and these p-values are optimistic."
            " Both",
            "limits apply equally to the section above.",
            "",
        ]
    if failures:
        lines += [
            "A FAIL means the generator produced Help rows that lean on a question",
            "marker far more than real users do. The lever is the intent's own spec and",
            "the FRAGMENT instruction in the prompt, not this list -- widening the",
            "marker list would hide the failure rather than fix it.",
            "",
        ]
    return "\n".join(lines), failures


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
    marker_text, marker_failures = report_markers(base, gen)
    text = text + marker_text
    failures += marker_failures
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
        print(f"wrote {args.markdown}")
    else:
        print(text)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
