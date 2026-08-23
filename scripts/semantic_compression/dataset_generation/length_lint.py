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
numbers decide it:

  short share   generated share at or under the intent's cap, against its target
  mean words    generated mean against deployed mean

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
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CHECKPOINTS = HERE / ".checkpoints" / "stage1"
TARGETS = HERE / "length_targets.yaml"

# --- Outside this directory ---------------------------------------------
DEPLOYED = HERE.parents[2] / "language_packs" / "en" / "train.csv"
# ------------------------------------------------------------------------

# Floors are approximately honoured, never met: the measured pattern across five
# runs is that generation lands below the ask. Demanding the full target would
# fail every intent forever, so the gate is a fraction of it. 0.70 is set from
# the best-behaved observed run (Help_Volume: target 36%, achieved 24% = 0.67)
# rounded up -- it asks for slightly better than the best we have seen, which is
# the point of a gate, and is re-set once the full run gives 60 intents of data.
MIN_FRACTION_OF_TARGET = 0.70
MAX_MEAN_EXCESS_WORDS = 3.0


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


def build(only=None, root=None) -> tuple[str, int]:
    dep, gen, targets = load_deployed(), load_generated(only, root), load_targets()
    if not gen:
        return f"no generated rows under {root or CHECKPOINTS}\n", 0

    lines = [
        "# Utterance length lint",
        "",
        "`target` is the intent's deployed short share (`length_targets.yaml`).",
        "`deployed` columns are what real users produce for that same intent.",
        "",
        f"A run passes an intent when its short share reaches "
        f"{MIN_FRACTION_OF_TARGET:.0%} of target and its mean is within "
        f"+{MAX_MEAN_EXCESS_WORDS:.0f} words of deployed. Floors are approximately "
        "honoured, never met, so the gate is a fraction of the ask rather than the ask.",
        "",
        "| Intent | Rows | Cap | Target | Gen short | Dep short | Gen mean | Dep mean | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:-:|",
    ]
    failures = 0
    review: list[str] = []
    for intent in sorted(gen):
        rec = targets.get(intent)
        if not rec:
            lines.append(
                f"| `{intent}` | {len(gen[intent])} | — | — | — | — | — | — | *no target* |"
            )
            continue
        cap = int(rec["short_max_words"])
        target = float(rec["target_short_share"])
        g, d = gen[intent], dep.get(intent, [])
        gs, ds = share(g, cap), (share(d, cap) if d else float("nan"))
        gm = statistics.mean(g)
        dm = statistics.mean(d) if d else float("nan")

        # A target derived from a family median is a stand-in, not a measurement:
        # the intent it describes has no deployed rows. Missing it is worth seeing
        # and is not the same evidence as missing a target taken from real speech,
        # so it is reported separately rather than counted as a failure.
        inferred = not rec.get("deployed_rows")
        short_bad = target > 0 and gs < target * MIN_FRACTION_OF_TARGET
        mean_bad = bool(d) and gm > dm + MAX_MEAN_EXCESS_WORDS
        bad = (short_bad or mean_bad) and not inferred
        failures += bad
        if inferred and (short_bad or mean_bad):
            review.append(intent)
        why = "**FAIL**" if bad else ("*review*" if inferred and short_bad else "ok")
        lines.append(
            f"| `{intent}` | {len(g)} | {cap} | {target:.0%} | {gs:.0%} | "
            f"{'—' if not d else f'{ds:.0%}'} | {gm:.1f} | "
            f"{'—' if not d else f'{dm:.1f}'} | {why} |"
        )
        if bad or (inferred and short_bad):
            reasons = []
            if short_bad:
                reasons.append(
                    f"short share {gs:.0%} is below {target * MIN_FRACTION_OF_TARGET:.0%}"
                )
            if mean_bad:
                reasons.append(
                    f"mean {gm:.1f} exceeds deployed {dm:.1f} by more than "
                    f"{MAX_MEAN_EXCESS_WORDS:.0f} words"
                )
            lines.append(f"| | | | | | | | | {'; '.join(reasons)} |")

    lines += ["", f"**{failures} intent(s) failed.**", ""]
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
            "Length drift is not visible in diversity, near-duplicate rate or type mix --",
            "all three stay healthy while it happens. Check the difficulty section of the",
            "prompt first: tying Hard to length is what caused this once already.",
            "",
        ]
    return "\n".join(lines) + "\n", failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--checkpoints",
        type=Path,
        default=None,
        help="Checkpoint directory to read (default .checkpoints/stage1). Point this at\n.checkpoints-pilot/stage1 to score a pilot run.",
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
