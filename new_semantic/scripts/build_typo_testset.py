#!/usr/bin/env python3
"""
Build the eval set that measures ROBUSTNESS TO TYPOS.

Nothing in the repo measures this. `increae volume` and `slince` were noticed by
hand during an interactive session — two anecdotes, not a scored set — and every
model in the backbone comparison got them wrong, including the 133 MB ones. So
"typo robustness" has been on the plan with an effort estimate and no baseline,
which means any fix for it would have been unmeasurable.

This builds the missing measurement. Character-level noise is applied to rows
from the EXISTING eval sets, so the result inherits their independence from
training: those rows are already leak-guarded, and corrupting a character cannot
reintroduce a training phrase.

FOUR CORRUPTIONS, chosen because they are what real input actually produces:

    substitute   volume -> volime     finger hits an adjacent key
    drop         volume -> volme      key not registered
    duplicate    volume -> vollume    key bounced
    transpose    volume -> voulme     two keys in the wrong order

Applied to ONE word per utterance by default. Words of 3 characters or fewer are
left alone — corrupting "up" or "the" changes the meaning rather than misspelling
it, and a human could not recover those either, so scoring them would measure
noise instead of robustness.

DETERMINISTIC. Fixed seed, so the file regenerates identically and two models can
be compared on the same corruptions.

PROVENANCE
----------
Derived, not collected. It answers one question: does a single mistyped
character break an utterance the model otherwise gets right? It is a diagnostic,
never a training source and never a headline accuracy number.

Usage:
    python scripts/build_typo_testset.py
    python scripts/build_typo_testset.py --per-utterance 2 --out data/eval/typo2.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.common import load_rows, token_key, tokenize  # noqa: E402

SEED = 20260810

# QWERTY physical neighbours — a substitution has to be a plausible slip, not a
# random letter, or the set measures something no keyboard produces.
NEIGHBOURS = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}


def corrupt(word: str, rng: random.Random) -> str:
    """One character-level slip. Returns the word unchanged if it is too short
    to corrupt meaningfully."""
    if len(word) <= 3:
        return word
    kinds = ["substitute", "drop", "duplicate", "transpose"]
    for _ in range(6):  # retry until the word actually changes
        kind = rng.choice(kinds)
        i = rng.randrange(len(word))
        if kind == "substitute":
            opts = NEIGHBOURS.get(word[i], "")
            if not opts:
                continue
            out = word[:i] + rng.choice(opts) + word[i + 1:]
        elif kind == "drop":
            out = word[:i] + word[i + 1:]
        elif kind == "duplicate":
            out = word[:i] + word[i] + word[i:]
        else:
            if i >= len(word) - 1:
                continue
            out = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        if out != word:
            return out
    return word


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-utterance", type=int, default=1,
                    help="how many words to corrupt in each utterance")
    ap.add_argument("--out", type=Path,
                    default=config.DATA / "eval" / "typo_test_en.csv")
    ap.add_argument("--sources", nargs="+", default=["stress", "locked"])
    args = ap.parse_args()

    rng = random.Random(SEED)
    paths = {"stress": config.STRESS_TEST, "locked": config.LOCKED_TEST}
    train_keys = {token_key(t) for t, _ in load_rows(config.TRAIN_CSV)}

    rows, skipped = [], collections.Counter()
    for src in args.sources:
        p = paths.get(src)
        if not p or not p.exists():
            skipped["missing source"] += 1
            continue
        for text, intent in load_rows(p):
            words = text.split()
            idx = [i for i, w in enumerate(words) if len(w.strip(".,?!'")) > 3]
            if not idx:
                skipped["no word long enough"] += 1
                continue
            picks = rng.sample(idx, min(args.per_utterance, len(idx)))
            new = list(words)
            for i in picks:
                new[i] = corrupt(new[i], rng)
            typo = " ".join(new)
            if typo == text:
                skipped["corruption was a no-op"] += 1
                continue
            # a corruption that lands on a real training phrase would be
            # measuring memorisation, not robustness
            if token_key(typo) in train_keys:
                skipped["collided with training"] += 1
                continue
            rows.append((typo, intent, text, src))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "intent", "original", "source"])
        w.writerows(rows)

    print(f"seed              : {SEED} (deterministic)")
    print(f"words corrupted   : {args.per_utterance} per utterance")
    print(f"rows written      : {len(rows)}")
    by = collections.Counter(s for *_, s in rows)
    for s, n in sorted(by.items()):
        print(f"   from {s:<8}{n:>6}")
    if skipped:
        print("skipped:")
        for why, n in skipped.most_common():
            print(f"   {why:<28}{n:>5}")

    print("\nsample:")
    for typo, intent, orig, _ in rows[:8]:
        print(f"   {orig!r}\n     -> {typo!r}   [{intent}]")
    print(f"\nwrote {args.out}")
    print("\nDERIVED SET — never train on it, never quote it as headline accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
