#!/usr/bin/env python3
"""
Is the vocabulary actually made of words this app will hear?

WHY THIS EXISTS
---------------
`build_semantic_vocab.py` fills the space beyond the training corpus from the
teacher's tokenizer *in token-id order*, which for BERT-family models is roughly
Wikipedia/BookCorpus frequency. That was meant to add "common English words" for
generalisation. What it actually adds is corpus-common words:

    in the shipped 8,000-token vocabulary:
        present : aaron, abdul, abbey, abraham, output
        ABSENT  : elevate, heighten, diminish, dampen, magnify, lessen

75% of the vocabulary never appears in a single training row. Those entries are
not neutral. Their embeddings are teacher-initialised and (with
--freeze-embeddings) never updated, while the transformer above them never sees
them in training, so nothing ever teaches the model what to do when one shows
up. The frozen vector wins by default:

    "turn up the volume"  -> Cmd.VolumeIncrease    0.89
    "turn up the output"  -> Cmd.ActivityCalories  0.96      <- 'output', 0 train rows

That is worse than an out-of-vocabulary word would have been. [UNK] at least has
learned behaviour, because UNK augmentation put it in training on purpose.

WHAT IT REPORTS
---------------
1. COMPOSITION   how much of the vocabulary training ever exercises
2. DOMAIN GAPS   in-domain words a user would plausibly say that are missing
3. DEAD-WORD HARM  accuracy on eval rows that contain an untrained vocabulary
                   word vs rows that do not, WITH the confidence interval, so a
                   gap that is inside the noise is reported as inside the noise

No torch required — runs the installed ONNX through the runtime class.

Usage:
    python scripts/vocab_health.py
    python scripts/vocab_health.py --vocab models/en/vocab_semfz_s1.json
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.common import load_rows, load_vocab, tokenize  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
INSTALL = REPO / "models" / "semantic_student" / "en"

# Words a hearing-aid user could plausibly say for the four volume intents.
# Authored, and used ONLY to report coverage — never written to any data file.
DOMAIN_WORDS = {
    "louder": ["elevate", "heighten", "magnify", "amplify", "boost", "intensify",
               "strengthen", "augment", "raise", "increase", "louder", "up"],
    "quieter": ["diminish", "dampen", "subdue", "attenuate", "lessen", "soften",
                "lower", "decrease", "quieter", "reduce", "down"],
    "mute": ["mute", "silence", "deaden", "quell", "off", "stop"],
    "unmute": ["unmute", "reinstate", "restore", "resume", "back", "again", "on"],
}


def load_runtime():
    spec = importlib.util.spec_from_file_location(
        "_sem", REPO / "packages" / "runtime" / "nlu_engine" / "semantic.py")
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load semantic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wilson(p: float, n: int) -> float:
    """Half-width of the 95% Wilson interval.

    NOT the normal approximation `1.96*sqrt(p(1-p)/n)`, which this used first and
    which is wrong exactly where small buckets land: at p=1.0 it returns ZERO
    width, so any gap looks significant. That produced a bogus "clears" verdict
    on a 12-row bucket where every row happened to be correct. Wilson keeps a
    sane width at the boundaries and widens properly for small n.
    """
    if not n:
        return float("nan")
    z = 1.96
    return (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=Path, default=None,
                    help="default: the installed student's vocab.json")
    args = ap.parse_args()

    if args.vocab:
        vocab, _ = load_vocab(args.vocab)
        source = args.vocab.name
    else:
        raw = json.loads((INSTALL / "vocab.json").read_text(encoding="utf-8"))
        vocab = raw["vocab"] if isinstance(raw, dict) and "vocab" in raw else raw
        source = f"installed ({INSTALL.name})"

    train = collections.Counter()
    for t, _ in load_rows(config.TRAIN_CSV):
        train.update(tokenize(t))

    words = [w for w in vocab if w not in ("[PAD]", "[UNK]", "<pad>", "<unk>")]
    dead = {w for w in words if train[w] == 0}

    print(f"vocabulary source : {source}")
    print("\n1. COMPOSITION")
    print(f"   total                    {len(words):>6}")
    print(f"   exercised by training    {len(words) - len(dead):>6}"
          f"  ({(len(words) - len(dead)) / len(words) * 100:.0f}%)")
    print(f"   never seen in training   {len(dead):>6}"
          f"  ({len(dead) / len(words) * 100:.0f}%)   <- no learned behaviour")

    print("\n2. DOMAIN COVERAGE  (words a user could say for the volume intents)")
    missing_all = []
    for group, ws in DOMAIN_WORDS.items():
        miss = [w for w in ws if w not in vocab]
        missing_all += miss
        have = len(ws) - len(miss)
        print(f"   {group:<9} {have:>2}/{len(ws)} present" +
              (f"   MISSING: {', '.join(miss)}" if miss else ""))
    if missing_all:
        print(f"\n   {len(missing_all)} in-domain words absent, while {len(dead)} "
              f"never-used words occupy the space.")

    if not (INSTALL / "student.onnx").exists():
        print("\n(no installed student — skipping section 3)")
        return 0

    mod = load_runtime()
    s = mod.StudentSemantic(INSTALL)
    print("\n3. DEAD-WORD HARM")
    print(f"   {'eval set':<10}{'rows':>6}{'clean':>9}{'has dead':>10}"
          f"{'gap':>8}{'CI':>8}  verdict")
    for name, path in (("stress", config.STRESS_TEST), ("locked", config.LOCKED_TEST)):
        if not path.exists():
            continue
        buckets = {True: [0, 0], False: [0, 0]}
        for t, g in load_rows(path):
            has = any(w in dead for w in tokenize(t))
            buckets[has][1] += 1
            if s.classify(t)[0] == g:
                buckets[has][0] += 1
        (c1, n1), (c0, n0) = buckets[True], buckets[False]
        if not n1 or not n0:
            continue
        if min(n1, n0) < 30:
            print(f"   {name:<10}{n1 + n0:>6}   only {min(n1, n0)} rows in the smaller "
                  f"bucket — too few to read")
            continue
        a1, a0 = c1 / n1, c0 / n0
        gap = a0 - a1
        ci = wilson(a1, n1) + wilson(a0, n0)
        verdict = "clears" if abs(gap) > ci else "inside the noise"
        print(f"   {name:<10}{n1 + n0:>6}{a0:>9.4f}{a1:>10.4f}"
              f"{gap:>+8.4f}{ci:>8.4f}  {verdict}")

    print("\n   'clean' = every word was seen in training. 'has dead' = at least")
    print("   one vocabulary word the model was never trained to use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
