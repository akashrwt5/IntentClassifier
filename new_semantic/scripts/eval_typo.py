#!/usr/bin/env python3
"""
How much damage does one mistyped character do?

Reports a PAIRED comparison, which is the only way to read this. Raw accuracy on
the typo set conflates two things: rows the model never got right anyway, and
rows a typo broke. Only the second is typo robustness.

    original correct, typo correct    robust
    original correct, typo WRONG      BROKEN BY THE TYPO   <- the number
    original wrong                    excluded, nothing to measure

Also splits by whether the corrupted word became out-of-vocabulary, because
those are two different failure modes with two different fixes:

    still in vocab   the model saw a real word and mis-read it
    now OOV          the word became [UNK] and the utterance lost a token

Runs the INSTALLED artifact through the runtime class, so the number is what the
device would do. No torch required.

Usage:
    python scripts/eval_typo.py
    python scripts/eval_typo.py --csv data/eval/typo2_test_en.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import importlib.util
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.common import tokenize  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
INSTALL = REPO / "models" / "semantic_student" / "en"


def wilson(p: float, n: int) -> float:
    if not n:
        return float("nan")
    z = 1.96
    return (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


class TagScorer:
    """Score an exported tag that is not installed, WHATEVER tokenizer it uses.

    Deliberately built on `scripts/common.encode` rather than the runtime class.
    Two different jobs:

      * the runtime class is the contract for what SHIPS — the installed model is
        scored through it so the number is what the device does;
      * `common.encode` is the contract for what was TRAINED — a candidate that
        has not shipped yet has no runtime path, and the honest way to score it
        is the tokenizer its own training used.

    This matters here specifically: subword is the main candidate for typo
    robustness (unseen words split into known pieces instead of collapsing to
    [UNK]), and `StudentSemantic` implements the word tokenizer only. Forcing
    subword through it would mis-tokenise every row and report a confidently
    wrong number — the exact failure this eval exists to catch.
    """

    def __init__(self, tag: str):
        import json

        import numpy as np
        import onnxruntime as ort

        from scripts.common import encode, load_vocab

        onnx = config.MODELS / f"student_{tag}.onnx"
        if not onnx.exists():
            raise SystemExit(f"{onnx} not found — export the tag first:\n"
                             f"    python scripts/export_onnx.py --tag {tag} "
                             f"--threshold 0.40 --skip-int8")
        summary = config.REPORTS / f"train_{tag}_summary.json"
        meta = json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else {}

        self._encode_fn = encode
        self._np = np
        self.vocab, mode = load_vocab(config.MODELS / f"vocab_{tag}.json")
        self.mode = meta.get("tokenizer", mode)
        self.max_len = meta.get("max_len", config.MAX_LEN)
        self.labels = json.loads(
            (config.MODELS / f"labels_{tag}.json").read_text(encoding="utf-8"))

        calib = config.REPORTS / f"calibration_{tag}.json"
        self.temperature = (
            float(json.loads(calib.read_text(encoding="utf-8")).get("temperature", 1.0))
            if calib.exists() else 1.0)
        self.threshold = 0.40

        self._sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
        self._in_ids = self._sess.get_inputs()[0].name
        self._in_mask = self._sess.get_inputs()[1].name
        print(f"scoring tag {tag!r}  tokenizer={self.mode}  max_len={self.max_len}  "
              f"T={self.temperature}"
              f"{'' if calib.exists() else ' (UNCALIBRATED)'}")

    def classify(self, text: str):
        np = self._np
        ids = np.array([self._encode_fn(text, self.vocab, self.max_len, self.mode)[0]],
                       dtype=np.int64)
        mask = ids != config.PAD_ID
        logits = self._sess.run(None, {self._in_ids: ids, self._in_mask: mask})[0][0]
        z = logits / self.temperature
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        top = int(np.argmax(p))
        return str(self.labels[top]), float(p[top])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=config.DATA / "eval" / "typo_test_en.csv")
    ap.add_argument("--tag", default=None,
                    help="score an exported tag (models/en/student_<tag>.onnx) "
                    "instead of the installed student. Needed to compare "
                    "candidates without installing each one.")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"{args.csv} not found — run scripts/build_typo_testset.py")

    spec = importlib.util.spec_from_file_location(
        "_sem", REPO / "packages" / "runtime" / "nlu_engine" / "semantic.py")
    mod = importlib.util.module_from_spec(spec)          # type: ignore[arg-type]
    spec.loader.exec_module(mod)                          # type: ignore[union-attr]

    if args.tag:
        s = TagScorer(args.tag)
    else:
        s = mod.StudentSemantic(INSTALL)

    with open(args.csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    stats = collections.Counter()
    by_source = collections.defaultdict(collections.Counter)
    broken_examples = []

    for r in rows:
        typo, gold, orig, src = r["text"], r["intent"], r["original"], r["source"]
        o_pred, _ = s.classify(orig)
        if o_pred != gold:
            stats["original already wrong"] += 1
            by_source[src]["excluded"] += 1
            continue
        t_pred, t_conf = s.classify(typo)
        ok = t_pred == gold
        stats["measurable"] += 1
        by_source[src]["measurable"] += 1
        # did the corruption push a word out of vocabulary?
        new_oov = any(w not in s.vocab for w in tokenize(typo)) and not any(
            w not in s.vocab for w in tokenize(orig))
        key = "now OOV" if new_oov else "still in vocab"
        stats[f"{key}: total"] += 1
        if ok:
            stats["robust"] += 1
            by_source[src]["robust"] += 1
            stats[f"{key}: robust"] += 1
        else:
            stats["broken"] += 1
            if len(broken_examples) < 8:
                broken_examples.append((orig, typo, gold, t_pred, t_conf))

    n = stats["measurable"]
    if not n:
        raise SystemExit("nothing measurable")
    p = stats["broken"] / n
    ci = wilson(p, n)

    print(f"set               : {args.csv.name}  ({len(rows)} rows)")
    print(f"installed student : T={s.temperature}, gate {s.threshold}\n")
    print(f"  original already wrong (excluded)   {stats['original already wrong']:>6}")
    print(f"  measurable                          {n:>6}")
    print(f"     survived the typo                {stats['robust']:>6}"
          f"   {stats['robust'] / n:>7.4f}")
    print(f"     BROKEN BY THE TYPO               {stats['broken']:>6}"
          f"   {p:>7.4f}  +-{ci:.4f}")

    print("\n  by what the corruption did to the word:")
    for key in ("still in vocab", "now OOV"):
        t = stats[f"{key}: total"]
        if not t:
            continue
        rb = stats[f"{key}: robust"]
        print(f"     {key:<16} {t:>6} rows   broken {(t - rb) / t:>7.4f}")

    print("\n  by source set:")
    for src, c in sorted(by_source.items()):
        m = c["measurable"]
        if m:
            print(f"     {src:<8} {m:>6} measurable   broken {(m - c['robust']) / m:>7.4f}")

    print("\n  examples broken by a single character:")
    for orig, typo, gold, pred, conf in broken_examples:
        print(f"     {orig!r}\n       -> {typo!r}")
        print(f"          {gold}  ->  {pred} ({conf:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
