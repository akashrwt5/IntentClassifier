#!/usr/bin/env python3
"""
Type an utterance, see what every model says — side by side.

Columns are whatever is trained/installed:

    e5-small / minilm / bge   frozen encoder + LogReg head, from train_backbones.py
    student                   the 2.5 MB distilled model that actually ships

The point is the DISAGREEMENTS. Where all four agree the input is easy and tells
you nothing. Where the 90 MB encoders agree with each other and the student does
not, that gap is what the student gives up for being 40x smaller.

Usage:
    python scripts/compare_backbones.py
    python scripts/compare_backbones.py "elevate the volume"
    python scripts/compare_backbones.py --file data/eval/probe_phrases.txt
    python scripts/compare_backbones.py --no-student
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

OUT_ROOT = config.MODELS.parent / "backbones"


class Backbone:
    """Frozen sentence encoder + the logistic head trained on top of it."""

    def __init__(self, key: str):
        from sentence_transformers import SentenceTransformer

        d = OUT_ROOT / key
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.labels = json.loads((d / "labels.json").read_text(encoding="utf-8"))
        blob = np.load(d / "head.npz")
        self.W, self.b = blob["coef"], blob["intercept"]
        self.enc = SentenceTransformer(self.meta["repo_id"])
        self.prefix = self.meta.get("prefix", "")
        self.name = key

    def predict(self, text: str):
        e = self.enc.encode([self.prefix + text], normalize_embeddings=True)[0]
        z = self.W @ e + self.b
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        i = int(p.argmax())
        return self.labels[i], float(p[i])


class InstalledStudent:
    """The shipping Stage 3, loaded through the engine's own class."""

    def __init__(self):
        import importlib.util

        repo = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "_sem", repo / "packages" / "runtime" / "nlu_engine" / "semantic.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        d = repo / "models" / "semantic_student" / "en"
        self.impl = m.StudentSemantic(d)
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.name = f"student ({self.meta.get('tag', '?')})"
        self.size_mb = (d / "student.onnx").stat().st_size / 1e6

    def predict(self, text: str):
        return self.impl.classify(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--file", type=Path)
    ap.add_argument("--models", nargs="+", default=["e5-small", "minilm", "bge"])
    ap.add_argument("--no-student", action="store_true")
    ap.add_argument("--gate", type=float, default=0.40)
    args = ap.parse_args()

    models = []
    for k in args.models:
        if not (OUT_ROOT / k / "head.npz").exists():
            print(f"  (skipping {k}: not trained — run train_backbones.py)")
            continue
        print(f"loading {k} ...", flush=True)
        models.append(Backbone(k))

    if not args.no_student:
        try:
            print("loading student ...", flush=True)
            models.append(InstalledStudent())
        except Exception as e:  # noqa: BLE001
            print(f"  (no installed student: {type(e).__name__})")

    if not models:
        raise SystemExit("nothing to compare")

    print()
    for m in models:
        if isinstance(m, Backbone):
            print(f"  {m.name:<22} {m.meta['encoder_mb_fp32']:>7.0f} MB  "
                  f"{m.meta['dim']}-d  {m.meta['repo_id']}")
        else:
            print(f"  {m.name:<22} {m.size_mb:>7.2f} MB  (the one that ships)")
    print(f"\ngate {args.gate} — below it, or predicting fallback, counts as REJECT\n")

    width = max(len(m.name) for m in models) + 2

    def show(text: str):
        text = text.strip()
        if not text:
            return
        out = []
        for m in models:
            intent, conf = m.predict(text)
            rejected = intent == config.FALLBACK_INTENT or conf < args.gate
            out.append((m.name, intent, conf, rejected))

        for name, intent, conf, rej in out:
            mark = "REJECT" if rej else "accept"
            print(f"  {name:<{width}} {intent:<32} {conf:.4f}  {mark}")

        decided = {i for _, i, _, r in out if not r}
        if len(decided) > 1:
            print(f"  {'':<{width}} ^^ MODELS DISAGREE: {sorted(decided)}")
        elif not decided:
            print(f"  {'':<{width}} ^^ all reject")
        print()

    if args.file:
        for ln in args.file.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                print(f"> {ln.strip()}")
                show(ln)
        return 0
    if args.text:
        t = " ".join(args.text)
        print(f"> {t}")
        show(t)
        return 0

    print("type an utterance, blank line or Ctrl-D to quit\n")
    while True:
        try:
            t = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not t.strip():
            return 0
        show(t)


if __name__ == "__main__":
    raise SystemExit(main())
