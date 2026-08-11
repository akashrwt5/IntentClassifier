#!/usr/bin/env python3
"""
Interactive tester for the NEW student — the 0.166 MB model trained in this folder.

Runs the SHIPPING ARTIFACT (`models/en/student_<tag>.onnx`) through onnxruntime,
not the PyTorch checkpoint, so what you see is what a device would compute. No
torch needed.

For every utterance it shows:

  * how the closed vocabulary tokenised it, and WHICH WORDS BECAME [UNK] —
    usually the real explanation when an answer looks stupid
  * the top-3 intents with probabilities
  * the gate decision at the student's threshold
  * (optional) what Stage 2 says, for comparison

Usage:
    python scripts/try_student.py
    python scripts/try_student.py "make it louder"
    python scripts/try_student.py --with-stage2
    python scripts/try_student.py --tag base_s1 --threshold 0.5
    python scripts/try_student.py --file phrases.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.common import encode, load_vocab, tokenize  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MIGRATION = REPO / "datasets" / "label_migration_map.json"


class Student:
    """Prefers the shipping ONNX artifact; falls back to the PyTorch checkpoint.

    The fallback exists so a freshly-trained tag can be inspected immediately —
    exporting just to look at predictions is friction during experimentation.
    The backend is printed, because ONNX is what a device runs and the .pt is
    not, and a claim made from one should not be reported as the other.
    """

    def __init__(self, tag: str):
        onnx = config.MODELS / f"student_{tag}.onnx"
        ckpt = config.MODELS / f"student_{tag}.pt"
        self.backend = "onnx" if onnx.exists() else "pytorch"
        if not onnx.exists() and not ckpt.exists():
            raise SystemExit(
                f"no model for tag '{tag}'.\nTrain it, or export an existing one:\n"
                f"  python scripts/export_onnx.py --tag {tag} --threshold 0.40 --skip-int8"
            )
        self.vocab, self.mode = load_vocab(config.MODELS / f"vocab_{tag}.json")
        self.labels = json.loads((config.MODELS / f"labels_{tag}.json").read_text(encoding="utf-8"))
        meta_p = config.REPORTS / f"train_{tag}_summary.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        self.max_len = meta.get("max_len", config.MAX_LEN)
        self.mode = meta.get("tokenizer", self.mode)
        self.meta = meta

        if self.backend == "onnx":
            import onnxruntime as ort

            self.sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
            self.size_mb = onnx.stat().st_size / 1e6
        else:
            import torch

            from scripts.train_en import build_student

            self.torch = torch
            self.model = build_student(
                len(self.vocab), len(self.labels), dim=meta.get("embed_dim", config.EMBED_DIM)
            )
            self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            self.model.eval()
            self.size_mb = ckpt.stat().st_size / 1e6

    def tokens(self, text):
        """(surface token, is_unknown) as the model sees it."""
        if self.mode == "subword":
            from scripts.common import tokenize_subword

            return [(p, p == "[UNK]") for p in tokenize_subword(text, self.vocab)]
        return [(w, w not in self.vocab) for w in tokenize(text)]

    def predict(self, text):
        ids, n = encode(text, self.vocab, self.max_len, self.mode)
        X = np.array([ids], dtype=np.int64)
        M = X != config.PAD_ID
        if self.backend == "onnx":
            logits = self.sess.run(None, {"input_ids": X, "attention_mask": M})[0][0]
        else:
            with self.torch.no_grad():
                logits = self.model(self.torch.tensor(X), self.torch.tensor(M)).numpy()[0]
        z = logits - logits.max()
        e = np.exp(z)
        return e / e.sum(), n


def stage2_probs(text):
    """Stage 2 exactly as classifier.py computes it: softmax(scores / T)."""
    import onnxruntime as ort

    w = json.loads((REPO / "models" / "intent_classifier_weights.json").read_text(encoding="utf-8"))
    T = float(w.get("temperature", 1.0))
    labels = json.loads((REPO / "models" / "intent_labels.json").read_text(encoding="utf-8"))
    sess = ort.InferenceSession(
        str(REPO / "models" / "intent_model.onnx"), providers=["CPUExecutionProvider"]
    )
    name = sess.get_inputs()[0].name
    _, scores = sess.run(None, {name: np.array([text], dtype=object).reshape(-1, 1)})
    z = np.array(scores, dtype=np.float64)[0] / T
    z -= z.max()
    e = np.exp(z)
    return e / e.sum(), labels, float(w.get("conf_threshold", 0.70))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--tag", default="unkaug_s1")
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument(
        "--with-stage2", action="store_true", help="also show what the TF-IDF stage predicts"
    )
    ap.add_argument("--file", type=Path)
    args = ap.parse_args()

    st = Student(args.tag)
    tagline = (
        "ONNX (shipping artifact)"
        if st.backend == "onnx"
        else "PyTorch checkpoint (NOT the shipping artifact)"
    )
    print(f"student : {args.tag}   {st.size_mb:.3f} MB   {tagline}")
    print(f"vocab   : {len(st.vocab)} tokens ({st.mode})   max_len {st.max_len}")
    print(f"gate    : {args.threshold}")
    if st.meta.get("synthetic_rows"):
        print(f"note    : trained with {st.meta['synthetic_rows']} synthetic rows")
    print()

    def show(text: str):
        text = text.strip()
        if not text:
            return
        toks = st.tokens(text)
        unk = [t for t, u in toks if u]
        rendered = " ".join(f"[{t}]" if u else t for t, u in toks)
        print(f"  tokens    {rendered}")
        if unk:
            print(
                f"            {len(unk)}/{len(toks)} unknown -> {unk}"
                f"   <- the model cannot see these words"
            )

        p, n = st.predict(text)
        order = np.argsort(-p)[: args.topk]
        for rank, i in enumerate(order):
            bar = "#" * int(p[i] * 30)
            print(
                f"  {'top' if rank == 0 else '   '} {rank + 1}. "
                f"{st.labels[i]:<34} {p[i]:.4f}  {bar}"
            )

        top = st.labels[order[0]]
        conf = p[order[0]]
        if top == config.FALLBACK_INTENT:
            verdict = "REJECT (predicted fallback)"
        elif conf < args.threshold:
            verdict = f"REJECT (below gate {args.threshold})"
        else:
            verdict = f"ACCEPT -> {top}"
        print(f"  verdict   {verdict}")

        if args.with_stage2:
            p2, l2, gate2 = stage2_probs(text)
            j = int(np.argmax(p2))
            fired = p2[j] >= gate2
            print(f"  stage2    {l2[j]:<34} {p2[j]:.4f}   " f"{'FIRES' if fired else 'hands over'}")
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
