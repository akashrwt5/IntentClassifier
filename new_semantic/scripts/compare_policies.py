#!/usr/bin/env python3
"""
The student loses to Stage 2 on the handover subset (0.3632 vs 0.6923 on stress).
Before retraining anything, find out WHY and whether the combination POLICY —
not the model — is what is broken.

The current policy throws Stage 2's opinion away the moment it hands over. But
Stage 2 still scores 0.6923 on exactly those rows. Discarding a 0.69 signal to
replace it with a 0.36 signal is a pipeline bug, not a model deficiency.

Policies compared on the handover subset:

  s2_only        no Stage 3 at all (the bar)
  replace        student wins; below its gate -> fallback        [CURRENT]
  s2_backstop    student if confident, else KEEP STAGE 2's answer
  max_conf       whichever of the two is more confident
  avg            average the two probability vectors
  s2_prior       geometric blend, weighted toward Stage 2

Also decomposes the student's losses into misclassification vs gate rejection —
if most failures are the gate, the fix is the threshold, not the weights.

Usage:
    python scripts/compare_policies.py --tag unkaug_s1 --threshold 0.40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.common import encode, load_rows, load_vocab  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
STAGE2_ONNX = REPO / "models" / "intent_model.onnx"
WEIGHTS = REPO / "models" / "intent_classifier_weights.json"
S2_LABELS = REPO / "models" / "intent_labels.json"


def stage2_probs(texts, temperature):
    """Full probability matrix in models/intent_labels.json column order
    (verified: argmax of that order reproduces the model's own label output)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(STAGE2_ONNX), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    _, scores = sess.run(None, {name: np.array(texts, dtype=object).reshape(-1, 1)})
    z = np.array(scores, dtype=np.float64) / temperature
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threshold", type=float, default=0.40)
    args = ap.parse_args()

    import torch

    from scripts.train_en import build_student

    meta_w = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    s2_T = float(meta_w.get("temperature", 1.0))
    s2_gate = float(meta_w.get("conf_threshold", 0.7))
    s2_labels = json.loads(S2_LABELS.read_text(encoding="utf-8"))

    vocab, tok_mode = load_vocab(config.MODELS / f"vocab_{args.tag}.json")
    meta = json.loads(
        (config.REPORTS / f"train_{args.tag}_summary.json").read_text(encoding="utf-8")
    )
    max_len = meta.get("max_len", config.MAX_LEN)
    tok_mode = meta.get("tokenizer", tok_mode)
    labels = json.loads((config.MODELS / f"labels_{args.tag}.json").read_text(encoding="utf-8"))
    model = build_student(len(vocab), len(labels), dim=meta.get("embed_dim", config.EMBED_DIM))
    model.load_state_dict(torch.load(config.MODELS / f"student_{args.tag}.pt"))
    model.eval()

    # align Stage 2's columns onto the student's label ordering
    col = {l: i for i, l in enumerate(s2_labels)}
    perm = [col[l] for l in labels]
    fb = labels.index(config.FALLBACK_INTENT)

    def student_probs(texts):
        X = np.array([encode(t, vocab, max_len, tok_mode)[0] for t in texts], dtype=np.int64)
        M = X != config.PAD_ID
        with torch.no_grad():
            return (
                torch.softmax(model(torch.tensor(X), torch.tensor(M)), -1)
                .numpy()
                .astype(np.float64)
            )

    print(f"Stage 2: T={s2_T} gate={s2_gate}   |   student gate={args.threshold}\n")
    out = {}

    for name, path, is_ood in (
        ("stress", config.STRESS_TEST, False),
        ("locked", config.LOCKED_TEST, False),
        ("ood", config.OOD_TEST, True),
    ):
        if not path.exists():
            continue
        rows = load_rows(path)
        if not is_ood:
            rows = [(t, g) for t, g in rows if g in labels]
        texts = [t for t, _ in rows]
        gold = [g for _, g in rows]

        P2 = stage2_probs(texts, s2_T)[:, perm]
        c2, i2 = P2.max(1), P2.argmax(1)
        hand = c2 < s2_gate
        idx = np.where(hand)[0]
        if len(idx) == 0:
            continue

        PS = student_probs([texts[i] for i in idx])
        cs, isx = PS.max(1), PS.argmax(1)
        p2 = P2[idx]

        def score(pred_idx, accept):
            if is_ood:
                rej = [(p == fb) or (not a) for p, a in zip(pred_idx, accept)]
                return float(np.mean(rej))
            return float(
                np.mean([a and labels[p] == gold[i] for p, a, i in zip(pred_idx, accept, idx)])
            )

        c2h = p2.max(1)
        policies = {}
        policies["s2_only"] = score(p2.argmax(1), [True] * len(idx))
        policies["replace"] = score(isx, cs >= args.threshold)
        keep = np.where(cs >= args.threshold, isx, p2.argmax(1))
        policies["s2_backstop"] = score(keep, [True] * len(idx))
        mx = np.where(cs >= c2h, isx, p2.argmax(1))
        policies["max_conf"] = score(mx, [True] * len(idx))
        avg = (PS + p2) / 2
        policies["avg"] = score(avg.argmax(1), avg.max(1) >= args.threshold)
        blend = (PS**0.35) * (p2**0.65)
        blend /= blend.sum(1, keepdims=True)
        policies["s2_prior"] = score(blend.argmax(1), blend.max(1) >= args.threshold)

        # ---- dual gate -------------------------------------------------
        # max_conf answers whenever EITHER model is confident, which is why its
        # OOD rejection is weak: an OOD utterance only has to fool one of them.
        #
        # NOTE ON A FAILED FIRST ATTEMPT: gating on `c2h >= s2_gate` (0.7) is
        # vacuous here — the handover subset is DEFINED by Stage 2 scoring below
        # 0.7, so that clause is always False and the policy degenerates into
        # `replace`. Stage 2 needs a SECONDARY, lower threshold on this subset.
        for s3g in (args.threshold, 0.5, 0.6):
            for s2g in (0.30, 0.45, 0.60):
                accept = (cs >= s3g) | (c2h >= s2g)
                policies[f"dual@s3={s3g:.2f},s2={s2g:.2f}"] = score(mx, accept)

        # agreement gate: trust the turn when both models point the same way,
        # otherwise demand real confidence from the student.
        agree = isx == p2.argmax(1)
        for s3g in (0.5, 0.6):
            accept = agree | (cs >= s3g)
            policies[f"agree_or_conf@{s3g:.2f}"] = score(mx, accept)

        # ---- end-to-end: what the user actually experiences -------------
        n = len(texts)
        kept = np.where(~hand)[0]
        if is_ood:
            # Stage 2 keeps it and does NOT call it fallback -> wrong command fires
            kept_ok = int(sum(1 for i in kept if i2[i] == fb))
        else:
            kept_ok = int(sum(1 for i in kept if labels[i2[i]] == gold[i]))

        def e2e(v):
            return (kept_ok + v * len(idx)) / n

        label = "OOD reject" if is_ood else "accuracy"
        base = policies["s2_only"]
        print(
            f"{name.upper()}  {n} rows | handover {len(idx)} ({hand.mean() * 100:.1f}%) "
            f"| Stage 2 keeps {len(kept)} ({kept_ok} good)"
        )
        print(f"   {'policy':<26} {'handover':>9} {'delta':>8}   {'END-TO-END':>10}")
        for k, v in sorted(policies.items(), key=lambda kv: -kv[1]):
            mark = "  <- CURRENT" if k == "replace" else ("  <- bar" if k == "s2_only" else "")
            print(f"   {k:<26} {v:>9.4f} {v - base:>+8.4f}   {e2e(v):>10.4f}{mark}")
        policies = {
            k: {"handover": round(v, 4), "end_to_end": round(e2e(v), 4)}
            for k, v in policies.items()
        }

        if not is_ood:
            correct = np.array([labels[p] == gold[i] for p, i in zip(isx, idx)])
            gated = cs < args.threshold
            print(f"   -- student failure breakdown on {len(idx)} rows --")
            print(f"      classified correctly       {correct.sum():>4}")
            print(
                f"      ...but rejected by gate    {(correct & gated).sum():>4}  <- recoverable by lowering the gate"
            )
            print(
                f"      misclassified              {(~correct).sum():>4}  <- needs a better model"
            )
        print()
        out[name] = {"handover_rows": int(len(idx)), "policies": policies}

    p = config.REPORTS / f"policies_{args.tag}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
