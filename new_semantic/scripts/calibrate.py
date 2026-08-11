#!/usr/bin/env python3
"""
Fit a temperature so the confidence number means something.

THE PROBLEM
-----------
The student ranks well and reports badly. Measured on stress:

    mean confidence when the answer is CORRECT   0.4861
    correct but rejected by the 0.40 gate        115 / 420   (27.4%)

Over a quarter of the right answers are thrown away by the gate. That is not a
capability failure — the argmax was correct — it is a calibration failure. The
same shows up as a cliff: moving the gate 0.60 -> 0.65 drops in-scope accuracy
from 0.798 to 0.552 for +0.5% OOD, because a large mass of correct answers sits
inside that narrow band.

THE FIX
-------
One scalar T, fitted by minimising NLL on held-out data:

    p = softmax(logits / T)

T is RANK-PRESERVING: it never changes which intent wins, only how confident the
model sounds. So it cannot make accuracy worse at a fixed decision rule — it
moves where the useful thresholds are. T < 1 sharpens, T > 1 softens.

Fitted on the DEV half only (dev_test_split.json). Reported on TEST.

Usage:
    python scripts/calibrate.py --tag semfz_s1
    python scripts/calibrate.py --tag semfz_s1 --apply     # write into meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.common import encode, load_rows, load_vocab, token_key  # noqa: E402
from scripts.select_policy import build_split  # noqa: E402


def logits_for(tag, texts):
    """Raw logits — NOT softmaxed. Temperature has to be fitted pre-softmax."""
    import onnxruntime as ort

    onnx = config.MODELS / f"student_{tag}.onnx"
    if not onnx.exists():
        raise SystemExit(f"{onnx} not found — export the tag first")
    vocab, mode = load_vocab(config.MODELS / f"vocab_{tag}.json")
    meta = json.loads(
        (config.REPORTS / f"train_{tag}_summary.json").read_text(encoding="utf-8"))
    max_len = meta.get("max_len", config.MAX_LEN)
    mode = meta.get("tokenizer", mode)
    labels = json.loads(
        (config.MODELS / f"labels_{tag}.json").read_text(encoding="utf-8"))

    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    out = []
    for t in texts:
        ids, _ = encode(t, vocab, max_len, mode)
        X = np.array([ids], dtype=np.int64)
        out.append(sess.run(None, {"input_ids": X,
                                   "attention_mask": X != config.PAD_ID})[0][0])
    return np.array(out, dtype=np.float64), labels


def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(z, y, T):
    p = softmax(z, T)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))))


def ece(p, y, bins=10):
    """Expected calibration error: |confidence - accuracy| within each bin."""
    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == y).astype(float)
    total = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf >= lo) & (conf < lo + 1 / bins)
        if m.sum():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gate", type=float, default=0.40)
    ap.add_argument("--apply", action="store_true",
                    help="write T into the installed models/semantic_student/en/meta.json")
    args = ap.parse_args()

    split = build_split()["assign"]

    # In-scope rows only: temperature is fitted against true labels, and the OOD
    # set has none. OOD is used afterwards to check the fit did not cost
    # rejection.
    rows = []
    for p in (config.LOCKED_TEST, config.STRESS_TEST):
        if p.exists():
            rows += load_rows(p)

    dev = [(t, g) for t, g in rows if split.get(token_key(t)) == "dev"]
    test = [(t, g) for t, g in rows if split.get(token_key(t)) == "test"]
    print(f"dev {len(dev)} / test {len(test)} in-scope rows")

    Zd, labels = logits_for(args.tag, [t for t, _ in dev])
    keep = [i for i, (_, g) in enumerate(dev) if g in labels]
    Zd = Zd[keep]
    yd = np.array([labels.index(dev[i][1]) for i in keep])

    grid = np.concatenate([np.linspace(0.05, 1.0, 96), np.linspace(1.05, 5.0, 80)])
    losses = [nll(Zd, yd, T) for T in grid]
    T = float(grid[int(np.argmin(losses))])
    print(f"\nfitted T = {T:.4f}   ({'sharpens' if T < 1 else 'softens'} confidence)")
    print(f"  dev NLL  {nll(Zd, yd, 1.0):.4f} -> {nll(Zd, yd, T):.4f}")

    Zt, _ = logits_for(args.tag, [t for t, _ in test])
    keep = [i for i, (_, g) in enumerate(test) if g in labels]
    Zt = Zt[keep]
    yt = np.array([labels.index(test[i][1]) for i in keep])

    print(f"\n{'':<22}{'T=1':>10}{'T=' + f'{T:.2f}':>10}")
    for name, fn in (
        ("ECE (lower better)", lambda t: ece(softmax(Zt, t), yt)),
        ("mean conf", lambda t: float(softmax(Zt, t).max(1).mean())),
        ("accuracy (argmax)", lambda t: float((softmax(Zt, t).argmax(1) == yt).mean())),
        (f"accuracy @ {args.gate}", lambda t: float(np.mean(
            (softmax(Zt, t).argmax(1) == yt) & (softmax(Zt, t).max(1) >= args.gate)))),
    ):
        print(f"  {name:<20}{fn(1.0):>10.4f}{fn(T):>10.4f}")

    ood = {}
    if config.OOD_TEST.exists():
        orows = [t for t, _ in load_rows(config.OOD_TEST)
                 if split.get(token_key(t)) == "test"]
        Zo, _ = logits_for(args.tag, orows)
        fb = labels.index(config.FALLBACK_INTENT)
        for t in (1.0, T):
            P = softmax(Zo, t)
            ood[t] = float(np.mean((P.argmax(1) == fb) | (P.max(1) < args.gate)))
        print(f"  {'OOD reject @ ' + str(args.gate):<20}{ood[1.0]:>10.4f}{ood[T]:>10.4f}")

    rep = config.REPORTS / f"calibration_{args.tag}.json"
    rep.write_text(json.dumps({
        "tag": args.tag, "temperature": T, "gate": args.gate,
        "dev_rows": len(yd), "test_rows": len(yt),
        "test_ece_before": ece(softmax(Zt, 1.0), yt),
        "test_ece_after": ece(softmax(Zt, T), yt),
        "ood_reject_before": ood.get(1.0), "ood_reject_after": ood.get(T),
        "note": "rank-preserving; argmax accuracy is unchanged by construction",
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {rep}")

    if args.apply:
        inst = Path(__file__).resolve().parents[2] / "models" / "semantic_student" / "en"
        mp = inst / "meta.json"
        if not mp.exists():
            raise SystemExit(f"{mp} not found — install the student first")
        meta = json.loads(mp.read_text(encoding="utf-8"))
        meta["temperature"] = T
        mp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"wrote temperature into {mp}")
        print("StudentSemantic reads and applies this at inference time.")
        print()
        print("NEXT: the gate must be re-picked on this scale. T sharpens the")
        print("confidence, so a threshold chosen at T=1 is effectively looser")
        print("once T is applied:")
        print(f"    python scripts/select_policy.py --tags {args.tag} --reveal-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
