#!/usr/bin/env python3
"""
Dependency-free baseline: TF-IDF + multinomial logistic regression, pure numpy.

Why this exists: it needs no torch, no sklearn, no network — so it runs anywhere
and gives an HONEST floor number on the clean, leak-free data in minutes.

It answers one question: how much of the old 98.87% was real signal and how much
was the locked-test leak? Any neural student must comfortably beat this baseline
to justify itself.

Usage:
    python scripts/baseline_numpy.py
    python scripts/baseline_numpy.py --no-class-weights
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.common import assert_no_leak, load_rows, tokenize  # noqa: E402


def build_features(texts, vocab=None, idf=None, max_features=4000):
    if vocab is None:
        df = Counter()
        for t in texts:
            df.update(set(tokenize(t)))
        top = [w for w, _ in df.most_common(max_features)]
        vocab = {w: i for i, w in enumerate(top)}
        n = len(texts)
        idf = np.ones(len(vocab), dtype=np.float32)
        for w, i in vocab.items():
            idf[i] = np.log((1 + n) / (1 + df[w])) + 1.0

    X = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for r, t in enumerate(texts):
        c = Counter(tokenize(t))
        for w, k in c.items():
            j = vocab.get(w)
            if j is not None:
                X[r, j] = k
    X *= idf
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    np.divide(X, norms, out=X, where=norms > 0)
    return X, vocab, idf


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_logreg(X, y, n_classes, weights=None, epochs=220, lr=1.0, l2=1e-4, seed=42):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = np.zeros((d, n_classes), dtype=np.float32)
    b = np.zeros(n_classes, dtype=np.float32)
    Y = np.zeros((n, n_classes), dtype=np.float32)
    Y[np.arange(n), y] = 1.0
    sw = np.ones(n, dtype=np.float32) if weights is None else weights[y].astype(np.float32)
    sw = sw / sw.mean()

    idx = np.arange(n)
    batch = 2048
    for ep in range(epochs):
        rng.shuffle(idx)
        for s in range(0, n, batch):
            bi = idx[s : s + batch]
            P = softmax(X[bi] @ W + b)
            G = (P - Y[bi]) * sw[bi][:, None]
            W -= lr * (X[bi].T @ G / len(bi) + l2 * W)
            b -= lr * G.mean(0)
    return W, b


def evaluate(name, texts, gold, W, b, vocab, idf, label_list, fb_label):
    X, _, _ = build_features(texts, vocab, idf)
    P = softmax(X @ W + b)
    pred = [label_list[i] for i in P.argmax(1)]
    conf = P.max(1)

    per_class = defaultdict(list)
    for p, g in zip(pred, gold):
        per_class[g].append(p == g)
    acc = float(np.mean([p == g for p, g in zip(pred, gold)]))
    macro = float(np.mean([np.mean(v) for v in per_class.values()]))
    worst = sorted(
        ((g, float(np.mean(v)), len(v)) for g, v in per_class.items()),
        key=lambda x: x[1],
    )[:8]

    print(f"\n{name.upper():<8} rows={len(texts)}")
    print(f"  accuracy      {acc:.4f}")
    print(f"  macro recall  {macro:.4f}")
    print(f"  mean conf     {conf.mean():.4f}")
    print("  worst classes:")
    for g, r, n in worst:
        print(f"     {g:<34} recall {r:.3f}  (n={n})")
    return {
        "rows": len(texts),
        "accuracy": round(acc, 4),
        "macro_recall": round(macro, 4),
        "mean_confidence": round(float(conf.mean()), 4),
        "worst_classes": [{"intent": g, "recall": round(r, 3), "n": n} for g, r, n in worst],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--epochs", type=int, default=220)
    args = ap.parse_args()

    rows = load_rows(config.TRAIN_CSV)
    texts = [t for t, _ in rows]
    labels = [l for _, l in rows]
    print(f"train rows : {len(rows)}")

    for path, name in (
        (config.LOCKED_TEST, "locked test"),
        (config.STRESS_TEST, "stress test"),
        (config.OOD_TEST, "OOD test"),
    ):
        if path.exists():
            assert_no_leak(texts, [t for t, _ in load_rows(path)], name)
    print("leak guard : OK")

    label_list = sorted(set(labels))
    l2i = {l: i for i, l in enumerate(label_list)}
    y = np.array([l2i[l] for l in labels])

    X, vocab, idf = build_features(texts)
    print(f"features   : {X.shape[1]}  ({X.nbytes / 1e6:.0f} MB dense)")

    w = None
    if not args.no_class_weights:
        counts = Counter(labels)
        w = np.array(
            [len(labels) / (len(label_list) * counts[l]) for l in label_list],
            dtype=np.float32,
        )
        print(f"weights    : ON (min {w.min():.2f} / max {w.max():.2f})")
    else:
        print("weights    : OFF")

    print("training...")
    W, b = train_logreg(X, y, len(label_list), weights=w, epochs=args.epochs)

    results = {}
    for name, path in (("locked", config.LOCKED_TEST), ("stress", config.STRESS_TEST)):
        if not path.exists():
            continue
        r = load_rows(path)
        keep = [(t, g) for t, g in r if g in label_list]
        results[name] = evaluate(
            name,
            [t for t, _ in keep],
            [g for _, g in keep],
            W,
            b,
            vocab,
            idf,
            label_list,
            config.FALLBACK_INTENT,
        )

    if config.OOD_TEST.exists():
        r = load_rows(config.OOD_TEST)
        Xo, _, _ = build_features([t for t, _ in r], vocab, idf)
        Po = softmax(Xo @ W + b)
        pred = [label_list[i] for i in Po.argmax(1)]
        conf = Po.max(1)
        fb = np.array([p == config.FALLBACK_INTENT for p in pred])
        esc = Counter(p for p, f in zip(pred, fb) if not f)
        results["ood"] = {
            "rows": len(r),
            "fallback_rate": round(float(fb.mean()), 4),
            "escaped": int((~fb).sum()),
            "escaped_into": dict(esc.most_common(8)),
        }
        print(f"\nOOD      rows={len(r)}")
        print(f"  fallback rate {fb.mean():.4f}   ({int((~fb).sum())} escaped)")
        for k, v in esc.most_common(6):
            print(f"     escaped -> {k:<32} {v}")

        # threshold sweep — where does the confidence gate belong?
        print("\n  threshold sweep (OOD reject vs locked retained):")
        lr_rows = [(t, g) for t, g in load_rows(config.LOCKED_TEST) if g in label_list]
        Xl, _, _ = build_features([t for t, _ in lr_rows], vocab, idf)
        Pl = softmax(Xl @ W + b)
        gl = [g for _, g in lr_rows]
        pl = [label_list[i] for i in Pl.argmax(1)]
        cl = Pl.max(1)
        sweep = []
        for th in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            ood_rej = float(((conf < th) | fb).mean())
            in_ok = float(np.mean([(p == g) and (c >= th) for p, g, c in zip(pl, gl, cl)]))
            sweep.append(
                {"threshold": th, "ood_reject": round(ood_rej, 4), "in_scope_acc": round(in_ok, 4)}
            )
            print(f"     t={th:.1f}  OOD reject {ood_rej:.3f}  in-scope acc {in_ok:.3f}")
        results["threshold_sweep"] = sweep

    config.REPORTS.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS / "baseline_numpy.json"
    results["_meta"] = {
        "model": "TF-IDF + multinomial logistic regression (numpy)",
        "train_rows": len(rows),
        "features": int(X.shape[1]),
        "class_weights": not args.no_class_weights,
        "purpose": "leak-free floor; the neural student must beat this",
    }
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
