"""
Train the semantic intent model and export a single fused ONNX artifact.

    python -m semantic_intent.train --data balanced_intents_final.xlsx

Pipeline
    load -> audit -> grouped split -> embed (frozen MiniLM)
         -> linear head on train -> temperature on dev -> prototypes
         -> refit head on train+dev -> evaluate on untouched test
         -> fuse to ONNX -> parity check

The encoder is never trained. Everything task-specific is ~17 KB of head plus
~530 KB of OOD prototypes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import data as data_mod
from . import export as export_mod
from .encoder import MiniLMEncoder
from .eval_sets import ANTONYM_PAIRS, HARD_PARAPHRASES, OUT_OF_SCOPE
from .head import (
    SemanticHead,
    build_prototypes,
    choose_threshold,
    expected_calibration_error,
    fit_temperature,
)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"


def _fit_linear(x: np.ndarray, y: np.ndarray, c: float):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=3000, C=c, class_weight="balanced")
    clf.fit(x, y)
    return clf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--encoder", type=Path, default=MODEL_DIR / "minilm-l6-v2.onnx")
    ap.add_argument("--vocab", type=Path, default=MODEL_DIR / "minilm-vocab.txt")
    ap.add_argument("--out", type=Path, default=MODEL_DIR / "semantic_intent.onnx")
    ap.add_argument("--head-out", type=Path, default=MODEL_DIR / "semantic_intent_head.npz")
    ap.add_argument("--report", type=Path, default=MODEL_DIR / "semantic_intent_report.json")
    ap.add_argument(
        "--augment",
        type=int,
        default=400,
        metavar="N",
        help="contrastive state+action phrases per volume intent, "
        "added to the TRAIN split only (0 disables)",
    )
    ap.add_argument("--protos-per-class", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--c-grid", type=float, nargs="+", default=[1, 4, 10, 30, 100])
    args = ap.parse_args()

    report: dict = {}

    # ---------------------------------------------------------------- data
    print("== 1. Dataset ==")
    df = data_mod.load(args.data)
    report["audit"] = data_mod.audit(df)
    print("\n  polarity of ambiguous words:")
    data_mod.polarity_report(df)

    df = data_mod.grouped_split(df, seed=args.seed)
    print("\n  grouped split (no core group spans two splits):")
    print("   ", df.split.value_counts().to_dict())

    if args.augment:
        from .augment import augment_training_split, contrastive_coverage

        before = contrastive_coverage(df)
        df = augment_training_split(df, per_intent=args.augment, seed=args.seed)
        print(f"  contrastive coverage: {before:.1%} -> {contrastive_coverage(df):.1%}")
        report["contrastive_coverage"] = contrastive_coverage(df)

    labels = np.array(sorted(df.intent.unique()))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y = df.intent.map(label_to_idx).values
    tr, dv, te = ((df.split == s).values for s in ("train", "dev", "test"))

    # ------------------------------------------------------------- embed
    print("\n== 2. Embedding with frozen encoder ==")
    encoder = MiniLMEncoder(args.encoder, args.vocab)
    t0 = time.time()
    x = encoder.encode(df.text.tolist())
    print(f"  {len(df)} phrases -> {x.shape} in {time.time() - t0:.1f}s")

    # -------------------------------------------------------------- head
    print("\n== 3. Head selection (grouped dev) ==")
    scores = {}
    for c in args.c_grid:
        clf = _fit_linear(x[tr], y[tr], c)
        scores[c] = clf.score(x[dv], y[dv])
        print(f"  C={c:<6g} dev={scores[c]:.4f}")
    best_c = max(scores, key=scores.get)
    print(f"  -> C={best_c}")
    report["head_search"] = {str(k): float(v) for k, v in scores.items()}

    # ------------------------------------------------------- temperature
    print("\n== 4. Calibration ==")
    clf_tr = _fit_linear(x[tr], y[tr], best_c)
    logits_dev = x[dv] @ clf_tr.coef_.T + clf_tr.intercept_
    temperature = fit_temperature(logits_dev, y[dv])

    def _probs(logits, t):
        z = logits / t
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True)

    logits_te = x[te] @ clf_tr.coef_.T + clf_tr.intercept_
    ece_before = expected_calibration_error(_probs(logits_te, 1.0), y[te])
    ece_after = expected_calibration_error(_probs(logits_te, temperature), y[te])
    print(f"  T={temperature:.3f}   ECE(test) {ece_before:.4f} -> {ece_after:.4f}")
    report["temperature"] = temperature
    report["ece_before"], report["ece_after"] = ece_before, ece_after

    # --------------------------------------------- final fit on train+dev
    print("\n== 5. Final fit (train + dev) ==")
    fit = tr | dv
    clf = _fit_linear(x[fit], y[fit], best_c)
    prototypes = build_prototypes(
        x[fit], y[fit], labels, per_class=args.protos_per_class, seed=args.seed
    )
    print(f"  head {clf.coef_.shape} = {clf.coef_.size + clf.intercept_.size} params")
    print(f"  prototypes {prototypes.shape}")

    head = SemanticHead(
        weights=clf.coef_.astype(np.float32),
        bias=clf.intercept_.astype(np.float32),
        labels=labels,
        temperature=temperature,
        prototypes=prototypes,
        ood_threshold=0.0,  # set below
        conf_threshold=0.0,
    )

    # ------------------------------------------------------- thresholds
    # Tuned against *realistic* utterances, not the template corpus. Template
    # phrases all score ~0.99, so a percentile on them sets a bar real speech
    # never clears — the earlier version rejected the very utterance this work
    # started from ("it's too loudy here can you make it quiter", conf 0.62).
    # These probes are an operating-point validation set, not a test set; retune
    # on logged production traffic once it exists.
    print("\n== 6. Gate thresholds ==")
    hard_texts = [t for t, _ in HARD_PARAPHRASES]
    hard_gold = np.array([label_to_idx[g] for _, g in HARD_PARAPHRASES])
    x_hard = encoder.encode(hard_texts)
    oos = [t for t in OUT_OF_SCOPE if t.strip()]
    x_oos = encoder.encode(oos)

    # Only the hard paraphrases stand in for real speech. Including the 3k
    # template test rows would swamp them — they all score ~0.9 and the
    # optimiser would happily discard the realistic tail to gain separation.
    #
    # The two gates do different jobs and are tuned differently:
    #   ood_threshold  — the primary (and only real) rejector
    #   conf_threshold — a permissive floor for genuine intent ambiguity
    #
    # Confidence is measurably useless for rejection: out-of-scope utterances
    # are frequently *confidently* wrong ("turn on the kitchen lights" -> 0.97,
    # "set a timer for ten minutes" -> 1.00), and top1-minus-top2 margin is no
    # better (OOS median margin 0.59 vs hard-set 5th percentile 0.28). So the
    # confidence gate is set below the weakest correct in-domain prediction
    # with headroom, and does no rejection work of its own.
    ood_in, ood_out = head.ood_score(x_hard), head.ood_score(x_oos)
    head.ood_threshold = choose_threshold(ood_in, ood_out, min_retention=0.90)
    conf_hard = head.probabilities(x_hard).max(1)
    head.conf_threshold = float(np.clip(conf_hard.min() * 0.85, 0.30, 0.45))
    print(f"  in-domain ood: min={ood_in.min():.2f} median={np.median(ood_in):.2f}")
    print(f"  out-of-scope ood: median={np.median(ood_out):.2f} max={ood_out.max():.2f}")
    if ood_out.max() > ood_in.min():
        print(
            "  note: distributions overlap — no threshold separates them perfectly;"
            "\n        retune on logged production traffic when available"
        )
    print(f"  ood_threshold={head.ood_threshold:.3f}  " f"conf_threshold={head.conf_threshold:.3f}")
    report["ood_threshold"] = head.ood_threshold
    report["conf_threshold"] = head.conf_threshold

    # ---------------------------------------------------------- evaluate
    print("\n== 7. Held-out evaluation ==")
    ood_test = head.ood_score(x[te])
    pred_te = head.probabilities(x[te]).argmax(1)
    acc = float((pred_te == y[te]).mean())
    print(f"  grouped test accuracy: {acc:.4f}  (n={int(te.sum())})")
    report["test_accuracy"] = acc

    pred_hard = head.probabilities(x_hard).argmax(1)
    hard_acc = float((pred_hard == hard_gold).mean())
    print(
        f"  hard paraphrases:      {hard_acc:.3f}  "
        f"({int((pred_hard == hard_gold).sum())}/{len(hard_gold)})"
    )
    report["hard_accuracy"] = hard_acc
    for (text, gold), p in zip(HARD_PARAPHRASES, pred_hard):
        if labels[p] != gold:
            print(f"    miss: {text[:46]:48s} gold={gold:24s} got={labels[p]}")

    ok_pairs = 0
    for a, ga, b, gb in ANTONYM_PAIRS:
        pa, pb = head.probabilities(encoder.encode([a, b])).argmax(1)
        good = labels[pa] == ga and labels[pb] == gb
        ok_pairs += good
        print(f"  antonym {'OK ' if good else 'FAIL'}: {a[:38]:40s} -> {labels[pa]}")
        print(f"          {'   ' if good else '    '} {b[:38]:40s} -> {labels[pb]}")
    report["antonym_pairs"] = f"{ok_pairs}/{len(ANTONYM_PAIRS)}"

    ood_oos = head.ood_score(x_oos)
    rejected = float((ood_oos < head.ood_threshold).mean())
    kept_in = float((ood_test >= head.ood_threshold).mean())
    print(
        f"\n  out-of-scope rejection: {rejected:.1%} rejected "
        f"while keeping {kept_in:.1%} of in-domain"
    )
    report["oos_rejection"] = rejected
    report["in_domain_retention"] = kept_in

    # ------------------------------------------------------------ export
    print("\n== 8. Export ==")
    head.save(args.head_out)
    onnx_path = export_mod.fuse(args.encoder, head, args.out)
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"  {onnx_path.name}: {size_mb:.1f} MB")
    report["onnx_mb"] = size_mb

    parity = export_mod.verify_parity(onnx_path, encoder, head, hard_texts[:12] + oos[:4])
    print(
        f"  parity vs python: max|dprob|={parity['max_abs_diff_probs']:.2e} "
        f"argmax agreement={parity['argmax_agreement']:.0%} "
        f"-> {'PASS' if parity['passed'] else 'FAIL'}"
    )
    report["parity"] = parity

    args.report.write_text(json.dumps(report, indent=2))
    print(f"\n  report -> {args.report}")


if __name__ == "__main__":
    main()
