#!/usr/bin/env python3
"""
Evaluate a trained student against every eval set + the ship bar.

Reports the four numbers that matter:
  1. locked test accuracy      — headline
  2. locked test MACRO recall  — exposes ignored small classes (accuracy hides them)
  3. stress test accuracy      — unseen phrasing
  4. OOD fallback rate         — must never regress

Usage:
    python scripts/evaluate.py --tag v1
    python scripts/evaluate.py --tag v1 --threshold 0.55
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
from scripts.common import encode, load_rows, load_vocab, token_key  # noqa: E402


def predict(model, texts, vocab, torch, mode="word", max_len=config.MAX_LEN):
    X = np.array([encode(t, vocab, max_len, mode)[0] for t in texts], dtype=np.int64)
    M = X != config.PAD_ID
    with torch.no_grad():
        logits = model(torch.tensor(X), torch.tensor(M))
        probs = torch.softmax(logits, dim=-1).numpy()
    return probs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument(
        "--threshold", type=float, default=0.0, help="below this confidence -> route to fallback"
    )
    ap.add_argument(
        "--sweep", action="store_true", help="sweep thresholds and print the operating curve"
    )
    args = ap.parse_args()

    import torch

    from scripts.train_en import build_student

    vocab, tok_mode = load_vocab(config.MODELS / f"vocab_{args.tag}.json")
    tr_path = config.REPORTS / f"train_{args.tag}_summary.json"
    max_len = config.MAX_LEN
    if tr_path.exists():
        meta = json.loads(tr_path.read_text(encoding="utf-8"))
        max_len = meta.get("max_len", max_len)
        tok_mode = meta.get("tokenizer", tok_mode)
    print(f"tokenizer: {tok_mode}  vocab {len(vocab)}  max_len {max_len}")
    label_list = json.loads((config.MODELS / f"labels_{args.tag}.json").read_text(encoding="utf-8"))
    model = build_student(len(vocab), len(label_list), dim=meta.get("embed_dim", config.EMBED_DIM))
    model.load_state_dict(torch.load(config.MODELS / f"student_{args.tag}.pt"))
    model.eval()

    fb_idx = label_list.index(config.FALLBACK_INTENT)
    results = {}

    # ------------------------------------------------ in-scope eval sets
    for name, path in (("locked", config.LOCKED_TEST), ("stress", config.STRESS_TEST)):
        if not path.exists():
            continue
        rows = load_rows(path)
        texts = [t for t, _ in rows]
        gold = [l for _, l in rows]
        known = [g for g in gold if g in label_list]
        if len(known) != len(gold):
            print(f"  ! {name}: {len(gold) - len(known)} rows have unknown labels")

        probs = predict(model, texts, vocab, torch, tok_mode, max_len)
        pred_idx = probs.argmax(1)
        conf = probs.max(1)
        pred = [label_list[i] for i in pred_idx]
        if args.threshold > 0:
            pred = [config.FALLBACK_INTENT if c < args.threshold else p for p, c in zip(pred, conf)]

        correct = [p == g for p, g in zip(pred, gold)]
        acc = float(np.mean(correct))

        # macro recall over the classes actually present
        per_class = defaultdict(list)
        for p, g in zip(pred, gold):
            per_class[g].append(p == g)
        macro = float(np.mean([np.mean(v) for v in per_class.values()]))

        worst = sorted(
            ((g, float(np.mean(v)), len(v)) for g, v in per_class.items()),
            key=lambda x: x[1],
        )[:8]

        results[name] = {
            "rows": len(rows),
            "accuracy": round(acc, 4),
            "macro_recall": round(macro, 4),
            "mean_confidence": round(float(conf.mean()), 4),
            "worst_classes": [{"intent": g, "recall": round(r, 3), "n": n} for g, r, n in worst],
        }
        print(f"\n{name.upper():<8} rows={len(rows)}")
        print(f"  accuracy      {acc:.4f}")
        print(f"  macro recall  {macro:.4f}   <- small classes ka sach")
        print("  worst classes:")
        for g, r, n in worst:
            print(f"     {g:<34} recall {r:.3f}  (n={n})")

    # ------------------------------------------------ OOD
    if config.OOD_TEST.exists():
        rows = load_rows(config.OOD_TEST)
        texts = [t for t, _ in rows]
        probs = predict(model, texts, vocab, torch, tok_mode, max_len)
        pred_idx = probs.argmax(1)
        conf = probs.max(1)
        is_fb = np.array([(i == fb_idx) or (c < args.threshold) for i, c in zip(pred_idx, conf)])
        rate = float(is_fb.mean())
        leaked = Counter(label_list[i] for i, f in zip(pred_idx, is_fb) if not f)
        results["ood"] = {
            "rows": len(rows),
            "fallback_rate": round(rate, 4),
            "escaped": int((~is_fb).sum()),
            "escaped_into": dict(leaked.most_common(10)),
        }
        print(f"\nOOD      rows={len(rows)}")
        print(f"  fallback rate {rate:.4f}   ({int((~is_fb).sum())} escaped)")
        for k, v in leaked.most_common(8):
            print(f"     escaped -> {k:<32} {v}")

    # ------------------------------------------------ threshold sweep
    if args.sweep and config.OOD_TEST.exists():
        lrows = load_rows(config.LOCKED_TEST)
        lrows = [(t, g) for t, g in lrows if g in label_list]
        lp = predict(model, [t for t, _ in lrows], vocab, torch, tok_mode, max_len)
        lpred = [label_list[i] for i in lp.argmax(1)]
        lconf = lp.max(1)
        lgold = [g for _, g in lrows]

        orows = load_rows(config.OOD_TEST)
        op = predict(model, [t for t, _ in orows], vocab, torch, tok_mode, max_len)
        opred_i = op.argmax(1)
        oconf = op.max(1)

        print("\nTHRESHOLD SWEEP")
        print(f"  {'thr':>5} {'OOD reject':>11} {'in-scope acc':>13} {'harmonic':>9}")
        sweep = []
        for th in [0.0, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
            rej = float(np.mean([(i == fb_idx) or (c < th) for i, c in zip(opred_i, oconf)]))
            acc = float(np.mean([(p == g) and (c >= th) for p, g, c in zip(lpred, lgold, lconf)]))
            h = 0.0 if (rej + acc) == 0 else 2 * rej * acc / (rej + acc)
            sweep.append(
                {
                    "threshold": th,
                    "ood_reject": round(rej, 4),
                    "in_scope_acc": round(acc, 4),
                    "harmonic": round(h, 4),
                }
            )
            print(f"  {th:>5.2f} {rej:>11.4f} {acc:>13.4f} {h:>9.4f}")
        best = max(sweep, key=lambda s: s["harmonic"])
        print(
            f"\n  best harmonic operating point: threshold={best['threshold']} "
            f"(OOD reject {best['ood_reject']}, in-scope {best['in_scope_acc']})"
        )
        results["threshold_sweep"] = sweep
        results["suggested_threshold"] = best["threshold"]

    out = config.REPORTS / f"eval_{args.tag}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    print("\n--- honest read ---")
    if "locked" in results:
        a, m = results["locked"]["accuracy"], results["locked"]["macro_recall"]
        print(f"  locked accuracy {a:.4f} | macro recall {m:.4f}")
        if a - m > 0.05:
            print("  ! accuracy >> macro recall: model bade classes pe achha,")
            print("    chhote classes pe kharaab. Accuracy alone mat report karna.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
