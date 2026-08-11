#!/usr/bin/env python3
"""
Train the same head on three sentence encoders, so they can be compared.

    e5-small   intfloat/e5-small-v2                    (the current teacher)
    minilm     sentence-transformers/all-MiniLM-L6-v2  (what Stage 3 ships today)
    bge        BAAI/bge-small-en-v1.5                  (strongest small symmetric)

ARCHITECTURE: frozen encoder + logistic-regression head. The encoder is never
fine-tuned — exactly the SetFit-style setup the original Stage 3 used. That
keeps the comparison about the ENCODER, which is the question being asked, and
it trains in minutes on CPU.

These are NOT the tiny distilled student. Each of these ships its full encoder
(90-130 MB), so they are a quality ceiling to measure against, not deployment
candidates. The script prints the size so that stays visible.

Requires: torch, sentence-transformers, scikit-learn (training-time only).
First run downloads the encoders (~350 MB total).

Usage:
    python scripts/train_backbones.py
    python scripts/train_backbones.py --models bge
    python scripts/train_backbones.py --models e5-small minilm bge
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.common import assert_no_leak, load_rows  # noqa: E402

BACKBONES = {
    "e5-small": "intfloat/e5-small-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge": "BAAI/bge-small-en-v1.5",
}

# e5 was trained with asymmetric prefixes. For single-utterance classification
# both sides are the same kind of text, so "query: " is the right one; omitting
# it entirely measures the model off-distribution and understates it.
PREFIX = {"e5-small": "query: ", "minilm": "", "bge": ""}

OUT_ROOT = config.MODELS.parent / "backbones"


def encoder_size_mb(model) -> float:
    n = sum(p.numel() for p in model.parameters())
    return n * 4 / 1e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(BACKBONES),
                    choices=list(BACKBONES))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-iter", type=int, default=2000)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    rows = load_rows(config.TRAIN_CSV)
    texts = [t for t, _ in rows]
    labels = [l for _, l in rows]

    for path, name in ((config.LOCKED_TEST, "locked"), (config.STRESS_TEST, "stress"),
                       (config.OOD_TEST, "OOD"), (config.OOV_TEST, "OOV")):
        if path.exists():
            assert_no_leak(texts, [t for t, _ in load_rows(path)], name)
    print(f"train rows : {len(rows)}   leak guard: OK")

    label_list = sorted(set(labels))
    y = np.array([label_list.index(l) for l in labels])
    # inverse-frequency: the corpus is ~55x imbalanced by design
    counts = np.bincount(y, minlength=len(label_list))
    cw = {i: len(y) / (len(label_list) * max(c, 1)) for i, c in enumerate(counts)}

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {}

    for key in args.models:
        repo_id = BACKBONES[key]
        pre = PREFIX[key]
        print(f"\n{'=' * 62}\n  {key}  ({repo_id})\n{'=' * 62}")

        t0 = time.time()
        enc = SentenceTransformer(repo_id)
        size = encoder_size_mb(enc)
        dim = enc.get_sentence_embedding_dimension()
        print(f"encoder    : {size:.1f} MB fp32, {dim}-d"
              + (f", prefix {pre!r}" if pre else ""))

        E = enc.encode([pre + t for t in texts], batch_size=args.batch,
                       normalize_embeddings=True, show_progress_bar=True)
        print(f"embedded   : {len(E)} rows in {time.time() - t0:.0f}s")

        head = LogisticRegression(max_iter=args.max_iter, class_weight=cw, n_jobs=-1)
        head.fit(E, y)

        out = OUT_ROOT / key
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "head.npz",
                            coef=head.coef_.astype(np.float32),
                            intercept=head.intercept_.astype(np.float32))
        (out / "labels.json").write_text(json.dumps(label_list, indent=2),
                                         encoding="utf-8")
        (out / "meta.json").write_text(json.dumps({
            "key": key, "repo_id": repo_id, "prefix": pre, "dim": dim,
            "encoder_mb_fp32": round(size, 1), "train_rows": len(rows),
            "intents": len(label_list), "class_weighted": True,
            "encoder_finetuned": False,
        }, indent=2), encoding="utf-8")

        # ---- score every eval set -------------------------------------
        scores = {}
        for name, path, is_ood in (("locked", config.LOCKED_TEST, False),
                                   ("stress", config.STRESS_TEST, False),
                                   ("oov", config.OOV_TEST, False),
                                   ("ood", config.OOD_TEST, True)):
            if not path.exists():
                continue
            er = load_rows(path)
            if not is_ood:
                er = [(t, g) for t, g in er if g in label_list]
            if not er:
                continue
            Ev = enc.encode([pre + t for t, _ in er], batch_size=args.batch,
                            normalize_embeddings=True)
            pred = [label_list[i] for i in head.predict(Ev)]
            if is_ood:
                scores[name] = float(np.mean(
                    [p == config.FALLBACK_INTENT for p in pred]))
            else:
                scores[name] = float(np.mean(
                    [p == g for p, (_, g) in zip(pred, er)]))
            print(f"  {name:<7} {scores[name]:.4f}")

        summary[key] = {"encoder_mb": round(size, 1), "dim": dim, **scores}
        (out / "scores.json").write_text(json.dumps(summary[key], indent=2),
                                         encoding="utf-8")
        del enc

    rep = config.REPORTS / "backbones.json"
    rep.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'backbone':<12}{'MB':>8}{'dim':>6}" +
          "".join(f"{k:>9}" for k in ("locked", "stress", "oov", "ood")))
    print("-" * 62)
    for k, v in summary.items():
        print(f"{k:<12}{v['encoder_mb']:>8.0f}{v['dim']:>6}" +
              "".join(f"{v.get(m, float('nan')):>9.4f}"
                      for m in ("locked", "stress", "oov", "ood")))
    print(f"\nwrote {rep}")
    print("\nEncoders are 90-130 MB and are NOT deployment candidates — they are")
    print("the ceiling the 2.5 MB student is being measured against.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
