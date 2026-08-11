#!/usr/bin/env python3
"""
Evaluate the student WHERE IT ACTUALLY RUNS — as Stage 3, not standalone.

Production turn flow (.claude/memory/architecture.md):

    Stage 2  TF-IDF + LogReg  --confidence >= 0.7--> answer, DONE
                              --confidence <  0.7--> Stage 3 (semantic rescue)

So the student never sees the easy turns. Measuring it on 100% of inputs — which
is what evaluate.py does — answers a question production never asks.

This script uses the REAL Stage 2 artifact (models/intent_model.onnx), routes by
the REAL schema threshold (language_packs/en/nlu_schema.json), and reports:

  * how often Stage 2 hands over at all
  * Stage 2 accuracy on what it keeps
  * the student's accuracy ON THE HANDOVER SUBSET (the only number that matters)
  * end-to-end pipeline accuracy, with and without the student
  * OOD: how much Stage 2 already rejects, and what the student adds

Usage:
    python scripts/evaluate_cascade.py --tag unkaug_s1 --threshold 0.40
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

REPO = Path(__file__).resolve().parents[2]
STAGE2_ONNX = REPO / "models" / "intent_model.onnx"
SCHEMA = REPO / "language_packs" / "en" / "nlu_schema.json"
STAGE2_TRAIN = REPO / "datasets" / "04_GENERATED_MASTER_training_data.csv"


MIGRATION = REPO / "datasets" / "label_migration_map.json"


def _reverse_map() -> dict[str, str]:
    """Stage 2 emits the NEW ND-3 taxonomy (device.volume.increase); the student
    uses the OLD space (Cmd.VolumeIncrease). Map Stage 2 back."""
    raw = json.loads(MIGRATION.read_text(encoding="utf-8"))["map"]
    return {new: old for old, new in raw.items() if new}


WEIGHTS = REPO / "models" / "intent_classifier_weights.json"


def stage2_temperature_and_gate() -> tuple[float, float]:
    """Mirror packages/runtime/nlu_engine/classifier.py exactly."""
    meta = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    return float(meta.get("temperature", 1.0)), float(meta.get("conf_threshold", 0.7))


def run_stage2(texts, rev, temperature=1.0):
    """Stage 2 confidence, computed the way the runtime computes it.

    CAREFUL: the ONNX output named "probabilities" is NOT probabilities — it is
    the sklearn decision_function, i.e. raw scores (they sum to ~0 and range
    well outside [0,1]). classifier.py turns them into probabilities with
    softmax(scores / T). Taking .max() on the raw output makes every utterance
    look confident and drives the handover rate to 0%.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(STAGE2_ONNX), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    arr = np.array(texts, dtype=object).reshape(-1, 1)
    labels, scores = sess.run(None, {name: arr})
    z = (
        np.array(
            [[p[k] for k in sorted(p)] if isinstance(p, dict) else p for p in scores],
            dtype=np.float64,
        )
        / temperature
    )
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    probs = e / e.sum(axis=1, keepdims=True)
    mapped = [rev.get(str(l), str(l)) for l in labels]
    return mapped, probs.max(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threshold", type=float, required=True, help="student's own gate")
    ap.add_argument(
        "--stage2-threshold",
        type=float,
        default=None,
        help="default: confidence_threshold from nlu_schema.json",
    )
    args = ap.parse_args()

    import torch

    from scripts.train_en import build_student

    s2_T, s2_gate = stage2_temperature_and_gate()
    s2_thr = args.stage2_threshold if args.stage2_threshold is not None else s2_gate
    print(
        f"Stage 2      : softmax(logits / T={s2_T})  gate {s2_thr}   "
        f"(from intent_classifier_weights.json, same as classifier.py)"
    )
    print(f"Stage 3 gate : {args.threshold}")
    rev = _reverse_map()

    # ---- leak check: was Stage 2 trained on our eval rows? --------------
    s2_leak = {}
    if STAGE2_TRAIN.exists():
        import csv

        with open(STAGE2_TRAIN, encoding="utf-8-sig") as f:
            s2_keys = {token_key(r["text"]) for r in csv.DictReader(f)}
        for nm, p in (
            ("locked", config.LOCKED_TEST),
            ("stress", config.STRESS_TEST),
            ("ood", config.OOD_TEST),
        ):
            if p.exists():
                ev = {token_key(t) for t, _ in load_rows(p)}
                s2_leak[nm] = round(len(ev & s2_keys) / max(len(ev), 1), 4)
        print("\n! Stage 2 training overlap with each eval set:")
        for nm, v in s2_leak.items():
            flag = "  <- CONTAMINATED, cascade result not usable" if v > 0.20 else ""
            print(f"    {nm:<7} {v * 100:5.1f}%{flag}")

    # ---- student --------------------------------------------------------
    vocab, tok_mode = load_vocab(config.MODELS / f"vocab_{args.tag}.json")
    meta = json.loads(
        (config.REPORTS / f"train_{args.tag}_summary.json").read_text(encoding="utf-8")
    )
    max_len = meta.get("max_len", config.MAX_LEN)
    tok_mode = meta.get("tokenizer", tok_mode)
    label_list = json.loads((config.MODELS / f"labels_{args.tag}.json").read_text(encoding="utf-8"))
    model = build_student(len(vocab), len(label_list), dim=meta.get("embed_dim", config.EMBED_DIM))
    model.load_state_dict(torch.load(config.MODELS / f"student_{args.tag}.pt"))
    model.eval()

    def student(texts):
        X = np.array([encode(t, vocab, max_len, tok_mode)[0] for t in texts], dtype=np.int64)
        M = X != config.PAD_ID
        with torch.no_grad():
            p = torch.softmax(model(torch.tensor(X), torch.tensor(M)), -1).numpy()
        return [label_list[i] for i in p.argmax(1)], p.max(1)

    results = {
        "stage2_temperature": s2_T,
        "stage2_threshold": s2_thr,
        "stage3_threshold": args.threshold,
        "stage2_train_overlap": s2_leak,
    }

    # ---- in-scope sets --------------------------------------------------
    for name, path in (("locked", config.LOCKED_TEST), ("stress", config.STRESS_TEST)):
        if not path.exists():
            continue
        rows = [(t, g) for t, g in load_rows(path) if g in label_list]
        texts = [t for t, _ in rows]
        gold = [g for _, g in rows]

        s2_lab, s2_conf = run_stage2(texts, rev, s2_T)
        keep = s2_conf >= s2_thr
        hand = ~keep

        s2_kept_acc = (
            float(np.mean([s2_lab[i] == gold[i] for i in np.where(keep)[0]])) if keep.any() else 0.0
        )
        s2_all_acc = float(np.mean([a == b for a, b in zip(s2_lab, gold)]))
        # what Stage 2 would score on the handover subset if there were no Stage 3
        s2_hand_acc = (
            float(np.mean([s2_lab[i] == gold[i] for i in np.where(hand)[0]])) if hand.any() else 0.0
        )

        idx = np.where(hand)[0]
        st_lab, st_conf = student([texts[i] for i in idx]) if len(idx) else ([], np.array([]))
        st_ok = [
            (st_lab[j] == gold[i]) and (st_conf[j] >= args.threshold) for j, i in enumerate(idx)
        ]
        st_hand_acc = float(np.mean(st_ok)) if st_ok else 0.0

        combined = (s2_kept_acc * keep.sum() + st_hand_acc * hand.sum()) / len(rows)
        s2_only = s2_all_acc

        results[name] = {
            "rows": len(rows),
            "stage2_contaminated": s2_leak.get(name, 0.0) > 0.20,
            "handover_rate": round(float(hand.mean()), 4),
            "stage2_accuracy_all": round(s2_all_acc, 4),
            "stage2_accuracy_on_kept": round(s2_kept_acc, 4),
            "stage2_accuracy_on_handover": round(s2_hand_acc, 4),
            "student_accuracy_on_handover": round(st_hand_acc, 4),
            "pipeline_with_student": round(combined, 4),
            "pipeline_stage2_only": round(s2_only, 4),
            "delta": round(combined - s2_only, 4),
        }
        r = results[name]
        warn = "   [Stage 2 CONTAMINATED - ignore]" if r["stage2_contaminated"] else ""
        print(f"\n{name.upper()}  ({len(rows)} rows){warn}")
        print(
            f"  handover to Stage 3        {r['handover_rate'] * 100:5.1f}%  ({int(hand.sum())} rows)"
        )
        print(f"  Stage 2 acc on what it keeps {r['stage2_accuracy_on_kept']:.4f}")
        print(
            f"  Stage 2 acc on handover set  {r['stage2_accuracy_on_handover']:.4f}   <- what the student must beat"
        )
        print(f"  STUDENT acc on handover set  {r['student_accuracy_on_handover']:.4f}")
        print(f"  pipeline, Stage 2 only       {r['pipeline_stage2_only']:.4f}")
        print(
            f"  pipeline, with student       {r['pipeline_with_student']:.4f}   ({r['delta']:+.4f})"
        )

    # ---- OOD ------------------------------------------------------------
    if config.OOD_TEST.exists():
        texts = [t for t, _ in load_rows(config.OOD_TEST)]
        s2_lab, s2_conf = run_stage2(texts, rev, s2_T)
        keep = s2_conf >= s2_thr
        hand = ~keep
        # Stage 2 "rejects" if it hands over or already says fallback
        s2_rej = float(
            np.mean(
                [(not keep[i]) or (s2_lab[i] == config.FALLBACK_INTENT) for i in range(len(texts))]
            )
        )
        idx = np.where(hand)[0]
        st_lab, st_conf = student([texts[i] for i in idx]) if len(idx) else ([], np.array([]))
        st_rej = [
            (st_lab[j] == config.FALLBACK_INTENT) or (st_conf[j] < args.threshold)
            for j in range(len(idx))
        ]
        kept_bad = int(
            sum(1 for i in range(len(texts)) if keep[i] and s2_lab[i] != config.FALLBACK_INTENT)
        )
        pipeline_rej = (sum(st_rej) + (len(texts) - len(idx) - kept_bad)) / len(texts)

        results["ood"] = {
            "rows": len(texts),
            "handover_rate": round(float(hand.mean()), 4),
            "stage2_rejects": round(s2_rej, 4),
            "stage2_wrongly_accepts": kept_bad,
            "student_rejects_on_handover": round(float(np.mean(st_rej)) if st_rej else 0.0, 4),
            "pipeline_reject_rate": round(pipeline_rej, 4),
        }
        r = results["ood"]
        print(f"\nOOD  ({len(texts)} rows)")
        print(f"  handover to Stage 3        {r['handover_rate'] * 100:5.1f}%")
        print(f"  Stage 2 already rejects    {r['stage2_rejects']:.4f}")
        print(f"  Stage 2 WRONGLY accepts    {kept_bad} rows  <- student never sees these")
        print(f"  student rejects (handover) {r['student_rejects_on_handover']:.4f}")
        print(f"  PIPELINE reject rate       {r['pipeline_reject_rate']:.4f}")

    out = config.REPORTS / f"cascade_{args.tag}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
