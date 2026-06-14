#!/usr/bin/env python3
"""
P0-1 conformance test: ONNX Runtime (Python) vs. hand-rolled iOS scorer.

Both paths must agree on top-1 intent for every test utterance.
A probability divergence >0.01 on the top-1 class also fails.

Usage:
    python scripts/test_ios_conformance.py
    python scripts/test_ios_conformance.py --verbose

This is a CI gate — exit code 1 on any mismatch.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort

BASE_DIR = Path(__file__).parent.parent
WEIGHTS_PATH = BASE_DIR / "models" / "intent_classifier_weights.json"
MODEL_PATH   = BASE_DIR / "models" / "intent_model.onnx"
LABELS_PATH  = BASE_DIR / "models" / "intent_labels.pkl"

# Representative utterances — short commands, compound phrases, edge cases.
TEST_UTTERANCES = [
    "mute",
    "unmute",
    "turn it up",
    "turn it down",
    "increase volume",
    "decrease the volume",
    "change to restaurant",
    "switch to music",
    "send a message",
    "find my phone",
    "battery level",
    "start a run",
    "start walking",
    "how many steps",
    "show my calories",
    "set a reminder",
    "open translate",
    "start transcribe",
    "start streaming",
    "stop streaming",
    "check my heart rate",
    "open health",
    "help with volume",
    "help with pairing",
    "help with battery",
    "what is mask mode",
    "tell me about edge mode",
    "help with tinnitus",
    "show thrive score",
    "how do i insert my device",
]


# ---------------------------------------------------------------------------
# iOS hand-rolled scorer (mirrors export_ios_weights.py + iOS Swift logic)
# ---------------------------------------------------------------------------

def _ios_predict(text: str, weights: dict) -> tuple[str, float]:
    vocab: dict = weights["vocab"]
    idf: list = weights["idf"]
    coef: list = weights["coef"]
    intercept: list = weights["intercept"]
    labels: list = weights["labels"]

    t = text.lower().strip()
    tokens = t.split()

    # TF (raw counts) for unigrams and bigrams
    counts: dict[int, int] = {}
    for tok in tokens:
        idx = vocab.get(tok)
        if idx is not None:
            counts[idx] = counts.get(idx, 0) + 1
    for i in range(len(tokens) - 1):
        bg = tokens[i] + " " + tokens[i + 1]
        idx = vocab.get(bg)
        if idx is not None:
            counts[idx] = counts.get(idx, 0) + 1

    # TF-IDF with sublinear_tf (log(1+count) * idf)
    vec: dict[int, float] = {idx: math.log(1 + cnt) * idf[idx] for idx, cnt in counts.items()}

    # L2 normalise
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm > 0:
        vec = {idx: v / norm for idx, v in vec.items()}

    # Dot product + intercept per class
    scores = []
    for i, (row, b) in enumerate(zip(coef, intercept)):
        s = b + sum(vec.get(idx, 0.0) * w for idx, w in enumerate(row) if w != 0.0)
        scores.append(s)

    # Softmax
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    probs = [e / total for e in exps]

    top = int(np.argmax(probs))
    return labels[top], probs[top]


# ---------------------------------------------------------------------------
# ONNX Runtime scorer
# ---------------------------------------------------------------------------

def _onnx_predict(text: str, session: ort.InferenceSession, labels: list) -> tuple[str, float]:
    inp = session.get_inputs()[0]
    t = text.lower().strip()
    arr = (np.array([[t]], dtype=object)
           if len(inp.shape) == 2
           else np.array([t], dtype=object))
    outputs = session.run(None, {inp.name: arr})
    scores = None
    for o in outputs:
        if hasattr(o, "shape") and o.shape and o.shape[-1] == len(labels):
            scores = o[0]
            break
    if scores is None:
        scores = outputs[-1][0]
    scores = np.asarray(scores, dtype=float)
    top = int(np.argmax(scores))
    return labels[top], float(scores[top])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="Max allowed probability difference on top-1 class (default 0.01)")
    args = parser.parse_args()

    if not WEIGHTS_PATH.exists():
        print(f"ERROR: iOS weights not found at {WEIGHTS_PATH}")
        print("Run: python scripts/export_ios_weights.py")
        sys.exit(1)

    weights = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(MODEL_PATH))
    labels  = joblib.load(str(LABELS_PATH))

    failures = []
    for utt in TEST_UTTERANCES:
        ios_intent,  ios_prob  = _ios_predict(utt, weights)
        onnx_intent, onnx_prob = _onnx_predict(utt, session, labels)

        intent_match = ios_intent == onnx_intent
        prob_diff    = abs(ios_prob - onnx_prob)
        ok = intent_match and prob_diff <= args.tolerance

        if args.verbose or not ok:
            status = "✅" if ok else "❌"
            print(f"{status} {utt!r}")
            print(f"    ONNX: {onnx_intent} ({onnx_prob:.4f})")
            print(f"    iOS:  {ios_intent}  ({ios_prob:.4f})")
            if not intent_match:
                print("    ↑ INTENT MISMATCH")
            elif prob_diff > args.tolerance:
                print(f"    ↑ PROB DIFF {prob_diff:.4f} > tolerance {args.tolerance}")

        if not ok:
            failures.append(utt)

    print(f"\n{'='*50}")
    print(f"Conformance: {len(TEST_UTTERANCES) - len(failures)}/{len(TEST_UTTERANCES)} passed")
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f!r}")
        sys.exit(1)
    else:
        print("All utterances agree between ONNX Runtime and iOS hand-rolled scorer.")


if __name__ == "__main__":
    main()
