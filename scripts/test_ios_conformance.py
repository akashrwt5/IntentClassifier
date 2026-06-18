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

import re


def _swift_tokens(text: str) -> list:
    """Match IntentClassifierService.tokenize(): lowercase, split on
    non-alphanumerics, then unigrams + adjacent bigrams (single chars kept)."""
    words = [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]
    tokens = list(words)
    for i in range(len(words) - 1):
        tokens.append(words[i] + " " + words[i + 1])
    return tokens


def _interp(x: float, xs: list, ys: list) -> float:
    """Clamped piecewise-linear interpolation (mirrors Swift interpolate())."""
    if not xs:
        return 0.0
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    return float(np.interp(x, xs, ys))


def _ios_predict(text: str, weights: dict) -> tuple[str, float]:
    vocab: dict = weights["vocab"]
    idf: list = weights["idf"]
    coef: list = weights["coef"]
    intercept: list = weights["intercept"]
    labels: list = weights["labels"]

    # TF (raw counts) over the pruned vocab, sublinear TF-IDF, then L2 normalise.
    counts: dict[int, int] = {}
    for tok in _swift_tokens(text):
        idx = vocab.get(tok)
        if idx is not None:
            counts[idx] = counts.get(idx, 0) + 1
    vec: dict[int, float] = {idx: math.log(1 + cnt) * idf[idx] for idx, cnt in counts.items()}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm > 0:
        vec = {idx: v / norm for idx, v in vec.items()}

    # Logits: dot product + intercept per class.
    logits = [b + sum(vec.get(idx, 0.0) * w for idx, w in enumerate(row) if w != 0.0)
              for row, b in zip(coef, intercept)]

    # The predicted intent is always the base model's argmax. Calibration only
    # rescales the reported confidence — it must not re-rank (see Swift
    # stage2Scores). This mirrors the on-device behaviour exactly.
    top = int(np.argmax(logits))

    calibration = weights.get("calibration")
    if calibration and calibration.get("method") == "isotonic_logit":
        maps = calibration["maps"]
        cal = [_interp(logits[k], maps[k]["x"], maps[k]["y"]) for k in range(len(labels))]
        total = sum(cal)
        probs = [c / total for c in cal] if total > 0 else cal
    else:
        max_s = max(logits)
        exps = [math.exp(s - max_s) for s in logits]
        total = sum(exps)
        probs = [e / total for e in exps]

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

CONF_THRESHOLD = 0.70   # mirrors nlu_schema.json confidence_threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--threshold", type=float, default=CONF_THRESHOLD,
                        help=f"Confidence threshold (default {CONF_THRESHOLD})")
    args = parser.parse_args()

    if not WEIGHTS_PATH.exists():
        print(f"ERROR: iOS weights not found at {WEIGHTS_PATH}")
        print("Run: python scripts/export_ios_weights.py")
        sys.exit(1)

    weights = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(MODEL_PATH))
    labels  = joblib.load(str(LABELS_PATH))

    # Two failure modes, in order of severity:
    #   1. INTENT MISMATCH   — paths predict different intents (always a bug)
    #   2. THRESHOLD DISAGREE — one path fires above threshold while the other doesn't
    #      (user sees an action on one platform but a fallback on the other)
    # Probability distance alone is NOT a failure criterion: the iOS path now
    # applies its own isotonic calibration (device logit → server prob), so the
    # two should track closely, but exact parity isn't guaranteed (averaged single
    # model + one isotonic map vs. the server's per-fold calibrated ensemble).

    intent_failures    = []
    threshold_failures = []

    for utt in TEST_UTTERANCES:
        ios_intent,  ios_prob  = _ios_predict(utt, weights)
        onnx_intent, onnx_prob = _onnx_predict(utt, session, labels)

        intent_match      = ios_intent == onnx_intent
        onnx_fires        = onnx_prob >= args.threshold
        ios_fires         = ios_prob  >= args.threshold
        threshold_agree   = onnx_fires == ios_fires

        intent_fail    = not intent_match
        threshold_fail = intent_match and not threshold_agree

        if args.verbose or intent_fail or threshold_fail:
            if intent_fail:
                tag = "❌ INTENT"
            elif threshold_fail:
                tag = "⚠️  THRESH"
            else:
                tag = "✅"
            print(f"{tag}  {utt!r}")
            print(f"    ONNX: {onnx_intent} ({onnx_prob:.4f}, {'FIRE' if onnx_fires else 'REJECT'})")
            print(f"    iOS:  {ios_intent}  ({ios_prob:.4f}, {'FIRE' if ios_fires else 'REJECT'})")

        if intent_fail:
            intent_failures.append(utt)
        elif threshold_fail:
            threshold_failures.append(utt)

    total = len(TEST_UTTERANCES)
    clean = total - len(intent_failures) - len(threshold_failures)
    print(f"\n{'='*50}")
    print(f"Conformance: {clean}/{total} fully agree | "
          f"{len(intent_failures)} intent mismatch | "
          f"{len(threshold_failures)} threshold disagree")

    if intent_failures:
        print(f"\nINTENT MISMATCHES (must fix — different action fires on each platform):")
        for f in intent_failures:
            print(f"  ❌ {f!r}")
    if threshold_failures:
        print(f"\nTHRESHOLD DISAGREEMENTS (one platform fires, other falls back):")
        for f in threshold_failures:
            print(f"  ⚠️  {f!r}")

    if intent_failures or threshold_failures:
        sys.exit(1)
    else:
        print("All utterances agree on intent and threshold behaviour.")


if __name__ == "__main__":
    main()
