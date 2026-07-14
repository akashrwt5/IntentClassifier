#!/usr/bin/env python3
"""
Intent classification inference using ONNX Runtime.

Usage:
    python scripts/predict.py
"""

import os
import sys
import urllib.parse
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unknown_log import record_unknown  # noqa: E402  (privacy stance: A#10/ND-5)

CONF_THRESHOLD = 0.70
CONF_GAP_THRESHOLD = 0.20
# No placeholder default: a GenAI URL is only emitted when explicitly
# configured (Review-F5 Appendix A #5, RK1 — same guard as scripts/nlu/engine.py).
GENAI_BASE_URL = os.environ.get("NLU_GENAI_URL") or None

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "intent_model.onnx"
LABELS_PATH = BASE_DIR / "models" / "intent_labels.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH.resolve()}")
if not LABELS_PATH.exists():
    raise FileNotFoundError(f"Labels not found: {LABELS_PATH.resolve()}")

session = ort.InferenceSession(str(MODEL_PATH))
inp = session.get_inputs()[0]
input_name = inp.name
LABELS = joblib.load(str(LABELS_PATH))


def save_unknown(text, confidence):
    # Privacy stance (docs/privacy-unknown-data.md): counters by default,
    # raw text only behind the NLU_COLLECT_RAW_UNKNOWN opt-in.
    record_unknown(text, confidence)


def _genai_result(text: str, confidence: float) -> dict:
    r = {"type": "GENAI", "intent": "GENAI", "confidence": confidence}
    if GENAI_BASE_URL:
        r["url"] = GENAI_BASE_URL + urllib.parse.quote(text)
    return r


def _format_input(text: str):
    t = text.lower().strip()
    return (np.array([[t]], dtype=object) if len(inp.shape) == 2
            else np.array([t], dtype=object))


def _get_scores(ort_outputs):
    if len(ort_outputs) == 1:
        return ort_outputs[0]
    candidates = [o for o in ort_outputs
                  if hasattr(o, "shape") and o.shape and o.shape[-1] == len(LABELS)]
    return candidates[0] if candidates else ort_outputs[-1]


def _keyword_match(text: str):
    t = text.lower().strip()
    if "translate" in t:                              return "TRANSLATE"
    if "transcribe" in t or "transcription" in t:    return "TRANSCRIBE"
    if "telehear" in t:                               return "TELEHEARAI"
    if "selfcheck" in t or "self check" in t:         return "SELFCHECK"
    if t in ("mute", "silence"):                      return "VOLUME_MUTE"
    if t == "unmute":                                  return "VOLUME_UNMUTE"
    return None


def predict(text: str) -> dict:
    keyword_intent = _keyword_match(text)
    if keyword_intent:
        return {"type": "INTENT", "intent": keyword_intent, "confidence": 1.0}

    X = _format_input(text)
    ort_outputs = session.run(None, {input_name: X})
    scores = _get_scores(ort_outputs)[0]

    if isinstance(scores, dict):
        scores = np.array([scores[label] for label in LABELS], dtype=float)
    else:
        scores = np.asarray(scores, dtype=float)

    top1 = int(np.argmax(scores))
    conf1 = float(scores[top1])

    if scores.shape[0] > 1:
        sorted_idx = np.argsort(scores)
        conf2 = float(scores[int(sorted_idx[-2])])
        gap = conf1 - conf2
    else:
        gap = conf1

    intent = LABELS[top1]

    if conf1 >= CONF_THRESHOLD and gap >= CONF_GAP_THRESHOLD:
        if intent == "OUT_OF_SCOPE":
            save_unknown(text, conf1)
            return _genai_result(text, conf1)
        return {"type": "INTENT", "intent": intent, "confidence": conf1}

    save_unknown(text, conf1)
    return _genai_result(text, conf1)


if __name__ == "__main__":
    print("=== Intent Classifier (type 'exit' to quit) ===")
    print(f"    Known intents: {LABELS}\n")
    while True:
        text = input("Enter text: ").strip()
        if text.lower() == "exit":
            break
        if not text:
            continue
        r = predict(text)
        if r["type"] == "INTENT":
            print(f"  ✅ INTENT → {r['intent']}  (confidence: {r['confidence']:.2f})\n")
        else:
            dest = r.get("url", "(no GenAI endpoint configured — set NLU_GENAI_URL)")
            print(f"  🤖 GENAI  ({r['confidence']:.2f}) → {dest}\n")
