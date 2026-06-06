#!/usr/bin/env python3
"""
Intent classification inference using ONNX Runtime.

Usage:
    python scripts/predict.py
"""

import onnxruntime as ort
import numpy as np
import joblib
import urllib.parse
import csv
from datetime import datetime
from pathlib import Path

CONF_THRESHOLD = 0.70
CONF_GAP_THRESHOLD = 0.20
GENAI_BASE_URL = "https://genai.yourcompany.com/chat?query="

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "intent_model.onnx"
LABELS_PATH = BASE_DIR / "models" / "intent_labels.pkl"
UNKNOWN_PATH = BASE_DIR / "data" / "unknown_data.csv"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH.resolve()}")
if not LABELS_PATH.exists():
    raise FileNotFoundError(f"Labels not found: {LABELS_PATH.resolve()}")

session = ort.InferenceSession(str(MODEL_PATH))
inp = session.get_inputs()[0]
input_name = inp.name
LABELS = joblib.load(str(LABELS_PATH))

if not UNKNOWN_PATH.exists():
    with open(UNKNOWN_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["text", "confidence", "timestamp"])


def save_unknown(text, confidence):
    with open(UNKNOWN_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([text, f"{confidence:.6f}", datetime.now().isoformat()])


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
            return {"type": "GENAI", "intent": "GENAI",
                    "url": GENAI_BASE_URL + urllib.parse.quote(text), "confidence": conf1}
        return {"type": "INTENT", "intent": intent, "confidence": conf1}

    save_unknown(text, conf1)
    return {"type": "GENAI", "intent": "GENAI",
            "url": GENAI_BASE_URL + urllib.parse.quote(text), "confidence": conf1}


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
            print(f"  🤖 GENAI  ({r['confidence']:.2f}) → {r['url']}\n")
