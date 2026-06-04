#!/usr/bin/env python3
"""
Intent classification inference using ONNX Runtime.

This is the reference implementation. Port this logic to Android/iOS
using the ONNX Runtime mobile SDK.

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

# ---------- Config ----------
CONF_THRESHOLD = 0.70        # minimum confidence to return an intent
CONF_GAP_THRESHOLD = 0.20    # minimum gap between top-1 and top-2
GENAI_BASE_URL = "https://genai.yourcompany.com/chat?query="

# ---------- Paths ----------
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "intent_model.onnx"
LABELS_PATH = BASE_DIR / "models" / "intent_labels.pkl"
UNKNOWN_PATH = BASE_DIR / "data" / "unknown_data.csv"

# ---------- Load model ----------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH.resolve()}")
if not LABELS_PATH.exists():
    raise FileNotFoundError(f"Labels not found: {LABELS_PATH.resolve()}")

session = ort.InferenceSession(str(MODEL_PATH))
inp = session.get_inputs()[0]
input_name = inp.name
LABELS = joblib.load(str(LABELS_PATH))

# Ensure unknown file has header
if not UNKNOWN_PATH.exists():
    with open(UNKNOWN_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["text", "confidence", "timestamp"])


def save_unknown(text, confidence):
    """Log low-confidence inputs for review and future training."""
    with open(UNKNOWN_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([text, f"{confidence:.6f}", datetime.now().isoformat()])


def _format_input(text: str):
    """Format text for ONNX model input."""
    t = text.lower().strip()
    rank = len(inp.shape)
    if rank == 2:
        return np.array([[t]], dtype=object)
    else:
        return np.array([t], dtype=object)


def _get_scores(ort_outputs):
    """Extract probability scores from ONNX output."""
    if len(ort_outputs) == 1:
        return ort_outputs[0]

    candidates = [
        o for o in ort_outputs
        if hasattr(o, "shape") and o.shape and o.shape[-1] == len(LABELS)
    ]
    return candidates[0] if candidates else ort_outputs[-1]


def predict(text: str) -> dict:
    """
    Classify text into an intent.

    Returns:
        dict with keys:
            type: "INTENT" or "GENAI"
            intent: intent name (only if type == "INTENT")
            confidence: model confidence score
            url: fallback URL (only if type == "GENAI")
    """
    X = _format_input(text)
    ort_outputs = session.run(None, {input_name: X})
    scores = _get_scores(ort_outputs)[0]

    # Handle zipmap output format
    if isinstance(scores, dict):
        scores = np.array([scores[label] for label in LABELS], dtype=float)
    else:
        scores = np.asarray(scores, dtype=float)

    top1 = int(np.argmax(scores))
    conf1 = float(scores[top1])

    # Calculate gap to second-best
    if scores.shape[0] > 1:
        sorted_idx = np.argsort(scores)
        top2 = int(sorted_idx[-2])
        conf2 = float(scores[top2])
        gap = conf1 - conf2
    else:
        gap = conf1

    intent = LABELS[top1]

    # Confident prediction
    if conf1 >= CONF_THRESHOLD and gap >= CONF_GAP_THRESHOLD:
        return {
            "type": "INTENT",
            "intent": intent,
            "confidence": conf1
        }

    # Not confident — log and fallback
    save_unknown(text, conf1)
    return {
        "type": "GENAI",
        "url": GENAI_BASE_URL + urllib.parse.quote(text),
        "confidence": conf1
    }


if __name__ == "__main__":
    print("=== Intent Classifier (type 'exit' to quit) ===")
    print(f"    Confidence threshold: {CONF_THRESHOLD}")
    print(f"    Known intents: {LABELS}\n")

    while True:
        text = input("Enter text: ").strip()
        if text.lower() == "exit":
            break
        if not text:
            print("⚠️  Empty input\n")
            continue

        r = predict(text)

        if r["type"] == "INTENT":
            print(f"  ✅ INTENT → {r['intent']}  (confidence: {r['confidence']:.2f})\n")
        else:
            print(f"  ❌ LOW CONFIDENCE  ({r['confidence']:.2f}) → GenAI fallback")
            print(f"  🔗 {r['url']}\n")
