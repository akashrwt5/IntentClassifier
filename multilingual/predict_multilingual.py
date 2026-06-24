#!/usr/bin/env python3
"""
Interactive inference CLI for the multilingual intent models (ONNX Runtime).

Pick any generated model — a single language or the combined multilingual one —
and classify text from a prompt or one-shot from the command line. Mirrors
scripts/predict.py / scripts/nlu_cli.py, but for multilingual/models/.

Usage:
    # interactive REPL against the combined model (default)
    python multilingual/predict_multilingual.py

    # interactive REPL against a specific language model
    python multilingual/predict_multilingual.py --model fr

    # one-shot classification
    python multilingual/predict_multilingual.py --model da --text "skru op for lyden"

    # show the top-5 intents instead of top-3
    python multilingual/predict_multilingual.py --topk 5

Notes:
  * Input is normalised with the SAME text_norm.normalize_text (lowercase +
    ASCII accent-folding) used in training — required for the ONNX model to
    match the trained model on accented text.
  * The ONNX TfidfVectorizer needs the en_US.UTF-8 locale. If you hit a locale
    error, run:  localedef -i en_US -f UTF-8 en_US.UTF-8
                 export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

sys.path.insert(0, str(BASE_DIR))
from text_norm import normalize_text   # noqa: E402  (same fold used in training)

# Mirrors scripts/predict.py — below this the result is treated as low-confidence.
CONF_THRESHOLD = 0.70


def available_models():
    if not MODELS_DIR.exists():
        return []
    return sorted(d.name for d in MODELS_DIR.iterdir()
                  if d.is_dir() and next(d.glob("*_intent_model.onnx"), None))


def load_model(name):
    model_dir = MODELS_DIR / name
    onnx = next(model_dir.glob("*_intent_model.onnx"), None)
    labels_path = next(model_dir.glob("*_intent_labels.json"), None)
    if not onnx or not labels_path:
        sys.exit(f"Model '{name}' not found under {model_dir}. "
                 f"Available: {available_models() or '(none — run train_multilingual.py --all)'}")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    try:
        session = ort.InferenceSession(str(onnx))
    except Exception as e:  # noqa: BLE001
        if "locale" in str(e).lower():
            sys.exit("ONNX failed to load due to a missing locale. Run:\n"
                     "  localedef -i en_US -f UTF-8 en_US.UTF-8\n"
                     "  export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8")
        raise
    return session, session.get_inputs()[0].name, labels


def predict(session, input_name, labels, text, topk):
    """Return (top_intent, confidence, [(intent, prob), ...top-k])."""
    inp = session.get_inputs()[0]
    t = normalize_text(text)
    X = (np.array([[t]], dtype=object) if len(inp.shape) == 2
         else np.array([t], dtype=object))
    outputs = session.run(None, {input_name: X})

    scores = None
    for o in outputs:
        arr = np.asarray(o)
        if arr.ndim >= 1 and arr.shape[-1] == len(labels):
            scores = arr[0] if arr.ndim == 2 else arr
            break
    if scores is None:                       # model returned the label string
        lbl = str(np.asarray(outputs[0]).ravel()[0])
        return lbl, 1.0, [(lbl, 1.0)]

    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores)[::-1]
    top = [(labels[int(i)], float(scores[int(i)])) for i in order[:topk]]
    return top[0][0], top[0][1], top


def render(top_intent, conf, ranked):
    flag = "✅" if conf >= CONF_THRESHOLD else "🤖"
    note = "" if conf >= CONF_THRESHOLD else "  (low confidence → GenAI fallback)"
    print(f"  {flag} {top_intent}  (confidence {conf:.2f}){note}")
    if len(ranked) > 1:
        alts = "   ".join(f"{lbl} {p:.2f}" for lbl, p in ranked[1:])
        print(f"     also: {alts}")


def main():
    parser = argparse.ArgumentParser(description="Run a multilingual intent model")
    parser.add_argument("--model", "-m", default="multilingual",
                        help="Model name (e.g. en, fr, de, da, multilingual). Default: multilingual.")
    parser.add_argument("--text", "-t", help="One-shot text to classify (skips the REPL).")
    parser.add_argument("--topk", "-k", type=int, default=3, help="How many intents to show (default 3).")
    args = parser.parse_args()

    session, input_name, labels = load_model(args.model)

    if args.text:
        render(*predict(session, input_name, labels, args.text, args.topk))
        return

    print(f"=== Multilingual Intent Classifier — model '{args.model}' ({len(labels)} intents) ===")
    print(f"    Available models: {', '.join(available_models())}")
    print("    Type 'exit' to quit.\n")
    while True:
        try:
            text = input("you ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() == "exit":
            break
        render(*predict(session, input_name, labels, text, args.topk))
        print()


if __name__ == "__main__":
    main()
