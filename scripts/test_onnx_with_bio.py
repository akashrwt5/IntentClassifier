#!/usr/bin/env python3
"""
Full Joint Pipeline: Existing ONNX Intent Model + BIO Slot Tagger.

1. Uses `models/intent_model.onnx` via onnxruntime for Intent Classification.
2. Uses the BIO Tagger trained on `train.csv` for Title & Slot Extraction.

Usage:
    python scripts/test_onnx_with_bio.py
    python scripts/test_onnx_with_bio.py "bro set reminder to take medicine tomorrow 5 pm"
"""

import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "runtime"))

from scripts.bio_tagger_prototype import (
    build_training_data,
    train_bio_tagger,
    tokenize,
    predict_bio,
    extract_entities_from_bio,
)


def load_pipeline():
    print("⏳ [1/2] Loading existing ONNX intent model...")
    onnx_path = PROJECT_ROOT / "models" / "intent_model.onnx"
    session = ort.InferenceSession(str(onnx_path))

    print("⏳ [2/2] Training BIO slot tagger from train.csv...")
    corpus = build_training_data()
    vec, clf = train_bio_tagger(corpus)

    return session, vec, clf


def predict(session, vec, clf, phrase: str):
    # 1. Intent from ONNX
    raw_in = np.array([[phrase]], dtype=object)
    onnx_intent = session.run(["label"], {"input": raw_in})[0][0]

    # Map legacy label if needed
    if onnx_intent == "reminders.task.create":
        intent_display = "reminders.add (reminders.task.create)"
    else:
        intent_display = onnx_intent

    # 2. Slots from BIO Tagger
    tokens = tokenize(phrase)
    tags = predict_bio(tokens, vec, clf)
    entities = extract_entities_from_bio(tokens, tags)
    title = entities.get("TITLE")

    print("\n" + "=" * 65)
    print(f'📥 Input Phrase     : "{phrase}"')
    print("-" * 65)
    print(f"🎯 ONNX Intent       : {intent_display}")
    print(f"🏷️  Extracted Title   : {title or '(None)'}")
    print("-" * 65)
    print("🔬 Token Breakdown (BIO Tags):")
    for t, tag in zip(tokens, tags):
        tag_str = f"[{tag}]" if tag != "O" else "O"
        print(f"   {t:<15} -> {tag_str}")
    print("=" * 65)


def main():
    session, vec, clf = load_pipeline()

    args = sys.argv[1:]
    if args:
        predict(session, vec, clf, " ".join(args))
        return

    # Interactive loop
    print("\n" + "=" * 65)
    print("🚀 ONNX INTENT MODEL + BIO SLOT TAGGER (LIVE TESTER)")
    print("   Apna phrase likhein aur Enter dabayein. (Exit ke liye 'exit')")
    print("=" * 65)

    while True:
        try:
            phrase = input("\n👉 Enter phrase: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not phrase or phrase.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        predict(session, vec, clf, phrase)


if __name__ == "__main__":
    main()
