#!/usr/bin/env python3
"""
Interactive Testing Script (Distilled Model Only)
=================================================
Loads only the final 9 MB Distilled ONNX model and its classifier head.
Provides a clean, standalone interface to test latency and intent predictions.
"""

import sys
import time
import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from artifact import encode, head_path, load_encoder

# Paths
MODEL_PATH = (
    Path(__file__).parent
    / "output_models"
    / "stage2_contrastive_bge_small_onnx"
    / "model_quantized.onnx"
)
CLF_PATH = head_path(MODEL_PATH)


def main():
    print("Loading 9 MB Distilled Model... Please wait.")

    if not MODEL_PATH.exists():
        print(f"\n❌ Error: Distilled model not found at {MODEL_PATH}")
        sys.exit(1)
    if not CLF_PATH.exists():
        print(f"\n❌ Error: Classifier head not found at {CLF_PATH}")
        sys.exit(1)

    try:
        # Tokenizer, pooling and id-range come from the artifact's declaration.
        # The previous fallback to a hub tokenizer is gone: it would have
        # loaded a 30,522-token vocabulary for a 10,000-row model.
        encoder = load_encoder(MODEL_PATH)

        with open(CLF_PATH, "rb") as f:
            classifier = pickle.load(f)

        encode(encoder, ["warmup"])  # force ORT to allocate before timing

        print(f"✅ Ready!\n   {encoder.summary()}\n   head: {CLF_PATH.name}\n")
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        sys.exit(1)

    print("=" * 60)
    print(" Voice Command Inference (Distilled 9MB Model)")
    print(" Type any sentence to classify. Type 'exit' to stop.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nUser Command > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break

            # Measure Latency
            t0 = time.time()
            vec = encode(encoder, [user_input])[0]
            probs = classifier.predict_proba([vec])[0]
            best_idx = np.argmax(probs)
            intent = classifier.classes_[best_idx]
            confidence = probs[best_idx]
            t1 = time.time()

            latency_ms = (t1 - t0) * 1000
            vec_slice = ", ".join([f"{x:+.4f}" for x in vec[:4]])

            # Print Output
            print(f"  Intent     : {intent}")
            print(f"  Confidence : {confidence*100:.1f}%")
            print(f"  Latency    : {latency_ms:.2f} ms")
            print(f"  Vector     : [{vec_slice} ...]")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError processing input: {e}")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
