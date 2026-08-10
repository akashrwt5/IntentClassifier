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

from transformers import AutoTokenizer
from packages.runtime.nlu_engine.inference import OrtEmbedderBackend

# Paths
MODEL_PATH = (
    Path(__file__).parent
    / "output_models"
    / "stage2_contrastive_bge_small_onnx"
    / "model_quantized.onnx"
)
CLF_PATH = Path(__file__).parent / "output_models" / "classifier_head.pkl"
# The pruned tokenizer is inside the final ONNX folder!
TOKENIZER_PATH = Path(__file__).parent / "output_models" / "stage2_contrastive_bge_small_onnx"


def embed_onnx(backend, text, tokenizer):
    encoded = tokenizer(
        text, max_length=64, truncation=True, padding="max_length", return_tensors="np"
    )
    input_ids = encoded["input_ids"]

    # Safety clamp: if the tokenizer generates an out-of-bounds ID, clamp it to UNK (100)
    input_ids[input_ids >= 10000] = 100

    attention_mask = encoded["attention_mask"]
    token_type_ids = encoded.get("token_type_ids", np.zeros_like(input_ids))

    token_embeddings = backend.embed_tokens(input_ids, attention_mask, token_type_ids)

    vec = token_embeddings[0]

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def main():
    print("Loading 9 MB Distilled Model... Please wait.")

    if not MODEL_PATH.exists():
        print(f"\n❌ Error: Distilled model not found at {MODEL_PATH}")
        sys.exit(1)
    if not CLF_PATH.exists():
        print(f"\n❌ Error: Classifier head not found at {CLF_PATH}")
        sys.exit(1)

    try:
        # Load ONNX Backend
        backend = OrtEmbedderBackend(MODEL_PATH)

        # Load Tokenizer (use fallback to baseline if local pruned one fails to load)
        if TOKENIZER_PATH.exists():
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        else:
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        # Load Classifier Head
        with open(CLF_PATH, "rb") as f:
            classifier = pickle.load(f)

        # Warmup (forces ONNX Runtime to initialize memory)
        embed_onnx(backend, "warmup", tokenizer)

        print("✅ Ready!\n")
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
            vec = embed_onnx(backend, user_input, tokenizer)
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
