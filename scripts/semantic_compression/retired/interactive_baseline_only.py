#!/usr/bin/env python3
"""
Interactive Testing Script (Baseline Model Only) -- RETIRED
==========================================================
Loads the original 22.8 MB all-MiniLM-L6-v2 ONNX model and its classifier head.

RETIRED, and kept here for provenance rather than use.

It is the one script in this directory that runs a model living OUTSIDE the
directory (models/minilm-l6-v2.onnx). The active scripts dropped that
dependency so the directory can be lifted into a separate project, and the
plan replaces this reference with the real bge-small teacher baseline at P1.5.

It is therefore NOT covered by the artifact contract in artifact.py: that
contract requires an encoder to declare its own pooling, and a third-party
model is not obliged to carry our metadata. The mean pooling below is correct
for all-MiniLM-L6-v2 -- it is the pooling that model was trained with, per its
sentence-transformers 1_Pooling/config.json -- but it is asserted by this
comment rather than by the artifact, which is exactly why the active scripts
no longer work this way.

Do not extend this file. If the external reference is needed again, load it
through artifact.py with a declared pooling, or copy the model into this
directory with a pooling.json beside it.
"""

import sys
import time
import pickle
import numpy as np
from pathlib import Path

# retired/ is one level deeper than the rest of this directory
BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE_DIR))

from transformers import AutoTokenizer
from packages.runtime.nlu_engine.inference import OrtEmbedderBackend

# Paths
MODEL_PATH = BASE_DIR / "models" / "minilm-l6-v2.onnx"
CLF_PATH = BASE_DIR / "classifier_head.pkl"
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def embed_onnx(backend, text, tokenizer):
    encoded = tokenizer(
        text, max_length=64, truncation=True, padding="max_length", return_tensors="np"
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_type_ids = encoded.get("token_type_ids", np.zeros_like(input_ids))

    token_embeddings = backend.embed_tokens(input_ids, attention_mask, token_type_ids)
    mask = attention_mask[0]
    expanded = mask[:, np.newaxis].astype(np.float32)
    summed = (token_embeddings * expanded).sum(axis=0)
    vec = summed / expanded.sum()
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def main():
    print("Loading 22.8 MB Baseline Model... Please wait.")

    if not MODEL_PATH.exists():
        print(f"\n❌ Error: Baseline model not found at {MODEL_PATH}")
        sys.exit(1)
    if not CLF_PATH.exists():
        print(f"\n❌ Error: Classifier head not found at {CLF_PATH}")
        sys.exit(1)

    try:
        # Load ONNX Backend
        backend = OrtEmbedderBackend(MODEL_PATH)

        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

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
    print(" Voice Command Inference (Baseline 22.8MB Model)")
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
