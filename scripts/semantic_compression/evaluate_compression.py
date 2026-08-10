#!/usr/bin/env python3
"""
Comparative bake-off script to evaluate the compression models:
1. Baseline (all-MiniLM-L6-v2)
2. Track 1 (Pruned L3)
3. Track 3 (SVD L6)
"""

import sys
import time
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

# We will need the OrtEmbedderBackend from nlu_engine.inference
from packages.runtime.nlu_engine.inference import OrtEmbedderBackend

# Paths
MODELS = {
    "Baseline": BASE_DIR / "models" / "minilm-l6-v2.onnx",
    "Track 1 (Pruned L3)": Path(__file__).parent
    / "output_models"
    / "track1_pruned_l3"
    / "onnx_quantized"
    / "model_quantized.onnx",
    "Track 3 (SVD L6)": Path(__file__).parent
    / "output_models"
    / "track3_svd_l6"
    / "onnx_quantized"
    / "model_quantized.onnx",
}

TEST_PHRASES = [
    "turn on the lights",
    "turn up the lights",  # Identical twin
    "dim the audio",
    "increase the volume",
    "what is the weather today",  # Out of domain
    "who won the world cup",  # Out of domain
]

from transformers import AutoTokenizer


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


def evaluate_model(name, model_path):
    print(f"\n{'='*60}")
    print(f"Evaluating {name}")
    print(f"{'='*60}")

    if not model_path.exists():
        print(f"File not found: {model_path}")
        return

    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"Size: {size_mb:.2f} MB")

    tokenizer_path = model_path.parent.parent / "pytorch"
    if not tokenizer_path.exists():
        tokenizer_path = "sentence-transformers/all-MiniLM-L6-v2"

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"Vocab Size: {tokenizer.vocab_size} tokens")
    except Exception as e:
        print(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        return

    try:
        backend = OrtEmbedderBackend(model_path)

        # Warmup
        embed_onnx(backend, "warmup", tokenizer)

        # Measure latency & embedding drift
        print("\nEmbeddings:")
        for phrase in TEST_PHRASES:
            t0 = time.time()
            vec = embed_onnx(backend, phrase, tokenizer)
            t1 = time.time()
            latency = (t1 - t0) * 1000

            # Print first 3 dims to see if the semantic space is entirely broken
            sample_dims = ", ".join([f"{x:+.4f}" for x in vec[:3]])
            print(f"  [{latency:4.1f} ms] '{phrase[:25]:25s}' -> [{sample_dims} ...]")

    except Exception as e:
        print(f"Failed to run inference: {e}")


def main():
    for name, path in MODELS.items():
        evaluate_model(name, path)


if __name__ == "__main__":
    main()
