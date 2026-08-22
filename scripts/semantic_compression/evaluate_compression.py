#!/usr/bin/env python3
"""
Comparative bake-off script to evaluate the compression models:
1. Baseline (all-MiniLM-L6-v2)
2. Track 1 (Pruned L3)
3. Track 3 (SVD L6)
"""

import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from artifact import ArtifactContractError, encode, load_encoder

# Paths
MODELS = {
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

def evaluate_model(name, model_path):
    print(f"\n{'='*60}")
    print(f"Evaluating {name}")
    print(f"{'='*60}")

    if not model_path.exists():
        print(f"File not found: {model_path}")
        return

    try:
        encoder = load_encoder(model_path)
    except ArtifactContractError as exc:
        print(f"Artifact contract not satisfied:\n  {exc}")
        return
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    print(f"  {encoder.summary()}")

    try:
        encode(encoder, ["warmup"])  # force ORT to allocate before timing

        print("\nEmbeddings:")
        for phrase in TEST_PHRASES:
            t0 = time.time()
            vec = encode(encoder, [phrase])[0]
            latency = (time.time() - t0) * 1000

            # First 3 dims are enough to see whether the space is entirely broken
            sample_dims = ", ".join([f"{x:+.4f}" for x in vec[:3]])
            print(f"  [{latency:4.1f} ms] '{phrase[:25]:25s}' -> [{sample_dims} ...]")

    except Exception as e:
        print(f"Failed to run inference: {e}")


def main():
    for name, path in MODELS.items():
        evaluate_model(name, path)


if __name__ == "__main__":
    main()
