#!/usr/bin/env python3
"""
Convert NLU ONNX models to Core ML (.mlpackage) for on-device iOS inference.

This script must be run on a Mac with coremltools installed.
The converted .mlpackage files should be checked into the repo (or a release
artifact) so iOS engineers can bundle them directly in Xcode.

Usage:
    pip install coremltools
    python scripts/export_coreml.py

Reads:   models/intent_model.onnx
Writes:  models/intent_model.mlpackage
         models/semantic_head.mlpackage   (from semantic_head.json)

The MiniLM embedding model (minilm-l6-v2.onnx) conversion is commented out
because that file is not checked in to this repo (228 MB). Uncomment and
provide the path if you have it available.

Notes:
  - intent_model.onnx: The TF-IDF vectoriser is baked in as a string-input
    ONNX graph. If coremltools cannot handle the string-input operator, fall
    back to the hand-rolled iOS scorer (intent_classifier_weights.json).
  - minilm-l6-v2.mlpackage: Runs on the Apple Neural Engine (A12+) at ~3-5 ms
    vs ~15 ms on CPU. Specify RangeDim for the sequence axis to support
    variable-length inputs.
  - semantic_head.json: A 59x384 linear layer — trivial to express as a
    Core ML NeuralNetwork model via the builder API.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"


def _check_coremltools():
    try:
        import coremltools as ct  # noqa: F401
        return True
    except ImportError:
        print("ERROR: coremltools is not installed.")
        print("Run: pip install coremltools")
        print("This script must be run on macOS.")
        return False


def export_intent_model():
    """Convert intent_model.onnx → intent_model.mlpackage."""
    import coremltools as ct

    src = MODELS_DIR / "intent_model.onnx"
    dst = MODELS_DIR / "intent_model.mlpackage"
    if not src.exists():
        print(f"SKIP: {src} not found (run python scripts/train.py first)")
        return

    print(f"Converting {src} → {dst} ...")
    try:
        model = ct.converters.onnx.convert(
            model=str(src),
            minimum_ios_deployment_target="16",
        )
        model.save(str(dst))
        print(f"✅ Saved {dst}")
    except Exception as e:
        print(f"⚠️  ONNX→CoreML failed for intent_model: {e}")
        print("   The string-input TF-IDF graph may not be supported by coremltools.")
        print("   Fall back: use intent_classifier_weights.json with the hand-rolled Swift scorer.")


def export_semantic_head():
    """Convert semantic_head.json (linear layer) → semantic_head.mlpackage."""
    import coremltools as ct
    import numpy as np
    from coremltools.models.neural_network import NeuralNetworkBuilder

    src = MODELS_DIR / "semantic_head.json"
    dst = MODELS_DIR / "semantic_head.mlpackage"
    if not src.exists():
        print(f"SKIP: {src} not found")
        return

    print(f"Converting {src} → {dst} ...")
    head = json.loads(src.read_text(encoding="utf-8"))
    weights = np.array(head["weights"], dtype=np.float32)   # (n_classes, 384)
    bias    = np.array(head["bias"],    dtype=np.float32)   # (n_classes,)
    labels  = head["labels"]
    n_classes, n_features = weights.shape

    builder = NeuralNetworkBuilder(
        input_features=[("embedding", ct.models.datatypes.Array(n_features))],
        output_features=[("logits",   ct.models.datatypes.Array(n_classes))],
    )
    builder.add_inner_product(
        name="linear",
        W=weights,
        b=bias,
        input_channels=n_features,
        output_channels=n_classes,
        has_bias=True,
        input_name="embedding",
        output_name="logits",
    )
    builder.add_softmax(name="softmax", input_name="logits", output_name="probabilities")

    mlmodel = ct.models.MLModel(builder.spec)
    mlmodel.save(str(dst))
    print(f"✅ Saved {dst} ({n_classes} classes, {n_features}-dim input)")


def export_minilm():
    """
    Convert minilm-l6-v2.onnx → minilm-l6-v2.mlpackage.
    Uncomment and adjust MINILM_PATH when you have the file available.
    """
    # import coremltools as ct
    # MINILM_PATH = MODELS_DIR / "minilm-l6-v2.onnx"
    # if not MINILM_PATH.exists():
    #     print(f"SKIP: {MINILM_PATH} not found")
    #     return
    # print(f"Converting {MINILM_PATH} → minilm-l6-v2.mlpackage ...")
    # mlmodel = ct.convert(
    #     str(MINILM_PATH),
    #     inputs=[
    #         ct.TensorType(name="input_ids",      shape=(1, ct.RangeDim(1, 64)), dtype=int),
    #         ct.TensorType(name="attention_mask", shape=(1, ct.RangeDim(1, 64)), dtype=int),
    #         ct.TensorType(name="token_type_ids", shape=(1, ct.RangeDim(1, 64)), dtype=int),
    #     ],
    #     minimum_deployment_target=ct.target.iOS16,
    # )
    # mlmodel.save(str(MODELS_DIR / "minilm-l6-v2.mlpackage"))
    # print(f"✅ Saved minilm-l6-v2.mlpackage")
    print("SKIP: MiniLM conversion not enabled (file not in repo). See docstring.")


if __name__ == "__main__":
    if not _check_coremltools():
        sys.exit(1)
    export_intent_model()
    export_semantic_head()
    export_minilm()
