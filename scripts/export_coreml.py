#!/usr/bin/env python3
"""
Convert NLU model weights to CoreML (.mlpackage) for on-device iOS inference.

This script MUST be run on macOS with coremltools installed.

Usage:
    pip install coremltools onnxruntime numpy torch transformers
    python scripts/export_coreml.py              # full conversion (FP16 MiniLM)
    python scripts/export_coreml.py --quantize       # also emit INT8 MiniLM (~half size)
    python scripts/export_coreml.py --inspect-only   # inspect ONNX only, no conversion
    python scripts/export_coreml.py --skip-minilm    # skip the slow MiniLM step

After --quantize, ALWAYS measure the accuracy delta before shipping INT8:
    python scripts/compare_coreml_quant.py

Outputs (written to models/):
    IntentClassifier.mlpackage    Stage 2  TF-IDF LogReg (float-vector input)
    SemanticHead.mlpackage        Stage 3a linear head over MiniLM embeddings
    MiniLMEmbedder.mlpackage      Stage 3b MiniLM-L6-v2 encoder

── Stage 2 design note ──────────────────────────────────────────────────────
We build IntentClassifier.mlpackage from intent_classifier_weights.json, NOT
from intent_model.onnx. Two reasons:

  1. The sklearn pipeline uses CalibratedClassifierCV(method='isotonic').
     coremltools' sklearn converter does not support isotonic calibration,
     so ct.converters.sklearn.convert(pipeline) will fail or silently drop
     the calibration tables. (Confirmed: coremltools disables the sklearn
     conversion API entirely for scikit-learn > 1.5.1.)

  2. The ONNX export includes a string-input TF-IDF subgraph. coremltools
     cannot convert ONNX string tensors.

The intent_classifier_weights.json was already exported from the calibrated
model by export_ios_weights.py, so the weights ARE the calibrated values.
Building CoreML directly from JSON is deterministic and has zero conversion risk.

iOS side: Swift runs tfidfVector() as before, producing a float[n_features]
L2-normalised vector. That vector is the input to IntentClassifier.mlpackage.
The CoreML model runs the linear layer + softmax and returns classProbability.

── Stage 3b design note ──────────────────────────────────────────────────
coremltools 9 REMOVED the ONNX frontend — ct.convert() only accepts
tensorflow / pytorch / milinternal sources. Passing minilm-l6-v2.onnx fails
with "Unable to determine the type of the model".

So we convert from the original HuggingFace PyTorch model via torch.jit.trace.
The ONNX file was exported from this same network, so the traced model gives
the identical token-level last_hidden_state [1, seq, 384] output, and the
Swift mean-pool logic remains correct. The minilm-l6-v2.onnx file is still
used for introspection (confirming output shape) but not for conversion.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

MINILM_HF_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ─────────────────────────────────────────────────────────────────────────────
# Dependency guards
# ──────────────────────────────────────────────────────────────────────────────

def _require_coremltools():
    try:
        import coremltools as ct
        print(f"coremltools  {ct.__version__}")
        return ct
    except ImportError:
        print("ERROR: coremltools not installed.")
        print("  pip install coremltools")
        print("  (macOS only)")
        sys.exit(1)


def _require_onnxruntime():
    try:
        import onnxruntime as ort
        print(f"onnxruntime  {ort.__version__}")
        return ort
    except ImportError:
        print("WARN: onnxruntime not installed — skipping ONNX introspection.")
        print("  pip install onnxruntime")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# ONNX introspection  (runs first, before any conversion)
# ──────────────────────────────────────────────────────────────────────────────

def inspect_minilm_onnx(ort):
    """
    Print MiniLM ONNX input/output specs and run a test inference.
    The output shapes here are the ground truth for SemanticEmbedder.swift.
    """
    if ort is None:
        return None
    src = MODELS_DIR / "minilm-l6-v2.onnx"
    if not src.exists():
        print(f"  SKIP MiniLM introspection: {src} not found")
        print(f"  Run: python scripts/download_minilm.py")
        return None

    print(f"\n{'─'*60}")
    print(f"  MiniLM ONNX introspection")
    print(f"{'─'*60}")
    sess = ort.InferenceSession(str(src))

    print("  Declared inputs:")
    for inp in sess.get_inputs():
        print(f"    {inp.name:<22} shape={inp.shape}  type={inp.type}")

    print("  Declared outputs:")
    for out in sess.get_outputs():
        print(f"    {out.name:<22} shape={out.shape}  type={out.type}")

    ids  = np.array([[101, 7592, 1010, 2088, 102]], dtype=np.int64)
    mask = np.ones_like(ids)
    tids = np.zeros_like(ids)
    results = sess.run(None, {
        "input_ids":      ids,
        "attention_mask": mask,
        "token_type_ids": tids,
    })
    print("  Actual output shapes (seq_len=5 dummy run):")
    for info, val in zip(sess.get_outputs(), results):
        print(f"    {info.name:<22} shape={val.shape}  dtype={val.dtype}")

    primary_output = sess.get_outputs()[0].name
    primary_shape  = results[0].shape
    print()
    if len(primary_shape) == 3:
        print("  ✅ Output is token-level [batch, seq, dim].")
        print("     SemanticEmbedder.swift mean-pool logic is CORRECT.")
    elif len(primary_shape) == 2:
        print("  ⚠️  Output is already POOLED [batch, dim].")
        print("     SemanticEmbedder.swift must skip mean-pool.")
    else:
        print(f"  ❓ Unexpected output rank {len(primary_shape)} — inspect manually.")

    return {"name": primary_output, "shape": primary_shape}


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — IntentClassifier.mlpackage
# ──────────────────────────────────────────────────────────────────────────────

def export_intent_classifier(ct):
    """Build IntentClassifier.mlpackage from intent_classifier_weights.json."""
    src = MODELS_DIR / "intent_classifier_weights.json"
    dst = MODELS_DIR / "IntentClassifier.mlpackage"

    if not src.exists():
        print(f"\nSKIP Stage 2: {src} not found")
        print("  Run: python scripts/export_ios_weights.py")
        return

    print(f"\n{'─'*60}")
    print(f"  Stage 2: IntentClassifier")
    print(f"{'─'*60}")
    print(f"  Source : {src}")
    print(f"  Output : {dst}")

    data      = json.loads(src.read_text(encoding="utf-8"))
    labels    = [str(x) for x in data["labels"]]
    coef      = np.array(data["coef"],      dtype=np.float32)
    intercept = np.array(data["intercept"], dtype=np.float32)
    n_classes, n_features = coef.shape
    print(f"  Classes  : {n_classes}")
    print(f"  Features : {n_features}")

    from coremltools.models.neural_network import NeuralNetworkBuilder
    import coremltools.models.datatypes as dt

    builder = NeuralNetworkBuilder(
        input_features =[("tfidf_vector",     dt.Array(n_features))],
        output_features=[("classProbability", None)],
        mode="classifier",
    )
    builder.add_inner_product(
        name="logistic_regression",
        W=coef, b=intercept,
        input_channels=n_features, output_channels=n_classes,
        has_bias=True, input_name="tfidf_vector", output_name="logits",
    )
    builder.add_softmax(name="softmax", input_name="logits", output_name="classProbability")
    builder.set_class_labels(labels, predicted_feature_name="label", prediction_blob="classProbability")

    mlmodel = ct.models.MLModel(builder.spec)
    mlmodel.short_description = "TF-IDF + Logistic Regression intent classifier"
    mlmodel.input_description["tfidf_vector"] = (
        "L2-normalised TF-IDF feature vector — compute with Swift tfidfVector()"
    )
    mlmodel.output_description["classProbability"] = "Per-intent softmax probabilities"
    mlmodel.output_description["label"] = "Top-1 predicted intent label"
    mlmodel.save(str(dst))
    print(f"  Saved {dst}")
    _print_io(mlmodel)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3a — SemanticHead.mlpackage
# ──────────────────────────────────────────────────────────────────────────────

def export_semantic_head(ct):
    """Build SemanticHead.mlpackage from semantic_head.json."""
    src = MODELS_DIR / "semantic_head.json"
    dst = MODELS_DIR / "SemanticHead.mlpackage"

    if not src.exists():
        print(f"\nSKIP Stage 3a: {src} not found")
        print("  Run: python scripts/train_semantic_head.py")
        return

    print(f"\n{'─'*60}")
    print(f"  Stage 3a: SemanticHead")
    print(f"{'─'*60}")
    print(f"  Source : {src}")
    print(f"  Output : {dst}")

    head      = json.loads(src.read_text(encoding="utf-8"))
    weights   = np.array(head["weights"], dtype=np.float32)
    bias      = np.array(head["bias"],    dtype=np.float32)
    labels    = [str(x) for x in head["labels"]]
    n_classes, n_features = weights.shape
    print(f"  Classes  : {n_classes}")
    print(f"  Emb dim  : {n_features}")

    from coremltools.models.neural_network import NeuralNetworkBuilder
    import coremltools.models.datatypes as dt

    builder = NeuralNetworkBuilder(
        input_features =[("embedding",        dt.Array(n_features))],
        output_features=[("classProbability", None)],
        mode="classifier",
    )
    builder.add_inner_product(
        name="semantic_linear",
        W=weights, b=bias,
        input_channels=n_features, output_channels=n_classes,
        has_bias=True, input_name="embedding", output_name="logits",
    )
    builder.add_softmax(name="softmax", input_name="logits", output_name="classProbability")
    builder.set_class_labels(labels, predicted_feature_name="label", prediction_blob="classProbability")

    mlmodel = ct.models.MLModel(builder.spec)
    mlmodel.short_description = "MiniLM semantic classification head"
    mlmodel.input_description["embedding"] = (
        "L2-normalised 384-dim MiniLM-L6-v2 sentence embedding"
    )
    mlmodel.output_description["classProbability"] = "Per-intent softmax probabilities"
    mlmodel.output_description["label"] = "Top-1 predicted intent label"
    mlmodel.save(str(dst))
    print(f"  Saved {dst}")
    _print_io(mlmodel)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3b — MiniLMEmbedder.mlpackage  (from PyTorch via torch.jit.trace)
# ──────────────────────────────────────────────────────────────────────────────

def export_minilm(ct):
    """
    Convert MiniLM-L6-v2 → MiniLMEmbedder.mlpackage from the HuggingFace
    PyTorch model (coremltools 9 cannot convert ONNX directly).

    Output tensor is named 'last_hidden_state', shape [1, seq, 384],
    matching SemanticEmbedder.swift's featureValue(for: "last_hidden_state").
    FLOAT16 precision; RangeDim sequence axis 1..64 (matches Swift max_len=64).
    """
    dst = MODELS_DIR / "MiniLMEmbedder.mlpackage"

    print(f"\n{'─'*60}")
    print(f"  Stage 3b: MiniLMEmbedder  (PyTorch → CoreML)")
    print(f"{'─'*60}")
    print(f"  Output : {dst}")

    try:
        import torch
        from transformers import AutoModel
    except ImportError as e:
        print("  FAILED: torch + transformers required for MiniLM conversion.")
        print("    pip install transformers")
        print(f"    (import error: {e})")
        print("\n  Stage 3 is optional for the first ship — Stage 2 CoreML works without it.")
        return

    print(f"  Loading {MINILM_HF_NAME} (PyTorch, downloads ~90 MB on first run) ...")
    try:
        base = AutoModel.from_pretrained(MINILM_HF_NAME)
    except Exception as e:
        print(f"  FAILED to load model: {e}")
        print("    Check internet access (HuggingFace hub) or pre-cache the model.")
        return
    base.eval()

    class MiniLMWrapper(torch.nn.Module):
        """Returns only last_hidden_state so the traced graph has a tensor output."""
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask, token_type_ids):
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            return out.last_hidden_state

    wrapper = MiniLMWrapper(base).eval()

    # Trace with a representative fixed-length example (8 tokens).
    ids  = torch.ones((1, 8), dtype=torch.long)
    mask = torch.ones((1, 8), dtype=torch.long)
    tids = torch.zeros((1, 8), dtype=torch.long)
    print("  Tracing PyTorch model ...")
    try:
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, (ids, mask, tids), strict=False)
    except Exception as e:
        print(f"  FAILED to trace: {e}")
        return

    print("  Converting traced model to CoreML ... (30–90 s)")
    seq = ct.RangeDim(lower_bound=1, upper_bound=64, default=8)
    try:
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="input_ids",      shape=(1, seq), dtype=np.int32),
                ct.TensorType(name="attention_mask", shape=(1, seq), dtype=np.int32),
                ct.TensorType(name="token_type_ids", shape=(1, seq), dtype=np.int32),
            ],
            outputs=[ct.TensorType(name="last_hidden_state")],
            minimum_deployment_target=ct.target.iOS16,
            compute_precision=ct.precision.FLOAT16,
        )
        mlmodel.short_description = "MiniLM-L6-v2 sentence encoder (token-level output)"
        mlmodel.input_description["input_ids"]      = "BERT WordPiece token IDs"
        mlmodel.input_description["attention_mask"] = "1 for real tokens, 0 for padding"
        mlmodel.input_description["token_type_ids"] = "All zeros for single-sequence input"
        mlmodel.output_description["last_hidden_state"] = "Token embeddings [1, seq, 384] — mean-pool in Swift"
        mlmodel.save(str(dst))
        size_mb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
        print(f"  Saved {dst}  (on-disk {size_mb:.1f} MB)")
        _print_io(mlmodel)
    except Exception as exc:
        print(f"  FAILED to convert: {exc}")
        print()
        print("  Common causes and fixes:")
        print("  1. torch version incompatible with coremltools torch frontend")
        print("       coremltools 9 is tested against torch <= 2.7.")
        print("       Try: pip install 'torch==2.5.*' 'transformers' and re-run.")
        print("  2. transformers not installed")
        print("       pip install transformers")
        print()
        print("  Stage 3 is optional for the first ship — the iOS app still works")
        print("  via the pure-Swift SemanticEmbedder fallback or Stage 2 alone.")


# ──────────────────────────────────────────────────────────────────────────────
# INT8 weight quantization  (size optimization for MiniLMEmbedder)
# ──────────────────────────────────────────────────────────────────────────────

def quantize_minilm_int8(ct):
    """
    Post-training INT8 weight quantization of MiniLMEmbedder.mlpackage.

    Halves the on-disk size (~45 MB FP16 → ~22 MB INT8) by storing each weight
    as an 8-bit integer plus a per-channel scale. This is a SIZE optimization,
    not a speed one — on the Apple Neural Engine, FP16 is the native compute
    format, so INT8 weights are dequantized to FP16 before the matmul and
    latency is roughly unchanged.

    INT8 carries a small, real accuracy cost (typically 0.5–2% on a transformer
    like MiniLM). Do NOT ship the INT8 variant without running:
        python scripts/compare_coreml_quant.py
    which measures the holdout accuracy delta at the 0.55 rescue threshold.

    Output: models/MiniLMEmbedder_int8.mlpackage (the FP16 original is kept).
    """
    src = MODELS_DIR / "MiniLMEmbedder.mlpackage"
    dst = MODELS_DIR / "MiniLMEmbedder_int8.mlpackage"

    print(f"\n{'─'*60}")
    print(f"  INT8 quantization: MiniLMEmbedder")
    print(f"{'─'*60}")

    if not src.exists():
        print(f"  SKIP: {src} not found — run the MiniLM export first.")
        return

    try:
        import coremltools.optimize.coreml as cto
    except ImportError as e:
        print(f"  FAILED: coremltools.optimize.coreml unavailable ({e}).")
        print("    Requires coremltools >= 7. Try: pip install -U coremltools")
        return

    print(f"  Source : {src}")
    print(f"  Output : {dst}")
    try:
        mlmodel = ct.models.MLModel(str(src))
        op_cfg  = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        config  = cto.OptimizationConfig(global_config=op_cfg)
        print("  Quantizing weights to INT8 (linear_symmetric) ...")
        quantized = cto.linear_quantize_weights(mlmodel, config)
        quantized.short_description = "MiniLM-L6-v2 encoder (INT8-quantized weights)"
        quantized.save(str(dst))

        fp16_mb = sum(f.stat().st_size for f in src.rglob("*") if f.is_file()) / 1e6
        int8_mb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
        print(f"  Saved {dst}")
        print(f"  Size   : FP16 {fp16_mb:.1f} MB  →  INT8 {int8_mb:.1f} MB  "
              f"({100*(1-int8_mb/fp16_mb):.0f}% smaller)")
        print()
        print("  ⚠️  Measure accuracy before shipping INT8:")
        print("      python scripts/compare_coreml_quant.py")
    except Exception as exc:
        print(f"  FAILED to quantize: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_all(ct):
    print(f"\n{'─'*60}")
    print(f"  Validation")
    print(f"{'─'*60}")

    for name in ["IntentClassifier", "SemanticHead", "MiniLMEmbedder", "MiniLMEmbedder_int8"]:
        path = MODELS_DIR / f"{name}.mlpackage"
        if not path.exists():
            print(f"  {name}: not found (skipped)")
            continue
        try:
            m = ct.models.MLModel(str(path))
            print(f"  {name}: loaded")
            _print_io(m, indent=4)
        except Exception as e:
            print(f"  {name}: load error — {e}")

    ic  = MODELS_DIR / "IntentClassifier.mlpackage"
    wts = MODELS_DIR / "intent_classifier_weights.json"
    if ic.exists() and wts.exists():
        print()
        print("  IntentClassifier test inference (dummy one-hot vector):")
        try:
            data       = json.loads(wts.read_text())
            n_features = len(data["idf"])
            m          = ct.models.MLModel(str(ic))
            dummy      = np.zeros(n_features, dtype=np.float32)
            dummy[0]   = 1.0
            out        = m.predict({"tfidf_vector": dummy})
            probs      = out["classProbability"]
            top        = max(probs, key=probs.get)
            print(f"    Top prediction : '{top}' (dummy input — value is meaningless)")
            print(f"    Output keys    : {list(out.keys())}")
            print("    Inference OK")
        except Exception as e:
            print(f"    FAILED: {e}")

    # MiniLM end-to-end smoke test if present
    mm = MODELS_DIR / "MiniLMEmbedder.mlpackage"
    if mm.exists():
        print()
        print("  MiniLMEmbedder test inference (seq_len=8):")
        try:
            m   = ct.models.MLModel(str(mm))
            ids = np.ones((1, 8), dtype=np.int32)
            out = m.predict({
                "input_ids":      ids,
                "attention_mask": np.ones((1, 8), dtype=np.int32),
                "token_type_ids": np.zeros((1, 8), dtype=np.int32),
            })
            key   = next(iter(out))
            shape = np.array(out[key]).shape
            print(f"    Output '{key}' shape: {shape}")
            print("    Inference OK")
        except Exception as e:
            print(f"    FAILED: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _print_io(mlmodel, indent=2):
    pad  = " " * indent
    spec = mlmodel.get_spec()
    try:
        for inp in spec.description.input:
            print(f"{pad}input : {inp.name}")
        for out in spec.description.output:
            print(f"{pad}output: {out.name}")
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export NLU models to CoreML for iOS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--inspect-only", action="store_true",
                        help="Only print ONNX model specs, skip all conversions")
    parser.add_argument("--skip-minilm", action="store_true",
                        help="Skip MiniLM conversion (Stage 2 + 3a only)")
    parser.add_argument("--quantize", action="store_true",
                        help="Also emit MiniLMEmbedder_int8.mlpackage (INT8 weights, ~half size). "
                             "Measure accuracy with compare_coreml_quant.py before shipping.")
    args = parser.parse_args()

    print("=" * 60)
    print("  CoreML export — NLU intent pipeline")
    print("=" * 60)

    ct  = _require_coremltools()
    ort = _require_onnxruntime()

    inspect_minilm_onnx(ort)

    if args.inspect_only:
        print("\n--inspect-only: done.")
        return

    export_intent_classifier(ct)
    export_semantic_head(ct)

    if args.skip_minilm:
        print("\nSKIP MiniLM (--skip-minilm)")
    else:
        export_minilm(ct)
        if args.quantize:
            quantize_minilm_int8(ct)

    validate_all(ct)

    print(f"\n{'='*60}")
    print("  Copy these files to STT/STT/STT/Resources/ in the iOS repo:")
    print("    models/IntentClassifier.mlpackage     (required for Stage 2)")
    print("    models/SemanticHead.mlpackage          (required for Stage 3)")
    print("    models/MiniLMEmbedder.mlpackage        (required for Stage 3)")
    print("    models/intent_classifier_weights.json  (Swift fallback for Stage 2)")
    print("    models/semantic_head.json              (Swift fallback for Stage 3)")
    print("    models/minilm-vocab.txt                (Swift BERT tokeniser)")
    print(f"{'='*60}")
    print()
    print("  In Xcode: drag .mlpackage files into the STT target Resources group.")
    print("  Xcode compiles them to .mlmodelc at build time.")
    print("  At runtime, use withExtension: \"mlmodelc\" in Bundle.main.url().")
    print("  See docs/coreml-conversion-guide.md for full integration steps.")


if __name__ == "__main__":
    main()
