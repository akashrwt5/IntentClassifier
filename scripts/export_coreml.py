#!/usr/bin/env python3
"""
Convert NLU model weights to CoreML (.mlpackage) for on-device iOS inference.

This script MUST be run on macOS with coremltools installed.

Usage:
    pip install coremltools onnxruntime numpy
    python scripts/export_coreml.py              # full conversion
    python scripts/export_coreml.py --inspect-only   # inspect ONNX only, no conversion
    python scripts/export_coreml.py --skip-minilm    # skip the slow MiniLM step

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
     the calibration tables.

  2. The ONNX export includes a string-input TF-IDF subgraph. coremltools
     cannot convert ONNX string tensors — the export_intent_model() function
     in the original script even documented this as a known failure mode.

The intent_classifier_weights.json was already exported from the calibrated
model by export_ios_weights.py, so the weights ARE the calibrated values.
Building CoreML directly from JSON is deterministic and has zero conversion risk.

iOS side: Swift runs tfidfVector() as before, producing a float[n_features]
L2-normalised vector. That vector is the input to IntentClassifier.mlpackage.
The CoreML model runs the linear layer + softmax and returns classProbability.

── Stage 3 design note ──────────────────────────────────────────────────────
The MiniLM ONNX introspection runs BEFORE conversion and prints the exact
output tensor name and shape. This is critical: SemanticEmbedder.swift assumes
the output is 'last_hidden_state' shape [1, seq, 384] and mean-pools it.
If the actual output is already pooled ([1, 384]) or has a different name,
SemanticEmbedder.swift must be updated accordingly. Check the printed output.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"


# ─────────────────────────────────────────────────────────────────────────────
# Dependency guards
# ─────────────────────────────────────────────────────────────────────────────

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
        print("ERROR: onnxruntime not installed.")
        print("  pip install onnxruntime")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# ONNX introspection  (runs first, before any conversion)
# ─────────────────────────────────────────────────────────────────────────────

def inspect_minilm_onnx(ort):
    """
    Print MiniLM ONNX input/output specs and run a test inference.
    The output shapes here are the ground truth for SemanticEmbedder.swift.

    IMPORTANT: compare the printed output name and shape against what
    SemanticEmbedder.swift reads with:
        output.featureValue(for: "last_hidden_state")?.multiArrayValue
    If the name or rank differs, update the Swift file before shipping.
    """
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

    # Dummy inference with seq_len=5: [CLS] hello , world [SEP]
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
    if len(primary_shape) == 3:   # [1, seq, dim]
        print("  ✅ Output is token-level [batch, seq, dim].")
        print("     SemanticEmbedder.swift mean-pool logic is CORRECT.")
    elif len(primary_shape) == 2:  # [1, dim]
        print("  ⚠️  Output is already POOLED [batch, dim].")
        print("     SemanticEmbedder.swift must skip mean-pool.")
        print("     Update meanPoolAndNormalize() to just L2-normalise the [dim] vector.")
    else:
        print(f"  ❓ Unexpected output rank {len(primary_shape)} — inspect manually.")

    return {"name": primary_output, "shape": primary_shape}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — IntentClassifier.mlpackage
# ─────────────────────────────────────────────────────────────────────────────

def export_intent_classifier(ct):
    """
    Build IntentClassifier.mlpackage from intent_classifier_weights.json.

    CoreML model contract:
      Input:   tfidf_vector     float32[n_features]  L2-normalised TF-IDF vector
      Output:  classProbability dict<String, Double>  per-class softmax probability
               label            String                top-1 predicted class

    iOS integration:
      The Swift tfidfVector() function is unchanged — it still builds the
      L2-normalised feature vector. That vector is passed to this CoreML model
      instead of the manual logitScores() + softmax() functions.
    """
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
    labels    = data["labels"]
    coef      = np.array(data["coef"],      dtype=np.float32)   # (n_classes, n_features)
    intercept = np.array(data["intercept"], dtype=np.float32)   # (n_classes,)
    n_classes, n_features = coef.shape
    print(f"  Classes  : {n_classes}")
    print(f"  Features : {n_features}")

    from coremltools.models.neural_network import NeuralNetworkBuilder
    import coremltools.models.datatypes as dt

    builder = NeuralNetworkBuilder(
        input_features =[("tfidf_vector", dt.Array(n_features))],
        output_features=[("logits",       dt.Array(n_classes))],
        mode="classifier",
        class_labels=labels,
    )
    # Linear: logits = coef @ tfidf_vector + intercept
    builder.add_inner_product(
        name="logistic_regression",
        W=coef,
        b=intercept,
        input_channels=n_features,
        output_channels=n_classes,
        has_bias=True,
        input_name="tfidf_vector",
        output_name="logits",
    )
    # Softmax → classProbability (dict output added automatically by classifier mode)
    builder.add_softmax(
        name="softmax",
        input_name="logits",
        output_name="classProbability",
    )

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


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3a — SemanticHead.mlpackage
# ─────────────────────────────────────────────────────────────────────────────

def export_semantic_head(ct):
    """
    Build SemanticHead.mlpackage from semantic_head.json.

    CoreML model contract:
      Input:   embedding        float32[384]  L2-normalised MiniLM sentence vector
      Output:  classProbability dict<String, Double>
               label            String
    """
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
    weights   = np.array(head["weights"], dtype=np.float32)  # (n_classes, 384)
    bias      = np.array(head["bias"],    dtype=np.float32)  # (n_classes,)
    labels    = head["labels"]
    n_classes, n_features = weights.shape
    print(f"  Classes  : {n_classes}")
    print(f"  Emb dim  : {n_features}")

    from coremltools.models.neural_network import NeuralNetworkBuilder
    import coremltools.models.datatypes as dt

    builder = NeuralNetworkBuilder(
        input_features =[("embedding", dt.Array(n_features))],
        output_features=[("logits",    dt.Array(n_classes))],
        mode="classifier",
        class_labels=labels,
    )
    builder.add_inner_product(
        name="semantic_linear",
        W=weights,
        b=bias,
        input_channels=n_features,
        output_channels=n_classes,
        has_bias=True,
        input_name="embedding",
        output_name="logits",
    )
    builder.add_softmax(
        name="softmax",
        input_name="logits",
        output_name="classProbability",
    )

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


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3b — MiniLMEmbedder.mlpackage
# ─────────────────────────────────────────────────────────────────────────────

def export_minilm(ct):
    """
    Convert minilm-l6-v2.onnx → MiniLMEmbedder.mlpackage.

    Uses modern ct.convert() API (coremltools 7+).
    FLOAT16 precision: halves model size, preferred by ANE on A12+.
    RangeDim on the sequence axis: the model accepts any seq length 1..64,
    matching the max_len=64 limit in SemanticEmbedder.swift.

    ONNX declares int64 inputs; coremltools inserts int64→int32 cast nodes
    automatically when we specify dtype=np.int32 in TensorType.

    After conversion, inspect the output name in the printed I/O spec.
    SemanticEmbedder.swift reads:
        output.featureValue(for: "last_hidden_state")
    If the CoreML model uses a different output name, update that line.
    """
    src = MODELS_DIR / "minilm-l6-v2.onnx"
    dst = MODELS_DIR / "MiniLMEmbedder.mlpackage"

    if not src.exists():
        print(f"\nSKIP Stage 3b: {src} not found")
        print("  Run: python scripts/download_minilm.py")
        return

    print(f"\n{'─'*60}")
    print(f"  Stage 3b: MiniLMEmbedder")
    print(f"{'─'*60}")
    print(f"  Source : {src}  ({src.stat().st_size / 1e6:.1f} MB)")
    print(f"  Output : {dst}")
    print("  Converting ... (first run may take 60–120 s)")

    seq = ct.RangeDim(minimum_val=1, maximum_val=64)
    try:
        mlmodel = ct.convert(
            str(src),
            inputs=[
                ct.TensorType(name="input_ids",
                              shape=(1, seq),
                              dtype=np.int32),
                ct.TensorType(name="attention_mask",
                              shape=(1, seq),
                              dtype=np.int32),
                ct.TensorType(name="token_type_ids",
                              shape=(1, seq),
                              dtype=np.int32),
            ],
            minimum_deployment_target=ct.target.iOS16,
            compute_precision=ct.precision.FLOAT16,
        )
        mlmodel.short_description = "MiniLM-L6-v2 sentence encoder"
        mlmodel.input_description["input_ids"]      = "BERT WordPiece token IDs"
        mlmodel.input_description["attention_mask"] = "1 for real tokens, 0 for padding"
        mlmodel.input_description["token_type_ids"] = "All zeros for single-sequence input"
        mlmodel.save(str(dst))
        size_mb = sum(
            f.stat().st_size for f in dst.rglob("*") if f.is_file()
        ) / 1e6
        print(f"  Saved {dst}  (on-disk {size_mb:.1f} MB)")
        _print_io(mlmodel)

    except Exception as exc:
        print(f"  FAILED: {exc}")
        print()
        print("  Common causes and fixes:")
        print("  1. Unsupported ONNX op in INT8-quantized model")
        print("       Try the FP32 export: python scripts/download_minilm.py --fp32")
        print("       Then re-run this script.")
        print("  2. coremltools version too old")
        print("       pip install --upgrade coremltools")
        print("  3. ONNX input name mismatch")
        print("       Run --inspect-only first and compare input names above.")
        print()
        print("  If conversion keeps failing, the iOS app still works via")
        print("  the pure-Swift SemanticEmbedder fallback (no CoreML needed).")
        print("  Only Stage 2 (IntentClassifier.mlpackage) is strictly required")
        print("  for the CoreML upgrade — Stage 3 is optional for the first ship.")


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_all(ct):
    """
    Load all generated .mlpackage files, print their I/O specs, and run
    a trivial test inference on IntentClassifier to confirm it works end-to-end.
    """
    print(f"\n{'─'*60}")
    print(f"  Validation")
    print(f"{'─'*60}")

    for name in ["IntentClassifier", "SemanticHead", "MiniLMEmbedder"]:
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

    # IntentClassifier test inference
    ic  = MODELS_DIR / "IntentClassifier.mlpackage"
    wts = MODELS_DIR / "intent_classifier_weights.json"
    if ic.exists() and wts.exists():
        print()
        print("  IntentClassifier test inference (dummy zero vector):")
        try:
            data       = json.loads(wts.read_text())
            n_features = len(data["idf"])
            m          = ct.models.MLModel(str(ic))
            dummy      = np.zeros(n_features, dtype=np.float32)
            dummy[0]   = 1.0   # must be non-zero for softmax to pick a class
            out        = m.predict({"tfidf_vector": dummy})
            top        = max(out["classProbability"], key=out["classProbability"].get)
            print(f"    Top prediction : '{top}' (dummy input — value is meaningless)")
            print(f"    Output keys    : {list(out.keys())}")
            print("    Inference OK")
        except Exception as e:
            print(f"    FAILED: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export NLU models to CoreML for iOS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--inspect-only", action="store_true",
        help="Only print ONNX model specs, skip all conversions",
    )
    parser.add_argument(
        "--skip-minilm", action="store_true",
        help="Skip MiniLM conversion (Stage 2 + 3a only — fastest iteration)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  CoreML export — NLU intent pipeline")
    print("=" * 60)

    ct  = _require_coremltools()
    ort = _require_onnxruntime()

    # Always introspect first — confirms output names/shapes for Swift code
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
    print("  IMPORTANT — verify MiniLM output shape from introspection above:")
    print("    shape [1, seq, 384] → SemanticEmbedder.swift mean-pool is CORRECT")
    print("    shape [1, 384]      → remove mean-pool in SemanticEmbedder.swift")
    print()
    print("  In Xcode: drag .mlpackage files into the STT target Resources group.")
    print("  Xcode compiles them to .mlmodelc at build time.")
    print("  At runtime, use withExtension: \"mlmodelc\" in Bundle.main.url().")
    print("  See docs/coreml-conversion-guide.md for full integration steps.")


if __name__ == "__main__":
    main()
