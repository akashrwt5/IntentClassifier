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
    python scripts/export_coreml.py --seq-len 32     # fixed seq length (ANE-resident); 0 = legacy dynamic

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

The intent_classifier_weights.json holds the averaged base-LR weights plus a
per-class isotonic calibration table (device logit → server-calibrated prob),
both exported by export_ios_weights.py. The linear weights go into the CoreML
graph; the calibration is applied on-device in Swift.

iOS side: Swift runs tfidfVector() as before, producing a float[n_features]
L2-normalised vector. That vector is the input to IntentClassifier.mlpackage.
The CoreML model runs the linear layer + softmax and returns BOTH classProbability
and the raw "logits". Swift feeds the logits through the isotonic tables and
renormalises, matching the server's CalibratedClassifierCV(method='isotonic').

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

# Fixed sequence length for the exported CoreML encoder.
# Why fixed instead of a flexible RangeDim(1, 64): a dynamic sequence axis is
# typically evicted from the Apple Neural Engine onto GPU/CPU and makes CoreML
# allocate for the worst-case length — the likely root cause of the >100 MB iOS
# RAM use. A single fixed length stays ANE-resident and sizes memory exactly.
# 32 is empirically safe: across every file in data/ the max WordPiece length is
# 33 tokens (one garbled training sample, clipped by 1 token — immaterial), with
# p99 <= 20 and p95 <= 15. Override with --seq-len; pass --seq-len 0 to restore
# the legacy dynamic RangeDim(1, 64) shape.
DEFAULT_SEQ_LEN = 32


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

    # "logits" is exposed as a model output (not just an intermediate blob) so the
    # Swift side can apply the isotonic calibration tables shipped in
    # intent_classifier_weights.json and recalibrate to server-level confidence.
    # Older bundles without this output make Swift fall back to its own logit calc.
    builder = NeuralNetworkBuilder(
        input_features =[("tfidf_vector",     dt.Array(n_features))],
        output_features=[("classProbability", None), ("logits", dt.Array(n_classes))],
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
    mlmodel.output_description["logits"] = "Raw pre-softmax logits — input to Swift isotonic calibration"
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

def export_minilm(ct, seq_len=DEFAULT_SEQ_LEN):
    """
    Convert MiniLM-L6-v2 → MiniLMEmbedder.mlpackage from the HuggingFace
    PyTorch model (coremltools 9 cannot convert ONNX directly).

    Output tensor is named 'last_hidden_state', shape [1, seq, 384],
    matching SemanticEmbedder.swift's featureValue(for: "last_hidden_state").
    FLOAT16 precision.

    seq_len > 0  → fixed input shape (1, seq_len): stays resident on the Apple
                   Neural Engine and sizes memory exactly. Swift MUST pad every
                   input to this length (mask=0 on pads; the mean-pool already
                   ignores masked positions, so accuracy is unchanged).
    seq_len <= 0 → legacy dynamic RangeDim(1, 64) shape (ANE-hostile; kept only
                   as an escape hatch).
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
        # Force EAGER attention. SDPA routes masking through
        # scaled_dot_product_attention, whose mask handling coremltools converts
        # less predictably; the eager path uses the additive mask we patch below.
        try:
            base = AutoModel.from_pretrained(MINILM_HF_NAME, attn_implementation="eager")
        except TypeError:
            base = AutoModel.from_pretrained(MINILM_HF_NAME)  # older transformers
    except Exception as e:
        print(f"  FAILED to load model: {e}")
        print("    Check internet access (HuggingFace hub) or pre-cache the model.")
        return
    base.eval()

    # ── FP16-safe attention mask ──────────────────────────────────────────────
    # Recent transformers builds the additive attention mask as
    #     (1 - attention_mask) * torch.finfo(model.dtype).min
    # For an FP32 model that sentinel is -3.4e38. coremltools casts it to -inf
    # when converting to FLOAT16, and then at every UNMASKED position
    #     (1 - 1) * -inf  =  0 * -inf  =  NaN
    # which propagates through softmax and makes the ENTIRE output NaN (observed:
    # 12288/12288 NaN on any input). Older transformers used a finite -10000.0
    # sentinel and did not hit this. Override the method to use a finite,
    # FP16-representable value so the masked-attention math survives FP16
    # conversion. Masking is unchanged (exp(-1e4) ≈ 0) and the mean-pool already
    # skips pad positions, so embeddings — and accuracy — are identical.
    import types

    def _fp16_safe_extended_mask(self, attention_mask, input_shape, *args, **kwargs):
        ext = attention_mask[:, None, None, :].to(dtype=torch.float32)
        return (1.0 - ext) * -1e4

    base.get_extended_attention_mask = types.MethodType(_fp16_safe_extended_mask, base)

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

    if seq_len and seq_len > 0:
        shape = (1, seq_len)
        shape_desc = f"fixed (1, {seq_len}) — ANE-resident; Swift must pad to {seq_len}"
    else:
        # Legacy escape hatch: flexible shape. Typically evicted from the ANE.
        shape = (1, ct.RangeDim(lower_bound=1, upper_bound=64, default=8))
        shape_desc = "dynamic RangeDim(1, 64) — legacy, ANE-hostile"
    print("  Converting traced model to CoreML ... (30–90 s)")
    print(f"  Input shape: {shape_desc}")
    try:
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="input_ids",      shape=shape, dtype=np.int32),
                ct.TensorType(name="attention_mask", shape=shape, dtype=np.int32),
                ct.TensorType(name="token_type_ids", shape=shape, dtype=np.int32),
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
        if seq_len and seq_len > 0:
            print()
            print(f"  ⚠️  This model now requires inputs padded to EXACTLY {seq_len} tokens.")
            print(f"      Update SemanticEmbedder.swift to pad input_ids / attention_mask /")
            print(f"      token_type_ids to length {seq_len} (mask=0 on pad positions). The")
            print(f"      existing mean-pool already skips masked positions, so embeddings —")
            print(f"      and therefore accuracy — are unchanged. Do NOT ship this model until")
            print(f"      the Swift padding change is in, or on-device prediction will fail.")
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
    Post-training weight compression of MiniLMEmbedder.mlpackage via 6-bit
    palettization (~45 MB FP16 → ~22 MB compressed).

    WHY PALETTIZATION INSTEAD OF LINEAR INT8:
    linear_quantize_weights (INT8, linear_symmetric) generates QLinearMatMul
    ops in the CoreML MIL graph. The Metal MLIR backend's pass manager cannot
    compile these at inference time and crashes with:
        "Error: MLIR pass manager failed"
    Palettization stores each weight as a 6-bit lookup-table index referencing
    64 FP16 centroids per channel. The matmul ops remain standard FP16 — only
    the weight STORAGE changes. Metal compiles and runs this without issue.

    6-bit gives ~3× compression (vs 16-bit) with negligible accuracy loss.
    8-bit palettization gives ~2× compression but more accuracy headroom.

    Accuracy impact is small but real — measure before shipping:
        python scripts/compare_coreml_quant.py

    Output: models/MiniLMEmbedder_int8.mlpackage (the FP16 original is kept).
    The name stays *_int8 for consistency with the rest of the toolchain, even
    though the compression method is palettization rather than linear INT8.
    """
    src = MODELS_DIR / "MiniLMEmbedder.mlpackage"
    dst = MODELS_DIR / "MiniLMEmbedder_int8.mlpackage"

    print(f"\n{'─'*60}")
    print(f"  Weight compression (6-bit palettization): MiniLMEmbedder")
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
        # Load with cpuOnly — avoids Metal MLIR crash during the optimization pass.
        mlmodel = ct.models.MLModel(str(src), compute_units=ct.ComputeUnit.CPU_ONLY)

        # 6-bit palettization: 64 FP16 centroids per weight tensor.
        # Matmul ops remain FP16; only storage is compressed → Metal-compatible.
        op_cfg = cto.OpPalettizerConfig(mode="kmeans", nbits=6)
        config = cto.OptimizationConfig(global_config=op_cfg)
        print("  Palettizing weights to 6-bit (kmeans) ... this takes ~2–5 min")
        compressed = cto.palettize_weights(mlmodel, config)
        compressed.short_description = "MiniLM-L6-v2 encoder (6-bit palettized weights)"
        compressed.save(str(dst))

        fp16_mb = sum(f.stat().st_size for f in src.rglob("*") if f.is_file()) / 1e6
        cmp_mb  = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
        print(f"  Saved {dst}")
        print(f"  Size   : FP16 {fp16_mb:.1f} MB  →  6-bit {cmp_mb:.1f} MB  "
              f"({100*(1-cmp_mb/fp16_mb):.0f}% smaller)")
        print()
        print("  ⚠️  Measure accuracy before shipping:")
        print("      python scripts/compare_coreml_quant.py")
    except Exception as exc:
        print(f"  FAILED to compress: {exc}")
        import traceback; traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_all(ct, seq_len=DEFAULT_SEQ_LEN):
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

    # MiniLM end-to-end smoke test — run BOTH the FP16 base and (if present) the
    # palettized int8, and check the output is FINITE. A shape-only check misses
    # the failure mode that matters most: a model that loads and returns the
    # right shape but is full of NaN/Inf (FP16 overflow, bad palettization),
    # which then crashes head training with an opaque sklearn error downstream.
    test_len = seq_len if (seq_len and seq_len > 0) else 8
    for name in ["MiniLMEmbedder", "MiniLMEmbedder_int8"]:
        mm = MODELS_DIR / f"{name}.mlpackage"
        if not mm.exists():
            continue
        print()
        print(f"  {name} test inference (seq_len={test_len}):")
        try:
            m   = ct.models.MLModel(str(mm), compute_units=ct.ComputeUnit.CPU_ONLY)
            ids = np.ones((1, test_len), dtype=np.int32)
            out = m.predict({
                "input_ids":      ids,
                "attention_mask": np.ones((1, test_len), dtype=np.int32),
                "token_type_ids": np.zeros((1, test_len), dtype=np.int32),
            })
            key = next(iter(out))
            arr = np.array(out[key])
            print(f"    Output '{key}' shape: {arr.shape}")
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            if n_nan or n_inf:
                print(f"    ❌ NON-FINITE OUTPUT: {n_nan} NaN, {n_inf} Inf of {arr.size}.")
                print(f"       This model is broken and must NOT ship. Diagnose with:")
                print(f"         python scripts/diagnose_embedder_nan.py --model {mm}")
            else:
                print(f"    Inference OK — all {arr.size} values finite, "
                      f"range [{arr.min():.2f}, {arr.max():.2f}]")
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
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                        help=f"Fixed CoreML sequence length (default {DEFAULT_SEQ_LEN}). Keeps the "
                             f"encoder resident on the Apple Neural Engine; Swift must pad inputs to "
                             f"this length. Pass 0 to restore the legacy dynamic RangeDim(1, 64).")
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
        export_minilm(ct, args.seq_len)
        if args.quantize:
            quantize_minilm_int8(ct)

    validate_all(ct, args.seq_len)

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
