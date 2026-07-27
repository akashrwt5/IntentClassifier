#!/usr/bin/env python3
"""
Export the multilingual TF-IDF + Logistic-Regression intent classifiers to
CoreML (.mlpackage) for on-device iOS inference.

This is a NEW, standalone exporter for the six multilingual models under
``multilingual/models/<m>/``. It deliberately does NOT touch the legacy
production exporter ``scripts/export_coreml.py`` — see
``multilingual/COREML_EXPORT_IMPLEMENTATION_PROMPT.md`` for the rationale.

Graph contract (mirrors the proven dense-matmul design):

    tfidf_vector  (1, V) float32   ──►  Linear(V → 59)  ──►  logits (1, 59) float32

The TF-IDF vectorisation (sublinear TF-IDF over the pruned vocab, then L2
normalisation) happens in Swift; CoreML receives the pre-computed, normalised
float vector and runs ONLY the matrix-vector product + bias. There is no sparse
op — CoreML cannot express sparse inputs — so the matmul is dense. For these
tiny (~0.1 %-dense) inputs that is hundreds-of-X redundant arithmetic, costing
microseconds / tens of KB, which is the accepted trade for reusing the
already-linked Core ML inference surface at zero added binary cost.

Device confidence contract: read ``logits``, compute ``softmax(logits / T)`` in
Swift, and gate on that. ``T`` (the per-model "temperature") is embedded in the
package's ``user_defined_metadata``. A package WITHOUT a temperature key means
T = 1.0 (plain softmax). This exporter does NOT emit a baked T=1
``classProbability`` output — that is a calibration trap; the only output is the
raw ``logits``.

Precision:
  --fp16  (default, REQUIRED precision)  FLOAT16 weights/compute. Measured
          decision-safe: 0/30 argmax flips, 0/30 0.70-gate flips on every model.
          Halves package size and matmul bandwidth at no decision cost.
  --fp32  (optional fallback artifact)   FLOAT32, written alongside as
          IntentClassifier_<m>_fp32.mlpackage.

Build path:
  Preferred — torch.nn.Linear → torch.jit.trace → ct.convert(convert_to=
  "mlprogram"). Produces a true ML-Program package with a fixed (1, V) input
  shape (no flexible/RangeDim shapes, which would disqualify the ANE).
  Fallback (only if torch is unavailable) — NeuralNetworkBuilder +
  quantization_utils.quantize_weights(nbits=16); this produces a
  neuralnetwork-format package, recorded as a limitation.

Usage:
    python multilingual/export_coreml_multilingual.py --model all --fp16
    python multilingual/export_coreml_multilingual.py --all --fp16 --fp32
    python multilingual/export_coreml_multilingual.py --model en
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent           # multilingual/
MODELS_DIR = BASE_DIR / "models"
MODEL_NAMES = ["en", "fr", "de", "da", "multilingual", "multilingual_small"]

SWIFT_CONTRACT_NOTE = (
    "Read the 'logits' output and compute softmax(logits / temperature) in Swift; "
    "gate on that probability. Do NOT add an external softmax that bakes T=1. A "
    "missing 'temperature' key means T=1.0 (plain softmax)."
)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _weights_path(model: str) -> Path:
    return MODELS_DIR / model / f"{model}_intent_classifier_weights.json"


def _load_weights(model: str) -> dict:
    p = _weights_path(model)
    if not p.exists():
        print(f"  ERROR: weights not found at {p}")
        print(f"  Train/export the '{model}' model first (multilingual/train_multilingual.py).")
        sys.exit(1)
    return json.loads(p.read_text(encoding="utf-8"))


def _metadata(weights: dict) -> dict:
    """The user_defined_metadata to embed in every package (string-valued)."""
    coef = weights["coef"]
    return {
        "temperature": repr(float(weights.get("temperature", 1.0))),
        "conf_threshold": repr(float(weights.get("conf_threshold", 0.7))),
        "conf_gap_threshold": repr(float(weights.get("conf_gap_threshold", 0.2))),
        "normalize": str(weights.get("normalize", "l2")),
        "sublinear_tf": str(bool(weights.get("sublinear_tf", True))),
        "ngram_range": json.dumps(list(weights.get("ngram_range", [1, 2]))),
        "vocab_size": str(len(coef[0])),
        "num_classes": str(len(coef)),
        "class_labels": json.dumps([str(x) for x in weights["labels"]]),
        "device_contract": SWIFT_CONTRACT_NOTE,
    }


def _apply_descriptions(mlmodel, n_features, n_classes, temperature):
    mlmodel.short_description = (
        "TF-IDF + Logistic Regression intent classifier (logits only; "
        f"calibrate Swift-side with softmax(logits / T={temperature:.6f}))"
    )
    try:
        mlmodel.input_description["tfidf_vector"] = (
            f"L2-normalised sublinear TF-IDF feature vector, length {n_features} — "
            "compute in Swift; CoreML runs the dense linear layer only."
        )
        mlmodel.output_description["logits"] = (
            f"Raw pre-softmax logits, length {n_classes}. Confidence = "
            "softmax(logits / temperature) computed in Swift."
        )
    except Exception:
        # output_description may reject names before the model is loadable on a
        # non-macOS runner; metadata still carries the contract, so this is safe.
        pass


# ---------------------------------------------------------------------------
# Preferred build path: torch -> mlprogram
# ---------------------------------------------------------------------------

def _build_mlprogram(ct, weights: dict, model: str, fp16: bool):
    import torch

    coef = np.array(weights["coef"], dtype=np.float32)        # (K, V)
    intercept = np.array(weights["intercept"], dtype=np.float32)  # (K,)
    n_classes, n_features = coef.shape
    temperature = float(weights.get("temperature", 1.0))

    linear = torch.nn.Linear(n_features, n_classes)
    linear.eval()
    with torch.no_grad():
        linear.weight.copy_(torch.from_numpy(coef))
        linear.bias.copy_(torch.from_numpy(intercept))

    example = torch.zeros(1, n_features, dtype=torch.float32)
    traced = torch.jit.trace(linear, example)

    precision = ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        inputs=[ct.TensorType(name="tfidf_vector",
                              shape=(1, n_features), dtype=np.float32)],
        outputs=[ct.TensorType(name="logits")],
        compute_precision=precision,
        minimum_deployment_target=ct.target.iOS16,
    )

    for k, v in _metadata(weights).items():
        mlmodel.user_defined_metadata[k] = v
    _apply_descriptions(mlmodel, n_features, n_classes, temperature)
    return mlmodel, "mlprogram"


# ---------------------------------------------------------------------------
# Fallback build path: NeuralNetworkBuilder + 16-bit quantization
# ---------------------------------------------------------------------------

def _build_neuralnetwork(ct, weights: dict, model: str, fp16: bool):
    from coremltools.models.neural_network import NeuralNetworkBuilder
    from coremltools.models.neural_network import quantization_utils
    import coremltools.models.datatypes as dt

    coef = np.array(weights["coef"], dtype=np.float32)
    intercept = np.array(weights["intercept"], dtype=np.float32)
    n_classes, n_features = coef.shape
    temperature = float(weights.get("temperature", 1.0))

    builder = NeuralNetworkBuilder(
        input_features=[("tfidf_vector", dt.Array(n_features))],
        output_features=[("logits", dt.Array(n_classes))],
    )
    builder.add_inner_product(
        name="logistic_regression",
        W=coef, b=intercept,
        input_channels=n_features, output_channels=n_classes,
        has_bias=True, input_name="tfidf_vector", output_name="logits",
    )
    mlmodel = ct.models.MLModel(builder.spec)
    if fp16:
        mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=16)

    for k, v in _metadata(weights).items():
        mlmodel.user_defined_metadata[k] = v
    _apply_descriptions(mlmodel, n_features, n_classes, temperature)
    return mlmodel, "neuralnetwork(fp16-quantized)"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _hash_package(pkg: Path) -> str:
    """Deterministic combined SHA-256 over all files in the .mlpackage dir."""
    h = hashlib.sha256()
    for f in sorted(pkg.rglob("*")):
        if f.is_file():
            h.update(f.relative_to(pkg).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def _refresh_manifest(model: str, packages: list):
    manifest_path = MODELS_DIR / model / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for pkg in packages:
        manifest[pkg.name] = _hash_package(pkg)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"  manifest: refreshed {manifest_path.name}")


# ---------------------------------------------------------------------------
# Per-model export
# ---------------------------------------------------------------------------

def _pkg_size_kb(pkg: Path) -> float:
    return sum(f.stat().st_size for f in pkg.rglob("*") if f.is_file()) / 1024.0


def export_model(ct, model: str, fp16: bool, fp32: bool, use_torch: bool):
    weights = _load_weights(model)
    coef = weights["coef"]
    n_classes, n_features = len(coef), len(coef[0])
    temperature = float(weights.get("temperature", 1.0))
    out_dir = MODELS_DIR / model

    print(f"\n{'-'*64}\n  {model}: K={n_classes}  V={n_features}  T={temperature:.6f}\n{'-'*64}")

    produced = []
    builder = _build_mlprogram if use_torch else _build_neuralnetwork

    targets = []
    if fp16:
        targets.append((True, out_dir / f"IntentClassifier_{model}.mlpackage"))
    if fp32:
        targets.append((False, out_dir / f"IntentClassifier_{model}_fp32.mlpackage"))
    if not targets:  # default to FP16 if neither flag given
        targets.append((True, out_dir / f"IntentClassifier_{model}.mlpackage"))

    for is_fp16, dst in targets:
        mlmodel, fmt = builder(ct, weights, model, is_fp16)
        if dst.exists():
            import shutil
            shutil.rmtree(dst)
        mlmodel.save(str(dst))
        produced.append(dst)
        print(f"  saved {dst.name}  [{fmt}, {'FP16' if is_fp16 else 'FP32'}]  "
              f"{_pkg_size_kb(dst):.1f} KB")

    _refresh_manifest(model, produced)
    return produced


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _require_coremltools():
    try:
        import coremltools as ct
        return ct
    except ImportError:
        print("ERROR: coremltools not installed.  pip install coremltools")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Export multilingual intent classifiers to CoreML (.mlpackage).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", "-m", default="all",
                        choices=["all"] + MODEL_NAMES,
                        help="Which model to export (default: all).")
    parser.add_argument("--all", action="store_true",
                        help="Alias for --model all.")
    parser.add_argument("--fp16", action="store_true",
                        help="Emit the FP16 package (default precision).")
    parser.add_argument("--fp32", action="store_true",
                        help="Also emit an optional FP32 fallback package.")
    args = parser.parse_args()

    ct = _require_coremltools()

    use_torch = True
    try:
        import torch  # noqa: F401
    except ImportError:
        use_torch = False
        print("WARN: torch unavailable — using NeuralNetworkBuilder + 16-bit "
              "quantization fallback (neuralnetwork format, not mlprogram).")

    fp16 = args.fp16 or not args.fp32   # FP16 is the default precision
    fp32 = args.fp32
    models = MODEL_NAMES if (args.all or args.model == "all") else [args.model]

    print("=" * 64)
    print(f"  CoreML export — multilingual intent classifiers")
    print(f"  precision: {'FP16 ' if fp16 else ''}{'FP32' if fp32 else ''}".rstrip())
    print(f"  build path: {'torch -> mlprogram' if use_torch else 'NeuralNetworkBuilder fallback'}")
    print("=" * 64)

    for model in models:
        export_model(ct, model, fp16, fp32, use_torch)

    print(f"\n{'='*64}\n  Done. Verify numeric equivalence with:")
    print("    python multilingual/test/test_coreml_multilingual.py")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
