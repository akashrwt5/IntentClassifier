#!/usr/bin/env python3
"""Export the intent classifier as a TFLite model — the linear HEAD only.

Why head-only (float-vector in -> logits out), and NOT a string-in graph
=======================================================================
The shipping ONNX graph is string-in and built from ONNX-ML ops
(``StringNormalizer`` / ``Tokenizer`` / ``TfIdfVectorizer`` / ``LinearClassifier``)
that have NO TFLite equivalents, so the whole pipeline cannot be represented in
TFLite. It does not need to be: the on-device contract already splits
vectorisation from classification. ``intent_classifier_weights.json`` ships the
TF-IDF ``vocab`` + ``idf`` and the runtime computes the L2-normalised TF-IDF
vector natively; the CoreML ``IntentClassifier.mlpackage`` is likewise a
"TF-IDF LogReg (float-vector input)" head. TFLite mirrors that exactly.

So this exporter builds a single Dense layer, ``logits = X @ W.T + b``, seeded
DIRECTLY from the fitted sklearn ``LogisticRegression`` (``coef_`` / ``intercept_``
in ``pipeline.pkl`` — the same trained object ``train.py`` exports to ONNX). The
map is linear, so fp32 TFLite is numerically identical to the ONNX
``LinearClassifier`` (which ``train.py`` emits with ``raw_scores=True``, i.e. raw
decision-function logits) up to float rounding — parity by construction, no
transcoding through ONNX or CoreML.

Output is LOGITS. The per-language temperature is applied at runtime as
``softmax(logits / T)``; it is not baked into the graph (mirrors ONNX/CoreML).

Artifacts (per language, beside model.onnx):
    models/intent/<lang>/model.tflite        fp32 (exact)
    models/intent/<lang>/model_int8.tflite   dynamic-range int8 weights (~4x smaller)

Usage:
    PYTHONPATH=packages/buildtime python -m nlu_export.export_tflite --lang en
    PYTHONPATH=packages/buildtime python -m nlu_export.export_tflite --all --no-int8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[3]
MODELS_DIR = BASE_DIR / "models"

# fp32 TFLite must reproduce the sklearn decision function to float precision.
# This is a linear map, so the only error is float32 rounding; anything above
# this means the graph is not the head we think it is.
FP32_LOGIT_TOL = 1e-4


def _rel(p: Path) -> Path:
    """Display path relative to the repo root when possible (never crash on it)."""
    try:
        return p.relative_to(BASE_DIR)
    except ValueError:
        return p


def _require_tf():
    # Quiet TF's C++ logging BEFORE import, and fail with a clear install hint
    # rather than a bare ModuleNotFoundError three frames deep.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    try:
        import tensorflow as tf  # noqa: F401
    except ModuleNotFoundError:
        sys.exit(
            "tensorflow is required for TFLite export and is an EXPORT-ONLY dep "
            "(like coremltools/torch), not on the inference path. Install it:\n"
            "    pip install tensorflow"
        )
    return tf


def _load_head(lang: str):
    """Return (W, b, n_features, n_classes) from the fitted sklearn pipeline."""
    pkl = MODELS_DIR / "intent" / lang / "pipeline.pkl"
    if not pkl.exists():
        sys.exit(
            f"no fitted pipeline at {_rel(pkl)} — train first "
            f"(python -m nlu_training.train --lang {lang})"
        )
    pipe = joblib.load(str(pkl))
    clf = pipe.named_steps.get("clf") or pipe.named_steps.get("classifier")
    if clf is None or not hasattr(clf, "coef_"):
        sys.exit(f"{pkl} has no linear classifier with coef_ — cannot export a head")
    W = clf.coef_.astype(np.float32)          # (n_classes, n_features)
    b = clf.intercept_.astype(np.float32)     # (n_classes,)
    return W, b, W.shape[1], W.shape[0], pipe


def _build_keras(tf, W: np.ndarray, b: np.ndarray):
    """A single Dense layer: logits = X @ W.T + b. Static batch size 1."""
    n_classes, n_features = W.shape
    inp = tf.keras.Input(shape=(n_features,), batch_size=1, dtype=tf.float32, name="tfidf")
    dense = tf.keras.layers.Dense(n_classes, activation=None, name="logits")
    model = tf.keras.Model(inp, dense(inp))
    dense.set_weights([W.T, b])               # kernel (n_features, n_classes), bias (n_classes,)
    return model


def _run_tflite(tf, blob: bytes, X: np.ndarray) -> np.ndarray:
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    i = it.get_input_details()[0]
    o = it.get_output_details()[0]
    out = []
    for row in X:
        it.set_tensor(i["index"], row[None, :].astype(np.float32))
        it.invoke()
        out.append(it.get_tensor(o["index"])[0].copy())
    return np.asarray(out)


def _fit_int8_temperature(tf, blob: bytes, pipe, lang: str):
    import json
    import pandas as pd
    from scipy.optimize import minimize_scalar

    data_path = BASE_DIR / "language_packs" / lang / "train.csv"
    if not data_path.exists():
        print("  [WARN] train.csv not found — cannot fit int8 temperature.")
        return None

    print("  fitting int8 temperature (this may take ~10 seconds)...")
    data = pd.read_csv(data_path, encoding="utf-8-sig", header=0)
    data.columns = [c.strip().lower() for c in data.columns]
    data = data.dropna(subset=["text", "intent"])

    # Downsample for speed: max 50 per intent is plenty for a scalar temperature
    data = data.groupby("intent").head(50)
    texts = data["text"].astype(str).tolist()
    intents = data["intent"].astype(str).str.strip().tolist()

    clf = pipe.named_steps.get("clf") or pipe.named_steps.get("classifier")
    lbl_idx = {lbl: i for i, lbl in enumerate(clf.classes_)}
    keep = [i for i, c in enumerate(intents) if c in lbl_idx]
    y_idx = np.array([lbl_idx[intents[i]] for i in keep])
    texts_keep = [texts[i] for i in keep]

    tfidf = pipe.named_steps.get("tfidf") or pipe.named_steps.get("vectorizer")
    X = tfidf.transform(texts_keep).toarray().astype(np.float32)
    logits = _run_tflite(tf, blob, X)

    def _nll(T: float) -> float:
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z / T)
        p = e / e.sum(axis=1, keepdims=True)
        p_true = p[np.arange(len(y_idx)), y_idx]
        return float(-np.log(np.clip(p_true, 1e-12, 1.0)).mean())

    res = minimize_scalar(_nll, bounds=(0.05, 10.0), method="bounded")
    T_int8 = float(res.x)

    # Update calibration.json in place
    calib_file = MODELS_DIR / "intent" / lang / "calibration.json"
    if calib_file.exists():
        payload = json.loads(calib_file.read_text(encoding="utf-8"))
        payload["temperature_int8"] = round(T_int8, 6)
        calib_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"  updated calibration.json with temperature_int8 = {T_int8:.4f}")

    return T_int8


def export(lang: str, *, fp32: bool = True, int8: bool = True) -> int:
    tf = _require_tf()
    W, b, n_features, n_classes, pipe = _load_head(lang)
    out_dir = MODELS_DIR / "intent" / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _build_keras(tf, W, b)

    # A handful of real TF-IDF vectors + one all-zeros row, as a self-check that
    # the exported graph equals sklearn's decision_function.
    tfidf = pipe.named_steps.get("tfidf") or pipe.named_steps.get("vectorizer")
    probe_texts = [
        "what is my battery", "turn up the volume", "start streaming",
        "how do i pair my hearing aids", "asdfqwer zzz",
    ]
    X = tfidf.transform(probe_texts).toarray().astype(np.float32)
    ref = pipe.named_steps.get("clf").decision_function(X).astype(np.float32)

    print(f"language     : {lang}")
    print(f"features     : {n_features}   classes: {n_classes}")

    if fp32:
        blob = tf.lite.TFLiteConverter.from_keras_model(model).convert()
        dst = out_dir / "model.tflite"
        dst.write_bytes(blob)
        diff = float(np.abs(_run_tflite(tf, blob, X) - ref).max())
        status = "OK" if diff <= FP32_LOGIT_TOL else "FAIL"
        print(f"fp32         : {_rel(dst)}  ({len(blob):,} B)  "
              f"max|Δlogit|={diff:.2e} [{status}]")
        if diff > FP32_LOGIT_TOL:
            return 1  # a linear head that doesn't match sklearn is a broken export

    if int8:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic-range weight quant
        blob = conv.convert()
        dst = out_dir / "model_int8.tflite"
        dst.write_bytes(blob)
        t8 = _run_tflite(tf, blob, X)
        diff = float(np.abs(t8 - ref).max())
        agree = bool((t8.argmax(1) == ref.argmax(1)).all())
        print(f"int8         : {_rel(dst)}  ({len(blob):,} B)  "
              f"max|Δlogit|={diff:.3f}  argmax==sklearn:{agree}")
        _fit_int8_temperature(tf, blob, pipe, lang)

    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="en", help="language code (default: en)")
    ap.add_argument("--all", action="store_true",
                    help="export every language that has a fitted pipeline.pkl")
    ap.add_argument("--no-int8", action="store_true", help="skip the int8 variant")
    ap.add_argument("--no-fp32", action="store_true", help="skip the fp32 variant")
    a = ap.parse_args(argv)

    if a.all:
        langs = sorted(p.parent.name for p in (MODELS_DIR / "intent").glob("*/pipeline.pkl"))
        if not langs:
            sys.exit(f"no */pipeline.pkl under {MODELS_DIR / 'intent'} — train first")
    else:
        langs = [a.lang]

    rc = 0
    for lang in langs:
        rc |= export(lang, fp32=not a.no_fp32, int8=not a.no_int8)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
