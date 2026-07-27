"""Tier-A numeric parity for the TFLite intent head.

The TFLite artifact is the LINEAR HEAD only (float TF-IDF vector -> logits),
built straight from the fitted sklearn ``LogisticRegression``. Because the map
is linear, fp32 TFLite must equal sklearn's ``decision_function`` to float
precision — that is the whole point of exporting the head rather than
transcoding the string-in ONNX graph. int8 is dynamic-range quantised, so we
require argmax agreement (0 gate disagreements), not bit parity.

Skips cleanly when TensorFlow is not installed (it is an export-only dep; the
Linux quality job does not install it, the release train-gate job does).
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow", reason="tensorflow is export-only; not on the inference path")

import sys  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "buildtime"))

from nlu_export import export_tflite as X  # noqa: E402

_PIPE = _ROOT / "models" / "intent" / "en" / "pipeline.pkl"
pytestmark = pytest.mark.skipif(
    not _PIPE.exists(), reason="no trained en pipeline (run nlu_training.train --lang en)")

_PROBE = [
    "what is my battery", "turn up the volume", "start streaming",
    "how do i pair my hearing aids", "asdfqwer zzz", "",
]


def _setup():
    import tensorflow as tf
    W, b, n_features, n_classes, pipe = X._load_head("en")
    model = X._build_keras(tf, W, b)
    tfidf = pipe.named_steps.get("tfidf") or pipe.named_steps.get("vectorizer")
    clf = pipe.named_steps.get("clf")
    feats = tfidf.transform(_PROBE).toarray().astype(np.float32)
    ref = clf.decision_function(feats).astype(np.float32)
    return tf, model, feats, ref


def test_fp32_tflite_matches_sklearn_logits_bit_for_bit():
    """fp32 head == sklearn decision_function to float rounding (the ONNX parity)."""
    tf, model, feats, ref = _setup()
    blob = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    out = X._run_tflite(tf, blob, feats)
    assert out.shape == ref.shape
    assert np.abs(out - ref).max() <= X.FP32_LOGIT_TOL


def test_int8_tflite_preserves_argmax():
    """Dynamic-range int8 keeps the predicted class (0 gate disagreements)."""
    tf, model, feats, ref = _setup()
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    out = X._run_tflite(tf, conv.convert(), feats)
    assert (out.argmax(1) == ref.argmax(1)).all()


def test_exporter_writes_both_variants(tmp_path, monkeypatch):
    """`export('en')` produces model.tflite + model_int8.tflite and self-passes fp32."""
    monkeypatch.setattr(X, "MODELS_DIR", tmp_path / "models")
    # stage the fitted pipeline where the exporter expects it
    dst = tmp_path / "models" / "intent" / "en"
    dst.mkdir(parents=True)
    import shutil
    shutil.copy(_PIPE, dst / "pipeline.pkl")
    rc = X.export("en", fp32=True, int8=True)
    assert rc == 0
    assert (dst / "model.tflite").exists()
    assert (dst / "model_int8.tflite").exists()
