"""The device-weights exporter can emit either a pruned or a FULL-vocab head.

`export_ios_weights.py` loads the trained pipeline and, by default, prunes the
vocabulary to the top-N features per class (small on-device head). Passing
`--top-per-class 0` disables pruning so the exported head matches the trained
pipeline (and thus the ONNX/TFLite artifacts) feature-for-feature — that is how
the full-vocab CoreML variant is produced.

Uses only sklearn/numpy (no TensorFlow or coremltools), so it runs in the normal
quality job.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "buildtime"))

from nlu_export import export_ios_weights as W  # noqa: E402

_PIPE = _ROOT / "models" / "intent" / "en" / "pipeline.pkl"
_LABELS = _ROOT / "models" / "intent" / "en" / "labels.pkl"
pytestmark = pytest.mark.skipif(
    not (_PIPE.exists() and _LABELS.exists()),
    reason="no trained en pipeline/labels (run nlu_training.train --lang en)")


def _full_vocab_size():
    return len(joblib.load(str(_PIPE)).named_steps["tfidf"].vocabulary_)


def _export(tmp_path, top_per_class):
    # point the module's module-level paths at the committed en artifacts
    W.PIPELINE_PATH = _PIPE
    W.LABELS_PATH = _LABELS
    W.DATA_PATH = _ROOT / "language_packs" / "en" / "train.csv"
    out = tmp_path / f"w_{top_per_class}.json"
    W.export(out, top_per_class)
    return json.loads(out.read_text(encoding="utf-8"))


def test_no_prune_keeps_the_full_pipeline_vocab(tmp_path):
    d = _export(tmp_path, top_per_class=0)
    full = _full_vocab_size()
    assert len(d["vocab"]) == full
    assert np.array(d["coef"]).shape == (len(d["labels"]), full)


def test_pruning_shrinks_the_vocab(tmp_path):
    d = _export(tmp_path, top_per_class=30)
    full = _full_vocab_size()
    assert len(d["vocab"]) < full
    # every pruned token must be a real pipeline feature (a subset, not new terms)
    pv = set(joblib.load(str(_PIPE)).named_steps["tfidf"].vocabulary_)
    assert set(d["vocab"]).issubset(pv)


def test_full_vocab_refits_its_own_temperature(tmp_path):
    """T is fit on device-equivalent logits, so the full head gets its own T
    (not the pruned head's) — a stale T would mis-tune the confidence gate."""
    pruned = _export(tmp_path, top_per_class=30)
    full = _export(tmp_path, top_per_class=0)
    assert full["temperature"] != pruned["temperature"]
