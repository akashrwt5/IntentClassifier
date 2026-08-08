"""
Calibration hygiene (charter B4).

Confidence is `softmax(logits / T)`. `T` is rank-preserving — it cannot change
which intent wins — but it drives every gate in the system: the 0.70 fire
threshold, the 0.80 confirm gate, slot acceptance, flow interruption, semantic
agreement. A `T` fit on the wrong data or the wrong featurizer silently
mis-tunes all of them, and nothing else in the suite would notice.

These lock the two failures that actually happened (Review-F5 blocker B8):
  - the shipped `T` was fit on a set that was 99.6% training data;
  - a DEVICE-featurizer `T` (pruned 1370-term vocab) was applied to full-vocab
    SERVER/ONNX logits.

and the durable defence against both: provenance, required by test. An
untraceable temperature is how two conflicting values coexisted unnoticed.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for p in ("packages/buildtime", "packages/runtime"):
    if str(_ROOT / p) not in sys.path:
        sys.path.insert(0, str(_ROOT / p))

CALIB = _ROOT / "models" / "intent" / "en" / "calibration.json"

pytestmark = pytest.mark.skipif(
    not CALIB.exists(),
    reason="no fitted calibration (python -m nlu_training.fit_calibration --lang en --write)")


@pytest.fixture(scope="module")
def calib():
    return json.loads(CALIB.read_text(encoding="utf-8"))


# ----------------------------- artifact shape ------------------------------

def test_temperature_is_plausible(calib):
    T = calib["temperature"]
    assert isinstance(T, (int, float)) and 0.05 <= T <= 10.0, f"implausible T: {T}"


def test_temperature_is_not_the_identity_placeholder(calib):
    """T = 1.0 means no calibration was applied at all."""
    assert calib["temperature"] != 1.0, (
        "temperature is the identity placeholder — run "
        "`python -m nlu_training.fit_calibration --lang en --write`")


def test_calibration_actually_improves_ece(calib):
    """A calibration that does not calibrate is worse than none: it looks fixed."""
    assert calib["ece"] < calib["ece_uncalibrated"], (
        f"ECE did not improve: {calib['ece_uncalibrated']} -> {calib['ece']}")


def test_provenance_is_recorded(calib):
    """A temperature with no provenance is how two conflicting values coexisted."""
    p = calib.get("provenance", {})
    for key in ("method", "folds", "seed", "n_samples", "featurizer",
                "source", "source_sha256", "fitted_at", "fitted_by"):
        assert key in p, f"provenance missing {key!r}"
    assert p["n_samples"] > 0
    assert len(p["source_sha256"]) == 64


def test_provenance_matches_the_current_training_data(calib):
    """If train.csv changed, the temperature is stale and must be refit."""
    import hashlib
    src = _ROOT / calib["provenance"]["source"]
    assert src.exists(), f"calibration source is missing: {src}"
    actual = hashlib.sha256(src.read_bytes()).hexdigest()
    assert actual == calib["provenance"]["source_sha256"], (
        f"{src.name} changed since the temperature was fit — refit it, or every "
        f"confidence-gated behaviour is tuned to data that no longer exists")


# ------------------------------ leakage guard ------------------------------

def test_fitter_excludes_the_evaluation_sets():
    """Fitting on the holdout would burn the only honest measurement we have."""
    fc = importlib.import_module("nlu_training.fit_calibration")
    import pandas as pd

    data = pd.read_csv(_ROOT / "language_packs" / "en" / "train.csv",
                       encoding="utf-8-sig").dropna(subset=["text", "intent"])
    keep, _leaked, checked = fc.eval_leakage_mask(data["text"].astype(str).values, "en")
    assert checked > 0, "no evaluation utterances were checked — the guard is inert"
    assert keep.sum() > 0


def test_fitter_would_catch_a_planted_leak():
    """Mutation check: the guard must actually reject an overlapping row."""
    fc = importlib.import_module("nlu_training.fit_calibration")
    import csv

    holdout = _ROOT / "language_packs" / "en" / "holdout_honest.csv"
    first = next(csv.DictReader(holdout.open(encoding="utf-8-sig")))["text"]
    keep, leaked, _ = fc.eval_leakage_mask([first, "an utterance that is not held out"], "en")
    assert not keep[0], "a verbatim holdout row was not detected as a leak"
    assert keep[1]
    assert leaked


# --------------------------- the B8 regression -----------------------------

def test_featurizer_mirrors_the_trainer():
    """T only calibrates logits from the featurizer it was fit on.

    A drift between fit_calibration and train.py reproduces B8 in a new place,
    so the two definitions are compared directly rather than by eye.
    """
    fc = importlib.import_module("nlu_training.fit_calibration")
    train_src = (_ROOT / "packages" / "buildtime" / "nlu_training" / "train.py").read_text()
    assert f'ngram_range={fc.TFIDF_KW["ngram_range"]}'.replace(" ", "") in \
        train_src.replace(" ", "").replace("\n", ""), "ngram_range drifted from train.py"
    assert f'min_df={fc.TFIDF_KW["min_df"]}' in train_src, "min_df drifted from train.py"
    assert f'C={fc.LR_KW["C"]}' in train_src, "C drifted from train.py"
    # The cap can no longer drift by construction: train.py IMPORTS it from
    # fit_calibration rather than declaring its own literal. Asserting the
    # import is a stronger guarantee than comparing two numbers that a careless
    # edit could still desynchronise.
    #
    # It also has to be an import now, because the cap moved to `None`
    # (disabled) and lives behind `cap_per_intent`, so there is no literal in
    # train.py to compare against — the old assertion would fail on a
    # correctly-wired tree.
    assert "from nlu_training.fit_calibration import" in train_src \
        and "cap_per_intent" in train_src, \
        "train.py must take the per-intent cap from fit_calibration, not redeclare it"


def test_runtime_uses_the_fitted_temperature_not_the_device_one(calib):
    """The actual B8 regression: the engine must read calibration.json.

    The device weights carry T=0.796, fit on a pruned 1370-term vocabulary. The
    engine applies T to full-vocab ONNX logits, so it must use the SERVER value.
    """
    pytest.importorskip("onnxruntime")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nlu_engine import NLUEngine
        eng = NLUEngine(language="en", semantic_enabled=False)
    assert eng.classifier.temperature == pytest.approx(calib["temperature"]), (
        f"engine is using T={eng.classifier.temperature}, not the fitted "
        f"{calib['temperature']}")


def test_device_and_server_temperatures_are_kept_separate():
    """They calibrate different featurizers and must NOT be unified."""
    device = _ROOT / "models" / "intent_classifier_weights.json"
    if not device.exists():
        pytest.skip("no device weights present")
    dev_T = json.loads(device.read_text(encoding="utf-8")).get("temperature")
    srv_T = json.loads(CALIB.read_text(encoding="utf-8"))["temperature"]
    assert dev_T != srv_T, (
        "device and server temperatures are identical — one of them is being "
        "reused for the wrong featurizer, which is blocker B8")
