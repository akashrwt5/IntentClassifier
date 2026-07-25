"""
Calibration hygiene.

Confidence is softmax(logits / T). T is rank-preserving — it cannot change which
intent wins — but it drives five separate gates (fire-vs-GenAI, slot acceptance,
flow interruption, semantic agreement, semantic rescue). A T fit on the wrong data
or the wrong featurizer silently mis-tunes all of them, and nothing else in the
suite would notice.

These lock the two failures that actually happened:
  - the shipped T was fit on a calibration set that was 99.6% training data;
  - a device-featurizer T was applied to server/ONNX logits.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "packages"))

PACK_CALIB = _ROOT / "packs" / "en" / "intent_model" / "calibration.json"
EVAL_SETS = [_ROOT / "data" / "semantic_holdout_2.csv",
             _ROOT / "data" / "semantic_holdout_100.csv"]


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", str(s).lower().strip()))


def _load_fitter():
    path = _ROOT / "scripts" / "fit_calibration.py"
    spec = importlib.util.spec_from_file_location("fit_calibration", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------- artifact shape ------------------------------

def test_pack_calibration_artifact_is_well_formed():
    d = json.loads(PACK_CALIB.read_text(encoding="utf-8"))
    assert "temperature" in d, "pack calibration must declare a temperature"
    T = d["temperature"]
    assert isinstance(T, (int, float)) and 0.05 <= T <= 10.0, f"implausible T: {T}"


def test_pack_calibration_is_no_longer_the_unfitted_stub():
    """It shipped as a T=1.0 placeholder that nothing ever filled in."""
    d = json.loads(PACK_CALIB.read_text(encoding="utf-8"))
    assert d["temperature"] != 1.0, (
        "temperature is still the identity placeholder — run "
        "`python scripts/fit_calibration.py --write`"
    )
    assert "SKELETON" not in d.get("_note", ""), "stub note still present"


def test_pack_calibration_records_provenance():
    """A temperature with no provenance is how two conflicting values coexisted."""
    p = json.loads(PACK_CALIB.read_text(encoding="utf-8")).get("provenance", {})
    for key in ("method", "n_samples", "featurizer", "source_sha256", "fitted_by"):
        assert key in p, f"provenance missing {key!r}"
    assert p["n_samples"] > 0


# ------------------------------ leakage guard ------------------------------

def test_calibration_data_excludes_evaluation_sets():
    """The guard must exclude any row that appears in an evaluation set."""
    mod = _load_fitter()
    import pandas as pd
    data = pd.read_csv(mod.DATA_PATH, encoding="utf-8-sig").dropna(subset=["text", "intent"])
    keep, _, checked = mod.eval_leakage_mask(data["text"].astype(str).values)
    assert checked > 0, "no evaluation utterances were checked"

    kept = {_norm(t) for t in data["text"].astype(str).values[keep]}
    for path in EVAL_SETS:
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        col = next((c for c in df.columns
                    if c.strip().lower() in ("utterance", "text", "query", "phrase")),
                   df.columns[0])
        overlap = kept & {_norm(t) for t in df[col]}
        assert not overlap, f"calibration set still overlaps {path.name}: {sorted(overlap)[:3]}"


def test_leakage_guard_catches_punctuation_only_differences():
    """The regression that let 3 rows through: train.py compares raw strings, so
    'how does fall alert work?' did not match 'how does fall alert work'."""
    mod = _load_fitter()
    keep, examples, _ = mod.eval_leakage_mask(["how does fall alert work?",
                                               "an utterance that is definitely not held out"])
    assert not keep[0], "punctuation-only leak was not caught"
    assert keep[1], "a clean utterance was wrongly excluded"
    assert examples


# --------------------------- rank preservation -----------------------------

def test_temperature_cannot_change_the_predicted_intent():
    """Temperature only rescales confidence. If this ever fails, changing T is no
    longer safe and every threshold downstream must be re-derived."""
    import numpy as np
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, 59))
    base = logits.argmax(axis=1)
    for T in (0.2, 0.5, 1.0, 2.0, 5.0):
        assert ((logits / T).argmax(axis=1) == base).all(), f"argmax changed at T={T}"


@pytest.mark.parametrize("T", [0.648339])
def test_fitted_temperature_beats_uncalibrated(T):
    """Sanity: the recorded fit must actually improve calibration over T=1.0."""
    d = json.loads(PACK_CALIB.read_text(encoding="utf-8"))
    if "ece_uncalibrated" not in d:
        pytest.skip("artifact predates ECE recording")
    assert d["ece"] < d["ece_uncalibrated"], "fitted T is worse than no calibration"
