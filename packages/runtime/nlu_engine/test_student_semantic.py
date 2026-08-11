"""StudentSemantic — the distilled single-file Stage 3 backend.

Placed inside the package to sit next to the code it tests (repo convention is
`tests/`; `pyproject.toml` sets `testpaths = ["tests"]`, so this file is NOT
collected by `make check` unless that path is added or the file is moved).

WHAT MATTERS HERE
-----------------
This class replaces a 23 MB MiniLM encoder + LogReg head with one 2.5 MB ONNX
that has a different tokenizer and a different artifact layout. Three things can
go wrong silently, and each has a test below:

1. THE TOKENIZER DRIFTS from the one used at training time. Every id shifts and
   the model answers confidently wrong. Nothing crashes.
2. THE LABEL ORDER does not match the logit columns, so every intent is
   mislabelled. Nothing crashes.
3. THE WEIGHTS ARE MISSING because torch wrote them to a `.onnx.data` sidecar
   and only the graph was installed. This nearly shipped: 0.166 MB was recorded
   as the artifact size when the graph alone was 0.166 MB and the weights were a
   separate 0.787 MB file.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[3]

# Load the module by path: `nlu_engine.semantic` would drag in the package
# __init__ -> engine -> classifier -> joblib/sklearn, none of which this needs.
#
# NOTE: this keeps the module UNDER TEST light, but it no longer makes the file
# dependency-free. Now that `packages/runtime/nlu_engine` is in testpaths,
# pytest imports this file as `nlu_engine.test_student_semantic`, which runs the
# package __init__ and pulls in joblib/sklearn at COLLECTION time. Both are
# declared runtime dependencies, so `make check` after `make install-dev` is
# fine; a bare interpreter will fail to collect.
_spec = importlib.util.spec_from_file_location(
    "_nlu_semantic_under_test", Path(__file__).with_name("semantic.py")
)
assert _spec is not None and _spec.loader is not None, "cannot load semantic.py"
_sem = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sem)
StudentSemantic = _sem.StudentSemantic

INSTALL_DIR = _ROOT / "models" / "semantic_student" / "en"
installed = pytest.mark.skipif(
    not (INSTALL_DIR / "student.onnx").exists(),
    reason="no student installed; run new_semantic/scripts/install_student.py",
)


# --------------------------------------------------------------- tokenizer


def test_tokenizer_regex_matches_training():
    """The regex IS the input contract. new_semantic/scripts/common.py uses
    exactly this; if they diverge, every token id shifts silently."""
    assert _sem.StudentSemantic._TOKEN_RE.pattern == r"[a-z0-9]+(?:'[a-z0-9]+)?"


@installed
def test_punctuation_is_discarded():
    """'volume up' and 'volume up?' must be the same input — the model was
    trained on a tokenizer that drops punctuation."""
    s = StudentSemantic(INSTALL_DIR)
    a, _ = s._encode("turn it up")
    b, _ = s._encode("turn it up?!")
    assert np.array_equal(a, b)


@installed
def test_curly_apostrophe_normalises():
    """ASR and keyboards emit U+2019; the training tokenizer folded it to '."""
    s = StudentSemantic(INSTALL_DIR)
    a, _ = s._encode("it's too quiet")
    b, _ = s._encode("it’s too quiet")
    assert np.array_equal(a, b)


@installed
def test_unknown_words_follow_the_installed_tokenizer_contract():
    s = StudentSemantic(INSTALL_DIR)
    ids, _ = s._encode("zzqxplffff")
    if s.tokenizer_mode == "word":
        assert ids[0][0] == s.UNK_ID
    else:
        # Subword vocabularies contain character pieces, preserving signal from
        # unseen words instead of collapsing the whole word to [UNK].
        assert ids[0][0] not in {s.PAD_ID, s.UNK_ID}


@installed
def test_subword_encoding_matches_the_training_contract():
    """Installed subword artifacts must receive the exact ids used in training."""
    s = StudentSemantic(INSTALL_DIR)
    if s.tokenizer_mode != "subword":
        pytest.skip("installed model uses the legacy word tokenizer")

    sys.path.insert(0, str(_ROOT / "new_semantic"))
    from scripts.common import encode

    expected, _ = encode("make it quieter", s.vocab, s.max_len, "subword")
    actual, _ = s._encode("make it quieter")
    assert actual[0].tolist() == expected


@installed
def test_padding_and_truncation_hold_the_static_shape():
    """The ONNX graph is exported at a FIXED (1, max_len). A wrong length here
    is a runtime error on device, not a soft failure."""
    s = StudentSemantic(INSTALL_DIR)
    short, _ = s._encode("mute")
    long, _ = s._encode(" ".join(["volume"] * 200))
    assert short.shape == (1, s.max_len)
    assert long.shape == (1, s.max_len)
    assert short[0][-1] == s.PAD_ID


# --------------------------------------------------------------- contract


@installed
def test_classify_returns_a_known_label_and_a_probability():
    s = StudentSemantic(INSTALL_DIR)
    intent, conf = s.classify("turn up the volume")
    assert intent in s.labels
    assert 0.0 <= conf <= 1.0


@installed
def test_label_count_matches_the_logit_width():
    """If labels.json and the classifier head disagree, every prediction is
    mislabelled and nothing raises."""
    s = StudentSemantic(INSTALL_DIR)
    ids, mask = s._encode("volume up")
    logits = s._sess.run(None, {s._in_ids: ids, s._in_mask: mask})[0][0]
    assert len(logits) == len(s.labels)


@installed
def test_fallback_intent_is_in_the_label_space():
    """The engine routes on this exact string; a rename breaks rejection."""
    s = StudentSemantic(INSTALL_DIR)
    assert _sem.FALLBACK_INTENT in s.labels


@installed
def test_it_is_deterministic():
    s = StudentSemantic(INSTALL_DIR)
    a = s.classify("make it a bit louder")
    b = s.classify("make it a bit louder")
    assert a == b


# --------------------------------------------------------------- calibration


@installed
def test_temperature_is_read_from_meta():
    """meta.json carried `temperature: 0.68` for a while before any code read
    it. The number being present is not the same as it being applied."""
    meta = json.loads((INSTALL_DIR / "meta.json").read_text(encoding="utf-8"))
    s = StudentSemantic(INSTALL_DIR)
    assert s.temperature == pytest.approx(float(meta.get("temperature", 1.0)))


@installed
def test_temperature_actually_changes_the_confidence():
    """The regression this guards: reading T into an attribute and then never
    dividing by it. Every accuracy test still passes in that state, because T
    cannot move argmax — only the confidence, which is what the gate reads."""
    s = StudentSemantic(INSTALL_DIR)
    if s.temperature == pytest.approx(1.0):
        pytest.skip("installed model is uncalibrated (T=1); nothing to compare")

    _, calibrated = s.classify("turn up the volume")
    s.temperature = 1.0
    _, raw = s.classify("turn up the volume")
    assert calibrated != pytest.approx(raw), "temperature is stored but not applied"


@installed
def test_temperature_is_rank_preserving():
    """T must not change WHICH intent wins, on any input. If it does, the
    softmax is being applied somewhere it should not be, and calibration has
    silently become a behaviour change rather than a reporting fix."""
    s = StudentSemantic(INSTALL_DIR)
    probes = [
        "turn up the volume",
        "mute it",
        "switch to the restaurant program",
        "where is my phone",
        "what is the weather like tomorrow",
        "zzqxplffff",
    ]
    calibrated = [s.classify(t)[0] for t in probes]
    s.temperature = 1.0
    raw = [s.classify(t)[0] for t in probes]
    assert calibrated == raw


def test_a_non_positive_temperature_is_refused(tmp_path):
    """T <= 0 gives a divide-by-zero or an inverted distribution. Fail at load
    rather than serve nonsense confidences."""
    import shutil

    if not (INSTALL_DIR / "student.onnx").exists():
        pytest.skip("no student installed")
    for name in ("student.onnx", "vocab.json", "labels.json"):
        shutil.copy(INSTALL_DIR / name, tmp_path / name)
    meta = json.loads((INSTALL_DIR / "meta.json").read_text(encoding="utf-8"))
    meta["temperature"] = 0.0
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="temperature"):
        StudentSemantic(tmp_path)


# --------------------------------------------------------------- artifacts


@installed
def test_installed_artifact_is_self_contained():
    """A `.onnx.data` sidecar means the installed graph has no weights. Parity
    checks cannot catch this — both sides load the same file in one process."""
    assert not (INSTALL_DIR / "student.onnx.data").exists()


@installed
def test_meta_records_what_was_installed():
    meta = json.loads((INSTALL_DIR / "meta.json").read_text(encoding="utf-8"))
    assert meta["tokenizer"] in {"word", "subword"}
    assert meta["max_len"] >= 1
    assert 0.0 < meta["threshold"] <= 1.0
    assert meta["temperature"] > 0.0
    # synthetic_text is tracked repo-wide; it must survive into the install
    assert "synthetic_text" in meta


def test_missing_artifacts_raise_rather_than_serve_nothing(tmp_path):
    with pytest.raises(FileNotFoundError):
        StudentSemantic(tmp_path)


def test_a_sidecar_is_refused(tmp_path):
    """Installing a graph whose weights live outside it must fail loudly."""
    for name in ("student.onnx", "vocab.json", "labels.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "student.onnx.data").write_bytes(b"weights")
    with pytest.raises(FileNotFoundError, match="sidecar"):
        StudentSemantic(tmp_path)


# --------------------------------------------------------------- engine wiring


@installed
def test_engine_prefers_the_student_over_minilm():
    """_load_semantic must pick the student when one is installed. Without this
    the engine silently keeps using the 23 MB MiniLM stage and every measurement
    taken 'with the new model' is actually the old one."""
    joblib = pytest.importorskip("joblib")  # noqa: F841
    sys.path.insert(0, str(_ROOT / "packages" / "runtime"))
    from nlu_engine.engine import NLUEngine

    stage = NLUEngine._load_semantic(0.40, "en")
    assert stage is not None
    assert type(stage).__name__ == "StudentSemantic"
