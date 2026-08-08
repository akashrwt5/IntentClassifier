"""One confidence, one scale — and the head commands must not ask permission.

Every threshold in this engine (`confidence_threshold`, the confirmation band,
`interrupt_threshold`, `semantic_threshold`) is fitted against ONE quantity: the
temperature-calibrated softmax of the intent model. Anything else written into
the `confidence` field is a different unit wearing the same name, and comparing
the two is meaningless arithmetic that no accuracy test can see — temperature
scaling is rank-preserving, so which intent wins never changes.

That is exactly how the original defect survived: the keyword stage
short-circuited the model and returned a hardcoded 0.75 for its `regex` tier,
which was then compared against a band of 0.91 fitted on model probabilities.
`0.75 < 0.91` is true and means nothing. Every `regex` rule in the schema became
permanently un-fireable, and "increase volume" asked the user to confirm while
the model scored it 0.9992.

No test in the suite compared a score across stages, so nothing failed. These do.

See docs/confirm-gate-diagnosis.md.
"""

import sys
import warnings
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in ("packages/buildtime", "packages/runtime"):
    if str(_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(_ROOT / _p))

_MODEL = _ROOT / "models" / "intent" / "en" / "model.onnx"

pytestmark = pytest.mark.skipif(not _MODEL.exists(),
                                reason="trained English model absent")


@pytest.fixture(scope="module")
def engine():
    pytest.importorskip("onnxruntime")
    from nlu_engine import NLUEngine
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Semantic rescue off: it is a separate stage with its own threshold,
        # and these assertions are about the keyword/model boundary only.
        return NLUEngine(model_name="en", language="en", semantic_enabled=False)


# Utterances that trip a keyword rule AND that the model independently agrees
# with. Deliberately spread across tiers and intents so a regression in any one
# rule family shows up.
_CORROBORATED = [
    "increase volume",
    "turn up the volume",
    "volume up",
    "decrease volume",
    "turn it down",
    "raise the volume",
    "mute",
    "unmute",
    "start transcription",
]


# --------------------------- the scale invariant ----------------------------

@pytest.mark.parametrize("text", _CORROBORATED)
def test_corroborated_keyword_confidence_is_the_models_calibrated_value(engine, text):
    """A keyword hit the model agrees with must report the MODEL's probability.

    This is the assertion that makes the confidence field single-scale. If any
    constant is reintroduced into the keyword path, the reported value stops
    tracking the model and this fails immediately — regardless of whether the
    predicted intent is still correct.
    """
    clf = engine.classifier
    intent, conf = clf.classify(text)

    assert clf.last_stage == "keyword", (
        f"{text!r} no longer trips a keyword rule; the fixture has drifted and "
        f"this case is no longer testing the cross-stage boundary")
    assert clf.last_arbitration == "corroborated", (
        f"{text!r}: model and rule disagree ({clf.last_arbitration}); pick a "
        f"different fixture or investigate the divergence")

    p = clf._model_distribution(text)
    expected = float(p[list(clf.labels).index(intent)])
    assert conf == pytest.approx(expected, abs=1e-9), (
        f"{text!r} reported {conf:.4f} but the model's calibrated probability "
        f"for {intent} is {expected:.4f}. A keyword hit must not invent a "
        f"confidence — it is compared downstream against thresholds fitted on "
        f"the model's scale.")


def test_no_stage_reports_a_confidence_from_a_fixed_table(engine):
    """Corroborated confidences must be a distribution, not a handful of values.

    The tell for the original defect was that the low-confidence tail of CORRECT
    predictions was literally one repeated number (0.750 at both the 5th and
    10th percentile) rather than a spread. A stage returning table lookups
    produces very few distinct values across many different utterances.
    """
    clf = engine.classifier
    confs = set()
    for text in _CORROBORATED:
        _, c = clf.classify(text)
        confs.add(round(c, 6))
    assert len(confs) >= len(_CORROBORATED) - 1, (
        f"{len(_CORROBORATED)} distinct keyword-stage utterances produced only "
        f"{len(confs)} distinct confidences {sorted(confs)}. That is the "
        f"signature of a lookup table in the confidence path.")


def test_contested_is_the_only_constant_and_it_is_declared(engine):
    """Exactly one constant may remain, and it must be the declared one.

    A rule firing against the model's disagreement is genuine ambiguity, and
    until the joint (FIRE, FLOOR) fit lands there is no fitted value to report.
    That single placeholder is allowed — silently growing a second one is not.
    """
    from nlu_engine.classifier import IntentClassifier
    assert isinstance(IntentClassifier.CONTESTED_CONFIDENCE, float)
    assert 0.0 < IntentClassifier.CONTESTED_CONFIDENCE < 1.0


# ------------------------- head commands must fire --------------------------

# The highest-frequency commands in the product. A user raising the volume in a
# noisy room is the single worst place to spend a confirmation turn: the
# confirmation is delivered through the exact channel that is already failing.
#
# Volume up/down are also trivially reversible — the cost of getting one wrong
# is the user saying "no, down", which is cheaper than asking every time.
_HEAD_COMMANDS = [
    ("increase volume", "Cmd.VolumeIncrease"),
    ("turn up the volume", "Cmd.VolumeIncrease"),
    ("volume up", "Cmd.VolumeIncrease"),
    ("make it louder", "Cmd.VolumeIncrease"),
    ("decrease volume", "Cmd.VolumeDecrease"),
    ("turn down the volume", "Cmd.VolumeDecrease"),
    ("turn it down", "Cmd.VolumeDecrease"),
    ("mute", "Cmd.VolumeMute"),
    ("unmute", "Cmd.VolumeUnmute"),
]


@pytest.mark.parametrize("text,expected", _HEAD_COMMANDS)
def test_head_commands_fulfil_without_asking(engine, text, expected):
    engine.reset("head-cmd")
    result = engine.handle("head-cmd", text)
    assert result.intent == expected, (
        f"{text!r} classified as {result.intent}, expected {expected}")
    assert result.type == "FULFILL", (
        f"{text!r} returned {result.type} at confidence {result.confidence:.4f} "
        f"instead of firing. These are the most common commands in the product; "
        f"a confirmation here is friction a hearing-aid user pays in the worst "
        f"possible moment.")
