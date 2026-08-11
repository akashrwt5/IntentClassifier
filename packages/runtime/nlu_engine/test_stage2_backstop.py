"""Stage-2 backstop: keep TF-IDF's answer when the semantic stage declines.

Placed inside the package (rather than in tests/) on request. Note the repo
convention is `tests/` — `make check` collects that directory, so add this path
to the pytest run or move the file if it should gate CI.

WHAT IS BEING PINNED
--------------------
Before this change, a turn below the fire threshold that Stage 3 declined went
straight to GENAI, throwing away the TF-IDF prediction that had caused the
handover in the first place. Measured on the handover subset:

    stress accuracy    Stage 2 0.6923    Stage 3 0.3632

so the engine was substituting a 0.36 signal for a 0.69 one. End-to-end on the
held-out TEST half (mean of 3 seeds):

    previous policy   stress 0.6856   OOD reject 0.9088
    with backstop     stress 0.8298   OOD reject 0.8590

The backstop is OFF unless a language pack sets `stage2_backstop_confidence`.
The first test pins that, because a silent behaviour change for fr/de/da — which
have no semantic stage and were never measured — is the risk here.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
for _p in ("packages/buildtime", "packages/runtime"):
    if str(_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(_ROOT / _p))

from nlu_engine.engine import NLUEngine  # noqa: E402

FALLBACK = "Default Fallback Intent"
REAL_INTENT = "Cmd.VolumeIncrease"


class _Classifier:
    def __init__(self, intent, conf):
        self._i, self._c = intent, conf

    def classify(self, text):
        return self._i, self._c

    def calibrated_confidence(self, intent):
        return None

    def oov_ratio(self, text):
        return 0.0


class _Semantic:
    """Stands in for SemanticFallback so no model artifact is needed."""

    def __init__(self, intent, conf):
        self._i, self._c = intent, conf

    def classify(self, text):
        return self._i, self._c


def _engine(backstop, tfidf, sem):
    """A real engine with its two model stages replaced.

    __new__ + explicit attributes rather than __init__: constructing the real
    engine loads ONNX artifacts and a language pack, none of which this
    behaviour depends on. Every attribute _handle_new_intent touches is set
    here, so a future field it starts reading will fail loudly rather than
    silently defaulting.
    """
    e = NLUEngine.__new__(NLUEngine)
    e.classifier = _Classifier(*tfidf)
    e.semantic = _Semantic(*sem) if sem else None
    e._stage2_backstop = backstop
    e.semantic_threshold = 0.40
    e.agreement_threshold = 0.50
    e.threshold = 0.70  # the fire bar
    e._oov_reject_ratio = None
    e._oov_bypass_confidence = 1.01
    e.intents = {
        REAL_INTENT: {"action": "volume.increase", "slots": []},
        "Cmd.VolumeDecrease": {"action": "volume.decrease", "slots": []},
    }
    # guards disabled: they are tested elsewhere and would only add a second
    # reason for an intent to change, obscuring which stage produced the answer
    e._polarity_guards = []
    e._help_markers = None
    e._help_pairs = {}
    e._availability = {}  # runtime-contract-v1 §5 snapshot; empty = no gating
    return e


def _run(engine, text="turn it up a bit"):
    class _S:
        """Session stub: every call the fulfilment path makes is a no-op, so the
        assertions can only be about which STAGE answered."""

        def decrement_contexts(self):
            pass

        def set_context(self, *a, **k):
            pass

        def clear_context(self, *a, **k):
            pass

        def record_fulfillment(self, *a, **k):
            pass

        def start_slot_fill(self, *a, **k):
            pass

    return engine._handle_new_intent(_S(), text)


# --------------------------------------------------------------- off by default


def test_backstop_off_preserves_genai_fallback():
    """0.0 must reproduce the previous behaviour EXACTLY: declined rescue -> GENAI,
    even though TF-IDF had a perfectly usable answer."""
    e = _engine(0.0, (REAL_INTENT, 0.55), (FALLBACK, 0.9))
    r = _run(e)
    assert r.type == "FALLBACK" and r.intent == "GENAI"


def test_engine_default_is_off():
    assert NLUEngine.DEFAULT_STAGE2_BACKSTOP_CONFIDENCE == 0.0


# --------------------------------------------------------------- backstop fires


def test_backstop_keeps_tfidf_answer_when_semantic_declines():
    """The case the change exists for: semantic says fallback, TF-IDF is at 0.55,
    backstop is 0.30 -> TF-IDF's intent survives instead of going to GENAI."""
    e = _engine(0.30, (REAL_INTENT, 0.55), (FALLBACK, 0.9))
    r = _run(e)
    assert r.intent == REAL_INTENT
    assert r.confidence == pytest.approx(0.55)
    assert not r.semantic_rescue


def test_backstop_respects_its_own_floor():
    """Below the backstop bar the turn still goes to GENAI — the backstop widens
    the accept region, it does not remove the gate."""
    e = _engine(0.30, (REAL_INTENT, 0.20), (FALLBACK, 0.9))
    assert _run(e).intent == "GENAI"


def test_backstop_never_resurrects_the_fallback_intent():
    """TF-IDF itself predicting out-of-scope must not be turned into an action."""
    e = _engine(0.30, (FALLBACK, 0.95), (FALLBACK, 0.9))
    assert _run(e).intent == "GENAI"


def test_backstop_applies_when_no_semantic_stage_is_loaded():
    """fr/de/da ship without Stage 3. The backstop must still hold the TF-IDF
    answer rather than dropping every sub-threshold turn."""
    e = _engine(0.30, (REAL_INTENT, 0.55), None)
    assert _run(e).intent == REAL_INTENT


# --------------------------------------------------------------- max-confidence arm


def test_semantic_wins_when_it_is_the_more_confident_signal():
    e = _engine(0.30, (REAL_INTENT, 0.35), ("Cmd.VolumeDecrease", 0.80))
    r = _run(e)
    assert r.intent == "Cmd.VolumeDecrease"
    assert r.semantic_rescue
    assert r.tfidf_intent == REAL_INTENT


def test_tfidf_wins_when_it_is_the_more_confident_signal():
    """Even with the semantic head above its own 0.40 floor, it must not overrule
    a TF-IDF prediction that is more confident. This is the arm that stops the
    engine substituting the weaker of two available signals."""
    e = _engine(0.30, (REAL_INTENT, 0.65), ("Cmd.VolumeDecrease", 0.45))
    r = _run(e)
    assert r.intent == REAL_INTENT
    assert not r.semantic_rescue


def test_without_backstop_the_weaker_semantic_signal_still_wins():
    """Pins the OLD behaviour when the flag is off, so the two paths are not
    quietly conflated: with backstop 0.0 the 0.45 head overrules TF-IDF at 0.65."""
    e = _engine(0.0, (REAL_INTENT, 0.65), ("Cmd.VolumeDecrease", 0.45))
    r = _run(e)
    assert r.intent == "Cmd.VolumeDecrease"
    assert r.semantic_rescue


# --------------------------------------------------------------- schema wiring


def test_english_pack_enables_the_backstop_at_the_measured_value():
    import json

    schema = json.loads((_ROOT / "language_packs" / "en" / "nlu_schema.json").read_text("utf-8"))
    assert schema["stage2_backstop_confidence"] == 0.30
    # the pair the policy was selected as; changing one without the other makes
    # the shipped behaviour something that was never measured
    assert schema["semantic_threshold"] == 0.40
