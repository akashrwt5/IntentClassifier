"""Slot-flow thresholds and the slot-answer precedence rule.

Two things are locked here.

**The thresholds are content-owned and fitted.** `interrupt_threshold` used to be
`NLUEngine.INTERRUPT_THRESHOLD = 0.75`, a class constant justified as "lowered
from 0.85 after isotonic calibration" — a calibration method this pipeline no
longer uses. Being a constant, no language pack could override it either. Both it
and `slot_confidence_threshold` now live in `content/platform.yaml`, are fitted
by `nlu_training.fit_slot_thresholds`, and carry provenance.

**Answering the live question is not a topic switch.** Several memory names are
also commands. Asked "What is the name of the memory?", a user saying "mute" used
to MUTE the device at 0.980 instead of switching to the Mute memory; "quiet"
muted too, and "telephone" rang the phone. No threshold can fix it — "tinnitus"
and "mask" classify at 1.000 — so the engine resolves it by precedence. This is a
multi-turn defect, invisible to the single-turn holdout replay, so it needs its
own test.
"""

import json
import sys
import warnings
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for p in ("packages/buildtime", "packages/runtime"):
    if str(_ROOT / p) not in sys.path:
        sys.path.insert(0, str(_ROOT / p))

SCHEMA = json.loads((_ROOT / "content" / "nlu_schema.json").read_text(encoding="utf-8"))
FITTED = _ROOT / "models" / "intent" / "en" / "slot_thresholds.json"

# From `python -m nlu_training.fit_slot_thresholds --lang en --write`.
EXPECTED_SLOT = 0.5
EXPECTED_INTERRUPT = 0.67


# ---------------------------- configuration --------------------------------

def test_interrupt_threshold_is_content_owned_not_hardcoded():
    assert "interrupt_threshold" in SCHEMA, (
        "interrupt_threshold must come from content/platform.yaml so a language "
        "pack can override it; it used to be a class constant on NLUEngine")


def test_engine_reads_the_schema_value():
    from nlu_engine.engine import NLUEngine
    assert SCHEMA["interrupt_threshold"] != NLUEngine.DEFAULT_INTERRUPT_THRESHOLD, (
        "schema value equals the fallback, so this test cannot tell whether the "
        "engine reads the schema — pick a fitted value that differs")
    eng = NLUEngine.__new__(NLUEngine)
    eng.schema = SCHEMA
    eng.interrupt_threshold = SCHEMA.get(
        "interrupt_threshold", NLUEngine.DEFAULT_INTERRUPT_THRESHOLD)
    assert eng.interrupt_threshold == SCHEMA["interrupt_threshold"]


def test_thresholds_match_the_fitted_values():
    assert SCHEMA["slot_confidence_threshold"] == EXPECTED_SLOT
    assert SCHEMA["interrupt_threshold"] == EXPECTED_INTERRUPT


def test_slot_threshold_is_below_the_fire_threshold():
    """Entering a slot flow only asks a question; firing changes device state.

    A slot-bearing intent must therefore never need MORE confidence than a
    fire-and-forget one.
    """
    assert SCHEMA["slot_confidence_threshold"] <= SCHEMA["confidence_threshold"]


@pytest.mark.skipif(not FITTED.exists(), reason="thresholds not fitted yet")
def test_fitted_values_carry_provenance():
    data = json.loads(FITTED.read_text(encoding="utf-8"))
    prov = data["provenance"]
    for key in ("method", "folds", "temperature", "slot_confidence",
                "interrupt", "source", "fitted_at", "fitted_by"):
        assert key in prov, f"provenance missing {key!r}"
    assert data["slot_confidence_threshold"] == EXPECTED_SLOT
    assert data["interrupt_threshold"] == EXPECTED_INTERRUPT


def test_compiler_cannot_silently_drop_a_platform_key():
    """The layout list in content_source.assemble is a second copy of the key
    set; a key present in one and absent from the other is discarded silently,
    which is how interrupt_threshold failed to reach the compiled schema."""
    from nlu_compiler.content_source import PLATFORM_KEYS, assemble
    out = assemble(write=False)["schema"]
    import yaml
    platform = yaml.safe_load((_ROOT / "content" / "platform.yaml")
                              .read_text(encoding="utf-8"))
    for key in PLATFORM_KEYS:
        if key in platform:
            assert key in out, f"platform key {key!r} was dropped by assemble()"


# --------------------- slot-answer precedence (the bug) ---------------------

_MODEL = _ROOT / "models" / "intent" / "en" / "model.onnx"
pytestmark_model = pytest.mark.skipif(
    not _MODEL.exists(), reason="trained English model absent")


@pytest.fixture(scope="module")
def engine():
    pytest.importorskip("onnxruntime")
    if not _MODEL.exists():
        pytest.skip("trained English model absent")
    from nlu_engine import NLUEngine
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return NLUEngine(model_name="en", language="en", semantic_enabled=False)


def _answer(engine, session, answer):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine.handle(session, "change my memory")
        return engine.handle(session, answer)


# "mute"/"quiet" fired volume.mute, "telephone" rang the phone, and
# "tinnitus"/"mask" classify at 1.000 — the reason a threshold cannot fix this.
@pytest.mark.parametrize("answer", ["mute", "quiet", "telephone", "tinnitus",
                                    "mask", "custom 1", "custom one",
                                    "restaurant", "outdoors"])
def test_memory_name_that_is_also_a_command_fills_the_slot(engine, answer):
    r = _answer(engine, f"slot-{answer}", answer)
    assert r.intent == "device.memory.change", (
        f"answering the memory prompt with {answer!r} produced "
        f"{r.intent} / action={r.action} instead of changing the memory")
    assert r.interrupted_intent is None


@pytest.mark.parametrize("utterance,expected", [
    ("turn up the volume", "device.volume.increase"),
    ("make it louder", "device.volume.increase"),
    ("find my phone", "find.phone.locate"),
    ("stop the stream", "streaming.session.stop"),
])
def test_a_genuine_topic_switch_still_interrupts(engine, utterance, expected):
    """The precedence rule must not trap the user in the flow.

    Guarding with FUZZY matching on did exactly that: the memory entity resolved
    "turn up the volume" to the memory "three" via "the" at 0.6, so the switch was
    silently swallowed. Hence the strict match floor.
    """
    r = _answer(engine, f"switch-{utterance}", utterance)
    assert r.interrupted_intent == "device.memory.change", (
        f"{utterance!r} did not interrupt the memory flow (got {r.intent})")
    assert r.intent == expected


def test_open_text_slot_still_accepts_arbitrary_answers(engine):
    """@remind is an OPEN entity — anything is a valid reminder name, so the
    precedence rule deliberately does not apply and the bar is the only signal."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine.handle("open-slot", "remind me")
        r = engine.handle("open-slot", "water the plants")
    assert r.intent == "reminders.task.create", r.intent
