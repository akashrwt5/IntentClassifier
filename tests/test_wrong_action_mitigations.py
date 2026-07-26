"""ND-11 mitigations: polarity guards (b) + uncertainty confirmation gate (a).

Uses the real engine with the en multilingual model; skips when trained
artifacts are absent (fresh clone before `make train-multilingual`).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "runtime"))

MODEL = REPO_ROOT / "models" / "intent" / "en" / "model.onnx"


@pytest.fixture(scope="module")
def engine():
    if not MODEL.exists():
        pytest.skip("trained artifacts not present (make train-multilingual)")
    pytest.importorskip("onnxruntime")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nlu_engine import NLUEngine

        eng = NLUEngine(model_name="en", language="en", semantic_enabled=False)
    return eng


def test_polarity_guard_mute_never_fires_unmute(engine):
    engine.reset("pg1")
    r = engine.handle("pg1", "turn mute on")
    # Borderline confidence → ask-first is acceptable; firing unmute is not.
    assert r.intent == "device.volume.mute", (r.intent, r.type)
    assert r.type in ("FULFILL", "CONFIRM")


def test_quiet_request_is_at_least_gated(engine):
    """A "quieter" request must never SILENTLY raise the volume.

    This test used to assert the polarity guard redirected the intent. Those
    guards were deliberately RETIRED (see content/platform.yaml): replaying the
    1,461-utterance holdout with them OFF was as good or better — 3 vs 4 wrong
    state-changing actions — because the rules mostly flipped answers that were
    already correct.

    So the intent can still come back wrong here; what must hold is that the
    surviving mitigation catches it. The uncertainty-confirmation gate turns a
    borderline volume change into an ask-first turn, so the user is asked rather
    than having their hearing aid turned up on them. Closing the residual
    wrong-action budget is charter B1, not this guard.
    """
    engine.reset("pg2")
    r = engine.handle("pg2", "i need it more quiet")
    if r.intent == "device.volume.increase":
        assert r.type == "CONFIRM", (
            f"a 'quieter' request resolved to volume.increase and would have "
            f"fired without asking ({r.type}, {r.confidence:.2f})")


def test_uncertain_confirm_ask_then_yes_fires(engine, monkeypatch):
    monkeypatch.setattr(engine, "_confirm_below", 1.01)  # force the gate
    engine.reset("uc1")
    r1 = engine.handle("uc1", "mute my hearing aids")
    assert r1.type == "CONFIRM" and r1.intent == "device.volume.mute"
    assert "mute" in (r1.message or "").lower()
    r2 = engine.handle("uc1", "yes please")
    assert r2.type == "FULFILL" and r2.intent == "device.volume.mute"
    assert r2.action == "volume.mute"


def test_uncertain_confirm_no_cancels_without_action(engine, monkeypatch):
    monkeypatch.setattr(engine, "_confirm_below", 1.01)
    engine.reset("uc2")
    r1 = engine.handle("uc2", "mute my hearing aids")
    assert r1.type == "CONFIRM"
    r2 = engine.handle("uc2", "no")
    assert r2.type == "FULFILL" and r2.intent == "sys.confirm.cancelled"
    assert r2.action is None


def test_uncertain_confirm_unclear_reply_never_fires_held_action(engine, monkeypatch):
    monkeypatch.setattr(engine, "_confirm_below", 1.01)
    engine.reset("uc3")
    r1 = engine.handle("uc3", "turn the volume up")
    assert r1.type == "CONFIRM"
    # user says something unrelated — the held action must be dropped, not fired
    r2 = engine.handle("uc3", "what's the weather like")
    assert not (r2.type == "FULFILL" and r2.action and "volume" in (r2.action or "")), r2


def test_confident_commands_still_fire_without_friction(engine):
    """The gate must not add friction to clearly-confident commands."""
    engine.reset("cf1")
    r = engine.handle("cf1", "mute")
    assert r.type == "FULFILL" and r.intent == "device.volume.mute", (r.type, r.intent)
