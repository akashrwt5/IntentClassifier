"""Runtime Contract v1 conformance — the Python engine as executable spec.

Covers the seams the contract's gap table listed: session blob round-trip
(§3), turn_id threading (§1), notify_execution (§4), availability snapshot
push + recognized-but-unavailable routing (§5, ADR-002 A6), and the
InferenceBackend injection point (§2).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "runtime"))

MODEL = REPO_ROOT / "models" / "intent" / "en" / "model.onnx"

from nlu_engine.context import Session  # noqa: E402  (no heavy deps)


@pytest.fixture(scope="module")
def engine():
    if not MODEL.exists():
        pytest.skip("trained artifacts not present")
    pytest.importorskip("onnxruntime")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nlu_engine import NLUEngine

        eng = NLUEngine(model_name="en", language="en", semantic_enabled=False)
    return eng


# -- §3 session blob ----------------------------------------------------------

def test_session_blob_roundtrip_slot_flow():
    s = Session(session_id="s1")
    s.pending_intent = "reminders.task.create"
    s.pending_slots = {"what": "medication"}
    s.awaiting_slot = "when"
    s.slot_attempts = 1
    blob = s.to_blob()
    assert blob["v"] == 1 and blob["frames"][0]["awaiting_slot"] == "when"
    r = Session.from_blob(blob)
    assert (r.pending_intent, r.pending_slots, r.awaiting_slot, r.slot_attempts) == \
           ("reminders.task.create", {"what": "medication"}, "when", 1)


def test_session_blob_roundtrip_held_confirmation():
    s = Session(session_id="s2")
    s.pending_confirm = {"intent": "device.volume.mute", "action": "volume.mute",
                         "fulfillment": "Muted."}
    r = Session.from_blob(s.to_blob())
    assert r.pending_confirm["action"] == "volume.mute"


def test_session_blob_never_contains_utterance_text():
    s = Session(session_id="s3")
    s.pending_intent = "reminders.task.create"

    def all_keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from all_keys(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from all_keys(v)

    keys = set(all_keys(s.to_blob()))
    assert not keys & {"text", "utterance", "message"}, keys


def test_blob_survives_engine_restart(engine):
    """Persist mid-flow, restore into a FRESH store, continue the conversation."""
    engine.reset("blob1")
    r1 = engine.handle("blob1", "set a reminder")
    assert r1.type == "PROMPT"
    blob = engine.sessions.get("blob1").to_blob()

    engine.sessions.reset("blob1")           # simulated process death
    restored = Session.from_blob(blob)
    engine.sessions._sessions["blob1"] = restored
    r2 = engine.handle("blob1", "take my medication")
    assert r2.type == "PROMPT", "restored session must continue the slot flow"


# -- §1 turn_id / §4 notify_execution ----------------------------------------

def test_turn_id_accepted_and_execution_feedback_recorded(engine):
    engine.reset("t1")
    r = engine.handle("t1", "mute", turn_id="turn-0001")
    assert r.type == "FULFILL"
    engine.notify_execution("t1", "turn-0001", "Success")
    s = engine.sessions.get("t1")
    assert s.last_execution == {"turn_id": "turn-0001", "outcome": "Success",
                                "detail": {}}


# -- §5 availability snapshot -------------------------------------------------

def test_unavailable_capability_never_fires_action(engine):
    engine.push_availability({"snapshot_id": 1, "capabilities": {
        "device.volume": {"state": "unavailable",
                          "unavailable_response":
                              "I can't reach your hearing aids right now."}}})
    try:
        engine.reset("av1")
        r = engine.handle("av1", "mute")
        assert r.action is None, "unavailable capability must not fire an action"
        assert "can't reach" in r.message
        assert r.intent == "device.volume.mute", "intent stays recognized (A6)"
    finally:
        engine.push_availability({"snapshot_id": 2, "capabilities": {}})


def test_stale_snapshot_dropped(engine):
    engine.push_availability({"snapshot_id": 10, "capabilities": {}})
    engine.push_availability({"snapshot_id": 3, "capabilities": {
        "device.volume": {"state": "unavailable"}}})
    assert engine._availability["snapshot_id"] == 10
    engine.reset("av2")
    r = engine.handle("av2", "mute")
    assert r.action == "volume.mute", "stale unavailability must not apply"


# -- §2 backend injection ------------------------------------------------------

def test_inference_backend_is_injectable(engine):
    from nlu_engine.inference import InferenceBackend

    assert isinstance(engine.classifier.backend, InferenceBackend)

    class CannedBackend:
        def __init__(self, labels):
            self._labels = list(labels)

        def tfidf_logits(self, text):
            import numpy as np
            z = np.full(len(self._labels), -5.0)
            z[self._labels.index("device.status.battery")] = 10.0
            return z

        def embed_tokens(self, *a):
            raise NotImplementedError

    from nlu_engine.classifier import IntentClassifier

    clf = IntentClassifier(
        labels_path=REPO_ROOT / "models" / "intent" / "en" / "labels.pkl",
        backend=CannedBackend(engine.classifier.labels))
    intent, conf = clf.classify("zzz nonsense that matches no keyword rule zzz")
    assert intent == "device.status.battery" and conf > 0.9
