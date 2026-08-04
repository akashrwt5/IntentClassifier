"""App-compatibility label shim (boundary-only) — unit coverage.

Verifies that the modern ``domain.object.action`` labels are translated back to
the legacy Dialogflow contract the app still consumes, WITHOUT retraining or
touching the model:

  * simple renames (device.volume.mute -> Cmd.VolumeMute),
  * passthrough for sentinels / unmapped / None,
  * confirmation compound synthesis: a yes/no resolution of a send
    confirmation becomes ``Cmd.SendMessage - yes`` / ``Cmd.SendMessage - no``,
  * the internal confirmation tags never leak into the serialized payload,
  * the kill-switch (NLU_LEGACY_LABELS=0) makes the shim a no-op.

No trained model needed — this is pure boundary logic.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME = _ROOT / "packages" / "runtime"
if str(_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_RUNTIME))

from nlu_engine import label_compat  # noqa: E402
from nlu_engine.engine import NLUResult  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    """Each test sees a clean, ENABLED shim (opt-in, so set the flag; cache
    reset between cases)."""
    monkeypatch.setenv("NLU_LEGACY_LABELS", "1")
    label_compat._load.cache_clear()
    yield
    label_compat._load.cache_clear()


def test_simple_renames():
    assert label_compat.to_app_label("device.volume.mute") == "Cmd.VolumeMute"
    assert label_compat.to_app_label("help.battery.show") == "Help_Battery"
    assert label_compat.to_app_label("sys.oos.fallback") == "Default Fallback Intent"


def test_passthrough():
    assert label_compat.to_app_label("GENAI") == "GENAI"            # sentinel
    assert label_compat.to_app_label("some.new.intent") == "some.new.intent"  # unmapped
    assert label_compat.to_app_label(None) is None


def test_apply_rewrites_all_label_fields():
    r = NLUResult(type="FULFILL", intent="device.volume.mute",
                  interrupted_intent="help.battery.show",
                  tfidf_intent="reminders.task.create")
    label_compat.apply(r)
    assert r.intent == "Cmd.VolumeMute"
    assert r.interrupted_intent == "Help_Battery"
    assert r.tfidf_intent == "reminders.add"


def test_send_confirmation_yes_becomes_compound():
    r = NLUResult(type="FULFILL", intent="messaging.message.send",
                  action="message.compose", complete=True)
    r._confirm_polarity = "yes"
    r._confirmed_intent = "messaging.message.send"
    label_compat.apply(r)
    assert r.intent == "Cmd.SendMessage - yes"


def test_send_confirmation_no_becomes_compound():
    # Engine surfaces sys.confirm.cancelled, but the tags still carry which
    # intent was being confirmed, so the app gets the exact legacy "- no".
    r = NLUResult(type="FULFILL", intent="sys.confirm.cancelled", complete=True)
    r._confirm_polarity = "no"
    r._confirmed_intent = "messaging.message.send"
    label_compat.apply(r)
    assert r.intent == "Cmd.SendMessage - no"


def test_internal_tags_do_not_leak_into_payload():
    r = NLUResult(type="FULFILL", intent="messaging.message.send", complete=True)
    r._confirm_polarity = "yes"
    r._confirmed_intent = "messaging.message.send"
    payload = label_compat.apply(r).to_dict()
    assert payload["intent"] == "Cmd.SendMessage - yes"
    assert "_confirm_polarity" not in payload
    assert "_confirmed_intent" not in payload


def test_every_trained_intent_has_a_legacy_mapping():
    """Completeness guard: every modern intent the model can emit must have a
    legacy label. A new intent added without a mapping (or a rename that breaks
    one) fails here rather than silently reaching the app as a modern string."""
    import json

    labels_path = _ROOT / "dist" / "bundle-en" / "models" / "intent" / "en" / "labels.json"
    if not labels_path.exists():
        pytest.skip("built pack (dist/bundle-en) required")
    trained = set(json.loads(labels_path.read_text(encoding="utf-8")))
    mapped = set(label_compat._load_map())

    missing = trained - mapped
    assert not missing, f"intents with no legacy mapping: {sorted(missing)}"


def test_killswitch_disables_shim(monkeypatch):
    monkeypatch.setenv("NLU_LEGACY_LABELS", "0")
    label_compat._load.cache_clear()
    r = NLUResult(type="FULFILL", intent="device.volume.mute")
    label_compat.apply(r)
    assert r.intent == "device.volume.mute"  # untouched — modern label passes through
