"""App-compatibility label shim (boundary-only) — unit coverage.

Verifies that the modern ``domain.object.action`` labels are translated back to
the legacy Dialogflow contract the app still consumes, WITHOUT retraining or
touching the model:

  * simple renames (Cmd.VolumeMute -> Cmd.VolumeMute),
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
    assert label_compat.to_app_label("Cmd.VolumeMute") == "Cmd.VolumeMute"
    assert label_compat.to_app_label("Help_Battery") == "Help_Battery"
    assert label_compat.to_app_label("Default Fallback Intent") == "Default Fallback Intent"


def test_passthrough():
    assert label_compat.to_app_label("GENAI") == "GENAI"            # sentinel
    assert label_compat.to_app_label("some.new.intent") == "some.new.intent"  # unmapped
    assert label_compat.to_app_label(None) is None


def test_apply_rewrites_all_label_fields():
    r = NLUResult(type="FULFILL", intent="Cmd.VolumeMute",
                  interrupted_intent="Help_Battery",
                  tfidf_intent="reminders.add")
    label_compat.apply(r)
    assert r.intent == "Cmd.VolumeMute"
    assert r.interrupted_intent == "Help_Battery"
    assert r.tfidf_intent == "reminders.add"


def test_send_confirmation_yes_becomes_compound():
    r = NLUResult(type="FULFILL", intent="Cmd.SendMessage",
                  action="message.compose", complete=True)
    r._confirm_polarity = "yes"
    r._confirmed_intent = "Cmd.SendMessage"
    label_compat.apply(r)
    assert r.intent == "Cmd.SendMessage - yes"


def test_send_confirmation_no_becomes_compound():
    # Engine surfaces sys.confirm.cancelled, but the tags still carry which
    # intent was being confirmed, so the app gets the exact legacy "- no".
    r = NLUResult(type="FULFILL", intent="sys.confirm.cancelled", complete=True)
    r._confirm_polarity = "no"
    r._confirmed_intent = "Cmd.SendMessage"
    label_compat.apply(r)
    assert r.intent == "Cmd.SendMessage - no"


def test_internal_tags_do_not_leak_into_payload():
    r = NLUResult(type="FULFILL", intent="Cmd.SendMessage", complete=True)
    r._confirm_polarity = "yes"
    r._confirmed_intent = "Cmd.SendMessage"
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
    r = NLUResult(type="FULFILL", intent="Cmd.VolumeMute")
    label_compat.apply(r)
    assert r.intent == "Cmd.VolumeMute"  # untouched — modern label passes through


# --------------------------------------------------------------------------- #
# The published conformance vector
# --------------------------------------------------------------------------- #

_FIXTURES = _ROOT / "tests" / "fixtures" / "legacy_label_parity_en.csv"


@pytest.mark.skipif(not _FIXTURES.exists(), reason="conformance vector absent")
def test_the_engine_still_produces_the_published_conformance_vector(monkeypatch):
    """The CSV iOS and Android are told to reproduce must match this engine.

    Nothing read this file before — only `scripts/gen_legacy_label_fixtures.py`
    wrote it. That is how it came to assert `CONFIRM` for four volume
    utterances: the generator replays the engine and records whatever comes out,
    so a defect was captured as the cross-platform contract (ADR-011, parity by
    fixtures) with no test to notice the divergence afterwards.

    A generated golden still needs a consumer, or the engine and the contract
    drift apart in silence — which is precisely the failure mode this whole
    change set exists to remove.

    If this fails after an intentional behaviour change: regenerate with
    `python scripts/gen_legacy_label_fixtures.py`, and TELL THE CLIENT TEAMS.
    The file is their contract, not an internal snapshot.
    """
    import csv
    import warnings

    pytest.importorskip("onnxruntime")
    monkeypatch.setenv("NLU_LEGACY_LABELS", "1")
    label_compat._load.cache_clear()

    pack_dir = _ROOT / "dist" / "bundle-en"
    if not pack_dir.exists():
        pytest.skip("built pack (dist/bundle-en) required")

    from nlu_engine import NLUEngine
    from nlu_langpack import load_pack

    with _FIXTURES.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eng = NLUEngine(pack=load_pack(str(pack_dir)))
        seen = set()
        mismatches = []
        for row in rows:
            sid = row["session"]
            if sid not in seen:
                eng.reset(sid)
                seen.add(sid)
            r = eng.handle(sid, row["text"])
            got = (r.type, r.intent or "")
            want = (row["expected_type"], row["expected_intent"])
            if got != want:
                mismatches.append(f"  {row['text']!r}: expected {want}, got {got}")
        assert not mismatches, (
            "the engine no longer reproduces the published conformance vector:\n"
            + "\n".join(mismatches))
    finally:
        label_compat._load.cache_clear()
