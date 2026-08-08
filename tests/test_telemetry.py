"""Telemetry event generation + aggregation (Phase 3).

The two properties that matter: raw text CANNOT appear in any event (by
construction — the redaction guarantee lives in one audited place), and
aggregation yields bounded counters keyed on bundle_id.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "runtime"))

from nlu_engine.telemetry import (  # noqa: E402
    TelemetryAggregator, TurnEvent, confidence_bucket, domain_of)


def test_turn_event_has_no_text_field():
    fields = {f.name for f in dataclasses.fields(TurnEvent)}
    assert "text" not in fields and "utterance" not in fields and "message" not in fields
    # and the schema is closed: unknown kwargs are TypeErrors
    with pytest.raises(TypeError):
        TurnEvent(bundle_id="b", stage="tfidf", result_type="FULFILL",
                  intent_domain="device", confidence_bucket="0.9-1.0",
                  text="leak")  # type: ignore[call-arg]


def test_closed_enums_enforced():
    with pytest.raises(AssertionError):
        TurnEvent(bundle_id="b", stage="llm", result_type="FULFILL",
                  intent_domain="device", confidence_bucket="0.9-1.0")
    with pytest.raises(AssertionError):
        TurnEvent(bundle_id="b", stage="tfidf", result_type="ACTION",
                  intent_domain="device", confidence_bucket="0.9-1.0")


def test_aggregation_counts_and_flush(tmp_path):
    agg = TelemetryAggregator(bundle_id="golden-full-3.0-000001")
    for _ in range(3):
        agg.record(TurnEvent(bundle_id=agg.bundle_id, stage="tfidf",
                             result_type="FULFILL", intent_domain="device",
                             confidence_bucket=confidence_bucket(0.93)))
    agg.record(TurnEvent(bundle_id=agg.bundle_id, stage="uncertain_confirm",
                         result_type="CONFIRM", intent_domain="device",
                         confidence_bucket=confidence_bucket(0.62), guarded=True))
    agg.record_security("signature_invalid")

    snap = agg.snapshot()
    assert snap["bundle_id"] == "golden-full-3.0-000001"
    assert sum(v for k, v in snap["counters"].items() if "tfidf|FULFILL" in k) == 3
    assert any("security|signature_invalid" in k for k in snap["counters"])
    assert all(isinstance(v, int) for v in snap["counters"].values())

    out = tmp_path / "telemetry.json"
    agg.flush_to(out)
    assert out.exists() and agg.counters == {}


def test_domain_and_bucket_coarseness():
    assert domain_of("Cmd.VolumeMute") == "cmd"
    # The separator that matters: Help_* has no dot, and these topics are
    # health-adjacent. A guard that returns the whole label here is the
    # disclosure it exists to prevent.
    assert domain_of("Help_Tinnitus") == "help"
    assert domain_of("Help_FallAlert") == "help"
    assert domain_of("Default Fallback Intent") == "default"
    assert domain_of("GENAI") == "genai"
    assert domain_of(None) == "none"
    assert confidence_bucket(1.0) == "0.9-1.0"
    assert confidence_bucket(0.0) == "0.0-0.1"
