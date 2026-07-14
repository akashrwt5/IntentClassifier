"""Unified evaluate report (roadmap §9.2) — shape and schema conformance.

The evaluate JSON must validate against the bundle report-card schema so the
CI gate artifact is byte-compatible with what the compiler ingests (stage 13).
Skips gracefully when model artifacts or runtime deps are absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema", reason="dev dependency (make install-dev)")
pytest.importorskip("sklearn", reason="runtime dependency")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "buildtime"))

PIPELINE = REPO_ROOT / "multilingual" / "models" / "en" / "en_intent_pipeline.pkl"


@pytest.fixture(scope="module")
def report():
    if not PIPELINE.exists():
        pytest.skip("trained artifacts not present (make train-multilingual)")
    from nlu_training import evaluate_all

    return evaluate_all(["en"])


def test_report_validates_against_report_card_schema(report):
    from referencing import Registry, Resource

    schema_dir = REPO_ROOT / "spec" / "bundle" / "3.0"
    resources = []
    for f in schema_dir.glob("*.schema.json"):
        s = json.loads(f.read_text())
        resources.append((s["$id"], Resource.from_contents(s)))
        resources.append((f.name, Resource.from_contents(s)))
    schema = json.loads((schema_dir / "report_card.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(
        schema, registry=Registry().with_resources(resources))
    errors = list(validator.iter_errors(report))
    assert not errors, [e.message for e in errors[:3]]


def test_report_matches_recorded_baseline(report):
    """The en baseline recorded in label-migration-plan.md: acc .900, F1 .89."""
    en = report["per_language"]["en"]
    assert abs(en["holdout_accuracy"] - 0.900) <= 0.01
    assert abs(en["macro_f1"] - 0.89) <= 0.01
    assert 0 < en["ece"] < 0.05, "temperature scaling should keep ECE small"


def test_gate_logic(report):
    assert report["gates_passed"] is True  # en is above the 0.80 floor
    assert report["wrong_action_count"] >= 0
    assert 0.0 <= report["oos_recall"] <= 1.0
