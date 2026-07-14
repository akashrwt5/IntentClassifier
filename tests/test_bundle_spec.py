"""Spec conformance — ADR-005 AI#2/#6 (format 3.0).

Proves the three-way agreement the ADR demands: the JSON Schemas are valid,
every golden example bundle validates against them, and basic cross-artifact
parity (labels ≡ workflow intents ≡ plan_facts; responses cover referenced
keys) holds inside each golden bundle. This is the seed of the shared
validator library (compiler stages 1–3, 8, 9).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema", reason="dev dependency (make install-dev)")

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = REPO_ROOT / "spec" / "bundle" / "3.0"
EXAMPLES = sorted((REPO_ROOT / "spec" / "examples" / "3.0").iterdir())

# Archive-relative path pattern -> schema file (the normative mapping).
FILE_SCHEMA_MAP = [
    (r"^bundle\.json$", "bundle.schema.json"),
    (r"^models/intent/[a-z]{2}/labels\.json$", "labels.schema.json"),
    (r"^models/intent/[a-z]{2}/calibration\.json$", "calibration.schema.json"),
    (r"^models/semantic_head/[a-z]{2}/head\.json$", "semantic_head.schema.json"),
    (r"^runtime/cascade\.json$", "cascade.schema.json"),
    (r"^runtime/policies\.json$", "policies.schema.json"),
    (r"^runtime/routing\.json$", "routing.schema.json"),
    (r"^runtime/plan_facts\.json$", "plan_facts.schema.json"),
    (r"^capabilities/[^/]+/capability\.json$", "capability.schema.json"),
    (r"^capabilities/[^/]+/workflows\.json$", "workflows.schema.json"),
    (r"^capabilities/[^/]+/entities\.json$", "entities.schema.json"),
    (r"^capabilities/[^/]+/responses/[a-z]{2}\.json$", "responses.schema.json"),
    (r"^entities/shared/[^/]+\.json$", "entities.schema.json"),
    (r"^lexicons/[a-z]{2}\.json$", "lexicons.schema.json"),
    (r"^keywords/[a-z]{2}\.json$", "keywords.schema.json"),
    (r"^telemetry/schema\.json$", "telemetry.schema.json"),
    (r"^meta/report_card\.json$", "report_card.schema.json"),
    (r"^meta/lineage\.json$", "lineage.schema.json"),
]


def _registry():
    from referencing import Registry, Resource

    resources = []
    for f in SCHEMA_DIR.glob("*.schema.json"):
        schema = json.loads(f.read_text())
        resources.append((schema["$id"], Resource.from_contents(schema)))
        resources.append((f.name, Resource.from_contents(schema)))
    return Registry().with_resources(resources)


def _validator(schema_name: str):
    schema = json.loads((SCHEMA_DIR / schema_name).read_text())
    return jsonschema.Draft202012Validator(schema, registry=_registry())


def _bundle_files(bundle_dir: Path):
    return sorted(p.relative_to(bundle_dir).as_posix() for p in bundle_dir.rglob("*.json"))


def test_schemas_are_valid_draft_2020_12():
    schemas = list(SCHEMA_DIR.glob("*.schema.json"))
    assert len(schemas) >= 16, "expected the full 3.0 schema set"
    for f in schemas:
        jsonschema.Draft202012Validator.check_schema(json.loads(f.read_text()))


def test_every_golden_file_has_a_schema_and_validates():
    assert len(EXAMPLES) == 2, "ADR-005 AI#2: one minimal, one full-featured golden bundle"
    for bundle in EXAMPLES:
        for rel in _bundle_files(bundle):
            matches = [s for pat, s in FILE_SCHEMA_MAP if re.match(pat, rel)]
            assert len(matches) == 1, f"{bundle.name}/{rel}: no unique schema mapping"
            instance = json.loads((bundle / rel).read_text())
            errors = list(_validator(matches[0]).iter_errors(instance))
            assert not errors, f"{bundle.name}/{rel}: " + "; ".join(
                e.message for e in errors[:3])


@pytest.mark.parametrize("bundle", EXAMPLES, ids=lambda b: b.name)
def test_cross_artifact_parity(bundle: Path):
    """Stage-8 seed checks: labels ≡ compiled intents ≡ plan_facts."""
    manifest = json.loads((bundle / "bundle.json").read_text())

    workflow_intents: set[str] = set()
    for wf in bundle.glob("capabilities/*/workflows.json"):
        workflow_intents |= set(json.loads(wf.read_text())["intents"])

    for labels_file in bundle.glob("models/intent/*/labels.json"):
        labels = json.loads(labels_file.read_text())
        assert set(labels) == workflow_intents, f"{labels_file}: labels ≠ intent set"

    plan = json.loads((bundle / "runtime" / "plan_facts.json").read_text())
    assert set(plan["intents"]) == workflow_intents

    policies = json.loads((bundle / "runtime" / "policies.json").read_text())
    assert set(policies["confirmation"]) == workflow_intents, "confirmation matrix incomplete"

    # manifest capability registry ≡ capability dirs
    cap_dirs = {p.name for p in (bundle / "capabilities").iterdir() if p.is_dir()}
    assert cap_dirs == set(manifest["capabilities"])

    # plan_facts capabilities exist
    assert {v["capability"] for v in plan["intents"].values()} <= cap_dirs


@pytest.mark.parametrize("bundle", EXAMPLES, ids=lambda b: b.name)
def test_localization_completeness(bundle: Path):
    """Stage-9 seed: every referenced response key exists in every declared language."""
    manifest = json.loads((bundle / "bundle.json").read_text())
    langs = set(manifest["languages"])
    for cap_dir in (bundle / "capabilities").iterdir():
        cap = json.loads((cap_dir / "capability.json").read_text())
        cap_langs = set(cap["languages"]) & langs
        referenced: set[str] = set()
        wf = json.loads((cap_dir / "workflows.json").read_text())
        for intent in wf["intents"].values():
            for slot in intent.get("slots", []):
                referenced.add(slot["prompt"])
                referenced.update(v["failure_prompt"] for v in slot.get("validation", [])
                                  if "failure_prompt" in v)
                if "reprompt" in slot:
                    referenced.add(slot["reprompt"])
            if "confirmation" in intent:
                referenced.add(intent["confirmation"]["prompt"])
            referenced.add(intent["completion"]["response"])
            fu = intent.get("followup")
            if fu:
                referenced.update(v for k, v in fu.items() if k != "on_yes")
        avail = cap.get("availability", {})
        if "unavailable_response" in avail:
            referenced.add(avail["unavailable_response"])
        for lang in cap_langs:
            keys = set(json.loads((cap_dir / "responses" / f"{lang}.json").read_text()))
            missing = referenced - keys
            assert not missing, f"{cap['id']}/{lang}: missing response keys {sorted(missing)}"


def test_portable_regex_corpus_agrees_with_python_re():
    """The normative corpus of spec/bundle/portable-regex.md, row by row."""
    corpus = [
        (r"^mute$", ["mute"], ["unmute", "mute please"]),
        (r"\bvolume\b", ["the volume up", "volume"], ["voluminous"]),
        (r"set (?:an? )?alarm", ["set an alarm", "set alarm"], []),
        (r"[0-9]{1,2}:[0-9]{2}", ["8:30", "12:05"], ["8-30"]),
        (r"louder|quieter", ["louder", "quieter"], ["loud"]),
        (r"remind(er)?s?\b", ["remind", "reminders"], ["remainder"]),
        (r"\d+ ?(minutes?|mins?)\b", ["10 minutes", "5 min"], []),
        (r"[^a-z]", ["A", "9"], ["a"]),
    ]
    for pattern, must, must_not in corpus:
        rx = re.compile(pattern)
        for s in must:
            assert rx.search(s), f"{pattern!r} should match {s!r}"
        for s in must_not:
            assert not rx.search(s), f"{pattern!r} should NOT match {s!r}"


def test_golden_patterns_stay_inside_portable_subset():
    """No forbidden constructs in any pattern shipped in the golden bundles."""
    forbidden = re.compile(
        r"\\[1-9]|\(\?[=!<P#i]|\*\?|\+\?|\?\?|\{\d+,?\d*\}\?|\\p\{")
    for bundle in EXAMPLES:
        for kw in bundle.glob("keywords/*.json"):
            data = json.loads(kw.read_text())
            for rule in data["rules"]:
                for pat in [rule["pattern"], *rule.get("guards", [])]:
                    assert not forbidden.search(pat), f"{kw}: {pat!r} leaves the subset"
                    re.compile(pat)  # and it must at least compile
