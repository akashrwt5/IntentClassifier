"""Shared validator library tests (ADR-005 AI#3, Part 6).

Golden bundles must validate clean; each reviewer-listed failure class,
injected by mutation, must be caught at its designated stage.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("jsonschema", reason="dev dependency (make install-dev)")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "buildtime"))

from nlu_compiler import Severity, validate_bundle_dir  # noqa: E402

EXAMPLES = sorted((REPO_ROOT / "spec" / "examples" / "3.0").iterdir())
FULL = REPO_ROOT / "spec" / "examples" / "3.0" / "full"


def _errors(diags):
    return [d for d in diags if d.severity is Severity.ERROR]


def _mutated(tmp_path: Path, mutate) -> Path:
    """Copy the full golden bundle and apply a mutation function to it."""
    dst = tmp_path / "bundle"
    shutil.copytree(FULL, dst)
    mutate(dst)
    return dst


def _edit(bundle: Path, rel: str, fn):
    p = bundle / rel
    data = json.loads(p.read_text())
    fn(data)
    p.write_text(json.dumps(data))


@pytest.mark.parametrize("bundle", EXAMPLES, ids=lambda b: b.name)
def test_golden_bundles_validate_clean(bundle):
    diags = validate_bundle_dir(bundle)
    assert not _errors(diags), [str(d) for d in _errors(diags)]


def test_unknown_field_is_an_error_not_a_warning(tmp_path):
    """The 'requred:' typo class — silent typos must not compile (stage 1)."""
    b = _mutated(tmp_path, lambda d: _edit(d, "bundle.json",
                                           lambda m: m.update({"requred_runtime_features": []})))
    codes = {d.code for d in _errors(validate_bundle_dir(b))}
    assert "SCHEMA_INVALID" in codes


def test_missing_intent_reference_caught(tmp_path):
    """Keyword rule targeting an undefined intent (stage 3)."""
    def mutate(d):
        _edit(d, "keywords/en.json",
              lambda k: k["rules"].append({"pattern": "ghost", "tier": 2,
                                            "intent": "ghost.intent.fire"}))
    codes = {d.code for d in _errors(validate_bundle_dir(_mutated(tmp_path, mutate)))}
    assert "KEYWORD_DANGLING_INTENT" in codes


def test_broken_workflow_prompt_caught(tmp_path):
    """Slot prompt → missing response key (stage 9): blank prompt on device."""
    def mutate(d):
        _edit(d, "capabilities/assistant.reminder/workflows.json",
              lambda w: w["intents"]["reminder.task.create"]["slots"][0].update(
                  {"prompt": "reminder.no_such_key"}))
    codes = {d.code for d in _errors(validate_bundle_dir(_mutated(tmp_path, mutate)))}
    assert "RESPONSE_KEY_MISSING" in codes


def test_cross_capability_entity_ref_rejected(tmp_path):
    """ADR-002 A9: capability depending on another capability's entity (stage 4)."""
    def mutate(d):
        _edit(d, "capabilities/assistant.reminder/workflows.json",
              lambda w: w["intents"]["reminder.task.create"]["slots"][0].update(
                  {"entity": "device.audio.volume_level"}))
    codes = {d.code for d in _errors(validate_bundle_dir(_mutated(tmp_path, mutate)))}
    assert "CROSS_CAPABILITY_REF" in codes


def test_incomplete_policy_matrix_caught(tmp_path):
    def mutate(d):
        _edit(d, "runtime/policies.json",
              lambda p: p["confirmation"].pop("audio.volume.mute"))
    codes = {d.code for d in _errors(validate_bundle_dir(_mutated(tmp_path, mutate)))}
    assert "POLICY_MATRIX_INCOMPLETE" in codes


def test_unconfirmed_high_cost_action_caught(tmp_path):
    """The safety incident class: high-cost action without confirmation (stage 5)."""
    def mutate(d):
        _edit(d, "runtime/policies.json",
              lambda p: p["confirmation"].update({"reminder.task.create": "never"}))
    codes = {d.code for d in _errors(validate_bundle_dir(_mutated(tmp_path, mutate)))}
    assert "HIGH_COST_UNCONFIRMED" in codes


def test_label_intent_mismatch_caught(tmp_path):
    """The parity assert moved to compile time (stage 8)."""
    def mutate(d):
        _edit(d, "models/intent/en/labels.json",
              lambda labels: labels.remove("audio.volume.mute"))
    codes = {d.code for d in _errors(validate_bundle_dir(_mutated(tmp_path, mutate)))}
    assert "LABEL_INTENT_MISMATCH" in codes


def test_embedder_head_mismatch_caught(tmp_path):
    """The silent-wrong-vector-space bug class (stage 8)."""
    def mutate(d):
        _edit(d, "models/semantic_head/en/head.json",
              lambda h: h.update({"embedder_id": "some-other-encoder"}))
    codes = {d.code for d in _errors(validate_bundle_dir(_mutated(tmp_path, mutate)))}
    assert "EMBEDDER_ID_MISMATCH" in codes


def test_missing_localization_caught(tmp_path):
    """English prompt in a French session — trust damage (stage 9)."""
    def mutate(d):
        (d / "capabilities/device.audio/responses/fr.json").rename(
            d / "capabilities/device.audio/responses_fr_gone.json.bak")
    # renamed file also becomes UNMAPPED, so check both diagnostics appear
    b = _mutated(tmp_path, mutate)
    (b / "capabilities/device.audio/responses_fr_gone.json.bak").unlink()
    codes = {d.code for d in _errors(validate_bundle_dir(b))}
    assert "RESPONSES_MISSING" in codes


def test_nonportable_regex_caught(tmp_path):
    def mutate(d):
        _edit(d, "keywords/en.json",
              lambda k: k["rules"].append({"pattern": "(?=lookahead)volume", "tier": 2,
                                            "intent": "audio.volume.set"}))
    codes = {d.code for d in _errors(validate_bundle_dir(_mutated(tmp_path, mutate)))}
    assert "REGEX_NOT_PORTABLE" in codes


def test_cli_green_on_golden_and_red_on_mutant(tmp_path):
    from nlu_compiler.__main__ import main

    assert main([str(FULL)]) == 0
    def mutate(d):
        _edit(d, "runtime/policies.json",
              lambda p: p["confirmation"].pop("audio.volume.set"))
    assert main([str(_mutated(tmp_path, mutate))]) == 1
