"""
Release-pack assembly (charter A10).

`scripts/ci/assemble_pack.py` turns an unpacked `spec/bundle/3.0` tree plus
freshly trained artifacts into a versioned, signed single-language `.nlu` — a
Language Pack in the only sense this repo recognises (ADR-005 Part 11).

The property that matters most here is the ND-8 one: the signing key id and the
channel are PARAMETERS, so moving from dev keys to production keys is a change
to the release workflow's inputs and not a change to any code.
"""

import importlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_MINIMAL = _ROOT / "spec" / "examples" / "3.0" / "minimal"
_WORKFLOW = _ROOT / ".github" / "workflows" / "release-pack.yml"

sys.path.insert(0, str(_ROOT / "packages" / "runtime"))
_assemble = importlib.import_module("importlib.util").spec_from_file_location(
    "assemble_pack", _ROOT / "scripts" / "ci" / "assemble_pack.py")
assemble_pack = importlib.util.module_from_spec(_assemble)
_assemble.loader.exec_module(assemble_pack)


def _build(tmp_path, **kw):
    rc = assemble_pack.assemble(_MINIMAL, kw.pop("version", "1.2.3"), tmp_path, **kw)
    assert rc == 0, "assemble_pack failed"
    return next(tmp_path.glob("pack-*.nlu"))


def _manifest(nlu: Path) -> dict:
    return json.loads(zipfile.ZipFile(nlu).read("bundle.json"))


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def test_produces_a_versioned_single_language_pack(tmp_path):
    nlu = _build(tmp_path)
    assert nlu.name == "pack-en-v1.2.3.nlu"
    m = _manifest(nlu)
    assert m["bundle_id"] == "pack-en-v1.2.3"
    assert list(m["languages"]) == ["en"], "a release pack declares exactly one language"
    assert m["format_version"] == "3.0", "it is the same format, not a new one"


def test_narrows_a_multi_language_bundle_to_one(tmp_path):
    """The golden `full` bundle has en + fr; a pack must carry only the one asked for."""
    rc = assemble_pack.assemble(_ROOT / "spec" / "examples" / "3.0" / "full",
                                "2.0.0", tmp_path, language="en")
    # `full` requires an unimplemented runtime feature, but assembly is about
    # packaging, not loading — narrowing must still work.
    assert rc == 0
    m = _manifest(next(tmp_path.glob("pack-*.nlu")))
    assert list(m["languages"]) == ["en"]


def test_ambiguous_language_is_refused(tmp_path):
    """Silently picking one of several languages would be a shipping hazard."""
    rc = assemble_pack.assemble(_ROOT / "spec" / "examples" / "3.0" / "full",
                                "2.0.0", tmp_path)
    assert rc == 1


def test_non_semver_version_is_refused(tmp_path):
    assert assemble_pack.assemble(_MINIMAL, "v1", tmp_path) == 1
    assert assemble_pack.assemble(_MINIMAL, "latest", tmp_path) == 1


def test_missing_artifact_is_refused_not_skipped(tmp_path):
    """A release that silently omits the model would be worse than a failure."""
    assert assemble_pack.assemble(_MINIMAL, "1.0.0", tmp_path,
                                  model=tmp_path / "nope.onnx") == 1


# --------------------------------------------------------------------------- #
# Signing — the ND-8 property
# --------------------------------------------------------------------------- #

def test_defaults_to_the_dev_key_and_dev_channel(tmp_path):
    m = _manifest(_build(tmp_path))
    assert m["channel"] == "dev"
    assert m["signature_info"]["key_id"] == "dev-key-golden"


def test_key_id_and_channel_are_parameters_not_constants(tmp_path):
    """The ND-8 cutover must be a settings change, not a code change.

    If this passes, swapping to production signing means passing different
    workflow inputs — nothing in the compiler or the assembler has to move.
    """
    nlu = _build(tmp_path, key_id="kms-prod-2026-01", channel="production")
    m = _manifest(nlu)
    assert m["signature_info"]["key_id"] == "kms-prod-2026-01"
    assert m["channel"] == "production"


def test_the_signed_pack_verifies(tmp_path):
    nlu = _build(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "nlu_compiler.verify", str(nlu)],
        cwd=_ROOT, capture_output=True, text=True,
        env={**__import__("os").environ,
             "PYTHONPATH": str(_ROOT / "packages" / "buildtime")})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "VERIFIED" in proc.stdout


def test_the_pack_loads_through_the_language_pack_contract(tmp_path):
    """End to end: assemble -> sign -> verify -> load. The whole point."""
    nlu = _build(tmp_path)
    unpacked = tmp_path / "unpacked"
    zipfile.ZipFile(nlu).extractall(unpacked)

    lp = importlib.import_module("nlu_langpack")
    pack = lp.load_pack(unpacked)
    assert pack.language == "en"
    assert pack.channel == "dev"


# --------------------------------------------------------------------------- #
# The workflow itself
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))


def test_workflow_is_valid_yaml_with_the_three_jobs(workflow):
    assert set(workflow["jobs"]) == {"train-gate", "coreml-export", "release"}


def test_workflow_does_not_release_from_main(workflow):
    """Releasing from the default branch is an owner decision tied to ND-8."""
    # PyYAML parses the `on:` key as the boolean True.
    triggers = workflow.get("on") or workflow.get(True)
    branches = triggers["push"]["branches"]
    assert "main" not in branches, "release-pack must not trigger on main"


def test_workflow_exposes_key_id_and_channel_as_inputs(workflow):
    triggers = workflow.get("on") or workflow.get(True)
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert "key_id" in inputs and "channel" in inputs, (
        "the ND-8 cutover must be reachable without editing the workflow"
    )
    assert inputs["channel"]["default"] == "dev"


def test_workflow_verifies_before_publishing(workflow):
    """Publishing an unverified artifact would defeat the signing pipeline."""
    steps = workflow["jobs"]["release"]["steps"]
    names = [s.get("name", "") for s in steps]
    verify = next(i for i, n in enumerate(names) if "Verify" in n)
    publish = next(i for i, n in enumerate(names) if "Publish" in n)
    assert verify < publish, "verification must precede publication"
