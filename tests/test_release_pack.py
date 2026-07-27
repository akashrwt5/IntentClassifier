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


def _workflow_without_comments() -> str:
    """Workflow text with COMMENT LINES removed, quoted strings left intact.

    The workflow documents the dead paths and schema errors it fixed, so a naive
    substring scan flags its own explanation. Splitting each line on `#` is the
    wrong fix: `echo "### Data grade..."` writes a markdown heading, and cutting
    at that `#` silently deleted the rest of the line — which made a test assert
    the absence of something that was there. Only whole-line comments are
    dropped.
    """
    return "\n".join(
        line for line in _WORKFLOW.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#"))


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


# --------------------------------------------------------------------------- #
# Calibration must travel with the model
# --------------------------------------------------------------------------- #

_FITTED_CALIB = _ROOT / "models" / "intent" / "en" / "calibration.json"


@pytest.mark.skipif(not _FITTED_CALIB.exists(), reason="no fitted calibration")
def test_calibration_is_translated_into_the_bundle_contract(tmp_path):
    """The build artifact and the in-bundle artifact are different shapes.

    `nlu_training.fit_calibration` writes a rich record (fit provenance, excluded
    eval sets, fitter identity). The bundle form
    (spec/bundle/3.0/calibration.schema.json) is the lean on-device contract:
    `additionalProperties: false` and `conf_threshold` required. Copying the build
    file in raw fails stage-1 validation, so assemble_pack translates it — this
    test is what stops someone "simplifying" that back into a copy.
    """
    nlu = _build(tmp_path, calibration=_FITTED_CALIB)
    packed = json.loads(zipfile.ZipFile(nlu).read("models/intent/en/calibration.json"))
    fitted = json.loads(_FITTED_CALIB.read_text(encoding="utf-8"))

    assert packed["temperature"] == fitted["temperature"], (
        "the pack must carry the temperature actually fitted for this model")
    assert packed["method"] == "temperature_scaling"
    # The fire threshold ships alongside: a runtime holding one without the other
    # cannot reproduce a confidence gate.
    schema = json.loads((_ROOT / "language_packs" / "en" / "nlu_schema.json").read_text(encoding="utf-8"))
    assert packed["conf_threshold"] == schema["confidence_threshold"]
    # `fitted_on` is the leakage audit — it is what makes a stale T detectable.
    assert packed["fitted_on"] == fitted["provenance"]["source_sha256"]
    assert set(packed) <= {"temperature", "conf_threshold", "method",
                           "ece_raw", "ece_calibrated", "fitted_on"}, (
        "extra keys break the bundle schema's additionalProperties: false")


@pytest.mark.skipif(not _FITTED_CALIB.exists(), reason="no fitted calibration")
def test_a_pack_without_calibration_would_run_uncalibrated(tmp_path):
    """Guard against silently shipping a pack with no temperature.

    A consumer that finds no calibration.json falls back to T = 1.0 (plain
    softmax), which mis-tunes the fire threshold, the confirm band and slot
    acceptance at once — blocker B8 in a new place. The minimal golden bundle
    ships its own placeholder, so what this asserts is that passing --calibration
    REPLACES it rather than leaving the placeholder in place.
    """
    placeholder = json.loads(
        (_MINIMAL / "models" / "intent" / "en" / "calibration.json")
        .read_text(encoding="utf-8"))
    fitted = json.loads(_FITTED_CALIB.read_text(encoding="utf-8"))
    assert placeholder["temperature"] != fitted["temperature"], (
        "fixture drifted: the golden placeholder now equals the fitted value, so "
        "this test can no longer tell them apart")
    nlu = _build(tmp_path, calibration=_FITTED_CALIB)
    packed = json.loads(zipfile.ZipFile(nlu).read("models/intent/en/calibration.json"))
    assert packed["temperature"] == fitted["temperature"]


def test_workflow_ships_calibration_and_uses_per_language_paths(workflow):
    """The paths the relayout retired must not come back.

    `dl/models/intent_model.onnx`, `dl/models/intent_labels.pkl` and
    `dl/models/intent_classifier_weights.json` were flat legacy locations. After
    the per-language relayout they cannot exist, so assemble_pack aborted on
    "artifact not found" and the CoreML staging step failed before it reached the
    exporter.
    """
    text = _WORKFLOW.read_text(encoding="utf-8")
    live = _workflow_without_comments()
    for stale in ("dl/models/intent_model.onnx",
                  "dl/models/intent_labels.pkl",
                  "dl/models/intent_classifier_weights.json"):
        assert stale not in live, f"stale pre-relayout path still referenced: {stale}"
    assert "--calibration" in text, (
        "the release must package the fitted temperature with the model")
    assert "fit_calibration" in text, (
        "T must be refit for the model trained in this run, not inherited")
    # The exporter has no --out flag; passing one aborts the step.
    assert "export_coreml --out" not in text


# --------------------------------------------------------------------------- #
# Bundle-schema limits the release job hit at stage 1
# --------------------------------------------------------------------------- #

def test_coreml_is_refused_rather_than_packaged_invalidly(tmp_path):
    """A second model format is not expressible in spec/bundle/3.0.

    `models` is a closed set of STAGES (intent/embedder/semantic_head) and
    `modelLangMap` allows exactly one artifact per language, so writing
    `models.coreml.<lang>` fails stage-1 validation, and the files inside a
    `.mlpackage` DIRECTORY have no schema mapping either. A release run reached
    the compiler and died on all three at once. Refusing loudly at the CLI beats
    building a bundle the validator will reject three steps later.
    """
    fake = tmp_path / "IntentClassifier.mlpackage"
    (fake / "Data").mkdir(parents=True)
    (fake / "Manifest.json").write_text("{}", encoding="utf-8")
    rc = assemble_pack.assemble(_MINIMAL, "1.2.3", tmp_path / "out", coreml=fake)
    assert rc != 0, "assemble_pack packaged CoreML into a bundle the spec forbids"


def test_models_schema_really_forbids_a_coreml_stage():
    """Guards the reason the test above exists.

    If the spec later gains a way to carry two formats, this fails and the
    refusal should be revisited rather than left in place forever.
    """
    schema = json.loads((_ROOT / "spec" / "bundle" / "3.0" / "bundle.schema.json")
                        .read_text(encoding="utf-8"))
    models = schema["properties"]["models"]
    assert models["additionalProperties"] is False
    assert "coreml" not in models["properties"], (
        "spec/bundle/3.0 now allows a coreml stage — revisit assemble_pack's "
        "refusal and the release workflow instead of keeping the workaround")
    lang_map = schema["$defs"]["modelLangMap"]["patternProperties"]["^([a-z]{2}|shared)$"]
    assert lang_map["additionalProperties"] is False


def test_report_card_is_not_decorated_with_extra_keys(workflow):
    """report_card.schema.json is additionalProperties:false over six keys.

    The workflow used to inject `data_grade` into it, which passed the accuracy
    gate and then failed stage-1 validation in the release job — three steps
    later, in a different job. The grade now goes to the run summary.
    """
    allowed = set(json.loads(
        (_ROOT / "spec" / "bundle" / "3.0" / "report_card.schema.json")
        .read_text(encoding="utf-8"))["properties"])
    assert "data_grade" not in allowed
    live = _workflow_without_comments()
    assert 'd["data_grade"]' not in live, (
        "the workflow writes data_grade into report_card.json again — the bundle "
        "schema rejects it")
    assert "GITHUB_STEP_SUMMARY" in live, "the data grade must still be reported"


def test_release_job_does_not_pass_coreml_to_the_packer(workflow):
    live = _workflow_without_comments()
    assert "--coreml" not in live, (
        "the release job passes --coreml again; assemble_pack refuses it and the "
        "bundle would fail stage-1 validation")


# --------------------------------------------------------------------------- #
# labels.json must describe the model that ships beside it
# --------------------------------------------------------------------------- #

_TRAINED_LABELS = _ROOT / "models" / "intent" / "en" / "labels.pkl"


@pytest.mark.skipif(not _TRAINED_LABELS.exists(), reason="no trained labels")
def test_labels_json_is_derived_from_the_pickle_not_inherited(tmp_path):
    """The published pack-en-v1.0.0 shipped a 57-class ONNX beside a 2-entry
    labels.json — the golden fixture's placeholder
    (["audio.volume.mute", "audio.volume.set"], still in the superseded `audio.*`
    naming), because only labels.pkl was refreshed. iOS reads labels.json to map
    output indices, so every prediction would have been mislabelled.

    Assembly now DERIVES labels.json from labels.pkl. Against a golden fixture
    that means stage 8 rejects the pack (the fixture declares 2 intents, the model
    has 57) — which is the correct outcome: the mismatch is real and must fail
    loudly instead of shipping. That rejection is what this asserts.
    """
    out = tmp_path / "out"
    rc = assemble_pack.assemble(_MINIMAL, "1.0.0", out, labels=_TRAINED_LABELS)
    assert rc != 0, (
        "assembly succeeded with a 57-label model against a 2-intent source "
        "bundle — labels.json is being inherited from the fixture again")
    assert not list(out.glob("*.nlu")), "a mismatched pack must not be produced"


def test_the_golden_fixtures_are_not_this_products_content():
    """Pins WHY a real release needs a content->bundle compiler.

    `SRC_BUNDLE` in release-pack.yml is spec/examples/3.0/minimal, a test fixture
    with one capability. The product has 12 capabilities and 57 intents under
    content/. A pack built from the fixture proves the PIPELINE, not the product.
    If someone wires a real content bundle, this test should fail and be replaced.
    """
    schema = json.loads((_ROOT / "language_packs" / "en" / "nlu_schema.json").read_text(encoding="utf-8"))
    fixture_caps = list((_MINIMAL / "capabilities").glob("*/capability.json"))
    content_caps = [d for d in (_ROOT / "content" / "capabilities").iterdir() if d.is_dir()]
    assert len(schema["intents"]) == 57
    assert len(fixture_caps) < len(content_caps), (
        "the golden fixture now has as many capabilities as content/ — if a real "
        "content->bundle source exists, point SRC_BUNDLE at it and drop this test")
