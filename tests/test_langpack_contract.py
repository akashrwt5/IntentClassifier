"""
Language Pack contract (charter A6).

`nlu_langpack` is the locked boundary between the language-agnostic engine and
everything language-specific. The container is a single-language
`spec/bundle/3.0` bundle — ADR-005 Part 11 already declares a per-language
bundle a packaging profile of that format, not a new one — so these tests run
against the real golden bundles rather than a fixture of their own.
"""

import copy
import importlib
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME = str(_ROOT / "packages" / "runtime")
if _RUNTIME not in sys.path:
    sys.path.insert(0, _RUNTIME)

lp = importlib.import_module("nlu_langpack")

MINIMAL = _ROOT / "spec" / "examples" / "3.0" / "minimal"
FULL = _ROOT / "spec" / "examples" / "3.0" / "full"


def _bundle(tmp_path, src, **manifest_overrides):
    """Copy a golden bundle and patch its manifest."""
    dst = tmp_path / src.name
    import shutil
    shutil.copytree(src, dst)
    data = json.loads((dst / "bundle.json").read_text(encoding="utf-8"))
    data.update(copy.deepcopy(manifest_overrides))
    (dst / "bundle.json").write_text(json.dumps(data), encoding="utf-8")
    return dst


# --------------------------------------------------------------------------- #
# Loading a real bundle
# --------------------------------------------------------------------------- #

def test_minimal_golden_bundle_loads():
    pack = lp.load_pack(MINIMAL)
    assert pack.language == "en"
    assert pack.channel == "dev"
    # Real tables, not placeholders.
    assert pack.resources["lexicon"] and pack.resources["keyword_matcher"]
    assert pack.resources["capabilities"]
    for table in ("cascade", "policies", "plan_facts"):
        assert table in pack.resources, f"runtime table {table} not resolved"
    # `routing` is deliberately not among them — the loader stopped requiring a
    # table nothing read, so that packs which no longer carry one still load.
    # The golden bundle here still HAS the file; not resolving it is the point.
    assert "routing" not in pack.resources


def test_language_must_be_named_when_the_bundle_has_several(tmp_path):
    """`full` declares en and fr, so single_language() is ambiguous.

    Its `frames` requirement is cleared first: the compatibility gate runs
    before language selection (a bundle this runtime cannot run should not be
    inspected further), so it would otherwise mask the error under test.
    """
    src = _bundle(tmp_path, FULL, required_runtime_features=[])
    with pytest.raises(lp.PackManifestError, match="declares 2 languages"):
        lp.load_pack(src)


def test_unknown_language_is_refused(tmp_path):
    src = _bundle(tmp_path, MINIMAL, required_runtime_features=[])
    with pytest.raises(lp.PackLanguageError, match="not among them"):
        lp.load_pack(src, language="zz")


# --------------------------------------------------------------------------- #
# The compatibility gate
# --------------------------------------------------------------------------- #

def test_unimplemented_required_feature_is_refused():
    """The `full` golden bundle requires `frames`, which this runtime lacks.

    Capability-by-declaration: refused at load, not discovered mid-turn.
    """
    with pytest.raises(lp.PackCompatibilityError, match="frames"):
        lp.load_pack(FULL, language="en")


def test_runtime_below_min_contract_is_fatal(tmp_path):
    src = _bundle(tmp_path, MINIMAL, engine_compat={
        "min_runtime_contract": lp.RUNTIME_CONTRACT_VERSION + 1,
        "max_tested_runtime_contract": lp.RUNTIME_CONTRACT_VERSION + 2})
    with pytest.raises(lp.PackCompatibilityError, match="needs runtime contract"):
        lp.load_pack(src)


def test_runtime_above_max_tested_is_an_issue_not_an_error():
    """`max_TESTED_runtime_contract` is advisory, and the name is the reason.

    The bundle declares the newest runtime it was verified against, not a hard
    ceiling. Running newer is unverified, so it is surfaced as an issue — but
    refusing to load would strand every existing bundle on the next contract
    bump. `min_runtime_contract` is the hard floor; this is not.

    Exercised at the gate directly rather than through a manifest, because at
    contract version 1 the state is not yet expressible: max_tested >= min >= 1
    == RUNTIME_CONTRACT_VERSION, so no valid manifest can put the runtime above
    max_tested. This becomes reachable via a bundle at contract version 2.
    """
    issues = lp.check_compatibility(
        min_contract=1,
        max_tested=lp.RUNTIME_CONTRACT_VERSION - 1,
        required_features=frozenset(),
    )
    assert any("tested only up to" in i for i in issues)


def test_below_min_contract_is_fatal_at_the_gate():
    with pytest.raises(lp.PackCompatibilityError, match="needs runtime contract"):
        lp.check_compatibility(
            min_contract=lp.RUNTIME_CONTRACT_VERSION + 1,
            max_tested=lp.RUNTIME_CONTRACT_VERSION + 1,
            required_features=frozenset(),
        )


# --------------------------------------------------------------------------- #
# Channel strictness
# --------------------------------------------------------------------------- #

def test_production_channel_refuses_a_partial_language(tmp_path):
    src = _bundle(tmp_path, MINIMAL, channel="production",
                  languages={"en": {"status": "partial"}})
    with pytest.raises(lp.PackLanguageError, match="production"):
        lp.load_pack(src)


def test_dev_channel_tolerates_a_partial_language(tmp_path):
    src = _bundle(tmp_path, MINIMAL, channel="dev",
                  languages={"en": {"status": "partial"}})
    pack = lp.load_pack(src)
    assert any("partial" in i for i in pack.issues)


def test_production_channel_refuses_a_missing_resource(tmp_path):
    src = _bundle(tmp_path, MINIMAL, channel="production")
    (src / "lexicons" / "en.json").unlink()
    with pytest.raises(lp.PackResourceError, match="lexicon"):
        lp.load_pack(src)


def test_corrupt_table_fails_at_load_not_mid_conversation(tmp_path):
    src = _bundle(tmp_path, MINIMAL)
    (src / "lexicons" / "en.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(lp.PackResourceError, match="not valid JSON"):
        lp.load_pack(src)


# --------------------------------------------------------------------------- #
# Manifest validation
# --------------------------------------------------------------------------- #

def test_missing_bundle_json_is_named_clearly(tmp_path):
    with pytest.raises(lp.PackManifestError, match="no bundle.json"):
        lp.load_pack(tmp_path)


def test_inverted_engine_compat_is_rejected(tmp_path):
    src = _bundle(tmp_path, MINIMAL, engine_compat={
        "min_runtime_contract": 5, "max_tested_runtime_contract": 2})
    with pytest.raises(lp.PackManifestError, match="inverted"):
        lp.load_pack(src)


def test_invalid_channel_is_rejected(tmp_path):
    src = _bundle(tmp_path, MINIMAL, channel="staging")
    with pytest.raises(lp.PackManifestError, match="channel"):
        lp.load_pack(src)


def test_required_field_typo_does_not_pass_silently(tmp_path):
    src = _bundle(tmp_path, MINIMAL)
    data = json.loads((src / "bundle.json").read_text())
    data["bundle_idd"] = data.pop("bundle_id")
    (src / "bundle.json").write_text(json.dumps(data))
    with pytest.raises(lp.PackManifestError, match="bundle_id"):
        lp.load_pack(src)


# --------------------------------------------------------------------------- #
# Semantic stage — opt-in plugin, off by default
# --------------------------------------------------------------------------- #

def test_semantic_is_off_by_default():
    assert lp.load_pack(MINIMAL).semantic_available is False


def test_semantic_stage_absent_from_the_cascade_when_off():
    assert lp.load_pack(MINIMAL).stages == ("keyword", "intent_model")


def test_explicitly_requesting_an_undeclared_semantic_stage_is_fatal(tmp_path):
    """An explicit arg is a stated intent; failing it silently hides a misconfig."""
    src = _bundle(tmp_path, MINIMAL)
    with pytest.raises(lp.PackResourceError, match="declares no semantic model"):
        lp.load_pack(src, enable_semantic=True)


def test_broad_env_switch_does_not_crash_a_bundle_without_the_stage(tmp_path, monkeypatch):
    """A fleet-wide env switch must not break bundles that simply lack a head."""
    monkeypatch.setenv("NLU_ENABLE_SEMANTIC", "1")
    pack = lp.load_pack(_bundle(tmp_path, MINIMAL))
    assert pack.semantic_available is False
    assert any("declares none" in i for i in pack.issues)


def test_arg_beats_env(monkeypatch):
    monkeypatch.setenv("NLU_ENABLE_SEMANTIC", "1")
    assert lp.load_pack(MINIMAL, enable_semantic=False).semantic_available is False


# --------------------------------------------------------------------------- #
# The contract itself
# --------------------------------------------------------------------------- #

def test_contract_version_matches_the_single_anchor():
    """There must be exactly one version axis: runtime-contract-v1.md §7."""
    assert lp.RUNTIME_CONTRACT_VERSION == 1
    assert (_ROOT / "spec" / "contracts" / "runtime-contract-v1.md").exists()


def test_semantic_is_the_only_optional_component():
    assert lp.COMPONENT_NAMES - lp.REQUIRED_COMPONENTS == {"semantic"}


def test_no_second_manifest_format_was_introduced():
    """ADR-005 Part 11: a per-language bundle is a packaging profile of
    spec/bundle/3.0, not a new format. A `packs/` tree with its own pack.json
    would be a second container with its own versioning and no signing."""
    assert not (_ROOT / "packs").exists(), (
        "a parallel packs/ format reappeared — the contract binds to "
        "spec/bundle/3.0 (see nlu_langpack/manifest.py)"
    )


def test_contract_module_imports_nothing_heavy():
    """The contract must stay dependency-free so both sides can import it."""
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); import nlu_langpack; "
            "bad = {'numpy','onnxruntime','torch','sklearn','joblib'} & set(sys.modules); "
            "print(sorted(bad))" % _RUNTIME)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "[]", f"nlu_langpack pulled in {out.stdout.strip()}"
