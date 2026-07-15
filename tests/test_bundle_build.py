"""Compiler stages 11–15: packaging, integrity, dev signing (ADR-005 §5, Part 11).

The §8.1 exit-gate property: one command builds a signed (dev-key) bundle
from a clean checkout, byte-identical from identical inputs; runtimes gate
on exactly signature → hashes → compatibility.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("jsonschema")
pytest.importorskip("cryptography")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "buildtime"))

from nlu_compiler.build import build_bundle, codegen_action_constants  # noqa: E402
from nlu_compiler.verify import BundleVerificationError, verify_bundle  # noqa: E402

FULL = REPO_ROOT / "spec" / "examples" / "3.0" / "full"
MINIMAL = REPO_ROOT / "spec" / "examples" / "3.0" / "minimal"


def test_build_is_byte_identical(tmp_path):
    a, b = tmp_path / "a.nlu", tmp_path / "b.nlu"
    build_bundle(FULL, a)
    build_bundle(FULL, b)
    assert a.read_bytes() == b.read_bytes(), "stage-14 determinism broken"


@pytest.mark.parametrize("src", [MINIMAL, FULL], ids=lambda p: p.name)
def test_build_then_verify_roundtrip(tmp_path, src):
    out = tmp_path / "bundle.nlu"
    report = build_bundle(src, out)
    manifest = verify_bundle(out)
    assert manifest["bundle_id"] == report["bundle_id"]
    assert manifest["signature_info"]["key_id"] == "dev-key-golden"


def test_tampered_content_fails_hash_gate(tmp_path):
    out = tmp_path / "bundle.nlu"
    build_bundle(FULL, out)
    with zipfile.ZipFile(out, "a") as zf:
        zf.writestr("lexicons/en.json", '{"lang":"en","affirmative":["yes"],"negative":["no"]}')
    with pytest.raises(BundleVerificationError):
        verify_bundle(out)


def test_tampered_manifest_fails_signature_gate(tmp_path):
    out = tmp_path / "bundle.nlu"
    build_bundle(FULL, out)
    names = {}
    with zipfile.ZipFile(out) as zf:
        for n in zf.namelist():
            names[n] = zf.read(n)
    names["bundle.json"] = names["bundle.json"].replace(b'"content_version":2',
                                                        b'"content_version":9')
    evil = tmp_path / "evil.nlu"
    with zipfile.ZipFile(evil, "w") as zf:
        for n, data in names.items():
            zf.writestr(n, data)
    with pytest.raises(BundleVerificationError, match="SIGNATURE"):
        verify_bundle(evil)


def test_production_runtime_refuses_dev_channel(tmp_path):
    out = tmp_path / "bundle.nlu"
    build_bundle(FULL, out)
    with pytest.raises(BundleVerificationError, match="production runtime refuses"):
        verify_bundle(out, runtime_channel="production")


def test_missing_required_feature_refused(tmp_path):
    out = tmp_path / "bundle.nlu"
    build_bundle(FULL, out)  # full bundle requires 'frames'
    with pytest.raises(BundleVerificationError, match="required features"):
        verify_bundle(out, runtime_features=set())


def test_invalid_source_refuses_to_package(tmp_path):
    import json
    import shutil

    src = tmp_path / "src"
    shutil.copytree(FULL, src)
    p = src / "runtime" / "policies.json"
    data = json.loads(p.read_text())
    data["confirmation"].pop("audio.volume.mute")
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="fails validation"):
        build_bundle(src, tmp_path / "never.nlu")


def test_codegen_side_outputs(tmp_path):
    files = codegen_action_constants(FULL, tmp_path)
    swift = Path(files[0]).read_text()
    kotlin = Path(files[1]).read_text()
    assert 'DEVICE_AUDIO_SET_VOLUME = "device.audio.set_volume"' in swift
    assert 'const val ASSISTANT_REMINDER_CREATE = "assistant.reminder.create"' in kotlin
