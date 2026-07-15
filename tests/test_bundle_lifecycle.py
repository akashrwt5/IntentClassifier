"""Two-slot lifecycle tests (ADR-005 Parts 7/9/11) against dev-signed bundles.

The invariant under test everywhere: the user-visible assistant NEVER
changes until a fully healthy candidate exists.
"""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("jsonschema")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "runtime"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "buildtime"))

from nlu_compiler.build import build_bundle  # noqa: E402  (test fixture only)
from nlu_engine.bundle_manager import BundleError, BundleManager  # noqa: E402

DEV_PUB = REPO_ROOT / "spec" / "keys" / "dev" / "dev_signing_key.pub"
MINIMAL = REPO_ROOT / "spec" / "examples" / "3.0" / "minimal"
FULL = REPO_ROOT / "spec" / "examples" / "3.0" / "full"


@pytest.fixture()
def bundles(tmp_path):
    """v1 (minimal, content_version 1) and v2 (full, content_version 2)."""
    v1, v2 = tmp_path / "v1.nlu", tmp_path / "v2.nlu"
    build_bundle(MINIMAL, v1)
    build_bundle(FULL, v2)
    return v1, v2


@pytest.fixture()
def manager(tmp_path):
    return BundleManager(tmp_path / "store", [DEV_PUB.read_bytes()],
                         runtime_features=frozenset({"frames", "plans",
                                                     "conditional_steps"}))


def test_baked_then_ota_then_lkg(manager, bundles):
    v1, v2 = bundles
    manager.install_baked(v1)
    assert manager.active_manifest()["content_version"] == 1
    m = manager.apply_update(v2)
    assert m["content_version"] == 2
    assert manager.active_manifest()["bundle_id"] == m["bundle_id"]
    # previous active is retained as... baked was active, so LKG stays empty
    # (baked is never demoted to LKG; it is its own slot)
    assert manager.slot("baked").exists()


def test_downgrade_refused_without_directive(manager, bundles):
    v1, v2 = bundles
    manager.install_baked(v1)
    manager.apply_update(v2)
    with pytest.raises(BundleError, match="downgrade refused"):
        manager.apply_update(v1)
    # ...but an authenticated rollback directive may name an older bundle
    m = manager.apply_update(v1, rollback_directive=True)
    assert m["content_version"] == 1


def test_corrupt_download_never_touches_active(manager, bundles, tmp_path):
    v1, v2 = bundles
    manager.install_baked(v1)
    active_before = manager.active_manifest()["bundle_id"]
    corrupt = tmp_path / "corrupt.nlu"
    data = bytearray(v2.read_bytes())
    data[len(data) // 2] ^= 0xFF
    corrupt.write_bytes(bytes(data))
    with pytest.raises(BundleError, match="corrupt archive"):
        manager.apply_update(corrupt)
    assert manager.active_manifest()["bundle_id"] == active_before


def test_tampered_bundle_is_security_event(manager, bundles, tmp_path):
    v1, v2 = bundles
    manager.install_baked(v1)
    names = {}
    with zipfile.ZipFile(v2) as zf:
        for n in zf.namelist():
            names[n] = zf.read(n)
    names["bundle.json"] = names["bundle.json"].replace(b'"channel":"dev"',
                                                        b'"channel":"pro"')
    evil = tmp_path / "evil.nlu"
    with zipfile.ZipFile(evil, "w") as zf:
        for n, data in names.items():
            zf.writestr(n, data)
    with pytest.raises(BundleError) as exc:
        manager.apply_update(evil)
    assert exc.value.security_event, "signature failure must be a security event"


def test_smoke_failure_discards_candidate(tmp_path, bundles):
    v1, v2 = bundles
    mgr = BundleManager(tmp_path / "store", [DEV_PUB.read_bytes()],
                        runtime_features=frozenset({"frames", "plans",
                                                    "conditional_steps"}),
                        smoke_suite=lambda p: False)  # device says no
    mgr.install_baked(v1)
    with pytest.raises(BundleError, match="smoke suite"):
        mgr.apply_update(v2)
    assert mgr.active_manifest()["content_version"] == 1
    assert not list((tmp_path / "store").glob("candidate-*"))


def test_rollback_to_last_known_good_then_baked(manager, bundles, tmp_path):
    v1, v2 = bundles
    manager.install_baked(v1)
    manager.apply_update(v2)
    m = manager.rollback()          # LKG empty (prev active was baked) → baked
    assert m["content_version"] == 1


def test_startup_disaster_falls_back_to_baked(manager, bundles):
    v1, v2 = bundles
    manager.install_baked(v1)
    manager.apply_update(v2)
    # corrupt the active slot at rest (storage corruption)
    active = manager.slot("active")
    data = bytearray(active.read_bytes())
    data[100] ^= 0xFF
    active.write_bytes(bytes(data))
    m = manager.startup()
    assert m["content_version"] == 1, "must degrade to a verified slot, never to none"


def test_missing_feature_bundle_refused_cleanly(tmp_path, bundles):
    v1, v2 = bundles  # v2 (full) requires the 'frames' feature
    mgr = BundleManager(tmp_path / "store", [DEV_PUB.read_bytes()],
                        runtime_features=frozenset())  # runtime has no features
    mgr.install_baked(v1)  # minimal requires none
    with pytest.raises(BundleError, match="missing runtime features"):
        mgr.apply_update(v2)


def test_production_channel_refuses_dev_bundles(tmp_path, bundles):
    v1, _ = bundles
    mgr = BundleManager(tmp_path / "store", [DEV_PUB.read_bytes()],
                        channel="production")
    with pytest.raises(BundleError, match="refuses 'dev'"):
        mgr.install_baked(v1)


def test_unpinned_key_rejected(tmp_path, bundles):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    v1, _ = bundles
    other = Ed25519PrivateKey.generate().public_key().public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    mgr = BundleManager(tmp_path / "store", [other])  # pins a DIFFERENT key
    with pytest.raises(BundleError) as exc:
        mgr.install_baked(v1)
    assert exc.value.security_event
