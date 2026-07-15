"""Two-slot bundle lifecycle manager (ADR-001.1 §9.2, ADR-005 Parts 7 & 9).

Reference implementation of the device-side lifecycle the iOS and Android
BundleManagers must mirror. Slots:

    baked            shipped in the app binary — the floor that always exists
    active           what the assistant runs right now
    last_known_good  the previously active, fully-verified bundle (one kept)
    candidate        a download being verified/warmed; never user-visible

Normative sequences implemented (Part 7):

    STARTUP     select active → verify → (corrupt? → last-known-good → baked)
    OTA UPDATE  download → candidate → full verify+gate → warm/smoke →
                atomic activate (pointer swap) → old active becomes LKG
    ROLLBACK    repoint to last-known-good (already verified; instant)
    PARTIAL     any candidate failure discards the candidate; the
    FAILURE     user-visible assistant NEVER changes until a fully healthy
                candidate exists
    DOWNGRADE   content_version ≤ active refused unless the instruction is
                an authenticated rollback directive (Part 11)

Verification is delegated to :mod:`nlu_compiler.verify` semantics —
duplicated here in runtime-local form so the RUNTIME package never imports
buildtime code (package-boundary rule). Signature/hash/compat only.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

SUPPORTED_FORMAT_MAJOR = 3
STATE_FILE = "slots.json"


class BundleError(Exception):
    """Verification or lifecycle failure. `security_event` marks signature
    failures — telemetry must report them distinctly from corruption."""

    def __init__(self, message: str, security_event: bool = False):
        super().__init__(message)
        self.security_event = security_event


class BundleManager:
    def __init__(self, store_dir: Path, trusted_pubkeys: list[bytes],
                 runtime_contract: int = 1,
                 runtime_features: frozenset = frozenset({"frames", "plans",
                                                          "conditional_steps"}),
                 channel: str = "dev",
                 smoke_suite=None):
        """`trusted_pubkeys`: PEM public keys pinned by the host app (two pins
        for rotation — Part 11). `smoke_suite`: callable(bundle_path) -> bool,
        the fixed-utterance check run after warm (host-provided)."""
        self.store = Path(store_dir)
        self.store.mkdir(parents=True, exist_ok=True)
        self.pubkeys = trusted_pubkeys
        self.runtime_contract = runtime_contract
        self.runtime_features = frozenset(runtime_features)
        self.channel = channel
        self.smoke_suite = smoke_suite or (lambda path: True)
        self._state = self._load_state()

    # -- state ---------------------------------------------------------------

    def _load_state(self) -> dict:
        p = self.store / STATE_FILE
        if p.exists():
            return json.loads(p.read_text())
        return {"active": None, "last_known_good": None, "baked": None}

    def _save_state(self) -> None:
        (self.store / STATE_FILE).write_text(json.dumps(self._state, indent=2) + "\n")

    def slot(self, name: str) -> Path | None:
        v = self._state.get(name)
        return self.store / v if v else None

    # -- verification (the exactly-three gates) ------------------------------

    def verify(self, path: Path) -> dict:
        """Signature → hashes → compatibility. ANY parse/decompress failure is
        corruption (BundleError), never an unhandled crash — a device must
        always be able to fall back."""
        try:
            return self._verify(path)
        except BundleError:
            raise
        except Exception as exc:  # zlib errors, truncated zips, bad JSON, ...
            raise BundleError(f"corrupt archive: {type(exc).__name__}: {exc}") from exc

    def _verify(self, path: Path) -> dict:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.serialization import load_pem_public_key

        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            for req in ("bundle.json", "integrity/manifest.sha256",
                        "integrity/signature.sig"):
                if req not in names:
                    raise BundleError(f"missing {req}")
            manifest_bytes = zf.read("bundle.json")
            sha_table = zf.read("integrity/manifest.sha256")
            signature = zf.read("integrity/signature.sig")

            for pem in self.pubkeys:  # any pinned key may verify (rotation window)
                try:
                    load_pem_public_key(pem).verify(signature, sha_table + manifest_bytes)
                    break
                except InvalidSignature:
                    continue
            else:
                raise BundleError("signature invalid under all pinned keys",
                                  security_event=True)

            manifest = json.loads(manifest_bytes)
            if hashlib.sha256(sha_table).hexdigest() != manifest["checksums_root"]:
                raise BundleError("checksums_root mismatch", security_event=True)
            listed = {}
            for line in sha_table.decode().splitlines():
                digest, rel = line.split("  ", 1)
                listed[rel] = digest
            for rel, digest in listed.items():
                if hashlib.sha256(zf.read(rel)).hexdigest() != digest:
                    raise BundleError(f"hash mismatch: {rel}")
            extra = names - set(listed) - {"bundle.json", "integrity/manifest.sha256",
                                           "integrity/signature.sig"}
            if extra:
                raise BundleError(f"unlisted files: {sorted(extra)}")

        major = int(manifest["format_version"].split(".")[0])
        if major > SUPPORTED_FORMAT_MAJOR:
            raise BundleError(f"format major {major} unsupported")
        if self.runtime_contract < manifest["engine_compat"]["min_runtime_contract"]:
            raise BundleError("runtime contract too old for bundle")
        missing = set(manifest["required_runtime_features"]) - self.runtime_features
        if missing:
            raise BundleError(f"missing runtime features: {sorted(missing)}")
        if self.channel == "production" and manifest["channel"] != "production":
            raise BundleError(f"production runtime refuses '{manifest['channel']}' channel")
        return manifest

    # -- lifecycle -----------------------------------------------------------

    def install_baked(self, bundle: Path) -> dict:
        """Register the app-binary bundle (the floor). Verified like any other."""
        manifest = self.verify(bundle)
        dest = f"baked-{manifest['bundle_id']}.nlu"
        shutil.copy2(bundle, self.store / dest)
        self._state["baked"] = dest
        if self._state["active"] is None:
            self._state["active"] = dest
        self._save_state()
        return manifest

    def active_manifest(self) -> dict | None:
        p = self.slot("active")
        if not p or not p.exists():
            return None
        with zipfile.ZipFile(p) as zf:
            return json.loads(zf.read("bundle.json"))

    def apply_update(self, downloaded: Path, rollback_directive: bool = False) -> dict:
        """OTA path: verify+gate candidate, warm/smoke, atomic swap.

        Raises on any failure — and in EVERY failure mode the active slot is
        untouched (the Part-7 invariant). Returns the new active manifest."""
        manifest = self.verify(downloaded)  # discard-on-fail: nothing persisted yet

        active = self.active_manifest()
        if active and not rollback_directive:
            if manifest["content_version"] <= active["content_version"]:
                raise BundleError(
                    f"downgrade refused: candidate content_version "
                    f"{manifest['content_version']} ≤ active "
                    f"{active['content_version']} (no authenticated directive)")

        candidate = self.store / f"candidate-{manifest['bundle_id']}.nlu"
        shutil.copy2(downloaded, candidate)
        try:
            self.verify(candidate)          # re-verify what we actually stored
            if not self.smoke_suite(candidate):
                raise BundleError("smoke suite failed on this device")
        except Exception:
            candidate.unlink(missing_ok=True)  # discard candidate; active untouched
            raise

        new_name = f"active-{manifest['bundle_id']}.nlu"
        (candidate).rename(self.store / new_name)
        prev = self._state["active"]
        self._state["active"] = new_name
        if prev and prev != self._state["baked"]:
            self._state["last_known_good"] = prev
        self._save_state()                  # the atomic pointer swap
        return manifest

    def rollback(self) -> dict:
        """Health trigger / remote directive: last-known-good, else baked."""
        for slot in ("last_known_good", "baked"):
            p = self.slot(slot)
            if p and p.exists():
                try:
                    manifest = self.verify(p)
                except BundleError:
                    continue
                self._state["active"] = self._state[slot]
                self._save_state()
                return manifest
        raise BundleError("no verified bundle available in any slot")

    def startup(self) -> dict:
        """Boot-time selection with corruption fallback (STARTUP DISASTER path)."""
        p = self.slot("active")
        if p and p.exists():
            try:
                return self.verify(p)
            except BundleError:
                pass  # fall through to recovery
        return self.rollback()
