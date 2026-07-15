"""Runtime-side bundle verification (ADR-005 Parts 7/11) — the three gates.

Verifies a `.nlu` archive exactly the way a device runtime must:
signature → hashes → compatibility. Nothing else is re-validated
(content semantics were proved at compile time — Part 6 entitlement).

Usage:
    PYTHONPATH=packages/buildtime python -m nlu_compiler.verify bundles/full.nlu \\
        [--runtime-contract 1] [--features frames plans] [--channel dev]
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEV_PUB = REPO / "spec" / "keys" / "dev" / "dev_signing_key.pub"
SUPPORTED_FORMAT_MAJOR = 3


class BundleVerificationError(Exception):
    pass


def verify_bundle(path: Path, runtime_contract: int = 1,
                  runtime_features: set[str] | None = None,
                  runtime_channel: str = "dev") -> dict:
    """Return the trusted manifest, or raise BundleVerificationError."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.serialization import load_pem_public_key

    runtime_features = runtime_features if runtime_features is not None else {
        "frames", "plans", "conditional_steps"}

    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        for required in ("bundle.json", "integrity/manifest.sha256",
                         "integrity/signature.sig"):
            if required not in names:
                raise BundleVerificationError(f"missing {required}")
        manifest_bytes = zf.read("bundle.json")
        sha_table = zf.read("integrity/manifest.sha256")
        signature = zf.read("integrity/signature.sig")

        # Gate 1 — signature over (manifest.sha256 ⊕ bundle.json).
        pub = load_pem_public_key(DEV_PUB.read_bytes())
        try:
            pub.verify(signature, sha_table + manifest_bytes)
        except InvalidSignature as exc:
            raise BundleVerificationError(
                "SIGNATURE INVALID — security event, not corruption") from exc

        manifest = json.loads(manifest_bytes)
        if hashlib.sha256(sha_table).hexdigest() != manifest["checksums_root"]:
            raise BundleVerificationError("checksums_root does not bind the hash table")

        # Gate 2 — every listed file hashes clean; no unlisted files.
        listed: dict[str, str] = {}
        for line in sha_table.decode().splitlines():
            digest, rel = line.split("  ", 1)
            listed[rel] = digest
        for rel, digest in listed.items():
            if hashlib.sha256(zf.read(rel)).hexdigest() != digest:
                raise BundleVerificationError(f"hash mismatch: {rel}")
        unlisted = names - set(listed) - {"bundle.json", "integrity/manifest.sha256",
                                          "integrity/signature.sig"}
        if unlisted:
            raise BundleVerificationError(f"unlisted files in archive: {sorted(unlisted)}")

    # Gate 3 — compatibility (format major, contract range, features, channel).
    major = int(manifest["format_version"].split(".")[0])
    if major > SUPPORTED_FORMAT_MAJOR:
        raise BundleVerificationError(f"format major {major} too new")
    ec = manifest["engine_compat"]
    if runtime_contract < ec["min_runtime_contract"]:
        raise BundleVerificationError("runtime contract below bundle minimum")
    missing = set(manifest["required_runtime_features"]) - runtime_features
    if missing:
        raise BundleVerificationError(f"runtime lacks required features: {sorted(missing)}")
    if runtime_channel == "production" and manifest["channel"] != "production":
        raise BundleVerificationError(
            f"production runtime refuses channel '{manifest['channel']}' "
            f"(key id '{manifest['signature_info']['key_id']}')")
    return manifest


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="nlu_compiler.verify", description=__doc__)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--runtime-contract", type=int, default=1)
    ap.add_argument("--features", nargs="*", default=None)
    ap.add_argument("--channel", default="dev")
    args = ap.parse_args(argv)
    try:
        m = verify_bundle(args.bundle, args.runtime_contract,
                          set(args.features) if args.features is not None else None,
                          args.channel)
    except BundleVerificationError as exc:
        print(f"REFUSED: {exc}")
        return 1
    print(f"VERIFIED: {m['bundle_id']} format={m['format_version']} "
          f"channel={m['channel']} key={m['signature_info']['key_id']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
