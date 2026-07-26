"""Bundle Compiler stages 11–15 — packaging, integrity, signing (ADR-005 §5).

Turns a VALIDATED unpacked bundle directory into a signed `.nlu` archive:

  11  optimization        canonical JSON (sorted keys, no float drift), NFC text
  13  integrity           integrity/manifest.sha256 over every entry;
                          checksums_root written back into bundle.json
  14  deterministic zip   fixed timestamps, sorted entries, no compression
                          extras → byte-identical rebuilds from identical inputs
  15  signing             Ed25519 over (manifest.sha256 ⊕ bundle.json)

Stage 12 (codegen side-outputs) emits Swift/Kotlin action-key constants
next to the archive.

SIGNING KEYS: this module signs with the committed DEV key
(spec/keys/dev/) whose key id is `dev-key-golden`. Per ADR-005 Part 11 a
production runtime categorically refuses dev-signed artifacts
(channel + signing-key id). Production signing (KMS/HSM, pinned rotation)
is ND-8 — approval-gated, NOT this code path.

Usage:
    PYTHONPATH=packages/buildtime python -m nlu_compiler.build \\
        spec/examples/3.0/full --out bundles/full.nlu
    PYTHONPATH=packages/buildtime python -m nlu_compiler.verify bundles/full.nlu
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
import zipfile
from pathlib import Path

from .diagnostics import Severity
from .validator import validate_bundle_dir

REPO = Path(__file__).resolve().parents[3]
DEV_KEY_DIR = REPO / "spec" / "keys" / "dev"
DEV_KEY_ID = "dev-key-golden"

# Deterministic packaging constants (stage 14).
FIXED_DATE_TIME = (2026, 1, 1, 0, 0, 0)


def canonical_json(data) -> bytes:
    """Stage 11: one canonical serialization — sorted keys, minimal separators,
    NFC-normalized strings, no NaN."""
    def norm(obj):
        if isinstance(obj, str):
            return unicodedata.normalize("NFC", obj)
        if isinstance(obj, dict):
            return {norm(k): norm(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [norm(v) for v in obj]
        return obj
    return json.dumps(norm(data), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def load_dev_key():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding, NoEncryption, PrivateFormat, PublicFormat, load_pem_private_key)

    pem = DEV_KEY_DIR / "dev_signing_key.pem"
    if pem.exists():
        return load_pem_private_key(pem.read_bytes(), password=None)
    DEV_KEY_DIR.mkdir(parents=True, exist_ok=True)
    key = Ed25519PrivateKey.generate()
    pem.write_bytes(key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()))
    (DEV_KEY_DIR / "dev_signing_key.pub").write_bytes(
        key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo))
    return key


def build_bundle(src: Path, out: Path, skip_validation: bool = False, *, key_id: str = DEV_KEY_ID, channel: str | None = None) -> dict:
    """Stages 11 + 13 + 14 + 15. Returns a build report."""
    src = Path(src)
    if not skip_validation:
        errors = [d for d in validate_bundle_dir(src) if d.severity is Severity.ERROR]
        if errors:
            raise ValueError("bundle fails validation; refusing to package:\n" +
                             "\n".join(str(e) for e in errors[:5]))

    # Stage 11: canonicalize every JSON file into an in-memory tree.
    entries: dict[str, bytes] = {}
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(src).as_posix()
        if rel.startswith("integrity/"):
            continue  # regenerated below, never trusted from the source dir
        if f.suffix == ".json":
            entries[rel] = canonical_json(json.loads(f.read_text(encoding="utf-8"))) + b"\n"
        else:
            entries[rel] = f.read_bytes()

    manifest = json.loads(entries["bundle.json"])

    # Stage 13: hash table over every entry except bundle.json + integrity/.
    lines = []
    for rel in sorted(entries):
        if rel == "bundle.json":
            continue
        digest = hashlib.sha256(entries[rel]).hexdigest()
        lines.append(f"{digest}  {rel}")
    sha_table = ("\n".join(lines) + "\n").encode("utf-8")

    manifest["checksums_root"] = hashlib.sha256(sha_table).hexdigest()
    # Key id and channel are PARAMETERS, not constants. The ND-8 cutover to
    # production signing must be a settings change in the release workflow, not
    # a code change here — so the workflow passes them in and this module never
    # needs to learn about KMS. Defaults keep the dev-key behaviour.
    if channel:
        manifest["channel"] = channel
    manifest["signature_info"] = {"scheme": "ed25519-v1", "key_id": key_id}
    manifest_bytes = canonical_json(manifest) + b"\n"
    entries["bundle.json"] = manifest_bytes
    entries["integrity/manifest.sha256"] = sha_table

    # Stage 15: signature over (manifest.sha256 ⊕ bundle.json).
    key = load_dev_key()
    signature = key.sign(sha_table + manifest_bytes)
    entries["integrity/signature.sig"] = signature

    # Stage 14: deterministic archive.
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(entries):
            info = zipfile.ZipInfo(rel, date_time=FIXED_DATE_TIME)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, entries[rel])

    return {"bundle_id": manifest["bundle_id"], "files": len(entries),
            "bytes": out.stat().st_size,
            "checksums_root": manifest["checksums_root"],
            "key_id": key_id, "channel": manifest.get("channel"),
            "out": str(out)}


def codegen_action_constants(src: Path, out_dir: Path) -> list[str]:
    """Stage 12 side-outputs: Swift + Kotlin action-key constants."""
    src, out_dir = Path(src), Path(out_dir)
    actions: list[tuple[str, str]] = []
    for cap_file in sorted(src.glob("capabilities/*/capability.json")):
        cap = json.loads(cap_file.read_text())
        for a in cap["actions"]:
            const = a["key"].replace(".", "_").upper()
            actions.append((const, a["key"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    swift = ["// Generated by nlu_compiler stage 12 — do not edit.",
             "public enum NLUActionKey {"]
    kotlin = ["// Generated by nlu_compiler stage 12 — do not edit.",
              "object NLUActionKey {"]
    for const, key in actions:
        swift.append(f'    public static let {const} = "{key}"')
        kotlin.append(f'    const val {const} = "{key}"')
    swift.append("}")
    kotlin.append("}")
    (out_dir / "NLUActionKey.swift").write_text("\n".join(swift) + "\n")
    (out_dir / "NLUActionKey.kt").write_text("\n".join(kotlin) + "\n")
    return [f"{out_dir}/NLUActionKey.swift", f"{out_dir}/NLUActionKey.kt"]


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="nlu_compiler.build", description=__doc__)
    ap.add_argument("src", type=Path, help="validated unpacked bundle directory")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--codegen-dir", type=Path, default=None)
    ap.add_argument("--key-id", default=DEV_KEY_ID,
                    help="signing key id recorded in the manifest "
                         f"(default: {DEV_KEY_ID}; production keys arrive with ND-8)")
    ap.add_argument("--channel", default=None, choices=[None, "dev", "beta", "production"],
                    help="override the manifest channel; production runtimes "
                         "refuse dev-signed artifacts, which is the intended gate")
    args = ap.parse_args(argv)

    out = args.out or (REPO / "bundles" / f"{args.src.name}.nlu")
    report = build_bundle(args.src, out, key_id=args.key_id, channel=args.channel)
    print(json.dumps(report, indent=2))
    codegen = codegen_action_constants(args.src, args.codegen_dir
                                       or out.parent / f"{args.src.name}-codegen")
    for f in codegen:
        print("codegen:", f)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
