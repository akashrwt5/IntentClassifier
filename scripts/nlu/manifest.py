"""
Model bundle manifest — SHA-256 checksums for all NLU artifacts.

generate_manifest() writes models/manifest.json after training.
verify_manifest()   is called at engine startup; raises if any file changed.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).parent.parent.parent
MANIFEST_PATH = BASE_DIR / "models" / "manifest.json"

# Artifacts that must stay in sync with each other.
TRACKED_FILES = [
    "models/intent_model.onnx",
    "models/intent_labels.json",
    "models/intent_labels.pkl",
    "models/intent_pipeline.pkl",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_manifest(base_dir: Path = BASE_DIR, extra_files: list = None) -> Path:
    """Write models/manifest.json with checksums of all tracked artifacts."""
    files = list(TRACKED_FILES) + (extra_files or [])
    manifest = {}
    for rel in files:
        p = base_dir / rel
        if p.exists():
            manifest[rel] = _sha256(p)
        else:
            print(f"  [manifest] skipped missing: {rel}")
    out = base_dir / "models" / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  [manifest] written → {out} ({len(manifest)} files)")
    return out


def verify_manifest(base_dir: Path = BASE_DIR, manifest_path: Optional[Path] = None) -> None:
    """Raise RuntimeError if any tracked artifact has changed since the manifest was written."""
    mp = manifest_path or (base_dir / "models" / "manifest.json")
    if not mp.exists():
        return  # no manifest yet — first run, skip check
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    mismatches = []
    for rel, expected in manifest.items():
        p = base_dir / rel
        if not p.exists():
            mismatches.append(f"  MISSING  {rel}")
            continue
        actual = _sha256(p)
        if actual != expected:
            mismatches.append(f"  CHANGED  {rel}")
    if mismatches:
        raise RuntimeError(
            "Model bundle integrity check failed — artifacts have changed without "
            "regenerating the manifest. Re-run `python scripts/train.py` to rebuild "
            "and re-sign the bundle.\n" + "\n".join(mismatches)
        )
