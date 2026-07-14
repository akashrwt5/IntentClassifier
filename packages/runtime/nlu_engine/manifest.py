"""
Model bundle manifest — SHA-256 checksums for all NLU artifacts.

generate_manifest() writes models/manifest.json after training.
verify_manifest()   is called at engine startup; raises if any file changed.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parents[3]
MANIFEST_PATH = BASE_DIR / "models" / "manifest.json"

# Artifacts that must stay in sync with each other. Covers BOTH the TF-IDF
# stage and the MiniLM semantic stage — a corrupted or swapped semantic model
# is as dangerous as a corrupted intent model and must be caught at startup.
TRACKED_FILES = [
    # Stage 2 — TF-IDF + LogReg
    "models/intent_model.onnx",
    "models/intent_labels.json",
    "models/intent_labels.pkl",
    "models/intent_pipeline.pkl",
    # Stage 3 — MiniLM semantic rescue
    "models/minilm-l6-v2.onnx",
    "models/minilm-vocab.txt",
    "models/semantic_head.json",
    "models/semantic_head.npz",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_manifest(base_dir: Path = BASE_DIR, extra_files: list = None,
                      meta: dict = None) -> Path:
    """
    Write models/manifest.json with checksums of all tracked artifacts.

    Optional `meta` (e.g. {"holdout_accuracy": 0.91}) is stored under the
    reserved "_meta" key so every shipped bundle carries its own report card.
    Reserved keys start with "_" and are ignored by verify_manifest.
    """
    files = list(TRACKED_FILES) + (extra_files or [])
    manifest = {}
    for rel in files:
        p = base_dir / rel
        if p.exists():
            manifest[rel] = _sha256(p)
        else:
            print(f"  [manifest] skipped missing: {rel}")
    if meta:
        manifest["_meta"] = meta
    out = base_dir / "models" / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    n_files = sum(1 for k in manifest if not k.startswith("_"))
    print(f"  [manifest] written → {out} ({n_files} files)")
    return out


def verify_manifest(base_dir: Path = BASE_DIR, manifest_path: Optional[Path] = None) -> None:
    """Raise RuntimeError if any tracked artifact has changed since the manifest was written."""
    mp = manifest_path or (base_dir / "models" / "manifest.json")
    if not mp.exists():
        return  # no manifest yet — first run, skip check
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    mismatches = []
    for rel, expected in manifest.items():
        if rel.startswith("_"):
            continue  # reserved metadata key, not a tracked file
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
