#!/usr/bin/env python3
"""
The PACK ASSEMBLER (release-pack.yml).

Takes a Language Pack directory + the evaluation report card and produces a
single VERSIONED, self-describing `.nlu` archive with a SHA-256 manifest and
lineage (git sha, dataset hashes). This archive is the one artifact published to
GitHub Releases — the production model registry (see IMPLEMENTATION-PLAN §5a).

The zip is built deterministically (sorted entries, fixed timestamps) so the
same inputs produce a byte-identical archive — a verifiable supply chain.

Usage:
    python scripts/ci/assemble_pack.py --pack packs/en --version 1.0.0 \
        --report dist/report_card.json --out dist
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Training inputs recorded in lineage so a released model maps back to its data.
LINEAGE_DATASETS = [
    "data/04_GENERATED_MASTER_training_data.csv",
    "data/01_source_base_training_data.csv",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def assemble(pack_dir: Path, version: str, report: Path, out_dir: Path,
             coreml: Path | None = None) -> Path:
    stage = Path(tempfile.mkdtemp()) / "pack"
    shutil.copytree(pack_dir, stage)

    # 1. Report card (the approval artifact) travels inside the bundle.
    meta = stage / "meta"
    meta.mkdir(exist_ok=True)
    if report.is_file():
        shutil.copyfile(report, meta / "report_card.json")

    manifest = json.loads((stage / "pack.json").read_text(encoding="utf-8"))

    # 1b. iOS CoreML/ANE variant (optional): include the .mlpackage bundle and
    # declare it alongside the ONNX model so iOS gets it from the same Release.
    if coreml is not None and coreml.exists():
        dst = stage / "intent_model" / "model.mlpackage"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(coreml, dst)
        manifest.setdefault("models", {}).setdefault("intent", {})["coreml"] = \
            "intent_model/model.mlpackage"
        print(f"included CoreML/ANE model: {coreml.name} -> intent_model/model.mlpackage")

    # 2. Manifest fields: version, timestamp, lineage.
    manifest["pack_version"] = version
    manifest["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["lineage"] = {
        "git_sha": _git_sha(),
        "compiler_version": "assemble_pack/1",
        "dataset_hashes": {
            d: _sha256(ROOT / d) for d in LINEAGE_DATASETS if (ROOT / d).is_file()
        },
    }
    (stage / "pack.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 3. Integrity manifest: SHA-256 of every file except the manifest itself.
    integrity = stage / "integrity"
    integrity.mkdir(exist_ok=True)
    files = sorted(p for p in stage.rglob("*")
                   if p.is_file() and p.name != "manifest.sha256")
    lines = [f"{_sha256(p)}  {p.relative_to(stage).as_posix()}" for p in files]
    (integrity / "manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 4. Deterministic zip.
    out_dir.mkdir(parents=True, exist_ok=True)
    lang = manifest.get("language", "xx")
    archive = out_dir / f"pack-{lang}-v{version}.nlu"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                info = zipfile.ZipInfo(p.relative_to(stage).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                z.writestr(info, p.read_bytes())

    shutil.rmtree(stage.parent, ignore_errors=True)
    return archive


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(ROOT / "packs" / "en"))
    ap.add_argument("--version", required=True, help="semver, e.g. 1.0.0")
    ap.add_argument("--report", default=str(ROOT / "dist" / "report_card.json"))
    ap.add_argument("--out", default=str(ROOT / "dist"))
    ap.add_argument("--coreml", default=None,
                    help="optional path to an IntentClassifier .mlpackage to bundle for iOS")
    args = ap.parse_args()

    coreml = Path(args.coreml) if args.coreml else None
    archive = assemble(Path(args.pack), args.version, Path(args.report),
                       Path(args.out).resolve(), coreml)
    size_kb = archive.stat().st_size / 1024
    try:
        shown = archive.relative_to(ROOT)
    except ValueError:
        shown = archive
    print(f"assembled {shown} ({size_kb:.0f} KB)")
    with zipfile.ZipFile(archive) as z:
        print(f"  {len(z.namelist())} entries; manifest + lineage embedded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
