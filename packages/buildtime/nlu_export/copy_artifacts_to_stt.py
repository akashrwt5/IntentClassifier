#!/usr/bin/env python3
"""
Copy the CoreML / NLU artifacts from models/ into the STT iOS app's Resources.

Handles the two renames the iOS app expects:
  * the shipped embedder (FP16 or palettized) -> MiniLMEmbedder.mlpackage
  * the chosen semantic head JSON              -> semantic_head.json

.mlpackage bundles are directories, so they are copied recursively. Anything
that doesn't exist in models/ is reported and skipped (so a partial export
won't silently ship a stale file).

Usage
-----
    # default: ship the palettized embedder + the CoreML-retrained head
    python scripts/copy_artifacts_to_stt.py /path/to/STT

    # you can pass the repo root, or the Resources dir directly:
    python scripts/copy_artifacts_to_stt.py ../STT
    python scripts/copy_artifacts_to_stt.py ../STT/STT/STT/Resources

    # ship the FP16 embedder and the ONNX-trained head instead:
    python scripts/copy_artifacts_to_stt.py ../STT \
        --embedder models/MiniLMEmbedder.mlpackage \
        --head     models/semantic_head.json

    # see what would happen without copying:
    python scripts/copy_artifacts_to_stt.py ../STT --dry-run
"""

import argparse
import shutil
import sys
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parents[3]
MODELS_DIR = BASE_DIR / "models"

# Relative path from the STT repo root to the app's Resources folder.
RESOURCES_SUBPATH = Path("STT") / "STT" / "Resources"

# Defaults reflect the shipping decision: palettized embedder + CoreML head.
DEFAULT_EMBEDDER = MODELS_DIR / "MiniLMEmbedder_int8.mlpackage"
DEFAULT_HEAD     = MODELS_DIR / "semantic_head_coreml.json"


def resolve_resources_dir(dest: Path) -> Path:
    """Accept either the STT repo root or the Resources dir itself."""
    if dest.name == "Resources":
        return dest
    candidate = dest / RESOURCES_SUBPATH
    if candidate.is_dir():
        return candidate
    # Fall back to searching for a unique Resources dir under the repo.
    matches = [p for p in dest.glob("**/Resources") if p.is_dir()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        sys.exit(f"ERROR: could not find a Resources/ folder under {dest}. "
                 f"Pass the Resources dir explicitly.")
    sys.exit(f"ERROR: multiple Resources/ folders under {dest}:\n  "
             + "\n  ".join(str(m) for m in matches)
             + "\nPass the exact one explicitly.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dest", help="STT repo root, or its Resources/ folder.")
    ap.add_argument("--embedder", default=str(DEFAULT_EMBEDDER),
                    help=f"Embedder .mlpackage to ship as MiniLMEmbedder.mlpackage "
                         f"(default: {DEFAULT_EMBEDDER.name}).")
    ap.add_argument("--head", default=str(DEFAULT_HEAD),
                    help=f"Semantic head JSON to ship as semantic_head.json "
                         f"(default: {DEFAULT_HEAD.name}).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print actions without copying.")
    args = ap.parse_args()

    res_dir = resolve_resources_dir(Path(args.dest).expanduser().resolve())
    print(f"Target Resources dir: {res_dir}\n")

    # (source path, destination filename). Destination may rename the source.
    plan = [
        (Path(args.embedder),                        "MiniLMEmbedder.mlpackage"),
        (Path(args.head),                            "semantic_head.json"),
        (MODELS_DIR / "IntentClassifier.mlpackage",  "IntentClassifier.mlpackage"),
        (MODELS_DIR / "SemanticHead.mlpackage",      "SemanticHead.mlpackage"),
        (MODELS_DIR / "intent_classifier_weights.json", "intent_classifier_weights.json"),
        (MODELS_DIR / "minilm-vocab.txt",            "minilm-vocab.txt"),
    ]

    copied, skipped = [], []
    for src, dst_name in plan:
        dst = res_dir / dst_name
        if not src.exists():
            skipped.append((src, dst_name))
            print(f"  SKIP  {dst_name:<32} (source missing: {src})")
            continue

        rename = " (renamed)" if src.name != dst_name else ""
        if args.dry_run:
            print(f"  COPY  {src.name:<32} -> {dst_name}{rename}")
            copied.append(dst_name)
            continue

        # Remove any existing destination first so .mlpackage dirs don't merge.
        if dst.is_dir():
            shutil.rmtree(dst)
        elif dst.exists():
            dst.unlink()

        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print(f"  COPY  {src.name:<32} -> {dst_name}{rename}")
        copied.append(dst_name)

    print(f"\n{'(dry run) ' if args.dry_run else ''}"
          f"{len(copied)} copied, {len(skipped)} skipped.")
    if skipped:
        print("\nSkipped files were not found in models/. If any are required for")
        print("the build, generate them (export_coreml.py / train_semantic_head_coreml.py)")
        print("and re-run. SemanticHead.mlpackage is only needed if iOS reads the")
        print("CoreML head instead of semantic_head.json.")


if __name__ == "__main__":
    main()
