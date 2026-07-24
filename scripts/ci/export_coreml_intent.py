#!/usr/bin/env python3
"""
Export the intent classifier to an ANE-eligible CoreML model for iOS.

Wraps `multilingual/export_coreml_multilingual.py` (the ONE exporter that emits a
true ML-Program package with a fixed (1, V) input shape, FP16, ComputeUnit.ALL —
i.e. Apple-Neural-Engine-resident) and feeds it the PRODUCTION weights so the
CoreML model matches the shipped ONNX model.

Pipeline:
  1. `scripts/export_weights.py` must have produced models/intent_classifier_weights.json
  2. stage it where the exporter looks: multilingual/models/<lang>/<lang>_intent_classifier_weights.json
  3. run the exporter (--model <lang> --fp16)
  4. copy the resulting IntentClassifier_<lang>.mlpackage to --out

REQUIRES coremltools (+ torch for the mlprogram path). Intended to run on a
macOS runner in CI, where the result can also be validated on the Core ML runtime.

Usage:
    python scripts/ci/export_coreml_intent.py --lang en --out dist/model.mlpackage
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en")
    ap.add_argument("--weights", default=str(ROOT / "models" / "intent_classifier_weights.json"),
                    help="production weights JSON (from scripts/export_weights.py)")
    ap.add_argument("--out", default=str(ROOT / "dist" / "model.mlpackage"))
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.is_file():
        print(f"ERROR: weights not found: {weights}\n"
              f"Run `python scripts/export_weights.py` first.")
        return 1

    # 2. stage weights where export_coreml_multilingual.py expects them.
    staged_dir = ROOT / "multilingual" / "models" / args.lang
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged = staged_dir / f"{args.lang}_intent_classifier_weights.json"
    shutil.copyfile(weights, staged)
    print(f"staged production weights -> {staged.relative_to(ROOT)}")

    # 3. run the ANE exporter (FP16 mlprogram).
    exporter = ROOT / "multilingual" / "export_coreml_multilingual.py"
    cmd = [sys.executable, str(exporter), "--model", args.lang, "--fp16"]
    print("running:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        print("ERROR: CoreML export failed.")
        return rc

    produced = staged_dir / f"IntentClassifier_{args.lang}.mlpackage"
    if not produced.exists():
        print(f"ERROR: expected artifact not found: {produced}")
        return 1

    # 4. copy to the requested output path (.mlpackage is a directory bundle).
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(produced, out)
    print(f"CoreML/ANE model -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
