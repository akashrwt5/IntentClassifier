#!/usr/bin/env python3
"""
Report the Core ML compute-device placement of each op in a multilingual
IntentClassifier .mlpackage (S5 — ANE eligibility check).

macOS-only: MLComputePlan needs the Core ML runtime (`libcoremlpython`), which
is absent on Linux. On a non-Darwin host this prints a skip notice and exits 0
so it can sit in a cross-platform CI matrix without failing.

Usage:
    python multilingual/test/ane_compute_plan.py            # all models
    python multilingual/test/ane_compute_plan.py --model en
"""

import argparse
import platform
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_NAMES = ["en", "fr", "de", "da", "multilingual", "multilingual_small"]


def check_model(ct, model: str) -> None:
    pkg = MODELS_DIR / model / f"IntentClassifier_{model}.mlpackage"
    if not pkg.exists():
        print(f"  [{model}] SKIP — {pkg.name} not found (run the exporter first).")
        return

    # MLComputePlan.load_from_path expects a COMPILED model (.mlmodelc), not a
    # raw .mlpackage — passing the package aborts in native code. Compile first.
    from coremltools.models.utils import compile_model
    compiled = compile_model(str(pkg))

    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        str(compiled), compute_units=ct.ComputeUnit.ALL,
    )
    program = plan.model_structure.program
    if program is None:
        print(f"  [{model}] no mlProgram structure (not an mlprogram package?).")
        return

    print(f"\n  {model}:")
    for function in program.functions.values():
        for op in function.block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is None:
                continue
            preferred = type(usage.preferred_compute_device).__name__
            supported = [type(d).__name__ for d in usage.supported_compute_devices]
            print(f"    {op.operator_name:24} preferred={preferred:32} "
                  f"supported={supported}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="all",
                        choices=["all"] + MODEL_NAMES)
    args = parser.parse_args()

    if platform.system() != "Darwin":
        print("ANE compute-plan check SKIPPED — requires macOS (this host is "
              f"{platform.system()}; the Core ML runtime is macOS-only).")
        print("Expected outcome on Apple hardware: the single 'linear'/'matmul' op "
              "reports preferred=CPU for these tiny models (too small to amortize "
              "ANE staging cost). See multilingual/COREML_RESULTS.md.")
        return

    import coremltools as ct
    print("=" * 70)
    print("  Core ML compute-plan (op -> preferred / supported compute devices)")
    print("=" * 70)
    models = MODEL_NAMES if args.model == "all" else [args.model]
    for model in models:
        try:
            check_model(ct, model)
        except Exception as e:
            print(f"  [{model}] ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
