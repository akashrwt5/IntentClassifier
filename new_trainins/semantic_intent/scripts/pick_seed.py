"""Fit the student head across seeds, keep the best one, on VALIDATION only.

WHY THIS EXISTS
---------------
The encoder is distilled once and is expensive. The head, the temperature, the
OOD scorer and every threshold are re-fitted on top of it in about six seconds,
and that re-fit turned out to carry more run-to-run variation than anything
downstream of it:

    run A   val ECE 0.0182   per-risk normal >= 0.940   test coverage 0.623
    run B   val ECE 0.0141   per-risk normal >= 0.860   test coverage 0.715

Same encoder quality, same data. A worse-calibrated head needs a higher
confidence threshold to reach the same precision, and the coverage is paid out
of that threshold. Refitting is cheap, so there is no reason to accept a bad
draw.

THE SELECTION RULE, FIXED BEFORE THE NUMBERS ARE SEEN
-----------------------------------------------------
Among seeds whose VALIDATION precision meets the target, keep the one with the
lowest VALIDATION calibrated ECE; ties broken by higher validation coverage.

Nothing here reads the test set. That matters more than usual: coverage is the
metric that failed the ship criteria, so picking a seed by test coverage would
be choosing the winner by the scoreboard. Validation ECE is the documented
tiebreak in the README and it is also the actual cause — the badly calibrated
head is why the threshold climbed.

    python scripts/pick_seed.py --encoder student-h256-l4
    python scripts/pick_seed.py --encoder student-h256-l4 --seeds 8

Writes the winning fit to --out and deletes the losing candidates.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ECE_RE = re.compile(r"ECE\s+[\d.]+\s*->\s*([\d.]+)")
VAL_RE = re.compile(r"val precision=([\d.]+)\s+coverage=([\d.]+)\s+"
                    r"fallback_leak=([\d.]+)\s+target_met=(\w+)")


def run_seed(seed: int, encoder: str, clf: str, train: str,
             out: Path, extra: list[str] | None = None) -> dict | None:
    cmd = [sys.executable, "scripts/train_classifier.py",
           "--encoder", encoder, "--classifier", clf,
           "--train", train, "--out", str(out), "--seed", str(seed)]
    cmd += extra or []
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"  seed {seed}: FAILED\n{p.stdout[-800:]}\n{p.stderr[-800:]}")
        return None
    text = p.stdout
    ece = ECE_RE.search(text)
    val = VAL_RE.search(text)
    if not (ece and val):
        print(f"  seed {seed}: could not parse output — has "
              f"train_classifier.py's printing changed?")
        return None
    return dict(seed=seed, val_ece=float(ece.group(1)),
                val_precision=float(val.group(1)),
                val_coverage=float(val.group(2)),
                val_leak=float(val.group(3)),
                target_met=val.group(4) == "True", out=out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="student-h256-l4")
    ap.add_argument("--classifier", default="mlp")
    ap.add_argument("--train", default="train_augmented")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="models/final_student_256")
    # Everything after `--` is forwarded verbatim to train_classifier.py.
    # The gate's thresholds live outside the ONNX graph precisely so they can be
    # retuned without re-exporting a model, and the run that produced coverage
    # 0.431 showed why that matters: the model was better on every quality
    # metric while the fitted thresholds (normal 0.97, high 0.99) destroyed
    # coverage. Retuning is a one-minute job; it should not require editing this
    # file or bypassing the seed selection.
    #
    #   python scripts/pick_seed.py -- --min-coverage 0.78
    #   python scripts/pick_seed.py -- --target-precision 0.97 --high-risk-precision 0.98
    ap.add_argument("passthrough", nargs="*", default=[],
                    help="args after -- are passed to train_classifier.py")
    args = ap.parse_args()
    extra = [a for a in args.passthrough if a != "--"]
    if extra:
        print(f"forwarding to train_classifier.py: {' '.join(extra)}")

    print(f"fitting the head on {args.encoder} across {args.seeds} seeds "
          f"(~6s each). The encoder is NOT retrained.\n")

    results = []
    for s in range(args.seeds):
        tmp = ROOT / f"models/_seed_{s}"
        r = run_seed(s, args.encoder, args.classifier, args.train, tmp, extra)
        if r:
            results.append(r)
            print(f"  seed {s}: val ECE {r['val_ece']:.4f}   "
                  f"precision {r['val_precision']:.4f}   "
                  f"coverage {r['val_coverage']:.4f}   "
                  f"leak {r['val_leak']:.4f}   "
                  f"target_met={r['target_met']}")

    if not results:
        raise SystemExit("no seed produced a usable fit")

    eligible = [r for r in results if r["target_met"]] or results
    if not any(r["target_met"] for r in results):
        print("\nWARNING: no seed met the precision target on validation. "
              "Falling back to all seeds — read the numbers before shipping.")

    best = min(eligible, key=lambda r: (r["val_ece"], -r["val_coverage"]))
    print(f"\nwinner: seed {best['seed']} "
          f"(val ECE {best['val_ece']:.4f}, val coverage {best['val_coverage']:.4f})")
    spread = max(r["val_coverage"] for r in results) - min(r["val_coverage"] for r in results)
    print(f"validation coverage spread across seeds: {spread:.4f} — "
          f"if this is large, a single fit was never a reliable reading")

    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    shutil.move(str(best["out"]), str(out))
    for r in results:
        if r["seed"] != best["seed"] and Path(r["out"]).exists():
            shutil.rmtree(r["out"])
    print(f"\nkept -> {out}")
    print("now: export_onnx.py, parity_test.py, evaluate_onnx.py")


if __name__ == "__main__":
    main()
