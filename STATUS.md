# Routine status — CoreML FP16 export (branch `claude/coreml-export`)

**State: TERMINAL (all checklist steps S1–S7 complete).** Future runs: do
nothing except re-verify; do not re-notify and do not make cosmetic changes.

## What is done

| Step | Result |
|---|---|
| S1 exporter | `multilingual/export_coreml_multilingual.py` — FP16 `mlprogram`, fixed `(1,V)`, `logits` output, temperature in metadata, optional `--fp32`. |
| S2 Tier-A | `multilingual/test/test_coreml_multilingual.py` — **6/6 PASS** on Linux. |
| S3 Tier-B | **6/6 PASS on the real Core ML runtime** (GitHub Apple-Silicon macOS CI, 2026-06-25): acc Δ ≈ 0, 0/30 gate disagreements. Auto-skips on Linux. |
| S4 report | `multilingual/COREML_RESULTS.md` — Tables 1 & 2 populated with real numbers. |
| S5 ANE | Documented (likely CPU/BNNS; eligibility maximized). Live compute-plan macOS-pending. |
| S6 iOS | `akashrwt5/STT` @ `claude/coreml-temperature-ios`: Swift switched to softmax(logits/T); golden fixtures + parity XCTest added. |
| S7 share | COREML_RESULTS.md + completion notification with the numbers. |

## macOS CI (Option A) — fills what Linux can't

`.github/workflows/coreml-macos.yml` runs on GitHub's Apple-Silicon
`macos-latest` and has already produced the real Tier-B numbers (see Table 2 in
COREML_RESULTS.md). The iOS XCTest workflow lives in the STT repo
(`.github/workflows/ios-coreml-parity.yml`) and needs a one-time
`INTENTCLASSIFIER_PAT` repo secret (read access to this repo) before it can run.

Still macOS-only (cannot run on this Linux routine):
- The iOS `IntentClassifierCoreMLParityTests` (it does `import CoreML`).
- Live ANE compute-plan placement (run on the macOS CI; non-blocking).

## Reproduce

```bash
python multilingual/export_coreml_multilingual.py --all --fp16 --fp32
python multilingual/test/test_coreml_multilingual.py --full
```

Note: `.mlpackage` bundles are gitignored (regenerated locally); the exporter
rebuilds them and the tests build any missing package on demand.
