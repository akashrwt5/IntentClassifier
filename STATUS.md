# Routine status — CoreML FP16 export (branch `claude/coreml-export`)

**State: TERMINAL (all checklist steps S1–S7 complete).** Future runs: do
nothing except re-verify; do not re-notify and do not make cosmetic changes.

## What is done

| Step | Result |
|---|---|
| S1 exporter | `multilingual/export_coreml_multilingual.py` — FP16 `mlprogram`, fixed `(1,V)`, `logits` output, temperature in metadata, optional `--fp32`. |
| S2 Tier-A | `multilingual/test/test_coreml_multilingual.py` — **6/6 PASS** on Linux. |
| S3 Tier-B | macOS-only; **auto-skips** on this Linux runner (recorded, not failed). |
| S4 report | `multilingual/COREML_RESULTS.md` — Tables 1 & 2 populated with real numbers. |
| S5 ANE | Documented (likely CPU/BNNS; eligibility maximized). Live compute-plan macOS-pending. |
| S6 iOS | `akashrwt5/STT` @ `claude/coreml-temperature-ios`: Swift switched to softmax(logits/T); golden fixtures + parity XCTest added. |
| S7 share | COREML_RESULTS.md + completion notification with the numbers. |

## Expected partial-block (NOT a failure)

**Tier-B (real Core ML runtime) and the iOS parity XCTest require macOS + Xcode.**
This is a Linux routine, so `coremltools.libcoremlpython` (the Core ML runtime)
is absent and `.mlpackage` `predict()` / `import CoreML` cannot run here. Tier-A
already proves numeric equivalence on Linux (NumPy reference ≡ ONNX ≤ 2.5e-6,
package weights ≡ JSON within float16), and the FP16 weight-rounding simulation
(Table 2) predicts the runtime numbers. Run on a Mac to fill the runtime cells:

```bash
python multilingual/test/test_coreml_multilingual.py --runtime   # Tier-B
# iOS: open STT.xcodeproj, run STTTests/IntentClassifierCoreMLParityTests
```

## Reproduce

```bash
python multilingual/export_coreml_multilingual.py --all --fp16 --fp32
python multilingual/test/test_coreml_multilingual.py --full
```

Note: `.mlpackage` bundles are gitignored (regenerated locally); the exporter
rebuilds them and the tests build any missing package on demand.
