---
description: Export mobile artifacts (CoreML/ONNX) and verify parity
---

Follow `.claude/memory/mobile.md`. Prefer the **mobile-ml-engineer** agent.

1. Make sure the weights are current and derived from the shipped pipeline:
   `python scripts/export_weights.py` (must read `models/intent_pipeline.pkl`).
2. Export: `python multilingual/export_coreml_multilingual.py --model en --fp16`
   (or `--all --fp16 --fp32`). Output lands in
   `multilingual/models/<lang>/IntentClassifier_<lang>.mlpackage`.
   For the CI path: `python scripts/ci/export_coreml_intent.py --lang en --out dist/model.mlpackage`.
3. Sanity-check the package before trusting it — this catches a stale/non-ANE model:
   `python scripts/ci/verify_coreml_parity.py --inspect`
4. Tier-A parity (Linux): `python scripts/ci/verify_coreml_parity.py`
   Swift conformance: `python scripts/test_ios_conformance.py --model production`
   Multilingual fixtures: `python multilingual/test/test_coreml_multilingual.py --full`
5. Confirm the gate: **acc Δ ≈ 0 and 0 gate disagreements**. Preserve fixed
   `(1,V)`, FP16 `mlprogram`, `ComputeUnit.ALL`, logits + temperature-in-metadata.

State explicitly which tiers could NOT run here — Tier-B runtime parity, live ANE
placement, and the iOS XCTest suite are macOS-only and auto-skip on Linux. Do not
commit generated `.mlpackage` artifacts. Report parity numbers.
