---
description: Export mobile artifacts (CoreML/ONNX) and verify parity
---

Follow `.claude/memory/mobile.md`. Prefer the **mobile-ml-engineer** agent.

1. Export: `make export-coreml` (FP16 + FP32) — or a single `--model <lang>`.
2. Tier-A numeric parity (Linux): `make export-coreml-test`.
3. Confirm the gate: **acc Δ ≈ 0 and 0 gate disagreements** vs. the golden
   fixtures (`multilingual/test/coreml_golden_fixtures.json`).
4. Note that Tier-B (real Core ML runtime) + ANE placement run on the macOS CI
   (`.github/workflows/coreml-macos.yml`) — flag if those need attention.
5. Preserve static `(1,V)` shapes, batch 1, logits + temperature-in-metadata.
   Do not commit `.mlpackage` artifacts. Report parity numbers.
