# Memory: Mobile Deployment (ONNX / CoreML / TFLite)

> Single responsibility: turning trained models into on-device artifacts and
> proving parity. For coremltools / onnxruntime / TFLite APIs use the
> **Context7 MCP**. Full report: `multilingual/COREML_RESULTS.md`; guide:
> `docs/coreml-conversion-guide.md`.

## Non-negotiable constraints

- **Static `(1, V)` input shapes, batch size 1** — embed one sentence at a time.
- Output is **logits**; the per-language **temperature** is stored in model
  metadata so the device computes `softmax(logits / T)`.
- Preserve numeric parity + the confidence gate across export/quantization.
- `.mlpackage` bundles are **gitignored** (regenerated locally / in CI).

## CoreML export

```bash
make export-coreml         # = python multilingual/export_coreml_multilingual.py --all --fp16 --fp32
python multilingual/export_coreml_multilingual.py --model en
```

FP16 `mlprogram` is the default/required precision; FP32 is an optional fallback
artifact written alongside.

## INT8 quantization

```bash
python scripts/SemanticSupport/quantize_multilingual.py
python scripts/compare_coreml_quant.py    # size vs. accuracy comparison
```

Outputs e.g. `models/MiniLMEmbedder_int8.mlpackage`.

## TFLite

The classifier is ONNX-first. TFLite is a supported on-device target; convert
from the ONNX graph, preserve the static-batch-1 contract, and re-verify numeric
parity + the confidence gate with the same methodology below.

## Parity + ANE verification

```bash
make export-coreml-test                                                  # Tier-A numeric (Linux)
python multilingual/test/test_coreml_multilingual.py --runtime --full    # Tier-B runtime (macOS)
python multilingual/test/ane_compute_plan.py --model all                 # ANE op placement (macOS)
```

Acceptance: **accuracy Δ ≈ 0 and 0 gate disagreements** vs. golden fixtures
(`multilingual/test/coreml_golden_fixtures.json`). Tier-B + ANE run on the
Apple-Silicon macOS CI (`.github/workflows/coreml-macos.yml`). iOS XCTest parity
lives in the STT repo (`akashrwt5/STT`).

## Memory-optimization roadmap

Distillation / compression plan (E5-small vs MiniLM teacher, student options,
Model2Vec static embeddings): `docs/on-device-memory-optimization-plan.md`
and `roadmap.md`.

## Related memory

Architecture -> `architecture.md` · Training -> `training.md` · Decisions ->
`decisions.md` · Known export issues -> `known-issues.md`.
