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
- Standalone `.mlpackage` bundles are **gitignored** (regenerated locally / in
  CI). The *release* pack embeds a copy inside the signed `.nlu` — see
  "Fat Bundle" below.

## CoreML export

```bash
make export-coreml         # = python multilingual/export_coreml_multilingual.py --all --fp16 --fp32
python multilingual/export_coreml_multilingual.py --model en
```

FP16 `mlprogram` is the default/required precision; FP32 is an optional fallback
artifact written alongside.

## Fat Bundle: CoreML inside the signed .nlu (ADR-014)

The release pack ships the CoreML `.mlpackage` INSIDE the signed bundle, not as a
loose side artifact. It is carried as `models.intent.<lang>.coreml_artifact` — a
schema-legal sibling of the ONNX `artifact`, NOT a second `models.coreml` stage
(`models` stays the closed set intent/embedder/semantic_head).

```bash
# release-pack.yml (release job) passes --coreml when the macOS export job
# produced one; assemble_pack copies the .mlpackage dir into the staging tree so
# its files are checksummed + signed with the rest of the pack.
python scripts/ci/assemble_pack.py --src dist/bundle-en --version 1.0.0 \
  --language en --calibration dl/models/intent/en/calibration.json \
  --coreml cml/intent/en/IntentClassifier.mlpackage --out dist
```

Guards: `tests/test_release_pack.py::test_coreml_is_packaged_into_the_bundle`,
`::test_release_job_passes_coreml_to_the_packer`,
`::test_models_schema_really_forbids_a_coreml_stage`.

**Caveat (parity):** the `.mlpackage` `nlu_export.export_coreml` emits today is
derived from the repo-committed DEVICE weights, not the ONNX trained in the same
run, so the bundled CoreML model is an iOS convenience artifact, not proof of
ONNX↔CoreML parity. Retargeting the exporter is the follow-up (see `known-issues.md`).

## INT8 quantization

```bash
python packages/buildtime/nlu_training/semantic_multilingual/quantize_multilingual.py
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
