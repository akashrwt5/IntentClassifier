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

### Intent head: pruned vs full vocab (two CoreML variants)

The intent classifier CoreML head is FP32 (`NeuralNetworkBuilder`, float-vector
input) and its size is set entirely by the vocabulary it is built from.
`export_ios_weights.py` produces the device weights from the trained
`pipeline.pkl`:

- default (`--top-per-class 30`) → PRUNED vocab (~1.3k features, ~290 KB head):
  the small on-device default, `IntentClassifier.mlpackage`.
- `--top-per-class 0` → FULL vocab (4718 features, ~1 MB head): matches the
  ONNX/TFLite feature space, `IntentClassifier_full.mlpackage`.

`export_coreml.py` builds the full head too whenever
`intent_classifier_weights_full.json` is present. Both ride in the signed `.nlu`
as `models.intent.<lang>.coreml_artifact` / `coreml_full_artifact`. The scalar
temperature T is refit per variant on its own device-equivalent logits (pruned
and full get different T; a stale T mis-tunes the 0.70 gate).

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

## TFLite (ADR-015)

TFLite is the **linear HEAD only** — float TF-IDF vector `(1, V)` in, logits
`(1, n_classes)` out — built DIRECTLY from the fitted sklearn
`LogisticRegression` (`coef_`/`intercept_`), the same trained object `train.py`
exports to ONNX. It is NOT transcoded from the ONNX or CoreML artifact: the ONNX
graph is string-in with ONNX-ML ops (`StringNormalizer`/`Tokenizer`/
`TfIdfVectorizer`/`LinearClassifier`) that have no TFLite equivalents. TF-IDF
vectorisation stays native on-device (from `vocab`+`idf`), exactly like the
CoreML "float-vector input" head. Output is logits; temperature is applied at
runtime.

```bash
make export-tflite LANG=en                                  # fp32 + int8
PYTHONPATH=packages/buildtime python -m nlu_export.export_tflite --all
```

Artifacts (beside `model.onnx`): `models/intent/<lang>/model.tflite` (fp32,
~1.08 MB) and `model_int8.tflite` (dynamic-range int8, ~271 KB). Because the map
is linear, **fp32 is bit-parity** with the ONNX `LinearClassifier`
(max |Δlogit| ≈ 1e-6); the exporter self-checks this and fails on divergence.
int8 keeps argmax (0 gate disagreements). Parity is guarded by
`tests/test_tflite_export.py`; TensorFlow is an export-only dep (Linux, so it
runs in the release-pack `train-gate` job, not a macOS runner). Both variants
ride in the signed `.nlu` as `models.intent.<lang>.tflite_artifact` /
`tflite_int8_artifact` (see "Fat Bundle" above).

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
