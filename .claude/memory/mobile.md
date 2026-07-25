# Memory: Mobile Deployment (ONNX / CoreML / ANE)

> Single responsibility: turning trained models into on-device artifacts and
> proving they agree with the server model. For coremltools / onnxruntime APIs
> use the **Context7 MCP**. Background: `multilingual/COREML_RESULTS.md`,
> `docs/coreml-conversion-guide.md`.

## Non-negotiable constraints

- **Static `(1, V)` input shape, batch size 1** — embed one sentence at a time.
- Output is **logits**; the per-language **temperature** rides in model metadata
  so the device computes `softmax(logits / T)`.
- FP16 `mlprogram` + `ComputeUnit.ALL` is what makes the model ANE-resident.
  A `neuralNetworkClassifier` package is the **non-ANE** shape — that is a bug,
  not a variant.
- Generated `.mlpackage` bundles are gitignored and regenerated.

## The one exporter

`multilingual/export_coreml_multilingual.py` is the **only** exporter that emits
a true ML-Program package. It reads per-language weights from
`multilingual/models/<lang>/<lang>_intent_classifier_weights.json` and writes
**into that same directory**:

```
multilingual/models/<lang>/IntentClassifier_<lang>.mlpackage        # fp16 (required)
multilingual/models/<lang>/IntentClassifier_<lang>_fp32.mlpackage   # optional fallback
```

```bash
python multilingual/export_coreml_multilingual.py --model en --fp16
python multilingual/export_coreml_multilingual.py --all --fp16 --fp32
```

It deliberately does **not** touch the legacy top-level `models/`.

## CI wrapper (production weights → ANE model)

`scripts/ci/export_coreml_intent.py` exists so the CoreML model is built from the
*shipped* weights rather than a side fit. It stages
`models/intent_classifier_weights.json` (produced by `scripts/export_weights.py`)
into the layout the exporter expects, runs it, and copies the result out:

```bash
python scripts/ci/export_coreml_intent.py --lang en --out dist/model.mlpackage
```

Needs coremltools + torch → runs on the macOS CI job.

## Three-featurizer parity (the drift trap)

sklearn pipeline / ONNX / Swift must all agree. This is where a real bug lived:
`scripts/train.py` (ONNX) and `scripts/export_weights.py` (CoreML weights) were
**two different fits** — different data, `min_df`, even upper-cased labels —
i.e. silent iOS↔Android drift. `export_weights.py` now derives everything
(labels, vocab, idf, coef, ngram, sublinear) from the **same
`models/intent_pipeline.pkl`** the ONNX is exported from. Keep it that way.

```bash
python scripts/ci/verify_coreml_parity.py            # Tier-A
python scripts/ci/verify_coreml_parity.py --runtime  # + Tier-B (macOS)
python scripts/ci/verify_coreml_parity.py --inspect  # format/dims, no macOS libs
python scripts/test_ios_conformance.py --model production
```

Defaults it assumes: `models/intent_pipeline.pkl`, `models/intent_model.onnx`,
`dist/models/model.mlpackage`, holdout `data/semantic_holdout_100.csv`,
`--max-argmax-disagree 2`. Report → `dist/coreml_parity.json`.

- **Tier-A** (anywhere, incl. Linux CI): ANE shape check; CoreML linear weights
  == pipeline classifier within FP16; ONNX top-1 == pipeline over the holdout
  within the skl2onnx budget. `--onnx-ref-only` runs just the ONNX check when
  coremltools is unavailable.
- **Tier-B** (`--runtime`, macOS): feeds `tfidf(x)` to the **real** Core ML
  runtime and compares top-1.
- **`--inspect`** parses the raw `.mlpackage` protobuf directly, so it works on
  Linux. Use it first when a package is suspect — it is how a stale checked-in
  model was caught (`neuralNetworkClassifier`, 1340 features vs the production
  5433: a different model entirely).
- **Swift conformance**: `scripts/test_ios_conformance.py` compares the Swift
  `swift_tokens` device path against ONNX (top-1 + fire/fallback), closing the
  loop on the third featurizer.

Any failure **blocks the release**.

## Multilingual parity fixtures

```bash
python multilingual/test/test_coreml_multilingual.py --full
```

Acceptance: **accuracy Δ ≈ 0 and 0 gate disagreements** vs
`multilingual/test/coreml_golden_fixtures.json`. Tier-B and ANE compute-plan
placement run on Apple-Silicon CI (`.github/workflows/coreml-macos.yml`); they
auto-skip on Linux. iOS XCTest parity lives in the STT repo (`akashrwt5/STT`)
and needs a one-time `INTENTCLASSIFIER_PAT` secret.

## Release packaging

`scripts/ci/assemble_pack.py` turns a pack directory into a versioned,
self-describing `.nlu` archive with a SHA-256 manifest, git-SHA lineage, and the
embedded report card:

```bash
python scripts/ci/assemble_pack.py --pack packs/en --version 1.0.0 \
    --report dist/report_card.json --coreml dist/model.mlpackage
# -> dist/pack-en-v1.0.0.nlu
```

With `--coreml` the archive carries **both** `intent_model/model.onnx` (Android)
and `intent_model/model.mlpackage` (iOS), each declared in the manifest.
`release-pack.yml` runs it as 3 jobs: train+gate (Linux) → CoreML/ANE export
(macOS) → assemble+publish (Linux), passing the `.mlpackage` between them.

## Related memory

Architecture -> `architecture.md` · Pack format -> `langpack.md` · Training +
the release gate -> `training.md` · Decisions -> `decisions.md` · Known export
issues -> `known-issues.md`.
