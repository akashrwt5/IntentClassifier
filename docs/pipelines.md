# Pipelines: Training, Evaluation, Inference, Export

Canonical, end-to-end reference for the four pipelines in this repo. Commands
have `make` shortcuts (see the `Makefile`) and direct-script equivalents.

```
data/  ──►  TRAIN  ──►  models/*.onnx (+labels, pipeline)  ──►  EXPORT  ──►  *.mlpackage / INT8
                │                                                  │
                └──►  EVALUATE (holdout / OOS / calibration)       └──►  PARITY TESTS (Tier-A/B, ANE)
                                                                   INFERENCE ◄── NLU engine (packages/runtime/nlu_engine/ (moved from packages/runtime/nlu_engine/ — ND-2 M1))
```

---

## 1. Training

### English core model (TF-IDF + LogisticRegression → ONNX)

```bash
make train                 # = python packages/buildtime/nlu_training/train.py   (version 3 data, default)
python packages/buildtime/nlu_training/train.py -v 1   # original files
python packages/buildtime/nlu_training/train.py -v 2   # corrected _2 files
python packages/buildtime/nlu_training/train.py -v 3   # enhanced: _2 + 02_source_manual_corrections.csv (default)
```

Data lineage (see `data/DATA_PIPELINE.md`):
`01_source_base_training_data.csv` → `02_source_manual_corrections.csv` →
`03_generated_augmented_phrases.csv` → `04_GENERATED_MASTER_training_data.csv`.

The script caps samples per intent, guards against holdout leakage, runs
cross-validation, enforces a **test-split accuracy floor** (`MIN_TEST_ACCURACY`),
and writes:

- `models/intent_model.onnx` — the shipped classifier (~16KB).
- `models/intent_labels.pkl` and `models/intent_labels.json` — label mapping.
- `models/intent_pipeline.pkl` — the sklearn pipeline (for calibration/analysis).
- `models/manifest.json` — artifact manifest.

### Multilingual model

```bash
make train-multilingual    # = python multilingual/train_multilingual.py
```

Trains per-language variants (en/fr/de/da) with shared text normalization
(`multilingual/text_norm.py`).

### Semantic heads (rescue layer)

```bash
python scripts/train_semantic_head.py            # English
python scripts/SemanticSupport/train_multilingual_semantic_head.py   # multilingual
```

Trains a lightweight head over MiniLM embeddings used to "rescue" low-confidence
classifier turns. Per-intent/per-language capping is applied before merge.

---

## 2. Evaluation

Training already prints a classification report, confusion matrix, and a
**permanent-holdout** accuracy check. Beyond that:

- **Holdout / OOS sets** — `data/semantic_holdout_100.csv`, `semantic_oos.csv`,
  and language holdouts in `multilingual/test/*_holdout.csv`. Keep train,
  holdout, and OOS strictly separate; never tune on the holdout.
- **Metrics** — macro-F1 (primary), per-class precision/recall, and ECE
  (expected calibration error).
- **Calibration** — per-language **temperature scaling**:
  ```bash
  make calibrate           # = python packages/buildtime/nlu_training/calibrate_languages.py
  ```
  Writes temperature, confidence thresholds, holdout macro-F1, and ECE per
  language to `config/calibration.json`. Decisions are recorded in
  `multilingual/TEMPERATURE_SCALING_DECISION.md` and
  `MODEL_CALIBRATION_DECISION.md`.
- **Ad-hoc checks** — `scripts/test_holdout.py`, `scripts/scan_mislabels.py`,
  `scripts/test_semantic.py`.

---

## 3. Inference

### Quick CLI (classifier only)

```bash
make predict               # = python apps/cli/predict.py
```

Loads `models/intent_model.onnx` + labels via ONNX Runtime. Applies a confidence
threshold (`CONF_THRESHOLD = 0.70`) and a top-1/top-2 gap threshold
(`CONF_GAP_THRESHOLD = 0.20`); low-confidence inputs are logged to
`data/unknown_data.csv` for later review/auto-labeling (`scripts/auto_label.py`).

### Full NLU engine (production path)

```bash
make nlu                   # = python apps/cli/nlu_cli.py
python apps/cli/nlu_cli_multilingual.py     # multilingual entry point
```

`packages/runtime/nlu_engine/engine.py` (`NLUEngine`) orchestrates each user turn in priority
order: **confirmation** (active yes/no context) → **slot-filling** (mid-collection,
with high-confidence interruption detection) → **classify** (fresh turn). It
combines `classifier.py`, `entities.py` (entity + datetime extraction),
`context.py` (session/slot state), and `semantic.py` (semantic rescue). Results
are returned as an `NLUResult` (intent, slots, confidence, `interrupted_intent`).

Schema/config: `content/nlu_schema.json`, `content/nlu_entities.json`,
`content/localization/`, and `config/calibration.json`.

---

## 4. Model export (mobile: ONNX / CoreML / TFLite)

The classifier is exported to **ONNX** during training. For Apple devices it is
converted to **CoreML**; INT8 quantization produces the smallest artifacts.

### CoreML

```bash
make export-coreml         # = python multilingual/export_coreml_multilingual.py --all --fp16 --fp32
python multilingual/export_coreml_multilingual.py --model en   # single language
```

Produces `.mlpackage` bundles: FP16 `mlprogram` (default, required precision),
optional FP32 fallback. Key constraints — **static `(1, V)` input shapes and
batch size 1** (embed one sentence at a time), `logits` output, and the
temperature stored in model metadata so the device applies `softmax(logits / T)`.
See `docs/coreml-conversion-guide.md` and `multilingual/COREML_RESULTS.md`.

### INT8 quantization

```bash
python scripts/SemanticSupport/quantize_multilingual.py
python scripts/compare_coreml_quant.py     # size vs. accuracy comparison
```

Outputs e.g. `models/MiniLMEmbedder_int8.mlpackage`.

### TFLite

The TF-IDF + LogisticRegression classifier is ONNX-first. TFLite is a supported
target for the on-device path; when a TFLite artifact is required, convert from
the ONNX graph and validate with the same parity methodology below. Preserve the
static-batch-1 contract and re-verify numeric parity + the confidence gate.

### Parity & ANE verification

```bash
make export-coreml-test    # Tier-A numeric equivalence (Linux)
python multilingual/test/test_coreml_multilingual.py --runtime --full   # Tier-B (macOS)
python multilingual/test/ane_compute_plan.py --model all                # ANE op placement (macOS)
```

Acceptance: accuracy Δ ≈ 0 and **0 gate disagreements** against the golden
fixtures (`multilingual/test/coreml_golden_fixtures.json`). Tier-B (real Core ML
runtime) and ANE placement run on the Apple-Silicon macOS CI
(`.github/workflows/coreml-macos.yml`); the iOS XCTest parity suite lives in the
STT repo. `.mlpackage` bundles are gitignored and rebuilt on demand.

---

## Reproduce the full loop

```bash
make train
make calibrate
make export-coreml
make export-coreml-test
```
