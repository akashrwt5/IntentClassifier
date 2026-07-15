# Memory: Training Pipeline

> Single responsibility: how models are trained + calibrated. Inference is in
> `inference.md`; export is in `mobile.md`. For library APIs (scikit-learn,
> skl2onnx) use the **Context7 MCP** rather than recalling from memory.

## English core model

```bash
make train                 # = python packages/buildtime/nlu_training/train.py   (version 3 data, default)
python packages/buildtime/nlu_training/train.py -v 1   # original files
python packages/buildtime/nlu_training/train.py -v 2   # corrected _2 files
python packages/buildtime/nlu_training/train.py -v 3   # enhanced: _2 + 02_source_manual_corrections (default)
```

Pipeline: `TfidfVectorizer` -> `LogisticRegression` (sklearn `Pipeline`) exported
via `skl2onnx` with a `StringTensorType` input. The script:
- caps samples per intent (`MAX_PER_INTENT`),
- runs a holdout-leakage guard,
- cross-validates,
- enforces a **test-split accuracy floor** (`MIN_TEST_ACCURACY`),
- prints classification report + confusion matrix + permanent-holdout accuracy.

Writes: `models/intent_model.onnx`, `models/intent_labels.pkl`,
`models/intent_labels.json`, `models/intent_pipeline.pkl`, `models/manifest.json`.

## Multilingual models

```bash
make train-multilingual    # = python multilingual/train_multilingual.py
```

Per-language (en/fr/de/da) variants; per-intent/per-language capping before
merge; shared `text_norm.py`.

## Semantic heads (rescue layer)

```bash
python scripts/train_semantic_head.py                                # English
python packages/buildtime/nlu_training/semantic_multilingual/train_multilingual_semantic_head.py   # multilingual
```

Lightweight classifier head trained over MiniLM ONNX embeddings; used to rescue
low-confidence classifier turns (see `inference.md`).

## Calibration (temperature scaling)

```bash
make calibrate             # = python packages/buildtime/nlu_training/calibrate_languages.py
```

Fits **per-language temperature** (rank-preserving) and writes to
`config/calibration.json`: `temperature`, `conf_threshold`, `conf_gap_threshold`,
`macro_f1_holdout`, `ece`. Rationale + history: `decisions.md`.

Current shipped values (holdout): en T≈0.62 F1≈0.90 ECE≈0.018 · fr T≈0.67
F1≈0.84 · de T≈0.68 F1≈0.83 · da T≈0.82 F1≈0.74. `config/calibration.json` is
the source of truth.

## Full retrain loop

```bash
make train && make calibrate && make export-coreml && make export-coreml-test
```

## Related memory

Datasets -> `datasets.md` · Inference -> `inference.md` · Mobile/export ->
`mobile.md` · Decisions -> `decisions.md`.
