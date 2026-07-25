# Memory: Training Pipeline

> Single responsibility: how models are trained + calibrated. Inference is in
> `inference.md`; export is in `mobile.md`. For library APIs (scikit-learn,
> skl2onnx) use the **Context7 MCP** rather than recalling from memory.

## English core model

```bash
python scripts/train.py           # version 3 data (default)
python scripts/train.py -v 1      # original files
python scripts/train.py -v 2      # corrected _2 files
python scripts/train.py -v 3      # enhanced: _2 + 02_source_manual_corrections
```

`-v` also selects the leakage-guard holdout: `-v 1` → `semantic_holdout_100.csv`,
`-v 2`/`-v 3` → `semantic_holdout_2.csv`.

Pipeline: `TfidfVectorizer` -> `LogisticRegression` (sklearn `Pipeline`) exported
via `skl2onnx` with `StringTensorType([None, 1])` (dynamic batch). The script:
- caps samples per intent at `MAX_PER_INTENT = 500` (keeps the newest rows),
- runs a **holdout-leakage guard** — raises if any training phrase also appears
  in the holdout,
- 3-fold cross-validates (`cross_val_score(..., cv=3)`),
- enforces a **test-split accuracy floor**: `MIN_TEST_ACCURACY` default **0.85**,
  overridable via env (`MIN_TEST_ACCURACY=0.80 python scripts/train.py`),
- prints classification report + confusion matrix.

> The floor is on the **test split**, not on the 100-utterance paraphrase
> holdout — that holdout is scored and recorded in the manifest for visibility
> only. Don't confuse the two when reading a failure.

Writes: `models/intent_model.onnx`, `models/intent_labels.pkl`,
`models/intent_labels.json`, `models/intent_pipeline.pkl`, `models/manifest.json`.

## Multilingual models

```bash
python multilingual/train_multilingual.py
```

Per-language (en/fr/de/da) variants; per-intent/per-language capping before
merge; shared `text_norm.py`.

## Semantic heads (rescue layer)

```bash
python scripts/train_semantic_head.py          # English
python scripts/train_semantic_head_coreml.py   # CoreML-targeted variant
python scripts/build_semantic_index.py
```

Lightweight classifier head trained over MiniLM ONNX embeddings; used to rescue
low-confidence classifier turns. **The stage ships off by default** — see
`inference.md` / `langpack.md`.

## Calibration (temperature scaling)

```bash
python scripts/calibrate_languages.py
```

Fits **per-language temperature** (rank-preserving) and writes to
`config/calibration.json`: `temperature`, `conf_threshold`, `conf_gap_threshold`,
`macro_f1_holdout`, `ece`. Rationale + history: `decisions.md` ADR-003/004.

Currently committed (holdout):

| Lang | T | conf_threshold | macro-F1 | ECE |
|---|---|---|---|---|
| en | 0.6214 | 0.60 | 0.9018 | 0.0184 |
| fr | 0.6699 | 0.60 | 0.8438 | 0.0223 |
| de | 0.6777 | 0.60 | 0.8318 | 0.0168 |
| da | 0.8156 | 0.60 | 0.7448 | 0.0352 |

`config/calibration.json` is the source of truth — read it rather than trusting
this table if they ever disagree.

## Release gate

`scripts/ci/evaluate_gate.py` + `config/gate_thresholds.json`. A model that
trains but does not clear the bar is **blocked, not shipped**:
`min_accuracy` 0.80, `min_macro_f1` 0.60, `max_wrong_action` 15, evaluated on
`data/semantic_holdout_100.csv` with semantic enabled. Last recorded dry run:
acc 0.87 / macro-F1 0.68 / 1 wrong-action — **PASS**.
Tighten these over time; **never loosen silently**.

## Full retrain loop

```bash
python scripts/train.py \
  && python scripts/calibrate_languages.py \
  && python multilingual/export_coreml_multilingual.py --all --fp16 --fp32 \
  && python scripts/test_holdout.py --strict
```

`test_holdout.py` takes `-v {1,2}` (default 2), `--strict` (exit 1 below budget —
this is the gate), `--verbose`, `--json FILE`, and `--no-semantic`.

### What the semantic stage is worth (measured 2026-07-25, v2 holdout, 341 rows)

| Config | Correct | Accuracy | Wrong-action | Safe GenAI fallbacks |
|---|---|---|---|---|
| Semantic **ON** | 320–323 | 93.8–94.7% | **6** | 12–15 |
| Semantic **OFF** (ships) | 280 | 82.1% | **6** | 55 |

Semantic is worth **~+12.6 points** and converts ~43 GenAI hand-offs into
direct answers, at **no wrong-action cost** — the safety-critical number is 6 in
both. That is a strong argument for revisiting the off-by-default posture
(ADR-007), weighed against its memory/latency cost on device.

The ON range is not measurement sloppiness — the score is genuinely
non-deterministic run to run. See `known-issues.md`.

A dry-run retrain overwrites `models/` — restore it if you were only testing the
pipeline.

## Related memory

Datasets -> `datasets.md` · Inference -> `inference.md` · Mobile/export ->
`mobile.md` · Decisions -> `decisions.md`.
