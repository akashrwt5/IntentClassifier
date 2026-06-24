# Intent Classifier — Active Pipeline Guide

This is the **single source of truth** for *which* data files are live, what each
one does, and how to train / retrain / test the model.

> **Doc-only by design.** To avoid stale duplicates, no data files were copied
> here. The files themselves still live in [`../data/`](../data/) and the models
> in [`../models/`](../models/). This README is the map.

---

## TL;DR — the commands

Run everything from the **repo root** (`IntentClassifier/`):

```bash
# 1. (only if you changed phrase banks or corrections) regenerate training set
python scripts/build_augmented_data.py

# 2. train the TF-IDF stage     -> models/intent_model.onnx, intent_pipeline.pkl
python scripts/train.py

# 3. train the semantic stage   -> models/semantic_head.npz, semantic_head.json
python scripts/train_semantic_head.py

# 4. validate the whole system  -> pass/fail gate
python scripts/test_holdout.py --strict
```

Inspect detail with:

```bash
python scripts/test_tfidf_only.py     # TF-IDF stage alone, on the holdout
python scripts/test_semantic.py       # 23 curated keyword/TF-IDF/semantic cases
python scripts/test_holdout.py -v     # every holdout phrase, per-intent breakdown
```

The three trainers/tests all **default to the active v3 configuration** — you do
not need to pass `--version` for normal work.

---

## The model in one paragraph

A user utterance flows through **three stages**, stopping at the first confident
answer: **(1) keyword rules** in [`../data/nlu_schema.json`](../data/nlu_schema.json),
**(2) the TF-IDF + logistic-regression classifier** (`intent_model.onnx`), and
**(3) the MiniLM semantic head** (`semantic_head.npz`) which *rescues* utterances
the TF-IDF stage was unsure about. If all three decline, the turn falls back to
GenAI. Stages 2 and 3 are the trained models; both learn from the **same** file.

---

## ACTIVE data files (these are the ones we use)

All in [`../data/`](../data/). Two kinds: **source** (hand-maintained) and
**generated** (rebuilt by a script — never edit by hand).

| File | Rows | Kind | What it is |
|---|---:|---|---|
| **`04_GENERATED_MASTER_training_data.csv`** | ~9 991 | generated | **The training set both models read.** Built by `build_augmented_data.py`. |
| `03_generated_augmented_phrases.csv` | ~1 378 | generated | Just the augmentation phrases (the new rows merged into the file above). Reference only. |
| `01_source_base_training_data.csv` | 8 325 | source | Base training corpus. Input to `build_augmented_data.py`. |
| `02_source_manual_corrections.csv` | 309 | source | Hand-written conversational paraphrases / fixes. Input to `build_augmented_data.py`. |
| `semantic_oos_2.csv` | 367 | source | Curated **out-of-scope** phrases — trained as the "Default Fallback Intent" class so the semantic head learns to *reject* off-domain queries. Used only by `train_semantic_head.py`. |
| `semantic_holdout_2.csv` | 341 | source | **The evaluation benchmark.** Held-out paraphrases never trained on. Used by `test_holdout.py`, `test_tfidf_only.py`, and as a leakage guard by the trainers. |
| `semantic_benchmark_250.csv` | 249 | source | In-distribution benchmark; used **only** in the augmentation leakage blocklist (not for training, not for the gate). |
| `semantic_holdout_100.csv` | 100 | source | Legacy v1 holdout; still consulted in the augmentation leakage blocklist. |

The phrase banks (`P`, `W2`, `W3`, `W4`) that augmentation adds are **inside**
[`../scripts/build_augmented_data.py`](../scripts/build_augmented_data.py) — that
script is where you add new training phrases.

## LEGACY / NOT used by the active pipeline

Kept only so older model versions stay reproducible (`--version 1` / `2`). **Do
not use these for new work.**

| File | Why it exists |
|---|---|
| `01_source_base_training_data.csv` | v1 training corpus |
| `semantic_oos.csv` | v1 OOS set |
| `semantic_holdout_*_expansion_template.csv` | scaffolding templates for growing the holdout |
| `unknown_data.csv` | scratch / unused |

---

## Dependency graph (active v3)

```
            SOURCE (edit these)                         GENERATED (don't edit)
  ┌─────────────────────────────────┐
  │ 01_source_base_training_data.csv  (base)   │
  │ 02_source_manual_corrections.csv     │ ──┐
  │ phrase banks in                 │   │  build_augmented_data.py
  │   build_augmented_data.py       │   ├──────────────►  03_generated_augmented_phrases.csv
  └─────────────────────────────────┘   │                 04_GENERATED_MASTER_training_data.csv  ◄─┐
                                         │                                                    │
  leakage blocklist (never trained on):  │                                                    │
    semantic_holdout_2.csv ──────────────┘                                                    │
    semantic_holdout_100.csv                                                                  │
    semantic_benchmark_250.csv                                                                │
                                                                                              │
  04_GENERATED_MASTER_training_data.csv ─────────────────────────────────────────────────────┬─────┘
                                                                                        │
                          train.py  ──────────►  intent_model.onnx, intent_pipeline.pkl │ (TF-IDF)
                                                                                        │
  semantic_oos_2.csv ──┐                                                                │
  04_GENERATED_MASTER_training_data.csv ──┴── train_semantic_head.py ──► semantic_head.npz/json  (semantic)
```

Key rule the trainers enforce: **nothing in the training set may appear in
`semantic_holdout_2.csv`/`_100`.** `build_augmented_data.py` filters it out and
`train.py` fails the build if any leak slips through — that keeps the benchmark an
honest generalisation test.

---

## Which script reads / writes what

| Script | Trains on | Evaluates on (never trained) | Produces |
|---|---|---|---|
| `build_augmented_data.py` | — | — | `03_generated_augmented_phrases.csv`, `04_GENERATED_MASTER_training_data.csv` |
| `train.py` *(default `-v 3`)* | `04_GENERATED_MASTER_training_data.csv` | `semantic_holdout_2.csv` *(guard + report)* | `intent_model.onnx`, `intent_pipeline.pkl`, `intent_labels.{json,pkl}`, `manifest.json` |
| `train_semantic_head.py` *(default `-v 3`)* | `04_GENERATED_MASTER_training_data.csv` + `semantic_oos_2.csv` | `semantic_holdout_2.csv` *(guard)* | `semantic_head.npz`, `semantic_head.json` |
| `test_holdout.py` *(default `-v 2`)* | — | `semantic_holdout_2.csv` *(full pipeline)* | gate pass/fail |
| `test_tfidf_only.py` | — | `semantic_holdout_2.csv` *(TF-IDF only)* | report |
| `test_semantic.py` | — | curated cases inside the script | pass/fail |

### ⚠️ The version-number gotcha
The trainers default to **`--version 3`** (the *enhanced* data). The end-to-end
test defaults to **`--version 2`**. These are **meant to pair up**: `--version 3`
training is evaluated against `semantic_holdout_2.csv`, which `test_holdout.py`
selects with its own `--version 2`. The numbers differ because they index
different things (data generation vs. holdout file) — just use the defaults.

---

## How to retrain from scratch

```bash
# 0. one-time: fetch the MiniLM embedder if models/minilm-l6-v2.onnx is missing
python scripts/download_minilm.py

# 1. regenerate the training set (ONLY needed if you edited phrase banks,
#    02_source_manual_corrections.csv, or 01_source_base_training_data.csv)
python scripts/build_augmented_data.py
#    -> prints leakage/dedupe counts and "leakage check: 0"

# 2. TF-IDF stage  (gate: test-split accuracy must clear 0.85 or the build aborts)
python scripts/train.py

# 3. semantic stage
python scripts/train_semantic_head.py

# 4. validate
python scripts/test_holdout.py --strict     # gate: total floor + wrong-action ceiling
python scripts/test_tfidf_only.py            # TF-IDF stage in isolation
python scripts/test_semantic.py              # curated keyword/TF-IDF/semantic cases
```

**To add training phrases for a missed intent:** edit the phrase banks in
`build_augmented_data.py` (or add rows to `02_source_manual_corrections.csv`), then run
steps 1→4. Add phrases in the *register* of the miss, never the verbatim holdout
phrase (the leakage filter will drop those anyway).

---

## Models produced (`../models/`)

| File | Stage | Notes |
|---|---|---|
| `intent_model.onnx` | TF-IDF | the deployable classifier (iOS/engine run this) |
| `intent_pipeline.pkl` | TF-IDF | sklearn pipeline, used by `test_tfidf_only.py` |
| `intent_labels.json` / `.pkl` | TF-IDF | class list; must match `nlu_schema.json` |
| `semantic_head.npz` / `.json` | semantic | MiniLM logistic head (npz = runtime, json = iOS) |
| `minilm-l6-v2.onnx`, `minilm-vocab.txt` | semantic | the frozen sentence embedder (from `download_minilm.py`) |
| `manifest.json` | both | bundle manifest + integrity signature (regenerated by the trainers) |

---

## Current accuracy snapshot

Measured on `semantic_holdout_2.csv` with the active v3 models:

| Stage | Score |
|---|---|
| TF-IDF only (`test_tfidf_only.py`) | ~86 % |
| End-to-end (`test_holdout.py`) | ~94 %, wrong-action ≈ 4 |
| Curated suite (`test_semantic.py`) | 23 / 23 |

**Wrong-action** = the pipeline fired the *wrong command* (worse than a safe GenAI
hand-off). It is the safety-critical metric for a hearing-aid app — watch it, not
just the headline accuracy.

> **Reproducibility note:** the semantic-head training has ~±2-phrase run-to-run
> variance (BLAS threading during logistic-regression fit), so end-to-end totals
> wobble by a phrase or two between identical retrains. This is expected; the gate
> in `test_holdout.py` has margin for it.
