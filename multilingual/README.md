# Multilingual Intent Models

Self-contained multilingual extension of the TF-IDF intent classifier. It does
**not** modify `scripts/train.py`; it mirrors that script's proven pipeline in a
parameterised generator that can build a model per language plus one combined
model.

## Layout

```
multilingual/
├── data/                       # per-language training data (text,intent)
│   ├── en.csv                  # ← data/04_GENERATED_MASTER_training_data.csv
│   ├── fr.csv                  # ← data/Generated_Master_training_French_Data.csv
│   └── de.csv                  # ← data/dialogflowData/de.csv
├── models/                     # generated models, one folder per model
│   ├── en/  fr/  de/           # per-language models
│   └── multilingual/           # combined model (all languages)
├── train_multilingual.py       # generator (accepts --language / --data / --all)
└── test/
    ├── test_multilingual_models.py
    └── <name>_holdout.csv       # held-out split written at train time
```

Each `models/<name>/` folder contains a complete, self-contained bundle:

| File | Purpose |
|---|---|
| `<name>_intent_model.onnx` | ONNX model — server / `onnxruntime` path |
| `<name>_intent_pipeline.pkl` | fitted sklearn `Pipeline` |
| `<name>_intent_labels.json` / `.pkl` | label list |
| `<name>_intent_classifier_weights.json` | raw TF-IDF + LR weights — on-device path |
| `manifest.json` | SHA-256 of every artifact in the folder |

The combined model's ONNX is named `multilingual_intent_model.onnx`.

## Data source mapping

| Lang | Source file |
|---|---|
| `en` | `data/04_GENERATED_MASTER_training_data.csv` |
| `fr` | `data/Generated_Master_training_French_Data.csv` |
| `de` | `pva_intent_german.csv` master (9,987 rows, 59 intents — promoted from `data/pending/`; replaced the weak `dialogflowData/de.csv` which scored 27%) |
| `da` | `pva_intent_danish.csv` master (9,987 rows, 59 intents — promoted from `data/pending/`) |

> **Staged data:** higher-quality German and a new Danish dataset are stashed
> under `data/pending/` (see `data/pending/README.md`) for later integration —
> not part of the current build.

Add a language with one line in the `LANGUAGES` registry at the top of
`train_multilingual.py` — the design target is "any number of languages". The
combined model is the concatenation of every registered language, so its label
space is the **union** of all per-language intents (these differ across
languages today).

## Usage

```bash
pip install -r ../requirements.txt   # from repo root: pip install -r requirements.txt

# build everything (per-language + combined)
python multilingual/train_multilingual.py --all

# one language from its registered file
python multilingual/train_multilingual.py --language fr

# one language from a custom file
python multilingual/train_multilingual.py --language fr --data some/fr.csv

# only the combined model
python multilingual/train_multilingual.py --language multilingual

# test all generated models (CI gate)
python multilingual/test/test_multilingual_models.py
python multilingual/test/test_multilingual_models.py --model fr --verbose
```

## ✅ Resolved — ONNX vs sklearn divergence on accented text

**Symptom (historical):** the exported `.onnx` models underperformed the
in-memory sklearn pipeline on non-ASCII languages, because `skl2onnx`'s
`TfidfVectorizer` tokenizer does not replicate Python's Unicode `\w`
word-boundary semantics — words with `æ ø å ä ö ü é è à ç …` tokenised
differently at ONNX inference than during training and missed the vocabulary.
Danish lost ~10 points (ONNX 0.695 vs sklearn 0.794; 225/1362 disagreements,
all on æ/ø/å). `lowercase=False` did not help — it is the tokenizer, not the
lowercasing op.

**Fix:** `multilingual/text_norm.py::normalize_text()` folds every input to
lowercase ASCII (`ø→o`, `æ→ae`, `ß→ss`, `é→e`, `å→a`, …) **before** the
vectorizer, applied identically in training (`train_multilingual.py`) and
inference (`test_multilingual_models.py`). Once the surface form is plain ASCII
the ONNX tokenizer and sklearn agree.

**Result — ONNX now matches sklearn, and accuracy recovered:**

| Model | Labels | sklearn test-split | ONNX holdout | sklearn↔ONNX agreement |
|---|---|---|---|---|
| `en` | 59 | 0.916 | 0.912 | ~1.00 |
| `fr` (é è à ç) | 59 | 0.866 | 0.866 | ~0.99 |
| `de` (ä ö ü ß) | 59 | 0.849 | 0.848 | 0.994 |
| `da` (æ ø å) | 59 | 0.791 | 0.791 | **1.000** |
| `multilingual` | 59 | 0.883 | 0.882 | ~0.99 |

The ONNX holdout now tracks the sklearn test-split for every language (the gap
was ~10 pts for Danish before the fix). The few remaining fr/de disagreements
(≤1%) are float-precision boundary cases, not the systematic Unicode bug.

> French covers all 59 intents (expanded from 54 after the
> `Generated_Master_training_French_Data.csv` data refresh); the larger,
> fuller-coverage set scores 0.866 vs the earlier narrower 54-intent set.

> ⚠️ **Swift port:** the on-device path must apply the *same* `normalize_text`
> logic (see `_FOLD` + NFKD in `text_norm.py`) before computing the TF-IDF
> vector, or iOS will diverge from the server the same way ONNX used to.

> **No leakage:** the per-model test split is written *before* the final fit and
> the exported model is trained on the train split only (no refit on all data),
> so `test_multilingual_models.py` measures genuine held-out generalisation.
> (This is why earlier "99%/97%" numbers fell to ~91%/89% — those were inflated
> by training on the holdout, not a real regression.)

### Environment note — locale

The exported ONNX `TfidfVectorizer` contains a `StringNormalizer` op that
requires the **`en_US.UTF-8` locale** at inference time. On a minimal container
this errors with *"Failed to construct locale with name: en_US.UTF-8"*. Fix:

```bash
localedef -i en_US -f UTF-8 en_US.UTF-8
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
```

### Accuracy gate

The accuracy gate defaults to `0.80` (vs `0.85` in `train.py`) because some
language sets generalise less — German has many intents with few rows each.
Override with `--min-accuracy` or `MIN_TEST_ACCURACY=`.

---

## How iOS / Core ML inference works today (and what's deferred)

**Question this answers:** is the iOS Core ML model built from the `.onnx` file
or from the weights JSON?

**Answer: the iOS intent classifier is built from
`intent_classifier_weights.json`, _not_ from `intent_model.onnx`.** This is by
design (see the Stage-2 note in `scripts/export_coreml.py`):

1. The production sklearn pipeline uses `CalibratedClassifierCV(method="isotonic")`.
   `coremltools` does **not** support isotonic calibration and disables its
   sklearn converter entirely for scikit-learn > 1.5.1.
2. The ONNX export contains a **string-input TF-IDF subgraph**, and `coremltools`
   cannot convert ONNX string tensors.

So the on-device path is:

| Stage | iOS source artifact | How it's converted |
|---|---|---|
| **Stage 2 — TF-IDF intent classifier** (`IntentClassifier.mlpackage`) | `intent_classifier_weights.json` | Swift computes the TF-IDF float vector natively (`tfidfVector()`), feeds it into a linear+softmax Core ML graph built from `coef`/`intercept`; the per-class **isotonic calibration tables** are applied **in Swift** to match the server's calibrated probabilities. Tables are emitted by `scripts/export_ios_weights.py`. |
| **Stage 3b — MiniLM embedder** (`MiniLMEmbedder.mlpackage`) | HuggingFace **PyTorch** model via `torch.jit.trace` | `coremltools` 9 removed the ONNX frontend, so MiniLM is traced from PyTorch, not converted from `minilm-l6-v2.onnx`. The ONNX file is kept only for shape introspection. |

In short: **`.onnx` drives the Python/server path (onnxruntime); the weights
JSON (+ isotonic tables) drives the iOS Core ML Stage-2 path.** That is exactly
why each multilingual model also emits a `classifier_weights.json`.

### ⚠️ Deferred — Swift / Core ML parity for multilingual (handle later)

Per the current decision, the Swift/Core ML side is **not** done here yet. The
multilingual generator intentionally trains a **plain (uncalibrated)
`LogisticRegression`** so a clean `classifier_weights.json` can be emitted, but
the following remain open and must be handled before shipping multilingual to
iOS:

- [ ] **Isotonic calibration tables** — `train_multilingual.py` writes
      `"calibration": null` in each `classifier_weights.json`. Production iOS
      relies on the isotonic tables from `scripts/export_ios_weights.py`; decide
      whether multilingual models need calibration and, if so, generate the
      tables per language / for the combined model.
- [ ] **Per-language / combined Core ML export** — extend / parameterise
      `scripts/export_coreml.py` to emit `*.mlpackage`s for each multilingual
      model (Stage 2 from weights JSON; Stage 3b MiniLM is multilingual-capable
      only if the embedder itself is multilingual — see below).
- [ ] **Multilingual MiniLM** — `all-MiniLM-L6-v2` is English-centric. For
      semantic-stage parity across FR/DE, evaluate a multilingual embedder
      (e.g. `paraphrase-multilingual-MiniLM-L12-v2` / multilingual E5/BGE) and
      its vocab/tokeniser impact on the Swift side.
- [ ] **Vocab size on-device** — multilingual TF-IDF vocab is larger; apply the
      `--top-per-class` pruning from `export_ios_weights.py` to keep the
      on-device weights file small.
- [ ] **iOS conformance test** — extend `scripts/test_ios_conformance.py` to
      cover the multilingual models.
