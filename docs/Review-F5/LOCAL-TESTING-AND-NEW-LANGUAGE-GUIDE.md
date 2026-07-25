# Local Testing & New-Language Guide

> **Branch:** `feature/refactoring-for-production-English-support`
>
> **Purpose:** How to run the full NLU pipeline locally — training, testing,
> validation, gates, CoreML export/parity, and pack assembly — exactly as CI
> runs it, plus the complete checklist for adding a new language.
>
> **Scope:** English is the production default. The engine is language-neutral;
> everything below runs against `packs/en` unless a different pack is selected.

---

## 0. Guidelines — read this first

The rules the pipeline enforces, so local runs match CI and nothing ships by accident:

1. **The engine never contains language.** All language-specific data lives in a
   Language Pack (`packs/<lang>/`). The neutrality guard (`check_language_neutral.py`)
   fails the build if any `if language == …` branch appears in engine code.
2. **English is the default.** When no language/pack is passed, the engine loads
   `packs/en`. Never re-bake English into the engine — put it in the pack.
3. **The semantic (MiniLM / Stage-3) layer is a plugin, OFF by default.** It is
   only enabled explicitly (constructor arg → `NLU_ENABLE_SEMANTIC` env → pack
   config → otherwise False). The accuracy gate turns it on deliberately.
4. **GitHub Releases are the ONLY production registry.** CI *artifacts* are
   transient and must never be treated as production distribution. Locally, the
   `dist/` folder is throwaway build output (gitignored).
5. **A model that trains but fails a gate is BLOCKED, not shipped.** Three gates:
   accuracy, Swift↔ONNX conformance, and CoreML↔ONNX parity. All must pass.
6. **Generated models go in a separate folder.** Local training writes to
   `models/`; CI build output (packs, CoreML) goes to `dist/`. Neither is the
   source of truth — the pack + data are.

---

## 1. One-time setup

```bash
# From the repo root
cd /path/to/IntentClassifier

# (Recommended) isolate dependencies
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install scikit-learn            # training/export dep (matches CI)
```

> The semantic layer needs the MiniLM model. It is **off by default**, so you
> only need this if you plan to run with `--semantic` / `enable_semantic=True`:
>
> ```bash
> python scripts/download_minilm.py
> ```

---

## 2. Train (produce the model locally)

Reproduces the CI `train` step. Writes to `models/` (the generated-models folder).

```bash
# Train the production English intent model (TF-IDF + LogisticRegression -> ONNX)
python scripts/train.py
```

Outputs written to `models/`:

- `intent_model.onnx` — the exported model (Android runtime path)
- `intent_labels.pkl` — label list
- `intent_pipeline.pkl` — the fitted sklearn pipeline (source of truth for weights)

Then export the on-device weights **from the same pipeline** (so ONNX, CoreML,
and the iOS scorer all agree):

```bash
python scripts/export_weights.py
# -> models/intent_classifier_weights.json  (vocab, idf, coef, intercept, temperature)
```

> **Why both steps:** `train.py` fits the model and writes the ONNX graph;
> `export_weights.py` derives the flat weight JSON *from the same
> `intent_pipeline.pkl`* rather than retraining. Running them together is what
> keeps every platform bit-for-bit consistent.

---

## 3. Test (correctness & parity checks)

Run these in the order CI does. Any non-zero exit = a real failure.

```bash
# 3a. Neutrality guard — no language baked into the engine
python scripts/ci/check_language_neutral.py

# 3b. Datetime golden-corpus parity (English), 77-case guardrail
python -m pytest tests/test_datetime_parity_en.py -q
python -m pytest tests/test_neutrality.py -q

# 3c. Held-out benchmark (paraphrases NOT in training data)
python scripts/test_holdout.py            # report only (v2, 345-row)
python scripts/test_holdout.py --strict   # GATE: exit 1 if below budget
python scripts/test_holdout.py --verbose  # show every phrase

# 3d. iOS/Swift <-> ONNX conformance (device featurizer parity)
#     top-1 agreement + 0.70 fire/fallback agreement on every utterance
python scripts/test_ios_conformance.py --model production
python scripts/test_ios_conformance.py --model production --verbose   # per-utterance trace

# 3e. Full functional suite (optional but recommended)
python -m pytest tests/ -q
```

Interactive smoke test — type utterances and see the resolved intent/action:

```bash
python scripts/nlu_cli.py
# you ▸ turn it up
# you ▸ set a reminder for 5pm
# you ▸ exit
```

---

## 4. Validate (the accuracy gate)

This is the gate that decides shippability. It runs the labeled holdout through
the engine and blocks if accuracy / macro-F1 / wrong-action budgets aren't met.

```bash
python scripts/ci/evaluate_gate.py --out dist/report_card.json
cat dist/report_card.json
```

Thresholds live in `config/gate_thresholds.json`:

| Key                 | Current value                    | Meaning                                  |
| ------------------- | -------------------------------- | ---------------------------------------- |
| `holdout`           | `data/semantic_holdout_100.csv`  | Labeled validation set                   |
| `enable_semantic`   | `true`                           | Gate runs with the semantic plugin ON    |
| `min_accuracy`      | `0.80`                           | Hard floor on holdout accuracy           |
| `min_macro_f1`      | `0.60`                           | Hard floor on macro-F1                   |
| `max_wrong_action`  | `15`                             | Max tolerated wrong-action count         |

> Prints `GATE PASSED` / `GATE FAILED`. Tighten thresholds over time; never
> loosen them silently.

---

## 5. CoreML export + parity (iOS / ANE path)

> **macOS only.** `coremltools` + the Core ML runtime need macOS; on Linux the
> export/runtime parity step can't run (you can still `--inspect` a package's
> protobuf, see below).

```bash
# 5a. Export the ANE-eligible CoreML intent model from the SAME production weights
python scripts/ci/export_coreml_intent.py --lang en --out dist/models/model.mlpackage

# 5b. CoreML <-> ONNX parity GATE (Tier-A structural + Tier-B real runtime)
python scripts/ci/verify_coreml_parity.py \
  --pipeline models/intent_pipeline.pkl \
  --onnx     models/intent_model.onnx \
  --coreml   dist/models/model.mlpackage \
  --holdout  data/semantic_holdout_100.csv \
  --runtime  --out dist/coreml_parity.json
```

Inspect any `.mlpackage` without a Mac runtime (pure-Python protobuf parse —
confirms ANE-eligible `mlProgram` format + input shape):

```bash
python scripts/ci/verify_coreml_parity.py --inspect --coreml dist/models/model.mlpackage
```

---

## 6. Assemble the versioned pack (what a Release ships)

Builds the deterministic `.nlu` archive (pack + SHA-256 manifest + report card +
models) into `dist/`. This is a local dry-run of what `release-pack.yml` publishes.

```bash
# ONNX-only
python scripts/ci/assemble_pack.py --version 1.0.0 --report dist/report_card.json --out dist

# ONNX + CoreML (bundles the mlpackage under intent_model/)
python scripts/ci/assemble_pack.py \
  --version 1.0.0 \
  --report dist/report_card.json \
  --coreml dist/models/model.mlpackage \
  --out dist
# -> dist/pack-en-v1.0.0.nlu
```

> `dist/` is gitignored throwaway output. Run `bash scripts/cleanup.sh` to remove it.

---

## 7. End-to-end local run (copy/paste)

The full sequence CI runs, condensed:

```bash
# --- train + export weights ---
python scripts/train.py
python scripts/export_weights.py

# --- gates that must pass ---
python scripts/ci/check_language_neutral.py
python scripts/test_ios_conformance.py --model production
python scripts/ci/evaluate_gate.py --out dist/report_card.json

# --- iOS / CoreML (macOS only) ---
python scripts/ci/export_coreml_intent.py --lang en --out dist/models/model.mlpackage
python scripts/ci/verify_coreml_parity.py \
  --pipeline models/intent_pipeline.pkl --onnx models/intent_model.onnx \
  --coreml dist/models/model.mlpackage --holdout data/semantic_holdout_100.csv \
  --runtime --out dist/coreml_parity.json

# --- assemble the pack ---
python scripts/ci/assemble_pack.py --version 1.0.0 --report dist/report_card.json \
  --coreml dist/models/model.mlpackage --out dist
```

If every step exits 0, the build is releasable. On `main`, `release-pack.yml`
does exactly this and publishes the `.nlu` to a GitHub Release.

---

## 8. Adding a new language

**The engine needs zero code changes.** Adding a language is a *pack + data +
model* task, plus light parameterization of the build pipeline. Two spots outside
the engine currently assume English (§8.4) — call those out honestly.

### 8.1 Author the Language Pack — `packs/<lang>/`

Mirror `packs/en/`. Every file is **native-authored**, not machine-translated:

| File                       | What it holds                                                        |
| -------------------------- | -------------------------------------------------------------------- |
| `pack.json`                | Manifest: language code, pack version, model refs, runtime-contract  |
| `config.json`              | Thresholds, `semantic_enabled` (keep `false`)                        |
| `lexicons.json`            | yes/no, carrier phrases, connectors, idioms                          |
| `keywords.json`            | Keyword pre-filter rules (`regex` / `not_regex` / `exact`)           |
| `normalizer.json`          | Text normalization rules                                             |
| `schema.json`              | Intents, slots, actions, prompts, fulfillment (translated)           |
| `entities/enums.json`      | Entity synonyms                                                      |
| `datetime/grammar.json`    | Weekday/month names, am/pm vs 24h, relative-time idioms              |
| `intent_model/`            | Trained `model.onnx`, `labels.pkl`, `weights.json` (from §8.3)       |
| `semantic/` (optional)     | MiniLM artifacts if the plugin is enabled for this language          |

### 8.2 Provide training data + a native holdout

- Training utterances for the new language under `data/`.
- A **native-authored holdout** (not machine-translated — that was the Danish
  lesson: translated holdouts hide real errors). This becomes the language's
  accuracy-gate set.

### 8.3 Train the model for the language

Train on the new-language data, then export weights from that pipeline (same two
steps as §2, pointed at the new data). Copy the resulting `model.onnx`,
`labels.pkl`, and `weights.json` into `packs/<lang>/intent_model/`.

### 8.4 Parameterize the build (the only code/config edits)

These are **workflow/config** edits, not engine edits:

1. **CI workflows are hardcoded to `en`** — `export_coreml_intent.py --lang en`,
   `packs/en` paths, `--model production`, tag `pack-en-v…`. Add the language to
   a matrix or a `language` dispatch input so the pipeline builds/releases it.
2. **Datetime is not yet fully pack-self-contained for non-English.** English
   datetime reads from the pack's `datetime/grammar.json`; the existing fr/de/da
   read from the legacy `data/localization/nlu_lexicon.<lang>.json` via the
   generic interpreter. A new language needs its datetime either as a
   localization lexicon (existing path) or a small consolidation so the pack's
   `datetime/grammar.json` drives the generic interpreter too.

### 8.5 Validate the new language locally

```bash
# Neutrality still clean (engine untouched)
python scripts/ci/check_language_neutral.py

# Accuracy gate against the native holdout (point gate_thresholds.json at it,
# or pass the language's threshold config)
python scripts/ci/evaluate_gate.py --out dist/report_card_<lang>.json

# Device parity for the language's model
python scripts/test_ios_conformance.py --model <lang>

# Assemble the language's pack
python scripts/ci/assemble_pack.py --pack packs/<lang> --version 1.0.0 \
  --report dist/report_card_<lang>.json --out dist
```

### 8.6 Definition of done for a new language

- [ ] `packs/<lang>/` fully authored (all files in §8.1), native-reviewed
- [ ] Training data + native holdout committed
- [ ] Model trained; `intent_model/` populated from the same pipeline
- [ ] Neutrality guard passes (no engine edits)
- [ ] Accuracy gate passes on the native holdout
- [ ] Swift↔ONNX conformance passes for the language's model
- [ ] CoreML↔ONNX parity passes (if shipping iOS)
- [ ] Pipeline parameterized for the language; datetime wired (§8.4)

---

## Appendix — file map

| Path                                     | Role                                            |
| ---------------------------------------- | ----------------------------------------------- |
| `scripts/train.py`                       | Train intent model → ONNX                       |
| `scripts/export_weights.py`              | Export flat weights from the fitted pipeline    |
| `scripts/test_holdout.py`                | Held-out paraphrase benchmark                   |
| `scripts/test_ios_conformance.py`        | Swift↔ONNX device parity gate                   |
| `scripts/nlu_cli.py`                     | Interactive REPL smoke test                     |
| `scripts/ci/check_language_neutral.py`   | Neutrality guard (no language in engine)        |
| `scripts/ci/evaluate_gate.py`            | Accuracy gate                                   |
| `scripts/ci/export_coreml_intent.py`     | ANE CoreML export (macOS)                       |
| `scripts/ci/verify_coreml_parity.py`     | CoreML↔ONNX parity gate + `--inspect`           |
| `scripts/ci/assemble_pack.py`            | Build versioned `.nlu` pack                     |
| `config/gate_thresholds.json`            | Accuracy-gate thresholds                        |
| `packs/en/`                              | The English Language Pack (default)             |
| `models/`                                | Generated models (local training output)        |
| `dist/`                                  | Throwaway build output (gitignored)             |
| `.github/workflows/pr.yml`               | Neutrality guard + tests on any branch/PR       |
| `.github/workflows/train-and-gate.yml`   | Retrain + gate on data/pack/content change      |
| `.github/workflows/release-pack.yml`     | Producer of production Releases (main only)     |
