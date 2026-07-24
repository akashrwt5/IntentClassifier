# Developer Guide — repository map, training, export, local testing

A practical orientation to the repo: where things live, how to train, how to
export to CoreML, how to run and debug locally (incl. PyCharm), which CLI does
what, and how multilingual support works end to end. Paths are relative to the
repo root.

> One-line mental model: **`content/` + `datasets/` are the source of truth →
> `packages/buildtime/` compiles/trains them into artifacts in
> `multilingual/models/` → `packages/runtime/nlu_engine/` loads those at
> runtime → `apps/cli/` are the interactive entry points.**

---

## 1. Directory layout (what each top-level folder is)

| Folder | Purpose |
|---|---|
| `packages/runtime/nlu_engine/` | **The engine** the device/app runs. Pure inference + dialogue. No trainers. |
| `packages/buildtime/nlu_training/` | Trainers, calibration, evaluation, the wrong-action harness, data tools. |
| `packages/buildtime/nlu_export/` | Export to ONNX / **CoreML** / iOS weights. |
| `packages/buildtime/nlu_compiler/` | Bundle compiler + validator + `content_source` (schema ⇄ source tree). |
| `content/` | **Source of truth for behavior**: `nlu_schema.json` (compiled), `platform*.yaml` (guards/thresholds), `capabilities/` (per-intent source), `localization/` (per-language overlays, entities, lexicons). |
| `datasets/` | Training CSVs (`multilingual/{en,fr,de,da}.csv`) + review files. DVC-tracked (`datasets.dvc`). |
| `multilingual/` | The multilingual trainer (`train_multilingual.py`), **`models/`** (trained artifacts), `test/` (holdouts). |
| `apps/cli/` | Interactive CLIs (the things you actually run by hand). |
| `spec/` | Bundle spec 3.0, JSON Schemas, signing keys, example bundles. |
| `tests/` | pytest suite (engine, guards, conformance, bundle lifecycle, parity). |
| `docs/` | ADRs, roadmap, runbooks, this guide, the Review-F5 tracker. |
| `scripts/` | Legacy/dev scripts. **`scripts/nlu/` is a deprecated import shim** — do not build on it. |

Note: `scripts/nlu_cli.py` **no longer exists** — it moved to
`apps/cli/nlu_cli.py` in the ND-2 restructure. If you have one locally, it is a
stale leftover and will load pre-migration `Cmd.*` labels.

---

## 2. Where the models are

Trained artifacts live under **`multilingual/models/<name>/`**, one folder per
model. `<name>` is a language (`en`, `fr`, `de`, `da`), or `multilingual`
(combined) / `multilingual_small` (compact).

Each folder contains:

```
<name>_intent_model.onnx              ONNX graph (server / onnxruntime path)
<name>_intent_pipeline.pkl            fitted sklearn Pipeline (TF-IDF + LogReg)
<name>_intent_labels.json / .pkl      the label list (57 intents, domain.object.action)
<name>_intent_classifier_weights.json TF-IDF + LR weights + temperature (on-device/Swift path)
manifest.json                         SHA-256 of every artifact above
```

Holdout test splits are written next door in `multilingual/test/<name>_holdout.csv`.

Model artifacts are **gitignored and regenerated** (except Danish, tracked as a
deliberate exception). If your labels read `Cmd.VolumeUnmute` you are on a
**stale pre-migration model** — retrain (§3).

Semantic-rescue (stage-3) artifacts: `models/semantic_head.*` +
`multilingual/SemanticSupport/` (the MiniLM encoder). Currently OFF by default
(`semantic_rescue_enabled: false`) pending local regeneration.

---

## 3. Training

Everything is wired through the **Makefile** (run from repo root):

```bash
make train-multilingual     # train en/fr/de/da + combined models -> multilingual/models/
make train                  # train the English root model (models/)
make calibrate              # fit per-language temperature scaling
make check                  # lint + typecheck + full pytest suite
```

Under the hood:
- Multilingual trainer: **`multilingual/train_multilingual.py`**
  (`python multilingual/train_multilingual.py --all`, or `--language fr`).
  TF-IDF (word 1–2 grams) + `LogisticRegression` + temperature-scaling
  calibration; 0.80 accuracy gate; reads `datasets/multilingual/<lang>.csv`.
- English root trainer: `packages/buildtime/nlu_training/train.py`.
- Evaluation JSON + wrong-action budget:
  `python -m nlu_training.evaluate` and
  `python -m nlu_training.wrong_action_harness` (needs
  `PYTHONPATH=packages/buildtime:packages/runtime`).

To change behavior (guards, thresholds, intents) you edit **`content/`**, then
recompile the schema:

```bash
PYTHONPATH=packages/buildtime:packages/runtime \
  python -m nlu_compiler.content_source assemble   # rebuild content/nlu_schema.json + overlays
PYTHONPATH=packages/buildtime:packages/runtime \
  python -m nlu_compiler.content_source check      # drift guard (must say "in sync")
```

---

## 4. CoreML export (iOS)

**Script: `packages/buildtime/nlu_export/export_coreml.py`** — **must run on
macOS with `coremltools`.** Makefile target:

```bash
make export-coreml          # FP16 + FP32 .mlpackage bundles -> models/
make export-coreml-test     # numeric-equivalence (Tier-A) check
```

Direct invocation and flags:

```bash
pip install coremltools onnxruntime numpy torch transformers
python packages/buildtime/nlu_export/export_coreml.py            # full (FP16 MiniLM)
python packages/buildtime/nlu_export/export_coreml.py --quantize # also INT8 MiniLM
python packages/buildtime/nlu_export/export_coreml.py --seq-len 32  # ANE-resident fixed seq
```

Outputs (to `models/`): `IntentClassifier.mlpackage` (stage-2 TF-IDF LogReg),
`SemanticHead.mlpackage`, `MiniLMEmbedder.mlpackage`. Important design note in
the script header: the CoreML intent model is built from
`*_intent_classifier_weights.json`, **not** the ONNX, because coremltools can't
convert the string-input TF-IDF subgraph or the sklearn calibration. Related:
`export_ios_weights.py` (emits the weights JSON), `compare_coreml_quant.py`
(measure INT8 accuracy delta before shipping), `copy_artifacts_to_stt.py`
(hand off to the iOS/STT repo).

This step is macOS-only, so it is an owner-machine action (can't run in the
Linux sandbox).

---

## 5. The CLIs — which script for what

All committed CLIs live in **`apps/cli/`**:

| Script | What it does | Run it |
|---|---|---|
| `nlu_cli.py` | Full multi-turn **English** NLU demo (engine end to end). | `python apps/cli/nlu_cli.py`  or  `make nlu` |
| `nlu_cli_multilingual.py` | Same, but with **language/model selection** (`--language fr` / `--model de`). | `python apps/cli/nlu_cli_multilingual.py -l fr` |
| `predict.py` | Lightweight intent-only prediction (English ONNX). | `python apps/cli/predict.py`  or  `make predict` |
| `unknown_log.py` | **Analytics / logging** for low-confidence ("unknown") turns. Aggregate counters by default (`data/unknown_counters.csv`); raw text only behind `NLU_COLLECT_RAW_UNKNOWN`. | imported by `predict.py`; see its header |

So: interactive analysis = `nlu_cli_multilingual.py`; the "analytics" data
capture is `unknown_log.py` (privacy-conscious counters, per ND-5).

Quick sanity check that you are on the migrated model: run
`python apps/cli/nlu_cli.py`, type `turn mute off` → expect
`device.volume.unmute` (NOT `Cmd.VolumeUnmute`).

---

## 6. Testing locally in PyCharm

The one thing that trips people up: the code is split across `packages/`, so
the interpreter needs **both package roots on `PYTHONPATH`**.

**Mark as Sources Root** (right-click → *Mark Directory as → Sources Root*):
- `packages/runtime`
- `packages/buildtime`

That makes `import nlu_engine`, `import nlu_training`, `import nlu_compiler`
resolve without a shim.

**Run/Debug configuration** (for a CLI or script):
- Script path: e.g. `apps/cli/nlu_cli_multilingual.py`
- Parameters: e.g. `--language fr`
- Working directory: the repo root
- Environment variables (if you run trainers/harness from a plain config):
  `PYTHONPATH=packages/buildtime:packages/runtime`

**pytest in PyCharm:** set the test runner to pytest (Settings → Tools → Python
Integrated Tools), working dir = repo root. Then you can run/debug any file in
`tests/` directly (e.g. `tests/test_help_marker_guard.py`), set breakpoints in
`packages/runtime/nlu_engine/engine.py`, and step through `handle()`.

From the terminal the equivalents are `make check` (all gates) or
`python -m pytest -q`.

---

## 7. Multilingual support — CLI, models, and the entity extractor

**Yes, it is multilingual end to end (en / fr / de / da).** Three independent
layers each carry per-language support:

1. **Intent models** — one trained model per language in
   `multilingual/models/{en,fr,de,da}/` (plus a combined `multilingual` model).
   The CLI picks one via `nlu_cli_multilingual.py --language fr` (or
   `--model de`); the engine is constructed as
   `NLUEngine(model_name=lang, language=lang)`.

2. **Schema / behavior overlays** — `content/localization/nlu_schema.<lang>.json`
   carries per-language guards, prompts, yes/no lexicons, and confirm texts;
   they overlay the base `content/nlu_schema.json` at load time. (This is why
   the polarity guards and the ND-14 help-marker guard have per-language
   pattern sets.)

3. **Entity extraction** — **yes, the entity extractor supports other
   languages.** `packages/runtime/nlu_engine/entities.py` (`EntityExtractor`)
   takes a `language` argument. For non-English it loads:
   - `content/localization/nlu_entities.<lang>.json` — the entity definitions,
   - `content/localization/nlu_lexicon.<lang>.json` — the datetime/number
     lexicon that drives date/time parsing for that language.

   The engine wires this automatically in `_load_entities(language)`: English
   uses the built-in path; `fr`/`de`/`da` each load their own entities +
   lexicon. Datetime parsing correctness is covered per-language by
   `tests/datetime_parity/nlu_datetime_parity_{fr,de,da}.csv` (e.g. French
   "huit heures moins le quart" → 07:45). So French (and German, Danish) get
   real localized number/date/relative-time handling, not just translated
   intent labels.

   Adding a new language = add its CSV to `datasets/multilingual/`, register it
   in `train_multilingual.py`'s `LANGUAGES`, author
   `nlu_entities.<lang>.json` + `nlu_lexicon.<lang>.json` + the schema overlay,
   then `make train-multilingual`.

---

## 8. Quick reference — common tasks

```bash
# run the engine interactively (French)
python apps/cli/nlu_cli_multilingual.py --language fr

# retrain everything on the current (migrated) label space
make train-multilingual

# recompile behavior after editing content/
PYTHONPATH=packages/buildtime:packages/runtime python -m nlu_compiler.content_source assemble

# measure the wrong-action budget end to end
PYTHONPATH=packages/buildtime:packages/runtime python -m nlu_training.wrong_action_harness

# all quality gates
make check

# CoreML (macOS only)
make export-coreml
```
