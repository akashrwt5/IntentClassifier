# Intent Classifier

On-device, multilingual (en/fr/de/da) intent classifier + NLU engine for a
hearing-aid app. Replaces a Dialogflow dependency with lightweight models that
run **offline** on iOS and Android.

```
User speaks -> Platform STT (offline) -> Text -> NLU engine -> Intent (+ slots)
```

## What's actually here (current generation)

- **NLU engine** — `packages/runtime/nlu_engine/`: per-turn priority *confirmation →
  slot-filling (with interruption detection) → classify*. Orchestrator in
  `engine.py`; classifier, entities/datetime, session state, semantic rescue
  alongside. CLIs: `apps/cli/nlu_cli.py`, `apps/cli/nlu_cli_multilingual.py`.
- **Core classifier** — TF-IDF + LogisticRegression exported to ONNX (~16 KB).
- **Multilingual** — `multilingual/`: training, prediction, text normalization,
  per-language temperature-scaling calibration (`config/calibration.json`).
- **Semantic rescue** — MiniLM embeddings via ONNX for low-confidence turns
  (`scripts/SemanticSupport/`, `multilingual/SemanticSupport/`). No
  torch/transformers on the inference path.
- **Mobile export** — CoreML `.mlpackage` + INT8 quantization
  (`multilingual/export_coreml_multilingual.py`); parity fixtures in
  `multilingual/test/`.

Schema/config sources of truth: `content/nlu_schema.json`, `content/nlu_entities.json`,
`content/localization/`, `config/calibration.json`.

## Setup & workflow

```bash
make install-dev     # deps + ruff/black/darker/mypy/pytest/pre-commit
make train           # train models
make nlu             # interactive NLU CLI
make calibrate       # per-language temperature scaling
make export-coreml   # CoreML export (macOS)
make format && make check   # before every commit
```

Full pipeline reference: `docs/pipelines.md`. Contributor rules:
`CONTRIBUTING.md` and `CLAUDE.md`.

## On-device constraints (first-class)

- ONNX graph is **static batch size 1** — embed one sentence at a time.
- Preserve numeric parity + calibration when touching export/quantization;
  verify against `multilingual/test/` fixtures.
- Generated artifacts (`*.onnx`, `*.pkl`, `*.mlpackage`) are gitignored and
  regenerated via `make train` / `make export-coreml`.

## Languages

English, French, German are supported. **Danish is flag-gated and must not
ship** until it passes a native-authored holdout (currently machine-translated,
macro-F1 ≈ 0.745).

## Data & privacy

- `datasets/01_source_base_training_data.csv` — labeled training data.
- `data/unknown_data.csv` — low-confidence inputs. Privacy stance: default is
  **aggregate counters only**; raw text collection is **opt-in**. See
  `docs/privacy-unknown-data.md`.
- `scripts/auto_label.py` is **quarantined** (writes a retired taxonomy —
  would poison training data). It refuses to run.

## Where the platform is heading

Architecture review, ADRs 001–005 (NLU Bundle, capabilities, orchestration,
GenAI routing, shared runtime) and the phased roadmap live in
`docs/Review-F5/`. Execution progress: `docs/Review-F5/EXECUTION_STATUS.md`.

## Project knowledge

Durable project knowledge lives in `.claude/memory/` (architecture, datasets,
training, inference, mobile, decisions, known issues) — read the file relevant
to your task rather than scanning the repo.
