# Memory: Architecture

> Single responsibility: how the system fits together. For code navigation
> (call graphs, symbol lookup, impact analysis) prefer the **Code Graph Memory
> MCP** over reading files. Update this file when component boundaries change.

## System purpose

On-device, multilingual (en/fr/de/da) intent classifier for a hearing-aid app.
Replaces a Dialogflow dependency with a lightweight model that runs **offline**
on iOS and Android.

```
User speaks -> Platform STT (offline) -> Text -> NLU engine -> Intent (+ slots)
```

## Components (where things live)

- **Core classifier** — TF-IDF + LogisticRegression exported to ONNX (~16KB).
  Trained by `packages/buildtime/nlu_training/train.py`, run by `apps/cli/predict.py`.
- **NLU engine** — `packages/runtime/nlu_engine/` (moved from `scripts/nlu/` in ND-2 M1; a deprecated `scripts/nlu` shim aliases the old import path for one release). The real Dialogflow replacement. Per-turn
  priority: **confirmation -> slot-filling (with interruption detection) ->
  classify**. Modules:
  - `engine.py` — orchestrator (`NLUEngine`, `NLUResult`).
  - `classifier.py` — intent classification over the ONNX model + schema-driven
    keyword pre-filter (no intent names hardcoded).
  - `entities.py` — entity + datetime extraction (lexicon-driven, per language).
  - `context.py` — session store / slot state (`SessionStore`).
  - `semantic.py` — embedding-based semantic rescue for low-confidence turns.
  - `manifest.py` — model/artifact manifest.
  - CLI entry points: `apps/cli/nlu_cli.py`, `apps/cli/nlu_cli_multilingual.py`.
- **Multilingual** — `multilingual/`: `train_multilingual.py`,
  `predict_multilingual.py`, `text_norm.py`, per-language temperature-scaling
  calibration -> `config/calibration.json` (**advisory only — no runtime reads
  it; see `inference.md` and Review-F5 blocker B8**).
- **Semantic support** — MiniLM embeddings via ONNX; heads trained by
  `scripts/train_semantic_head*.py`; multilingual variants under
  `packages/buildtime/nlu_training/semantic_multilingual/` and `multilingual/SemanticSupport/`.
- **Mobile export** — `multilingual/export_coreml_multilingual.py` (CoreML
  `.mlpackage`); INT8 quantization; parity tests in `multilingual/test/`.

## Runtime turn flow (NLUEngine.handle)

1. **Confirmation** — if a yes/no follow-up context is active, resolve it first.
2. **Slot-filling** — if mid-collection, fill the pending slot; a high-confidence
   topic switch (>= schema `interrupt_threshold`) abandons the flow and sets
   `NLUResult.interrupted_intent`.
3. **Classify** — fresh turn: ONNX classify -> confidence gate -> optional
   semantic rescue -> entity/datetime extraction -> slot prompts.

Config/schema: `content/nlu_schema.json`, `content/nlu_entities.json`,
`content/localization/`, `config/calibration.json` (moved from `data/` in
ND-2 M3; training CSVs now live in `datasets/`).

## Dependency posture

Runtime is intentionally lean: scikit-learn, skl2onnx, onnxruntime, scipy,
pandas, joblib, dateparser. **No torch/transformers on the inference path** —
embeddings run through ONNX. `coremltools`/`torch` are export-only (macOS/CI).

## On-device constraints (first-class)

- ONNX graph is **static batch size 1** — embed one sentence at a time.
- Preserve numeric parity + calibration when touching export/quantization;
  verify against golden fixtures in `multilingual/test/`.
- Artifacts (`*.onnx`, `*.pkl`, `*.mlpackage`) are gitignored — regenerated.

## Intents

57 intents in the `domain.object.action` taxonomy across 12 capabilities
(device.volume, device.memory, device.status, streaming, reminders,
messaging.ptt, translation, transcription, find, activity, help, sys) —
migrated 2026-07-14 (ND-3, map: `datasets/label_migration_map.json`).
`content/nlu_schema.json` is the source of truth.

## Related memory

Datasets -> `datasets.md` · Training -> `training.md` · Inference ->
`inference.md` · Mobile -> `mobile.md` · Decisions -> `decisions.md` ·
Known issues -> `known-issues.md`.
