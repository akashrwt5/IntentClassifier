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

## The organising idea on this branch

The engine is **language-neutral**; every language-specific input arrives from a
**Language Pack** (`packs/<lang>/`) through a locked contract
(`packages/nlu_langpack/`). A bare `NLUEngine()` loads `packs/en` by default.
Full detail: `langpack.md`.

## Components (where things live)

- **Language Pack contract** — `packages/nlu_langpack/`: `interfaces.py`
  (engine-facing Protocols), `version.py` (`RUNTIME_CONTRACT_VERSION = "1.0"` +
  compat gate), `manifest.py` (`pack.json` parse/validate), `pack.py`
  (`LanguagePack` container), `loader.py` (`load_pack()`), `errors.py`.
- **Reference pack** — `packs/en/`: `pack.json`, `config.json`, `schema.json`
  (59 intents, 32 keyword triggers), `keywords.json`, `lexicons.json`,
  `normalizer.json`, `entities/enums.json`, `datetime/grammar.json`,
  `intent_model/`, `semantic/`.
- **NLU engine** — `scripts/nlu/`. The real Dialogflow replacement. Per-turn
  priority: **confirmation -> slot-filling (with interruption detection) ->
  classify**. Modules:
  - `engine.py` — orchestrator (`NLUEngine`, `NLUResult`); always loads a pack.
  - `classifier.py` — intent classification over the ONNX model + pack-driven
    keyword pre-filter (no intent names hardcoded).
  - `entities.py` — entity + datetime extraction, driven by the pack's
    `datetime/grammar.json` and lexicons.
  - `context.py` — session store / slot state (`SessionStore`).
  - `semantic.py` — embedding-based semantic rescue for low-confidence turns.
  - `manifest.py` — model/artifact manifest.
  - CLI entry points: `scripts/nlu_cli.py`, `scripts/nlu_cli_multilingual.py`.
- **Core classifier training** — `scripts/train.py` (TF-IDF +
  LogisticRegression -> ONNX, ~16KB), run interactively by `scripts/predict.py`.
- **Multilingual** — `multilingual/`: `train_multilingual.py`,
  `predict_multilingual.py`, `text_norm.py`, per-language temperature-scaling
  calibration -> `config/calibration.json`.
- **Semantic support** — MiniLM embeddings via ONNX; heads trained by
  `scripts/train_semantic_head.py` / `train_semantic_head_coreml.py`;
  multilingual variants under `multilingual/SemanticSupport/`.
- **Mobile export** — `multilingual/export_coreml_multilingual.py` (CoreML
  `.mlpackage`, ANE-eligible); parity tests in `multilingual/test/`.
- **Build/release CI** — `scripts/ci/`: `check_language_neutral.py` (neutrality
  guard), `evaluate_gate.py` (accuracy gate), `assemble_pack.py` (deterministic
  versioned `.nlu` + SHA-256 manifest), `export_coreml_intent.py`,
  `verify_coreml_parity.py`. Workflows in `.github/workflows/`.

## Runtime turn flow (NLUEngine.handle)

1. **Confirmation** — if a yes/no follow-up context is active, resolve it first.
2. **Slot-filling** — if mid-collection, fill the pending slot; a high-confidence
   topic switch (>= `interrupt_threshold`, 0.75) abandons the flow and sets
   `NLUResult.interrupted_intent`.
3. **Classify** — fresh turn: ONNX classify -> confidence gate -> optional
   semantic rescue (**off by default**) -> entity/datetime extraction -> slot
   prompts.

Config/schema now come from the pack: `packs/en/schema.json`,
`packs/en/config.json`, `packs/en/entities/enums.json`. Legacy copies still
exist under `data/` (`nlu_schema.json`, `nlu_entities.json`, `localization/`)
and remain the source for the multilingual path. Calibration:
`config/calibration.json`.

## Dependency posture

Runtime is intentionally lean: scikit-learn, skl2onnx, onnxruntime, scipy,
pandas, joblib, dateparser. **No torch/transformers on the inference path** —
embeddings run through ONNX. `coremltools`/`torch` are export-only (macOS/CI).

## On-device constraints (first-class)

Be precise about which artifact a constraint applies to — they differ:

| Artifact | Shape contract |
|---|---|
| Classifier ONNX (`scripts/train.py`, skl2onnx) | `StringTensorType([None, 1])` — **dynamic** batch |
| CoreML intent model (ANE) | **fixed `(1, V)`** — required for ANE eligibility |
| MiniLM semantic embedder | **one sentence at a time** (`_embed(text: str)` → 384-dim L2-normalised) |

- Preserve numeric parity + calibration when touching export/quantization;
  verify against golden fixtures in `multilingual/test/`.
- Generated artifacts under `models/` and `dist/` are gitignored — regenerated.

## Intents

**59 intents**, `Cmd.*` / `Qry.*` style naming. `packs/en/schema.json` is the
source of truth on the pack path; `data/nlu_schema.json` for the legacy path.

> Note: `feature/production-work` migrated to a 57-intent
> `domain.object.action` taxonomy. **That migration is not on this branch.**
> Do not assume 57 intents or dotted names here.

## Related memory

Language Pack -> `langpack.md` · Datasets -> `datasets.md` · Training ->
`training.md` · Inference -> `inference.md` · Mobile -> `mobile.md` ·
Decisions -> `decisions.md` · Known issues -> `known-issues.md`.
