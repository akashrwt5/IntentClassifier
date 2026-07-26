# Memory: Inference Pipeline

> Single responsibility: runtime prediction path (classifier + NLU engine).
> Training is in `training.md`. For ONNX Runtime API details use the
> **Context7 MCP**. Trace call flow with the **Code Graph Memory MCP**.

## Quick CLI (classifier only)

```bash
make predict               # = python apps/cli/predict.py
```

Loads `models/intent_model.onnx` + labels via ONNX Runtime. Gates:
- `CONF_THRESHOLD = 0.70` (min top-1 confidence),
- `CONF_GAP_THRESHOLD = 0.20` (min top1-top2 margin).

Low-confidence inputs are appended to `data/unknown_data.csv` for later review.

## Full NLU engine (production path)

```bash
make nlu                              # = python apps/cli/nlu_cli.py
python apps/cli/nlu_cli_multilingual.py
```

`packages/runtime/nlu_engine/engine.py` -> `NLUEngine.handle(session_id, text)` returns an
`NLUResult` (intent, slots, confidence, `interrupted_intent`). Per-turn order:

1. **Confirmation** — resolve an active yes/no context first.
2. **Slot-filling** — fill the pending slot; a high-confidence topic switch
   (>= `INTERRUPT_THRESHOLD`) interrupts the flow and records the new intent.
3. **Classify** — ONNX classify -> confidence gate -> **semantic rescue** if
   low-confidence -> entity/datetime extraction -> slot prompts.

## Confidence + calibration

- The engine applies `softmax(logits / T)`.
- **`T` does NOT come from `config/calibration.json`** — nothing in
  `packages/runtime/nlu_engine/` reads that file. `classifier.py::_load_temperature`
  reads `temperature` out of the weights JSON, defaulting to
  `models/intent_classifier_weights.json`. That artifact is the **iOS/device**
  export (1370-term pruned vocab, 59-label pre-migration), so the engine
  currently calibrates full-vocab ONNX logits with a device-featuriser `T` of
  0.796286. `config/calibration.json` (en 0.6055) is read only by
  `nlu_training/evaluate.py`, so the report card and the runtime disagree.
- `engine.py:314` builds the multilingual classifier without `weights_path`, so
  **fr/de/da inherit the same English device `T`** rather than their fitted
  values. Per-language temperature scaling is computed and reported but has no
  runtime effect.
- This is Review-F5 blocker **B8**; the fix is charter steps B2/B3
  (`docs/Review-F5/ENGLISH-PRODUCTION-ROUTINE.md`). Do not describe the chain as
  working until those land.
- Semantic rescue threshold: `nlu_schema.json` key `semantic_threshold`
  (fallback `DEFAULT_SEMANTIC_THRESHOLD = 0.55`).

## Semantic rescue

`packages/runtime/nlu_engine/semantic.py` — embeds the utterance (MiniLM via ONNX, one sentence
at a time) and matches against a prebuilt index / semantic head to recover
intents the TF-IDF model misses. Multilingual variant under
`multilingual/SemanticSupport/`.

## Key config/data at runtime

`content/nlu_schema.json`, `content/nlu_entities.json`, `content/localization/`,
`models/intent_labels.json`, `models/intent_classifier_weights.json` (the
runtime's `temperature` source — see the calibration note above).
`config/calibration.json` is **not** a runtime input.

## Related memory

Architecture/turn-flow -> `architecture.md` · Calibration ->
`training.md` + `decisions.md` · Known runtime bugs -> `known-issues.md`.
