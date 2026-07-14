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

- Per-language `temperature`/`conf_threshold` come from `config/calibration.json`;
  the device applies `softmax(logits / T)`.
- Semantic rescue threshold: `nlu_schema.json` key `semantic_threshold`
  (fallback `DEFAULT_SEMANTIC_THRESHOLD = 0.55`).

## Semantic rescue

`packages/runtime/nlu_engine/semantic.py` — embeds the utterance (MiniLM via ONNX, one sentence
at a time) and matches against a prebuilt index / semantic head to recover
intents the TF-IDF model misses. Multilingual variant under
`multilingual/SemanticSupport/`.

## Key config/data at runtime

`content/nlu_schema.json`, `content/nlu_entities.json`, `content/localization/`,
`config/calibration.json`, `models/intent_labels.json`.

## Related memory

Architecture/turn-flow -> `architecture.md` · Calibration ->
`training.md` + `decisions.md` · Known runtime bugs -> `known-issues.md`.
