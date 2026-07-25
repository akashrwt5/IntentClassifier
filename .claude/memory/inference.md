# Memory: Inference Pipeline

> Single responsibility: runtime prediction path (classifier + NLU engine).
> Training is in `training.md`; the pack boundary is in `langpack.md`. For ONNX
> Runtime API details use the **Context7 MCP**. Trace call flow with the
> **Code Graph Memory MCP**.

## Quick CLI (classifier only)

```bash
python scripts/predict.py
```

Loads `models/intent_model.onnx` + labels via ONNX Runtime. Gates:
- `CONF_THRESHOLD = 0.70` (min top-1 confidence),
- `CONF_GAP_THRESHOLD = 0.20` (min top1-top2 margin).

Low-confidence inputs are appended to `data/unknown_data.csv` for later review.

## Full NLU engine (production path)

```bash
python scripts/nlu_cli.py
python scripts/nlu_cli_multilingual.py
```

`scripts/nlu/engine.py` -> `NLUEngine.handle(session_id, text)` returns an
`NLUResult` (intent, slots, confidence, `interrupted_intent`).

**A bare `NLUEngine()` loads `packs/en` by default** — English is the default
because it is the default *pack*, not because of a branch in the code. Per-turn
order:

1. **Confirmation** — resolve an active yes/no context first.
2. **Slot-filling** — fill the pending slot; a topic switch at or above
   `interrupt_threshold` (0.75) interrupts the flow and records the new intent.
3. **Classify** — ONNX classify -> confidence gate -> **semantic rescue** if
   enabled and low-confidence -> entity/datetime extraction -> slot prompts.

## Confidence + calibration

- Pack thresholds (`packs/en/config.json`): `confidence_threshold` 0.70,
  `slot_confidence_threshold` 0.60, `semantic_threshold` 0.55.
- Per-language `temperature`/`conf_threshold` come from `config/calibration.json`;
  the device applies `softmax(logits / T)`.

## Semantic rescue — OFF by default

`scripts/nlu/semantic.py` embeds the utterance (MiniLM via ONNX, one sentence at
a time) and matches against a prebuilt index / semantic head to recover intents
the TF-IDF model misses.

**It is a pack-declared stage that is disabled unless explicitly enabled**
(`enable_semantic=True`, env, or pack config). Do not assume it is active when
reasoning about a low-confidence result. Precedence and failure modes:
`langpack.md`.

## Privacy posture — this is a medical-context app

Treat these as invariants, not preferences. They are enforced in `engine.py`:

- **The raw user utterance is NEVER embedded into an `NLUResult`.** Do not add it
  to a result field, an error message, or a telemetry payload.
- **Raw-utterance logging is opt-in and off by default** — `NLU_LOG_UTTERANCES`
  (`1`/`true`/`yes`). It exists for dev only; hearing-aid speech must never be
  written to production logs.
- The GenAI escalation URL is **configuration, not a result field**:
  `schema["genai_url"]` → `NLU_GENAI_URL` env → `DEFAULT_GENAI_URL`. The default
  is a deliberate placeholder (`https://genai.yourcompany.com/chat?query=`) so an
  unconfigured deployment is obvious rather than silently shipping.

Any change that widens what leaves the device is an architectural decision — take
it to the **architect** agent, not a quick edit.

## Startup invariant

`NLUEngine._assert_label_schema_parity()` runs at construction: the model's label
set and the pack's schema intents must agree. A mismatch fails loudly at startup
rather than mispredicting at runtime — if you swap a model or a pack's
`schema.json`, expect this to catch a stale pairing.

## Key config/data at runtime

Pack path: `packs/en/{schema,config}.json`, `packs/en/entities/enums.json`,
`packs/en/datetime/grammar.json`, `packs/en/intent_model/`.
Legacy/multilingual path: `data/nlu_schema.json`, `data/nlu_entities.json`,
`data/localization/`. Shared: `config/calibration.json`,
`models/intent_labels.json`.

## Related memory

Architecture/turn-flow -> `architecture.md` · Pack contract -> `langpack.md` ·
Calibration -> `training.md` + `decisions.md` · Known runtime bugs ->
`known-issues.md`.
