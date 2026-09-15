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
2. **Slot-filling** — an awaited turn is read in fixed precedence and NEVER
   fabricates a value from a non-answer (ADR-012): (a) a valid strict value for
   the awaited slot is the ANSWER and never interrupts, whatever it classifies
   as; (b) else a high-confidence topic switch (>= schema `interrupt_threshold`)
   interrupts and records the new intent; (c) else a pure cancellation
   ("no"/"cancel", `_is_cancel`, cues from `schema cancel_cues`) abandons the
   flow; (d) else it is a no-match — re-prompt, and after `MAX_SLOT_ATTEMPTS`
   fall back. Recogniser guards behind this: enum fuzzy matching excludes
   function words (so "the" can't resolve to the memory "three"), and the
   `dateparser` fallback fires only on digit-bearing text (so "no" isn't read
   as November).
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

## BIO slot tagger (reminder title)

`packages/runtime/nlu_engine/slot_tagger.py` is the **single source of truth** for slot-tagger
behaviour. It exports `FEATURE_SPEC` — preprocessing rewrites, tokenizer pattern, affix sizes,
trigger/preposition/time word lists, the `rel_pos` rounding and the BIO labels — as DATA.

Three consumers read that one object, and none of them may redefine a rule:

| consumer | uses |
|---|---|
| `scripts/bio_tagger_prototype.py` (trainer) | `prepare`, `extract_token_features`, `extract_title_span` |
| `NLUEngine` (this runtime) | `extract_slot_title` |
| `nlu_export/export_slot_weights.py` | embeds `FEATURE_SPEC` in the weights JSON |

The export ships as `models/slot/<lang>/slot_tagger_weights.json` (payload `version` 2.0.0,
registered in `bundle.json` as `slot_tagger.<lang>.device_weights_artifact`). Because the spec
travels inside that file, the Swift and Kotlin ports hold **no word lists and no regexes of
their own** — changing a rule here reaches the devices through a pack update, with no app build.

Model shape: scikit-learn `LogisticRegression` over a `DictVectorizer`, classes
`B-TITLE / I-TITLE / O`, **no transition matrix** — each token is scored independently and the
label is the arg-max. There is no DATETIME class: date/time still comes from the entity path.

Decoding is `extract_title_span`: the LAST complete B-/I- run wins, and an `I-` tag with no
`B-` in front of it does not open a span. Training and inference share this function, so they
cannot drift. (Before this consolidation the runtime concatenated every span while the trainer
kept only the last — same model, two different answers.)

### Training caveats that still bite

- **Labels come from the rule-based `NLUEngine`**, so the tagger inherits that pipeline's
  mistakes along with its successes. It is imitating the system it replaces.
- The engine builds a title by DELETING date words, so `"remind me to call mom on christmas"`
  comes back as `"call mom christmas"` — a string that no longer occurs in the sentence.
  Matching it whole failed for **53 of 707** base sentences, and each one silently became an
  all-`O` sequence: 160 sentences in total were teaching the model that a normal reminder has
  no title. `best_span()` now takes the best contiguous run of title tokens that DOES occur
  (ranked by real words, then length, then earliest), which recovers all 53 with a BETTER
  label than the engine gave — `"call mom"`, not `"call mom christmas"`. All-`O` sequences
  drop from 160 to the 107 genuine negatives. Anything still unlocatable is dropped and
  counted, never kept as a negative.
- Augmentation happens at TEXT level and then goes through `prepare()`, so an injected filler
  is preprocessed exactly as production will preprocess it.
- `SEED = 42` and a `HOLDOUT_FRACTION` split; the trainer prints a token-level
  `classification_report` plus **exact title-match accuracy**. Judge a retrain on that number.
- `class_weight="balanced"` — roughly 70% of tokens are `O`.
- `isupper` / `istitle` were removed: `train.csv` is all lower-case, so both were always false
  and the exporter dropped their (zero) weights anyway.

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
