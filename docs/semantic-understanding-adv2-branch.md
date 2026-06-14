# Branch Overview — `feature/Adv2/AddSemanticUnderstanding`

## Cut from

`feature/stt-intent-integration-adv-2`

## Purpose

Add the MiniLM-based semantic rescue stage to the on-device NLU pipeline
so the engine can understand novel, conversational phrasings that the
TF-IDF model has never seen in training.

## Why this is needed

After the architectural improvements in `adv-2`, the holdout accuracy is:

```
*** HOLDOUT accuracy (never trained): 0.19 (19/100) ***
```

Test split accuracy is 0.94 (good — the TF-IDF model memorises training phrases),
but holdout is 0.19 (bad — novel phrasings all fall through to GenAI fallback).

Real examples that currently go to GenAI fallback instead of firing the correct intent:

| Utterance | Should be |
|-----------|-----------|
| "it sounds too low" | `Cmd.VolumeIncrease` |
| "your voice is too low" | `Cmd.VolumeIncrease` |
| "i can't hear properly" | `Cmd.VolumeIncrease` |
| "it sounds loudly" | `Cmd.VolumeDecrease` |

These fail because TF-IDF is a bag-of-words model — it matches tokens, not meaning.
MiniLM encodes semantic meaning, so "i can't hear" and "increase volume" land near
each other in embedding space even though they share no words.

## What is being added

Semantic rescue stage from `feature/stt-intent-integration-adv-semantic`:

| Component | File | Description |
|-----------|------|-------------|
| MiniLM tokenizer + embedder | `scripts/nlu/semantic.py` | WordPiece tokenizer, ONNX inference, mean-pool + L2-norm |
| Trained classification head | `models/semantic_head.json` | 59×384 logistic weights trained on intent embeddings |
| Vocab file | `models/minilm-vocab.txt` | WordPiece vocabulary for the tokenizer |
| ONNX model | `models/minilm-l6-v2.onnx` | MiniLM-L6-v2 INT8 quantised (22 MB) |
| Holdout CSV | `data/semantic_holdout_100.csv` | 100 never-trained utterances used as the permanent benchmark |
| Volume direction regex fixes | `scripts/nlu/classifier.py` | `_TOO_QUIET`, `_QUIET_COMPLAINT` patterns |

## Pipeline after this branch

```
User utterance
      │
      ▼
[1] Keyword pre-filter (schema-driven, ~0 ms)
      │ no match
      ▼
[2] TF-IDF + LogReg ONNX → calibrated probability
      │ conf >= 0.70 → FULFILL/PROMPT
      │ conf <  0.70
      ▼
[3] MiniLM semantic rescue (~8–15 ms on CPU, ~3–5 ms on ANE)
      │ conf >= 0.55 → FULFILL/PROMPT
      │ conf <  0.55
      ▼
[4] GenAI fallback (network)
```

## What is NOT changed here

- All architectural fixes from `adv-2` are preserved (calibration, manifest,
  parity assertion, declarative keywords, back-reference schema, to_dict fix)
- The TF-IDF model and training pipeline are unchanged
- The nlu_schema.json intent set is unchanged

## Target metric

After merging the semantic stage, re-run `data/semantic_holdout_100.csv`.
Target: ≥ 75/100 (matching the `adv-semantic` branch baseline).

## See also

- Architectural review: [`docs/architecture-review.md`](./architecture-review.md)
- Parent branch overview: [`docs/adv-2-branch-overview.md`](./adv-2-branch-overview.md)
- Source of semantic code: `feature/stt-intent-integration-adv-semantic`
