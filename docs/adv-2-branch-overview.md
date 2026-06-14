# Branch Overview — `feature/stt-intent-integration-adv-2`

## Cut from

`feature/stt-intent-integration-adv`

## Purpose

This branch applies the structural and correctness improvements identified in the architecture review of `feature/stt-intent-integration-adv`. It does not change the core two-stage cascade algorithm (keyword pre-filter → TF-IDF/ONNX → MiniLM semantic rescue). It fixes reliability gaps, silent failure modes, and maintainability issues that would block a production rollout.

## Problems being fixed

### P0 — Must fix before production

| # | Problem | Where |
|---|---------|-------|
| 1 | iOS runs a hand-rolled TF-IDF scorer while Python uses ONNX Runtime — two independent implementations of the same algorithm, divergence is silent | `export_ios_weights.py` |
| 2 | `LogisticRegression(C=15)` produces overconfident probabilities; threshold of 0.70 rejects almost nothing; calibration is not applied | `train.py` |
| 3 | `minilm-vocab.txt` and `minilm-l6-v2.onnx` must stay in sync but no checksum validates this at startup | `semantic.py` |

### P1 — Fix before next sprint

| # | Problem | Where |
|---|---------|-------|
| 4 | Intent names hardcoded as string literals in engine and classifier (Open/Closed violation) | `engine.py`, `classifier.py` |
| 5 | Label set (from trained model) and intent set (from schema) can diverge silently | `engine.py` init |
| 6 | `train.py` retrains on all data before export, inflating accuracy; no permanent holdout in CI | `train.py` |
| 7 | "at 6" → AM/PM heuristic silently picks wrong time; no regression tests for datetime parsing | `entities.py` |
| 8 | Back-reference resolution ("change back", "remind me again") is hardcoded per intent | `engine.py` |

### P2 — Quality improvements

| # | Problem | Where |
|---|---------|-------|
| 9 | MiniLM runs on CPU on iOS; Core ML / ANE can cut latency from ~15 ms to ~3–5 ms | needs `export_coreml.py` |
| 10 | `min_df=1` keeps noise terms; intent capping is non-deterministic (random sample) | `train.py` |
| 11 | `to_dict` dict comprehension drops falsy slot values (`0`, `""`) silently | `engine.py` |

## What is NOT changed here

- The cascade algorithm (keyword → TF-IDF → semantic) is kept as-is.
- Training data (`intent_data_new.csv`) is not modified unless fixing P0-3 requires a retrain.
- The `data/nlu_schema.json` intent set is not extended.
- The Dialogflow comparison tooling lives on `feature/dialogflow-compare--from-stt-adv` and is not touched here.

## See also

- Full review detail: [`docs/architecture-review.md`](./architecture-review.md)
- Semantic improvements (volume direction fixes, threshold tuning): `feature/stt-intent-integration-adv-semantic`
- Dialogflow comparison script: `feature/dialogflow-compare--from-stt-adv`
