# Memory: Training Pipeline

> Single responsibility: how models are trained + calibrated. Inference is in
> `inference.md`; export is in `mobile.md`. For library APIs (scikit-learn,
> skl2onnx) use the **Context7 MCP** rather than recalling from memory.

## English core model

```bash
python scripts/train.py           # version 3 data (default)
python scripts/train.py -v 1      # original files
python scripts/train.py -v 2      # corrected _2 files
python scripts/train.py -v 3      # enhanced: _2 + 02_source_manual_corrections
```

`-v` also selects the leakage-guard holdout: `-v 1` → `semantic_holdout_100.csv`,
`-v 2`/`-v 3` → `semantic_holdout_2.csv`.

Pipeline: `TfidfVectorizer` -> `LogisticRegression` (sklearn `Pipeline`) exported
via `skl2onnx` with `StringTensorType([None, 1])` (dynamic batch). The script:
- caps samples per intent at `MAX_PER_INTENT = 500` (keeps the newest rows),
- runs a **holdout-leakage guard** — raises if any training phrase also appears
  in the holdout,
- 3-fold cross-validates (`cross_val_score(..., cv=3)`),
- enforces a **test-split accuracy floor**: `MIN_TEST_ACCURACY` default **0.85**,
  overridable via env (`MIN_TEST_ACCURACY=0.80 python scripts/train.py`),
- prints classification report + confusion matrix.

> The floor is on the **test split**, not on the 100-utterance paraphrase
> holdout — that holdout is scored and recorded in the manifest for visibility
> only. Don't confuse the two when reading a failure.

Writes: `models/intent_model.onnx`, `models/intent_labels.pkl`,
`models/intent_labels.json`, `models/intent_pipeline.pkl`, `models/manifest.json`.

## Multilingual models

```bash
python multilingual/train_multilingual.py
```

Per-language (en/fr/de/da) variants; per-intent/per-language capping before
merge; shared `text_norm.py`.

## Semantic heads (rescue layer)

```bash
python scripts/train_semantic_head.py          # English
python scripts/train_semantic_head_coreml.py   # CoreML-targeted variant
python scripts/build_semantic_index.py
```

Lightweight classifier head trained over MiniLM ONNX embeddings; used to rescue
low-confidence classifier turns. **The stage ships off by default** — see
`inference.md` / `langpack.md`.

## Calibration (temperature scaling)

Confidence is `softmax(logits / T)`. **T is rank-preserving** — it cannot change
which intent wins, only how confident the engine is. But that confidence drives
**five** gates (fire-vs-GenAI, slot acceptance, interrupt, agreement, semantic
rescue), so changing T re-tunes all of them at once. Never change it in isolation.

**Temperature belongs to a `(model, featurizer)` pair — not to a language.**
English has three featurizers and each needs its own T:

| Artifact | Fit by | Featurizer |
|---|---|---|
| `packs/en/intent_model/calibration.json` | `scripts/fit_calibration.py` | server/ONNX, full vocab |
| `models/intent_classifier_weights.json` | `scripts/export_ios_weights.py` | iOS, 1,370-term pruned |
| `*.mlpackage` metadata | CoreML export | CoreML |

They are **expected to differ**. Do not unify them.

```bash
python scripts/fit_calibration.py            # report only
python scripts/fit_calibration.py --write    # writes the pack artifact
```

Fits out-of-fold (k-fold, each row scored by a model that never saw it), so no
data is sacrificed and nothing leaks. Excludes any row appearing in an evaluation
set and records full provenance (method, n, featurizer, source SHA-256).

Current English fit: **T = 0.648339**, ECE **0.0084** (vs 0.1160 uncalibrated).

**The runtime uses it.** `IntentClassifier` resolves T as: pack
`intent_model/calibration.json` → `weights.json` → 1.0. A `temperature` of 1.0 in
the calibration artifact is treated as unset (that was the un-fitted skeleton
value). Older packs without the artifact still work via the weights fallback.

Operating point after recalibration (`confidence_threshold` raised 0.70 → **0.75**
to match the corrected scale):

| | Before | After |
|---|---|---|
| Delivered (holdout, semantic off) | 280/341 (82.1%) | **292/341 (85.6%)** |
| Wrong-action | 5 | **5** |
| OOS misfires | 0 | **0** |

At the old 0.70 gate the new T delivers 295 but lets one OOS utterance
("play chess with me") fire a command. 0.75 keeps that at zero for the cost of
3 answers — the right trade on a medical device.

`config/calibration.json` is **deprecated** — nothing reads it, and its values
were fit on a set that is 99.6% training data. Retained only because the
fr/de/da measurements exist nowhere else. `scripts/calibrate_languages.py` is
kept until those languages are re-fit with `fit_calibration.py`.

## Release gate

`scripts/ci/evaluate_gate.py` + `config/gate_thresholds.json`. A model that
trains but does not clear the bar is **blocked, not shipped**:
`min_accuracy` 0.80, `min_macro_f1` 0.60, `max_wrong_action` 15, evaluated on
`data/semantic_holdout_100.csv` with semantic enabled. Last recorded dry run:
acc 0.87 / macro-F1 0.68 / 1 wrong-action — **PASS**.
Tighten these over time; **never loosen silently**.

## Full retrain loop

```bash
python scripts/train.py \
  && python scripts/calibrate_languages.py \
  && python multilingual/export_coreml_multilingual.py --all --fp16 --fp32 \
  && python scripts/test_holdout.py --strict
```

`test_holdout.py` takes `-v {1,2}` (default 2), `--strict` (exit 1 below budget —
this is the gate), `--verbose`, `--json FILE`, and `--no-semantic`.

### What the semantic stage is worth (measured 2026-07-25, v2 holdout, 341 rows)

| Config | Correct | Accuracy | Wrong-action | Safe GenAI fallbacks |
|---|---|---|---|---|
| Semantic **ON** | 320–323 | 93.8–94.7% | **6** | 12–15 |
| Semantic **OFF** (ships) | 280 | 82.1% | **6** | 55 |

Semantic is worth **~+12.6 points** and converts ~43 GenAI hand-offs into
direct answers, at **no wrong-action cost** — the safety-critical number is 6 in
both. That is a strong argument for revisiting the off-by-default posture
(ADR-007), weighed against its memory/latency cost on device.

The ON range is not measurement sloppiness — the score is genuinely
non-deterministic run to run. See `known-issues.md`.

A dry-run retrain overwrites `models/` — restore it if you were only testing the
pipeline.

## Related memory

Datasets -> `datasets.md` · Inference -> `inference.md` · Mobile/export ->
`mobile.md` · Decisions -> `decisions.md`.
