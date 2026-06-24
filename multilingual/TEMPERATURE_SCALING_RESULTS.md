# Temperature Scaling — Results

Outcome of the pivot from per-class isotonic calibration to single-parameter
**temperature scaling** (`softmax(logits / T)`, one scalar `T` per model).

See `TEMPERATURE_SCALING_DECISION.md` (rationale) and
`TEMPERATURE_SCALING_IMPLEMENTATION_PROMPT.md` (requirements). All numbers below
are reproducible with the commands in the **Reproduce** section.

## Method (no leakage)

- Each model: plain `LogisticRegression(max_iter=3000, class_weight="balanced",
  C=15.0)` over a TF-IDF pipeline. No `CalibratedClassifierCV`.
- **3-way split**: train (fit LR) → calibration (fit `T`) → test (report). `T`
  is fit ONLY on the calibration split via bounded NLL minimization
  (`scipy.optimize.minimize_scalar`, bounds `(0.05, 10.0)`) over the raw
  `decision_function` logits; every metric below is on the **untouched test
  split**.
- Metrics: **NLL** = mean cross-entropy of the true class (primary — it is what
  `T` optimizes). **ECE** = 15-bin equal-width, top-1 confidence vs. accuracy
  (diagnostic). Numerically stable softmax (per-row max subtracted).

## [BLOCKING] ONNX emits raw logits — confirmed

The pipeline is exported with `options={..., "zipmap": False, "raw_scores":
True}`, so the ONNX `probabilities` output is the raw `decision_function` (not a
baked-in softmax). Verified end-to-end on `models/intent_model.onnx`: per-row
output sum ≈ 0.0 (a softmax would sum to 1.0) with negative values present.
Applying `softmax(logits / T)` downstream is therefore mathematically correct.

## Device-path `T` is authoritative

For the **multilingual** models, `train_multilingual.py` exports the **full**
vocab (no pruning), so the `decision_function` logits ARE the device-equivalent
logits — the fitted `T` is the device-path `T` and the server adopts it
directly; there is no server/device `T` to reconcile.

For the **production** English model, `export_ios_weights.py` prunes the vocab
and re-L2-normalizes on the pruned subspace, so `T` is fit on the
**device-equivalent** (pruned) logits per requirement #2. Server↔device
confidence parity is then verified on the conformance set (below).

## Per-model results (test split)

NLL is the primary gate; ECE is diagnostic. Temperature scaling is
rank-preserving, so argmax accuracy is identical for raw and temperature.

| Model              |   `T`  | NLL (raw → temp)   | ECE (raw → temp)   | Accuracy |
|--------------------|:------:|:------------------:|:------------------:|:--------:|
| en                 | 0.6214 | 0.4769 → **0.3726** | 0.1162 → **0.0184** |  0.900   |
| fr                 | 0.6699 | 0.7178 → **0.5925** | 0.1495 → **0.0223** |  0.852   |
| de                 | 0.6777 | 0.8061 → **0.6787** | 0.1563 → **0.0168** |  0.833   |
| da                 | 0.8156 | 1.0375 → **0.9978** | 0.0820 → **0.0352** |  0.760   |
| multilingual       | 0.7182 | 0.7199 → **0.6352** | 0.1215 → **0.0109** |  0.845   |
| multilingual_small | 0.7554 | 0.6763 → **0.6214** | 0.0966 → **0.0075** |  0.848   |

Production English model (`models/`, device-fit on pruned logits):
`T = 0.7963`, NLL `0.3911 → 0.3663`, holdout accuracy `0.8944`.

**Every model: NLL improved (primary gate ✅) and ECE improved (diagnostic ✅).**
Argmax accuracy is unchanged by `T` (rank-preserving ✅).

`da` sits at the accepted ~0.76–0.79 pre-existing data floor and is out of scope
for this effort; temperature scaling still improves its NLL and ECE.

## Conformance — server (ONNX) vs. device (hand-rolled iOS scorer)

Both paths source raw logits and apply the same `softmax(logits / T)`. With the
shipped per-model `T`, threshold (fire/fallback at 0.70) parity is exact:

| Model              | Agree | Intent mismatch | Threshold disagree |
|--------------------|:-----:|:---------------:|:------------------:|
| production (en)    | 30/30 |        0        |         0          |
| en                 | 30/30 |        0        |         0          |
| fr                 | 30/30 |        0        |         0          |
| de                 | 30/30 |        0        |         0          |
| da                 | 30/30 |        0        |         0          |
| multilingual       | 30/30 |        0        |         0          |
| multilingual_small | 30/30 |        0        |         0          |

`0/30` threshold disagreements on `en` and `multilingual_small` (the required
minimum) — and in fact on every model. Server and device confidence stay on the
same side of the 0.70 GenAI-fallback gate.

## Backward compatibility & caveats

- A consumer that does not find a `"temperature"` key treats `T = 1.0` (plain
  softmax), so older artifacts keep working.
- `class_weight="balanced"` caveat: `T` corrects confidence **sharpness**, not
  the balanced-prior shift. This is acceptable for a confidence gate, where what
  matters is that the top-1 confidence is well-calibrated against the 0.70
  threshold.

## Reproduce

```bash
python multilingual/train_multilingual.py --all
python multilingual/train_multilingual.py --language da --min-accuracy 0.75
python multilingual/test/test_multilingual_models.py
python scripts/test_ios_conformance.py --verbose            # production model
python scripts/test_ios_conformance.py --model all          # all 6 multilingual
```
