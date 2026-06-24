# Temperature Scaling — Implementation Prompt

**Purpose:** Hand this to an implementation agent (Claude/Codex) to execute the
pivot from per-class isotonic calibration to single-parameter temperature
scaling. Read it top to bottom before writing code.
**Decision record:** `multilingual/TEMPERATURE_SCALING_DECISION.md` and
`multilingual/MODEL_CALIBRATION_DECISION.md`.

---

## ROLE
You are a **Principal Machine Learning Engineer** specializing in **on-device
NLU systems** — TF-IDF/linear intent classifiers, probability **calibration**,
and cross-platform parity (ONNX server ↔ Swift/CoreML & Android device). You
write production code that survives senior review: no data leakage, no
unvalidated assumptions, every claim backed by a measurement. You are cautious
about anything that could silently break server/device agreement.

## TASK
Replace per-class **isotonic calibration** with single-parameter **temperature
scaling** across the six multilingual intent models (`en`, `fr`, `de`, `da`,
`multilingual`, `multilingual_small`). Read both decision docs first.

Temperature scaling: at inference compute `softmax(logits / T)`, with one scalar
`T` per model fit to minimize NLL on a held-out calibration set. It is
rank-preserving (argmax unchanged), so intent selection is identical to raw
logits; only confidence is rescaled.

## CRITICAL CORRECTNESS REQUIREMENTS (do not violate)
1. **No leakage — use a 3-way split.** train (fit LR) → calibration (fit `T`) →
   test (report metrics). Never fit `T` and report calibration metrics on the
   same data. If data is scarce, fit `T` via cross-validation on the train
   portion and still report on an untouched test set.
2. **Fit and validate `T` on the device-equivalent logits**, not only the server
   full-vocab logits. The device path (`_device_logits` in
   `scripts/export_ios_weights.py`) uses the **pruned vocab, L2-normalized on the
   pruned subspace** — a different logit magnitude than the server. Confirm the
   shipped `T` keeps **both** the server confidence and the device confidence on
   the same side of the 0.70 gate for the conformance set.
   - **Prefer aligning the two logit computations** so the optimal Ts coincide.
   - **If the server-optimal and device-optimal `T` still diverge, the device
     `T` is authoritative** — the device is the real user-facing inference path
     and the 0.70 GenAI-fallback gate fires on-device. The server adopts the
     device `T`. Report all calibration metrics (NLL/ECE) on the **device path**,
     since that is the authoritative one.
3. **[BLOCKING] ONNX must expose raw decision scores (logits), not softmaxed
   probabilities.** With a plain `LogisticRegression`, the default skl2onnx graph
   outputs softmaxed probabilities — applying `T` to those is mathematically
   wrong (you'd be dividing probabilities, not logits). Before proceeding,
   **verify what the exported graph actually outputs** and adjust the export so
   the server receives raw logits. Do not continue past this check until raw
   logits are confirmed available end-to-end; treat failure here as a hard stop.
4. **Let the data set `T`'s direction.** Don't hardcode an expectation; verify
   the fitted `T` actually lowers NLL/ECE on the test set and is consistent
   across models.
5. **Numerically stable softmax** (subtract per-row max). Optimize `T` with a
   bounded method (e.g. `scipy.optimize.minimize_scalar(method="bounded",
   bounds=(0.05, 10.0))`) minimizing NLL on the calibration set.
6. **Fixed metric definitions** so raw / isotonic-baseline / temperature are
   apples-to-apples:
   - **NLL** = mean negative log-likelihood (cross-entropy) of the true class
     under the calibrated probabilities. This is the **primary** calibration
     metric (it is what `T` optimizes).
   - **ECE** = 15-bin equal-width, top-1 confidence vs. accuracy. **Secondary /
     diagnostic.**

## CHANGES

**`multilingual/train_multilingual.py`**
- Replace `CalibratedClassifierCV(...)` with plain
  `LogisticRegression(max_iter=3000, class_weight="balanced", C=15.0)`.
- Produce the 3-way split; fit `T` on the calibration set via bounded NLL
  minimization over `decision_function` logits.
- Persist `T` in the model metadata. Print per model: fitted `T`, and **both
  NLL and ECE** on the **test** set for raw vs temperature.
- Remove the fold-averaging branch in `extract_lr()` (only a plain LR exists now).

**`scripts/export_ios_weights.py`**
- Delete/skip `_fit_calibration()` and the
  `"calibration": {"method": "isotonic_logit", ...}` payload.
- Simplify `_extract_lr()` (no `CalibratedClassifierCV` branch).
- Add `"temperature": <float>` to the JSON. Keep `coef`, `intercept`, `idf`,
  `vocab`, `normalize: "l2"`, thresholds unchanged.
- **Validate (req. #2):** recompute `_device_logits` on the calibration texts,
  apply `T`, and confirm device confidences track the server within tolerance and
  don't straddle 0.70. If Ts diverge, ship the device-fit `T`.

**Server — `scripts/nlu/classifier.py` (and the multilingual classifier path)**
- Source **raw logits** (req. #3); compute `scaled = logits / T`;
  `argmax(scaled)` for intent (== raw argmax); `softmax(scaled)` for confidence;
  gate on that.

**Conformance — `scripts/test_ios_conformance.py`**
- In `_ios_predict()` and the ONNX path, drop isotonic interpolation; apply
  `logits / T` then stable softmax. Run all 6 models.

## VALIDATION / DEFINITION OF DONE
- **[BLOCKING]** ONNX confirmed to expose raw logits end-to-end (req. #3).
- Test-set **NLL improves** vs the raw/isotonic baseline per model (report the
  numbers). NLL is the primary gate.
- Test-set **ECE must improve vs baseline** per model (report the numbers);
  fixed estimator, untouched test set. (Diagnostic, not an absolute threshold.)
- Argmax accuracy **≥ raw-logit accuracy** (rank-preserving → should be identical).
- Conformance: **0/30 threshold disagreements** on `en` and `multilingual_small`
  minimum; intent mismatches confined to the known tokenizer / argmax-ordering
  cases.
- **Server vs device confidence parity** verified on the conformance set with the
  shipped `T`; device `T` authoritative if they diverge (req. #2).
- `da` stays ~0.79 (pre-existing data floor; out of scope).
- Per-model report: fitted `T`, NLL(raw)→NLL(temp), ECE(raw)→ECE(temp).

## CONSTRAINTS
- Do **not** touch the tokenizer or `multilingual/text_norm.py` (separate effort).
- Note the `class_weight="balanced"` caveat in code/PR: `T` corrects sharpness,
  not the balanced-prior shift; acceptable for a confidence gate.
- Backward compatibility: server/device treat a missing `temperature` key as
  `T = 1.0` (plain softmax).
- Commit with `git config user.email noreply@anthropic.com`. Work on the
  `...-TemperatureScaling` branch, not the decision-doc branch.

## REPRODUCE
```bash
python multilingual/train_multilingual.py --all
python multilingual/train_multilingual.py --language da --min-accuracy 0.75
python multilingual/test/test_multilingual_models.py
python scripts/test_ios_conformance.py --verbose
```
