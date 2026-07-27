# Multilingual Model Calibration — Approaches Considered & Decision

**Status:** Decided — implemented (`ensemble=False`, commit `b4b47e4`)
**Author:** NLU / ML
**Scope:** How the six multilingual TF-IDF intent models (`en`, `fr`, `de`,
`da`, `multilingual`, `multilingual_small`) produce **calibrated** confidence,
and how we keep the **server (ONNX)** and **on-device (Swift)** paths consistent.
**Audience:** anyone touching `train_multilingual.py`, the iOS scorer, or the
0.70 confidence/fallback gate.

---

## 0. TL;DR

> We ship **Option B**: each model is a **single Logistic Regression + a single
> isotonic calibrator per class** (`CalibratedClassifierCV(method="isotonic",
> cv=3, ensemble=False)`). It is not the highest raw-accuracy option, but it is
> the only one that is simultaneously **small**, **calibrated**, and
> **identical across server and device**, which is what the on-device design
> actually needs.

The deciding fact: **on the device, Swift does not run the ONNX.** It runs the
`coef` / `intercept` + isotonic `maps` from `*_intent_classifier_weights.json`.
The ONNX is the server/reference path. So "server vs device parity" means *two
implementations agreeing*, and the model design must make that agreement
provable rather than hopeful.

---

## 1. The three approaches considered

| | Pipeline classifier | What the ONNX bakes in | What Swift runs |
|---|---|---|---|
| **Previous** | plain `LogisticRegression(C=15)` | one plain LR (uncalibrated) | the same plain LR |
| **Option A** | `CalibratedClassifierCV(isotonic, cv=3, ensemble=True)` | a **3-fold calibrated ensemble** | one **averaged** LR + a **separately fit** isotonic map |
| **Option B** (shipped) | `CalibratedClassifierCV(isotonic, cv=3, ensemble=False)` | **one** LR + **one** isotonic calibrator | the **same** one LR + isotonic maps |

- **Previous** — no calibration. The softmax confidences are all bunched near
  ~0.9–0.95, so the 0.70 fallback gate cannot separate confident, correct
  predictions from ambiguous ones. Server and device match exactly, but the
  confidence number is not meaningful.
- **Option A** — calibrated, but `ensemble=True` bakes **three** fold-models
  into the ONNX. The device cannot run three models cheaply, so it **averages**
  the three LRs and fits a **separate** isotonic map to approximate the
  ensemble's `predict_proba`. The device is therefore an *approximation* of the
  server, and the two drift apart.
- **Option B** — calibrated, but `ensemble=False` keeps a **single** base LR
  (refit on all the training data; cross-validation is used only to generate
  unbiased scores for the calibrator). The device runs *that exact LR*, so the
  device logit equals the ONNX logit with no averaging. Parity holds by
  construction.

---

## 2. The numbers

### 2.1 Server ↔ device parity (the reason we changed)

Measured with the conformance scorer (`scripts/test_ios_conformance.py`),
30 representative utterances, comparing the ONNX output to the hand-rolled
iOS/Swift-equivalent scorer:

| Metric | Option A (ensemble) | **Option B** |
|---|---|---|
| Threshold disagreements (one path fires, other falls back) | **7 / 30** | **0 / 30** |
| Max top-1 confidence gap | ~0.53 (e.g. `open health`: ONNX 0.91 vs iOS 0.38) | **~0.10** |

Under Option A, ~30% of utterances behaved differently across server and
device. Under Option B the calibrated confidences track within ~0.10 and never
straddle the 0.70 gate.

### 2.2 Model size

The ONNX shrinks ~3× because one LR is baked in instead of three:

| Model | Option A ONNX | **Option B ONNX** |
|---|---|---|
| en | 4.3 MB | **1.5 MB** |
| fr | 4.6 MB | **1.6 MB** |
| de | 5.2 MB | **1.8 MB** |
| da | 4.3 MB | **1.5 MB** |
| multilingual | 15.2 MB | **5.4 MB** |
| multilingual_small | 4.2 MB | **1.5 MB** |

Option B is essentially the same size as the uncalibrated *Previous* models
(+~50 KB for the isotonic calibrators), but with real calibration.

### 2.3 Accuracy (holdout, post-calibration argmax)

| Model | Previous (plain LR) | Option A | **Option B** |
|---|---|---|---|
| en | 91.2% | ~90.8% | 90.6% |
| fr | 86.6% | ~84.3% | 84.8% |
| de | 84.8% | 84.1% | 84.6% |
| multilingual | 85.9% | 84.9% | 85.5% |
| multilingual_small | 86.3% | 85.9% | 85.7% |
| da | 79.1% | 78.7% | 79.3% |

**Read this carefully:** Previous (uncalibrated) is marginally *higher* on raw
accuracy (~0.5–1.8 points). That is the expected cost of calibration —
`predict()` argmaxes *after* isotonic, which re-ranks a few near-ties. Option B
and Option A are ~tied. Raw top-1 accuracy is **not** the metric that decides
this: Previous "scores higher" only because its confidences are meaningless, so
the fallback gate (the entire point of the on-device design) can't function.

---

## 3. Why Option B was finalized

- **vs Option A — strictly better.** Same accuracy, same calibration, but ~3×
  smaller *and* parity goes from 7/30 threshold splits to 0/30. There is no
  reason to keep A.
- **vs Previous — the right trade.** We give up ~1 point of raw accuracy to get
  a confidence score that actually means something. Without calibration the
  0.70 gate is dead weight; with it, the gate can distinguish a confident
  prediction from an ambiguous one. For an on-device classifier with a GenAI
  fallback, a trustworthy confidence is worth more than one point of top-1.
- **Operational simplicity.** One model, one calibrator, one set of weights —
  the device reproduces the server exactly, so the conformance gate is a real
  guarantee rather than a tolerance check.

**Known, accepted caveats:**

1. **`da` is below the 0.80 accuracy floor (~0.79).** Pre-existing and
   data-driven — true under all three approaches. It is exported with a relaxed
   gate to keep the model set complete; `--strict` CI will flag it. The fix is
   better/more Danish data, not a calibration change.
2. **A few near-tie intents re-rank** vs the uncalibrated baseline (the
   battery-vs-help cases). Inherent to adding calibration; the production model
   already behaves this way. See §4.

---

## 4. The residual intent-parity issue (explained)

Option B fixed **confidence** parity (threshold disagreements → 0). What remains
is a small number of **intent** (argmax) disagreements — ~27–29 of 30 utterances
agree. These come from **two distinct causes, neither of which is the ensemble
choice**, and both pre-date this work (they exist in the production model too):

### Cause 1 — Tokenizer divergence

sklearn's `TfidfVectorizer` and the iOS/Swift tokenizer do not tokenize
identically on edge cases. Example — `"set a reminder"`:

- **sklearn** drops the single-character token `"a"` (default token pattern
  keeps 2+ char words), so adjacent words form the bigram **`"set reminder"`**,
  which *is* in the vocabulary.
- **iOS** keeps `"a"` and forms `"set a"` / `"a reminder"`, which are *not* in
  the vocabulary, so it never produces `"set reminder"`.

Different feature vectors → different logits → the argmax can flip on a near-tie.

### Cause 2 — Argmax taken before vs after calibration

By design, the on-device scorer takes the predicted intent from the **raw LR
argmax** (pre-calibration) — "calibration must only rescale confidence, never
re-rank." The ONNX, however, emits the **calibrated** `predict_proba`, whose
argmax is **post-calibration**. Isotonic maps differ per class, so on a near-tie
they can re-order the top two. Example — `"open translate"`: raw-logit argmax is
`Help_Translate`, calibrated argmax is `Cmd.TranslationStart`. The tokens are
identical; only the *stage at which argmax is taken* differs.

> Net: the remaining mismatches are concentrated on genuine near-ties (two
> intents within a few hundredths of each other), and split between a
> tokenizer gap (Cause 1) and an argmax-ordering convention (Cause 2).

---

## 5. Can the residual fixes be pursued? — Feasibility

**Yes, both are fixable, and independently.** They do not require changing the
calibration approach.

| Fix | What it entails | Effort | Risk |
|---|---|---|---|
| **Cause 1 — tokenizer alignment** | Make the Swift/iOS tokenizer match sklearn's analyzer exactly: drop single-character tokens before forming bigrams (or, conversely, retrain with a `token_pattern` that keeps single chars). Lock it with a tokenizer-level unit test over a fixed corpus. | Low–moderate | Low — purely mechanical string handling; verifiable token-by-token |
| **Cause 2 — argmax ordering** | Choose one convention and apply it on *both* sides: either (a) device argmaxes the **calibrated** probabilities to match the ONNX, or (b) the server/conformance reference argmaxes the **raw logits** to match the device. (a) is simpler on-device and makes ONNX the source of truth. | Low | Low–moderate — it is a product decision (does calibration re-rank or not?), not just code |

**Recommended approach if we pursue it:**

1. Fix **Cause 1** first (tokenizer) — it is unambiguous and removes the
   feature-vector mismatch. Add a conformance assertion that sklearn-analyzer
   tokens == iOS tokens for a fixed utterance set.
2. Then make a **deliberate decision** on **Cause 2**. Recommendation: let the
   device argmax the calibrated scores (option (a)) so the ONNX is the single
   source of truth and intent + confidence are both taken post-calibration. This
   also recovers most of the ~1-point accuracy difference vs Previous, because
   the curated battery-vs-help cases stop depending on which side calibrates.
3. Re-run `scripts/test_ios_conformance.py` and the multilingual suite; target
   ≥29–30/30 intent agreement with 0 threshold disagreements.

**Expected outcome:** intent agreement from ~27/30 to ~30/30, with no regression
in size or confidence parity. The only true floor that remains is the `da`
dataset quality, which is out of scope for calibration.

---

## 6. How to reproduce

```bash
# Train all six models (Option B pipeline is the default now)
python multilingual/train_multilingual.py --all

# da is genuinely ~0.79; relax the floor if you want it exported
python multilingual/train_multilingual.py --language da --min-accuracy 0.75

# Holdout accuracy + curated smoke tests
python multilingual/test/test_multilingual_models.py

# Server (ONNX) vs device parity (production model wiring)
python scripts/test_ios_conformance.py --verbose
```
