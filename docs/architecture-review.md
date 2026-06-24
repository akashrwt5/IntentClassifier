# Architecture Review — `feature/stt-intent-integration-adv`

**Reviewer role:** Principal Conversational AI Architect, on-device NLP systems  
**Branch reviewed:** `feature/stt-intent-integration-adv`  
**Review date:** 2026-06-14  
**Implemented in:** `feature/stt-intent-integration-adv-2` (cut from this branch)

---

## Executive Summary

The branch delivers a working two-stage on-device NLU pipeline (keyword pre-filter → TF-IDF/LogReg ONNX → MiniLM semantic rescue) that correctly classifies the vast majority of commands. The core algorithmic design is sound. However, several structural issues create reliability, maintainability, and platform-parity risks that must be resolved before a production rollout.

---

## P0 — Must fix before any production release

### P0-1 · Dual-inference risk: ONNX Runtime vs. hand-rolled iOS scorer

**File:** `scripts/export_ios_weights.py`  
**Issue:** The iOS app performs TF-IDF + LogReg inference by hand (vocab lookup → IDF scaling → dot product → softmax) using the JSON weight export. The Python/server path uses `intent_model.onnx` via ONNX Runtime. These two code paths implement the same algorithm independently — any numerical or preprocessing difference (tokenisation, IDF formula, L2-norm application) produces silent accuracy divergence that is invisible until a user reports a wrong command on-device.

**Fix:** Either:
- (a) Run ONNX Runtime on iOS (eliminates the divergence entirely — preferred), or  
- (b) Add a conformance test suite that runs the same 50+ utterances through both paths and asserts the top-1 intent and probability agree within ±0.01. Make this test a CI gate.

**Risk if ignored:** Ship with systematically different accuracy on iOS vs. Python — the benchmark numbers become meaningless.

---

### P0-2 · Confidence calibration: LogisticRegression(C=15) is overconfident

**File:** `scripts/train.py`, line 83  
**Issue:** `C=15` is a very high regularisation relaxation for this vocabulary size (~1,800 pruned features, 60 classes). Logistic regression at C=15 fits hard to training data, producing probabilities that cluster near 0 and 1. The result: the 0.70 threshold rejects very few predictions (almost everything fires above threshold), and the semantic rescue stage is rarely reached. Apparent accuracy looks high, but calibration is poor.

**Evidence:** The `feature/stt-intent-integration-adv-semantic` branch had to raise the semantic threshold from 0.50 → 0.55 after retraining because out-of-scope inputs (e.g., "how is the weather today") scored ~0.51 on MemoryChange — a direct symptom of miscalibration.

**Fix:**
1. After training, apply isotonic regression or Platt scaling (`sklearn.calibration.CalibratedClassifierCV(method='isotonic')`) to produce calibrated probabilities.
2. Generate a reliability diagram (calibration curve) for the held-out test set.
3. Re-derive thresholds empirically against `data/semantic_holdout_100.csv` after calibration, rather than hardcoding 0.70.

---

### P0-3 · Embedder alignment: vocab and model must be versioned together

**File:** `scripts/nlu/semantic.py`  
**Issue:** `minilm-vocab.txt` and `minilm-l6-v2.onnx` were built together. If either is updated without updating the other, all semantic embeddings land in the wrong space. There is currently no checksum or version assertion to catch this at startup.

**Fix:** Add a versioned model bundle concept: a `models/manifest.json` (or similar) that records the SHA-256 of each model artifact. At startup, `semantic.py` asserts the hashes match. Fail loudly with a clear error if not.

---

## P1 — Fix before the next feature sprint

### P1-1 · Open/Closed violation: hardcoded intent names throughout engine

**Files:** `scripts/nlu/engine.py`, `scripts/nlu/classifier.py`  
**Issue:** Intent names like `"Cmd.MemoryChange"`, `"Cmd.SendMessage"`, `"Cmd.VolumeIncrease"` are string literals scattered through `_classify()`, `_handle_slot_filling()`, `_check_slot_filling()`, and the keyword pre-filter. Adding a new intent requires edits in multiple places with no compiler-enforced guarantee of completeness.

**Fix:** Move all intent-name constants to an `IntentName` enum or a frozen `INTENT_NAMES` dict loaded from `nlu_schema.json`. The keyword rules and back-reference logic should reference `IntentName.VOLUME_INCREASE` (or `schema.intents["Cmd.VolumeIncrease"]`), not bare strings. New intents added to the schema automatically become available without engine edits.

---

### P1-2 · Schema/model label parity not asserted at startup

**Files:** `scripts/nlu/engine.py` (init), `data/nlu_schema.json`  
**Issue:** The set of intents in `nlu_schema.json` and the set of labels in `intent_labels.json` (ONNX model output) can diverge silently. If a label exists in the model but not the schema, `engine.py` cannot look up its `action` or `fulfillment`. If a label exists in the schema but not the model, it can never be predicted.

**Fix:** At engine startup, assert:
```python
assert set(labels) == set(schema["intents"]), \
    f"Label/schema mismatch: {set(labels) ^ set(schema['intents'])}"
```
This makes the mismatch a hard startup error rather than a silent runtime bug.

---

### P1-3 · Train-on-test accuracy inflation / no permanent holdout

**File:** `scripts/train.py`  
**Issue:** `train.py` does cross-validation on the training split (good), but then retrains on **all** data (`pipeline.fit(X, y)`) before exporting — meaning the ONNX model has seen the test rows. Any accuracy numbers printed are therefore optimistic. There is also no permanent never-trained holdout set enforced by the training pipeline.

**Fix:**
1. Keep `data/semantic_holdout_100.csv` as the permanent holdout. Add an assertion in `train.py` that none of its utterances appear in `01_source_base_training_data.csv` (exact-match check).
2. Print holdout accuracy separately (clearly labelled "HOLDOUT — never trained").
3. Add this as a CI metric: fail if holdout accuracy drops below the previous run's score.

---

### P1-4 · DateTime entity: "at 6" → "6 PM" heuristic is fragile

**File:** `scripts/nlu/entities.py`  
**Issue:** The `"at 6"` → `6 PM` disambiguation uses a fixed cutoff (6 AM or earlier → AM, otherwise PM). This breaks for `"remind me at 7"` if the user says it at 7:30 PM (they probably mean 7 AM tomorrow), and also for `"at 12"` (noon vs midnight). The heuristic silently picks a time that may be 12 hours wrong, creating incorrect reminders with no user-visible error.

**Fix:** Add golden-file regression tests for datetime entity parsing covering at least: `"at 6"`, `"at 7"`, `"at 12"`, `"noon"`, `"midnight"`, `"in 10 minutes"`, `"tomorrow at 3"`. Make the ambiguity explicit — if the hour is ambiguous (6–12), return a `CONFIRM` response asking AM/PM rather than guessing.

---

### P1-5 · Back-reference resolution is hardcoded, not declarative

**File:** `scripts/nlu/engine.py`  
**Issue:** "change back" and "remind me again" back-reference resolution is implemented as special-case `if intent == "Cmd.MemoryChange"` blocks. Each new back-referenceable intent requires a new hardcoded block.

**Fix:** Add a `"back_reference"` field to the intent schema entry. The engine reads this field generically: if the current utterance matches a back-reference pattern, re-use `session.get_last_params(intent)` without any intent-specific code.

---

## P2 — Quality improvements

### P2-1 · Core ML / ANE path for iOS embedding model

**File:** `scripts/export_ios_weights.py` (or new `scripts/export_coreml.py`)  
**Issue:** MiniLM runs on ONNX Runtime in Python. On iOS, if the ONNX path is kept (via onnxruntime-mobile), inference takes ~15 ms on CPU. The Apple Neural Engine (A12+) can run the same transformer in ~3–5 ms via Core ML.

**Fix:** Add `scripts/export_coreml.py` that converts `minilm-l6-v2.onnx` → `minilm-l6-v2.mlpackage` using `coremltools`. This is a one-time offline step. The `.mlpackage` should be checked into the repo (or a release artifact) for iOS engineers to bundle.

---

### P2-2 · TF-IDF training determinism

**File:** `scripts/train.py`  
**Issue:** `min_df=1` retains hapax legomena (single-occurrence terms), inflating the vocabulary with noise and reducing generalisation. The intent-capping `pd.concat([g.sample(...)])` uses `random_state=42` which is fine, but if rows are added to an intent already at the cap, the sampled 500 changes — making model weights non-reproducible for the same logical dataset.

**Fix:** Set `min_df=2` in `TfidfVectorizer`. Change the capping strategy to deterministic keep-last (sort by row index, keep the last 500) so adding new rows always displaces the oldest, not a random subset.

---

### P2-3 · `to_dict` drops falsy slot values

**File:** `scripts/nlu/engine.py` (NLUResult or equivalent)  
**Issue:** If a slot value is an empty string, `0`, or `False`, a `{k: v for k, v in d.items() if v}` dict comprehension silently drops it. Slot values of `"0"` (e.g., volume level 0 for mute) or `""` (explicit empty name) would be lost.

**Fix:** Change the dict comprehension to `{k: v for k, v in d.items() if v is not None}` to preserve intentional falsy values.

---

## Summary Table

| ID | Priority | File(s) | Issue | Fix |
|----|----------|---------|-------|-----|
| P0-1 | P0 | `export_ios_weights.py` | Dual-inference risk (ONNX vs hand-rolled) | ONNX-RT on iOS OR conformance test suite |
| P0-2 | P0 | `train.py` | Overconfident probabilities (C=15) | Isotonic calibration + reliability diagram |
| P0-3 | P0 | `semantic.py` | Embedder alignment not checksummed | `models/manifest.json` + startup hash assert |
| P1-1 | P1 | `engine.py`, `classifier.py` | Hardcoded intent name strings | `IntentName` enum / schema-driven constants |
| P1-2 | P1 | `engine.py` | Label/schema parity not asserted | Startup `assert set(labels) == set(schema)` |
| P1-3 | P1 | `train.py` | Train-on-test accuracy inflation | Permanent holdout check + CI metric |
| P1-4 | P1 | `entities.py` | "at 6" → AM/PM heuristic is fragile | Golden-file tests + CONFIRM on ambiguity |
| P1-5 | P1 | `engine.py` | Back-reference hardcoded per intent | Declarative `back_reference` in schema |
| P2-1 | P2 | new `export_coreml.py` | iOS embedding on CPU only | Core ML / ANE export script |
| P2-2 | P2 | `train.py` | `min_df=1`, non-deterministic capping | `min_df=2`, keep-last deterministic cap |
| P2-3 | P2 | `engine.py` | `to_dict` drops falsy slot values | Use `is not None` guard |
