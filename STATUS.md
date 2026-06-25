# Routine STATUS — CoreML FP16 export (BLOCKED: repository mismatch)

**Run date:** 2026-06-25
**Routine:** "CoreML export + Tier-A/B parity + temperature contract" (multilingual intent classifier)
**Outcome this run:** BLOCKED — prerequisites do not exist in this repository. No code work performed.

## What the routine expects vs. what this repo actually contains

The routine prompt is written against a multilingual, temperature-scaled CoreML
codebase. **None of its required input artifacts exist in `akashrwt5/IntentClassifier`.**

| Routine expects (must read / build from) | Present in this repo? |
|---|---|
| `multilingual/` directory | **No** |
| `multilingual/TEMPERATURE_SCALING_RESULTS.md` | **No** |
| `scripts/export_coreml.py` (reference contract) | **No** |
| `scripts/test_ios_conformance.py` (numeric reference `_ios_predict`) | **No** |
| `multilingual/models/<m>/<m>_intent_classifier_weights.json` for `en, fr, de, da, multilingual, multilingual_small` | **No** |
| Six ONNX models + 59-class (K=59) linear classifiers | **No** |
| Per-model holdout CSVs `multilingual/test/<m>_holdout.csv` | **No** |
| Work branch `claude/coreml-export` on origin | **No** |

**What this repo IS:** a single-language **English** intent classifier —
`TfidfVectorizer + LogisticRegression` exported to ONNX via `skl2onnx`
(`scripts/train.py`, `scripts/predict.py`, `scripts/auto_label.py`,
`data/intent_data_new.csv`). It has **~10 intents** (Volume, Reminder,
Notifications, Push To Talk, Translate, Transcribe, TeleHearAI, SelfCheck,
Memory), **not 59 classes**, **no multilingual models**, **no temperature
scaling**, **no CoreML code**, and **no committed `models/` / ONNX artifacts**
(they are generated locally by `train.py`).

Verified across all branches/commits of both in-scope repos:
- `akashrwt5/IntentClassifier`: branches `main`, `claude/lucid-newton-5rnay9` — no
  `multilingual/`, `export_coreml`, `temperature`, or `test_ios_conformance`
  anywhere in history.
- `akashrwt5/STT`: the iOS Swift app. Branches `main`,
  `claude/trusting-planck-5rnay9`. The S6 base branch
  `feature/Adv2/AddSemanticUnderstanding-4-agentsfeedbackAddress` **does not
  exist** on origin, and there is no CoreML temperature/intent-classifier code.

## Why this is a hard block (not skippable)

- **S1** requires loading existing per-model weights JSON for six named models.
  Those files do not exist. The spec explicitly forbids modifying/creating
  weights, ONNX models, or temperature values — so they cannot be fabricated.
- **S2–S4** (Tier-A numeric equivalence, ONNX↔CoreML parity report) depend on the
  ONNX models and holdout CSVs that S1 mirrors. No inputs ⇒ nothing to verify.
- **S5/S6** depend on S1–S4 outputs and the (absent) iOS base branch.

There is no incomplete-but-runnable step: the routine is attached to a
repository that does not contain the project it describes.

## Branch note

The routine specifies a fixed work branch `claude/coreml-export`, but the
harness-level Git Development Branch Requirements designate
`claude/lucid-newton-5rnay9` for this repo and forbid pushing to other branches
without explicit permission. Because there is no CoreML work to accumulate, this
STATUS was committed to the designated branch `claude/lucid-newton-5rnay9`
rather than creating a conflicting `claude/coreml-export` branch.

## What the user needs to decide

One of the following — the routine cannot proceed without it:

1. **Wrong repository.** If the multilingual/CoreML/temperature project lives in
   a different repo, attach the routine to that repo (and add its iOS counterpart
   to scope for S6).
2. **Prerequisites not pushed.** If the multilingual artifacts
   (`multilingual/`, the six weights JSON, ONNX models, holdouts,
   `TEMPERATURE_SCALING_RESULTS.md`, `scripts/export_coreml.py`,
   `scripts/test_ios_conformance.py`) are expected to be in *this* repo, they must
   be committed/pushed first — the routine reads them, it does not create them.
3. **Different intent.** If the goal is actually to add a CoreML export for *this*
   English TF-IDF+LR model (a much smaller, different task than the spec), confirm
   that and the routine prompt should be rewritten accordingly (this repo has no
   temperature scaling or six-model fan-out).

Until one of the above is resolved, each scheduled run will re-detect the same
mismatch and re-record this STATUS without doing CoreML work.
