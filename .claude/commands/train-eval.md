---
description: Run the training + calibration + holdout evaluation loop
---

Follow `.claude/memory/training.md` and `.claude/memory/datasets.md`.
Prefer the **ml-engineer** agent for judgment calls.

1. If source data changed, regenerate the master set:
   `python scripts/build_augmented_data.py` (rebuilds `03_` and `04_`).
2. Train: `python scripts/train.py` (English core, `-v 3` by default) and/or
   `python multilingual/train_multilingual.py`. Retrain the semantic head if the
   data changed: `python scripts/train_semantic_head.py`.
3. Calibrate: `python scripts/calibrate_languages.py` (writes
   `config/calibration.json`).
4. Evaluate: `python scripts/test_holdout.py --strict` and report macro-F1 + ECE
   per language against the committed baselines in `config/calibration.json`.
5. Check the release gate: `python scripts/ci/evaluate_gate.py` against
   `config/gate_thresholds.json`.

Do not tune on the holdout. Do not loosen the gate thresholds. If a run fails,
say which floor it tripped — `MIN_TEST_ACCURACY` (test split, default 0.85) and
the release gate (100-utterance holdout) are different things. The retrain
overwrites `models/` — restore it if this was only a dry run. If metrics moved,
update `training.md`. Summarize the numbers.
