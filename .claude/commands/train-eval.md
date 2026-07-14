---
description: Run the training + calibration + holdout evaluation loop
---

Follow `.claude/memory/training.md` and `.claude/memory/datasets.md`.

1. If data changed, regenerate the master set: `python scripts/build_augmented_data.py`.
2. Train: `make train` (English) and/or `make train-multilingual`; retrain the
   semantic head if the data changed (`python scripts/train_semantic_head.py`).
3. Calibrate: `make calibrate` (writes `config/calibration.json`).
4. Evaluate on holdout/OOS: `python scripts/test_holdout.py --strict` and report
   macro-F1 + ECE per language, compared to the committed baselines.
5. Do not tune on the holdout. If metrics moved, update `training.md`. Summarize
   the numbers. Prefer the **ml-engineer** agent for judgment calls.
