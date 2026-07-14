---
description: Add a new language to the NLU pipeline
---

Follow `docs/adding-a-new-language.md` and `.claude/memory/datasets.md`.

Provide the language code, then:
1. Create the lexicon/entities/schema files: `data/localization/nlu_lexicon.<lang>.json`,
   `nlu_entities.<lang>.json`, `nlu_schema.<lang>.json`, and the training CSV.
2. Register the language in the multilingual trainer/config.
3. Train: `python multilingual/train_multilingual.py` (or `--model <lang>`).
4. Calibrate: `make calibrate` — add the `<lang>` block to `config/calibration.json`.
5. Add a per-language holdout (`multilingual/test/<lang>_holdout.csv`) and, for
   datetime, a parity fixture in `tests/datetime_parity/`.
6. Update `datasets.md`, `roadmap.md`, and `known-issues.md` as needed.
