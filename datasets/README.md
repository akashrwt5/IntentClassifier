# datasets/ — DVC-tracked training, holdout, and OOS data

**Tracked by DVC** (2026-07-14): git holds only the `datasets.dvc` pointer;
the data lives in the DVC cache and the local remote.

- Remote: `../dvc-store` relative to the repo (a directory next to the
  checkout — created on first `dvc push`). Migrate to cloud later with
  `dvc remote modify`.
- After cloning / pulling pointer changes: `dvc pull`
- After editing data: `dvc add datasets && git add datasets.dvc && dvc push`
- **Run `dvc push` on your machine now** to seed `../dvc-store` — until
  then the only blob copies are your working tree + `.dvc/cache` (+ full
  git history from before the DVC migration, as a backstop).

Layout notes (from the ND-2 M3 move): training CSVs feed
`packages/buildtime/nlu_training/`; multilingual per-language sets live in
`multilingual/`; holdout fixtures stay in `multilingual/test/` (test
fixtures, deliberately git-tracked, they gate CI). Lineage: `DATA_PIPELINE.md`.
