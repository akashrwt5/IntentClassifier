# datasets/ — training, holdout, and OOS data (ND-2 M3)

Moved 1:1 from `data/*.csv` and `multilingual/data/` (now
`datasets/multilingual/`). Lineage/pipeline notes: `DATA_PIPELINE.md`.

- Training CSVs feed `packages/buildtime/nlu_training/` trainers.
- Holdout fixtures used by the parity oracle live in `multilingual/test/`
  (they are test fixtures, deliberately not moved).
- `data/` now holds only runtime logs (`unknown_data.csv`,
  `unknown_counters.csv`) per docs/privacy-unknown-data.md.

**DVC status:** wiring pending — requires the dvc toolchain + a remote
decision (where dataset blobs live). Tracked in EXECUTION_STATUS as the
remaining M3 sub-item; the directory layout is already DVC-shaped.
