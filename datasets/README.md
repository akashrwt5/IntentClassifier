# datasets/ — tracked directly in git

**Storage decision (2026-07-26): these files live in git. DVC has been removed.**

At ~6.7 MB of CSV across 29 files this is well within what git handles
comfortably, and CSVs delta-compress well across revisions. DVC was buying
nothing here: its configured remote was a *local filesystem path*
(`../../dvc-store`) that existed only on one machine, so CI, a fresh clone and
every scheduled run got no data at all and every model-dependent gate silently
skipped (Review-F5 blocker B2). A tool that no other machine can read is worse
than no tool — and in the end the local cache turned out to be empty too, so it
was protecting nothing.

## How these files were recovered

The DVC cache was gone (`.dvc/cache` absent; `dvc status` reported
`not in cache: datasets`) and the local store was unreachable. The data was
recovered from **git history** instead — commit `a6cbb81c` ("migrate label space
to domain.object.action"), the last state in which `datasets/` was tracked
before commit `9f4e5481` handed it to DVC.

The recovery is provably exact. `datasets.dvc` recorded:

```
md5:    23a40044be29147b5de26753d90214af.dir
size:   7072263 bytes
nfiles: 29
```

and `a6cbb81c`'s `datasets/` tree is **29 files totalling 7,072,263 bytes** —
byte-for-byte what DVC was pointing at. DVC never captured a state newer than
the one git already held, so nothing was lost in the migration.

## Layout — one directory per language

```
datasets/<lang>/
    train.csv                  REQUIRED   columns: text,intent
    holdout_leakage_guard.csv  evaluation set train.py checks train.csv against
    holdout_paraphrase.csv     hard paraphrases (need the semantic stage)
    oos.csv, oos_2.csv         out-of-scope utterances
    benchmark_250.csv          scored benchmark
    sources/                   inputs that generate train.csv
```

**Adding a language is adding a directory.** Drop in `datasets/fr/train.csv`
and run `python -m nlu_training.train --lang fr`. No script learns the language
name; `train.py` resolves `datasets/<lang>/` and tells you what is available if
the directory is missing.

There is no combined "multilingual" model. Each Language Pack carries its own
model, so training is per language by construction.

### Why the data is not in the pack

A `.nlu` bundle is a RUNTIME artifact that ships to a hearing aid. Training data
is BUILD-TIME input. The bundle spec already draws this line: `bundle.json`'s
`training` block carries **`dataset_hashes`** — the sha256 of the data that
produced the model, never the data itself. So the pack stays small and carries
no user utterances, while lineage from model back to corpus is still provable.

Trained output goes to `models/intent/<lang>/`, mirroring the in-bundle path so
`assemble_pack.py` copies it across without renaming. Those artifacts are
gitignored; the pack is how they travel.

### `_archive/`

Retired by owner decision (2026-07-26): the combined-multilingual corpora
(`multilingual/`), the Dialogflow exports (`dialogflowData/`) and the
pre-migration Danish/French masters. English trains from `en/train.csv`; other
languages will supply their own data. Archived rather than deleted — this data
was recovered from near-loss, so it gets a grace period before removal.

## Rules

- **Training and evaluation data belongs here** and is committed.
- **Model artifacts do not.** `models/*.onnx`, `*.pkl` and friends stay
  gitignored and are regenerated (`make train`).
- `data/bootstrap/en/` is a provisional English-only snapshot kept from the
  window when this directory was empty. It is now redundant and can be deleted.
- Keep an eye on churn: a full regeneration that rewrites all 29 files each time
  will grow history. If that starts to hurt, the answer is fewer regenerated
  commits, not a return to an unreachable remote.

---

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
