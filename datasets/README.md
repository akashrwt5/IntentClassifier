# datasets/ — tracked directly in git

**Storage decision (2026-07-26): these files live in git. DVC has been removed.**

At ~6.7 MB of CSV across 29 files this is well within what git handles
comfortably, and CSVs delta-compress well across revisions. DVC was buying
nothing here: its configured remote was a *local filesystem path*
(`../../dvc-store`) that existed only on one machine, so CI, a fresh clone and
every scheduled run got no data at all and every model-dependent gate silently
skipped (Review-F5 blocker B2). A tool that no other machine can read is worse
than no tool.

## Provenance of the pre-git snapshot

The final DVC pointer, kept so the migration is auditable:

```
md5:    23a40044be29147b5de26753d90214af.dir
size:   7072263 bytes
nfiles: 29
```

If you ever need to prove the committed tree matches what DVC tracked, that is
the hash to reconcile against.

## CURRENT STATE — the authoritative data is not here yet

This directory currently holds **only the two docs you are reading** plus, if you
have run the bootstrap, five English CSVs generated from
`data/bootstrap/en/`. Those generated files are **deliberately untracked**: they
are a provisional English-only snapshot and committing them would put a
duplicate of `data/bootstrap/en/` into git under names that the real data also
uses.

**Owner action — commit the real tree** from the machine that has it (the old
DVC cache, ~29 files / 6.7 MB):

```bash
# 1. clear the provisional files so they cannot be mistaken for real data
rm -f datasets/*.csv

# 2. copy the authoritative files in (from your DVC cache / working copy)
cp -r <your-datasets>/* datasets/

# 3. sanity-check before committing
ls datasets/*.csv | wc -l        # expect ~25+, not 5
python scripts/ci/bootstrap_en_data.py   # must now say "Real datasets/ content is present"

# 4. commit
git add datasets/ && git commit -m "data: commit the authoritative datasets (DVC removed)"
```

Step 3 is the one that matters: the bootstrap script refuses to touch real data,
so if it still offers to materialise the snapshot, the real files are not there.

Once that lands, `data/bootstrap/` can be deleted and every model-dependent test
stops skipping.

## Rules

- **Training and evaluation data belongs here** and is committed.
- **Model artifacts do not.** `models/*.onnx`, `*.pkl` and friends stay
  gitignored and are regenerated (`make train`).
- Keep an eye on churn: a full regeneration that rewrites all 29 files each time
  will grow history. If that starts to hurt, the answer is fewer regenerated
  commits, not a return to an unreachable remote.

---

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
