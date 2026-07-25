# English bootstrap corpus — PROVISIONAL

A tracked, provenance-stamped English training snapshot that lets model,
calibration and evaluation work run **in any clone**, without DVC.

**This is a stopgap. It is not the authoritative dataset.** Read the limits
below before quoting any number derived from it.

Materialise it into `datasets/`:

```bash
python scripts/ci/bootstrap_en_data.py           # write datasets/
python scripts/ci/bootstrap_en_data.py --check   # verify only
python scripts/ci/bootstrap_en_data.py --build   # regenerate this snapshot
```

## Why it exists

The real datasets are DVC-managed and `.dvc/config` points the remote at
`../../dvc-store` — a local path that exists only on the owner's machine. Every
other environment (CI, a fresh clone, the scheduled routine) gets no data, so
all model-dependent gates silently skip. That is Review-F5 blocker **B2**.

Rather than leave the entire English workstream blocked on provisioning a shared
remote, this snapshot recovers the one corpus that *is* reachable from git.

## Where it came from

The English master committed on the reference branch
`claude/claude-setup-architecture-ebqobs-Temperaturescaling-fixes`, migrated
from the old 59-label space to the shipped 57-label `domain.object.action`
taxonomy via `docs/Review-F5/capability-map.json`.

The migration is deterministic and fully verified — the map covers all 59 old
labels exactly, and the 57 labels it produces match `content/nlu_schema.json`
with **zero** drift in either direction. Exact hashes, row counts and the source
commit are in `PROVENANCE.json`.

| File | Rows | Notes |
|---|---|---|
| `04_GENERATED_MASTER_training_data.csv` | 9,826 | training master (`text,intent`) |
| `semantic_holdout_2.csv` | 331 | trainer's leakage-guard holdout |
| `semantic_holdout_100.csv` | 100 | paraphrase holdout (needs the semantic stage) |
| `semantic_oos.csv` | 156 | out-of-scope set |
| `confirmation_fixtures.csv` | 170 | ND-3 dialogue-act rows, retained not discarded |

The 170 fixture rows carried `Cmd.SendMessage - yes|no` — labels the ND-3
migration dissolved into the confirmation flow. They are kept with an
`_origin_file` column for confirmation-flow regression use.

## Limits — read before quoting a number

1. **Provisional, not authoritative.** It predates this branch's post-migration
   data-quality passes. Numbers from it are directional, and must never be
   published as baseline-v2.
2. **English only.** The fr/de/da masters are not recoverable from git; those
   languages stay blocked on the real datasets.
3. **It is the leaked corpus.** `multilingual/test/en_holdout.csv` was drawn
   from this data — 99.9% of that "holdout" appears here verbatim (blocker
   **B9**). So this snapshot is the correct input for *building* an honest
   holdout, and must never be used *as* one. Partition it with removal.
4. **It never overwrites real data.** `bootstrap_en_data.py` refuses to touch
   `datasets/` once authoritative content is present. Silently swapping real
   data for a stale snapshot is exactly the failure mode Review-F5 exists to
   fix.

## When the real remote arrives

Provision the shared DVC remote, `dvc pull`, and this snapshot goes dormant on
its own — the script detects real content and no-ops. Delete
`data/bootstrap/` once the authoritative baseline is recorded.
