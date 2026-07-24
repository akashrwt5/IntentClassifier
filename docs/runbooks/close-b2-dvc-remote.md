# Runbook — Close B2: shared DVC remote + `dvc pull` in CI

**Goal:** make the training data reachable by CI and by teammates, so the
model-dependent safety/quality tests (wrong-action budget, unified `evaluate`,
semantic) actually run instead of silently skipping — which is the whole point
of B2 in `production-readiness-review.md`.

**Why it can't be automated from a code session:** `dvc push` needs (a) the
actual data blobs, which today exist only in your working tree + local DVC cache,
and (b) cloud credentials. Both live on your machine. So this runbook is executed
by you (the owner) locally; the repo changes it produces are then committed.

**Current state (the problem):**
- `.dvc/config` remote `localstore` → `../../dvc-store` — a *local folder*, not
  shared. `datasets.dvc` tracks 29 files / ~7 MB.
- Result: a fresh clone or CI has only the pointer, not the data → the safety
  tests skip. "Green CI" currently does not mean the safety gate passed.

**Privacy note (read first):** the training phrases are user-derived language
data for a medical-adjacent product. The shared store MUST be a private,
access-controlled bucket with encryption at rest, and (for EU users) in an
appropriate region. This is the same class of decision as ND-9/ND-12 — loop in
whoever owns data/privacy before picking the bucket.

---

## Step 1 — pick the store (owner decision)

| Option | Command family | Notes |
|---|---|---|
| **AWS S3** (recommended) | `dvc remote add`/`modify` `s3://…` | most common; IAM integrates with CI |
| GCP GCS | `gcs://…` | fine if you're already on GCP |
| Azure Blob | `azure://…` | fine if you're already on Azure |

Install the DVC extra for your choice, e.g. `pip install 'dvc[s3]'`.

## Step 2 — point the remote at it (on your machine)

```bash
# add the shared remote and make it the default
dvc remote add -d shared s3://<your-private-bucket>/intentclassifier
# encryption + region (S3 example)
dvc remote modify shared sse AES256
dvc remote modify shared region <your-region>

# push the blobs you already have locally
dvc push                     # uploads the 29 data files to the bucket

# commit the config change (NOT the data — data stays in DVC)
git add .dvc/config
git commit -m "chore(dvc): move datasets remote to shared cloud store (close B2)"
```

Keep the old `localstore` entry or remove it — the default `-d shared` is what
`pull`/`push` use now.

## Step 3 — verify a clean pull works

```bash
# in a fresh clone (or: rm -rf .dvc/cache datasets && dvc pull)
dvc pull
ls datasets/multilingual/*.csv     # the per-language training CSVs appear
```

## Step 4 — wire `dvc pull` into CI (the payoff)

Add to `.github/workflows/ci.yml` **before** the Pytest step, and provide the
bucket credentials as repo secrets (S3 example):

```yaml
      - name: Install DVC
        run: pip install 'dvc[s3]'

      - name: Pull datasets + models (DVC)
        env:
          AWS_ACCESS_KEY_ID:     ${{ secrets.DVC_AWS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.DVC_AWS_SECRET }}
        run: dvc pull --allow-missing
```

Then the model-dependent suites stop skipping. Optional but recommended: make
them **fail loudly** if data is unexpectedly absent in the CI context (so a
broken pull can't masquerade as a green run), and add the wrong-action replay as
a required gate once data is present (ties into B1).

## Step 5 — confirm the gates now run

```bash
# locally, after pull:
PYTHONPATH=packages/buildtime:packages/runtime \
  python -m nlu_training.wrong_action_harness --langs en
pytest tests/test_wrong_action_mitigations.py tests/test_unified_evaluate.py -q
```

These should now execute (not skip). In CI, the same suites should run on every
PR — that is B2 closed.

---

## Definition of done

- `.dvc/config` points at a shared, private, encrypted cloud remote; `dvc push`
  done; the local `../../dvc-store` is no longer the only copy.
- A fresh `git clone` + `dvc pull` yields the training CSVs.
- `ci.yml` pulls data and the wrong-action / evaluate suites **run** (no skip);
  credentials provisioned as repo secrets.
- Once green: turn the wrong-action replay into a required CI gate (B1) so the
  safety budget can never silently regress again.

## Owner to-dos (can't be done from a code session)

1. Choose the bucket + region; get data/privacy sign-off.
2. Run Steps 2–3 on your machine (you hold the data blobs + creds).
3. Provision the CI secrets (`DVC_AWS_KEY_ID` / `DVC_AWS_SECRET` or equivalent).
4. Commit the `.dvc/config` and `ci.yml` changes.
