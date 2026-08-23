# File register — P0 and P1

Every file created or changed while executing phases P0 and P1 of
`docs/Prod-Work-Documentation/semantic_compression_plan.md`, what it does, and why it
had to exist. `B1`–`B15` are defect IDs from §5 of that plan.

Read this when you are asking "why is this file here" — the answer is always a defect
that produced a confident wrong number rather than a crash. Nothing in P0 or P1 makes
the model better. Both phases make the numbers true, because until they were, no
improvement could be told apart from an error.

---

## The two contracts

P0 and P1 each establish one contract, and every file below belongs to one of them.

| | Question it answers | Enforced by |
|---|---|---|
| **P0 — the artifact contract** | Is this encoder what it says it is? | `artifact.py` |
| **P1 — the instrument contract** | Is this ruler what it says it is? | `check_instruments.py` |

The order matters. P0 first, because an instrument built with a mis-measured encoder
inherits the error; P1 second, because after P0 the encoder was trustworthy and the
evaluation sets were not.

---

## P0 — the artifact contract

### New

| File | What it does | Why it was needed |
|---|---|---|
| `artifact.py` | Loads an encoder and refuses it unless the artifact declares four things: tokenizer, pooling, vocabulary size, and the head beside it. Exposes `load_encoder` / `encode` / `head_path`, plus a CLI that describes any export. | Six scripts each re-derived tokenizer, pooling and id range independently, and **every one of them got at least one wrong**. B1 (tokenizer fell back to a 30.5k vocab), B2 (pooling hardcoded to CLS, scoring mean-pooled exports at 0.005), B3 (`vocab.txt` inconsistent with the matrix), B4 (an id clamp turning a fatal mismatch into silent 7% corruption), B12 (head written one level up, so two exports overwrote each other). One place to be right, and it **raises rather than guesses** when an artifact declares nothing. |
| `test_artifact_contract.py` | 16 tests, one per defect that actually occurred, asserting each now fails loudly. Builds its own fixtures — needs neither the 9 MB artifacts nor onnxruntime. | Every one of those defects survived for months because nothing crashed. A test that pins the *failure* is the only thing that keeps a silent defect from coming back silently. Includes a deliberately pinned test that `track1_pruned_l3` still fails, so if it ever starts passing someone must update the register. |
| `backfill_artifact_metadata.py` | One-time, idempotent: writes `pooling.json` beside each export (sourced from the originating `1_Pooling/config.json`, never guessed) and regenerates `vocab.txt` from the pruned `tokenizer.json`. | The exports predate the contract, so facts about them lived only in the sentence-transformers directories they came from, or nowhere. Deliberately skips `track1_pruned_l3` (no metadata fixes a tokenizer that was never pruned) and `track3_svd_l6` (its pooling would have to be inferred, not sourced). |
| `.gitignore` (folder-local) | Ignores `output_models/`, `*_l3/`, `*_l6/`, `*.zip`, scratch dirs. | 410 MB of model weights were untracked **but not ignored** — one `git add .` from entering a public repository. The plan and the README both already claimed they were ignored. Folder-local so it travels if the directory is lifted out. |
| `README.md` | Documents the artifact contract, what crosses the directory boundary, known-broken artifacts, and the one consumer living outside. | The directory is meant to be liftable into a separate project. That is a property someone has to maintain, and an undocumented property does not survive contact with the next person. |

### Changed

| File | What changed | Why |
|---|---|---|
| `train_experimental_head.py` | Dropped its private `embed_onnx` (id clamp + hardcoded CLS); loads through `artifact.py`; writes the head **beside** its model. | B4 and B12. Its clamp was the reason B1 could hide: an incompatible tokenizer became a 7% corruption instead of an error. |
| `test_distilled_holdout.py` | Dropped the `packages.runtime` import entirely, inlined an optional TF-IDF baseline, uses `head_path`, writes `holdout_results.md`. | The import was the only thing tying this directory to the rest of the repo. The report file exists because four encoders and two GPU sessions had produced **zero recorded measurements** — a run that leaves nothing behind did not happen. |
| `evaluate_compression.py` · `interactive_test.py` · `interactive_distilled_only.py` | Moved onto `load_encoder` / `encode` / `head_path`; each now names the artifact it is scoring. | Three more copies of the same three decisions. `interactive_test.py` additionally read `backend.model_path`, an attribute that does not exist, so it raised on every run (B11). |
| `build_pruned_l3.py` | Rewrites **both** `vocab.txt` and `tokenizer.json`, asserts the pruned vocab size, writes `pooling.json`. | B10. It rewrote `vocab.txt` only, and a fast tokenizer reads `tokenizer.json` — so the vocabulary pruning silently never happened and every number that artifact produced had ~7% of its input replaced by `[UNK]`. |
| `export_distilled_onnx.py` | Prunes `vocab.txt` alongside the tokenizer, copies `pooling_mode` from the source pooling config, verifies the contract before finishing. | So that new exports satisfy the contract at birth rather than needing a backfill. |
| `colab_distillation_stage1.py` · `colab_stage2_contrastive.py` | Pin the training file by sha256 and row count; refuse to start on a mismatch; write `provenance.json`. | Colab has no repository checkout, so a path cannot pin anything. Three files in this repo could be uploaded as `train.csv` and **two of them leak holdout rows into training** (585 and 1,461 rows). B0 exists because nobody recorded which one a historical run used. |
| `retired/interactive_baseline_only.py` | Moved to `retired/`, path depth fixed, header explains its status. | It runs a third-party model that is not obliged to carry our metadata, so it sits outside the contract on purpose. Moving it broke its own repo-root resolution — fixed in the same commit. |
| `scripts/evaluate_models.py` *(outside this directory)* | Loads through `artifact.py` instead of its own copy of the pooling, the clamp and the old head path. | It is the one consumer that cannot live here (it needs `packages/runtime`). It carried a second copy of every decision P0 had just centralised — and the B12 head move would have broken it silently. |

---

## P1 — the instrument contract

### New

| File | What it does | Why it was needed |
|---|---|---|
| `instruments.py` | The shared definitions: `normalize_text`, token-set Jaccard near-duplication with exact prefix filtering, and `minimum_detectable_effect` (McNemar power). | Three numbers steered the plan with **no script behind them**. Re-measured, the 44% near-duplicate share came back 44.7% and the difference could not be attributed, because the original method was unrecoverable. A value with no script is a memory, not a measurement. `normalize_text` is vendored, not imported, so the directory stays liftable — and a test asserts it has not drifted from the repository's copy. |
| `split_dev_sets.py` | Derives `dev_near.csv` (657 rows) and `dev_hard.csv` (813) from the honest holdout plus `train.csv`, with a manifest recording method, threshold, hashes, row counts and MDE. `--check` re-derives and diffs. | B7: `holdout_honest.csv` is 44.7% near-duplicate of training data, so its score is part generalisation and part recall in an unknown ratio. Shipped as a **script, not two CSVs**, because near-duplication is a relation between the holdout and `train.csv` — a hand-frozen file silently stops meaning what it says the moment training data moves. |
| `check_instruments.py` | Five CI guards: manifest freshness, exact train/holdout disjointness, partition integrity, split reproducibility, and **`dev_hard` contamination by training data**. `--refreeze` re-pins the holdout manifest while preserving the old hashes under `amendments[]`. | B9, and everything shaped like it. The manifest was frozen at `cc46010`; `ce0d469` then added 77 training rows and `af4a88b` rewrote every label in both files at an unchanged row count. Nothing failed — the manifest just stopped describing the files it named. The contamination guard is the one that matters later: when the Super Dataset enters training, it turns a rebuild of the instrument into a filter over new training rows. |
| `inventory_instruments.py` | Measures all nine English evaluation sets — rows, intents, exact leak, near-duplicate share, MDE — and generates `INSTRUMENTS.md` with a charter per set. | The plan's P1 asked for a charter per instrument. Generating it means no charter ever carries a hand-typed number. It found **B14** (the benchmark `evaluate_models.py` scores on is 85.5% leaked and covers 10 of 57 intents) and **B15** (two malformed labels making 10 rows unscoreable), and surfaced two broad clean sets the plan had been treating as an afterthought. |
| `score_instruments.py` | Scores the shipped encoder and its shipped head on every instrument, next to each one's MDE. Writes `BASELINE.md` and `baseline_predictions.csv`. | The P1 gate: produce the baseline every later phase is judged against. Before it, the project had one number (0.8578) on an instrument that is 44.7% near-duplicate. Excludes unscoreable rows by name rather than scoring them as errors, and averages macro-F1 over **gold** classes — sklearn's union convention scores a 10-intent instrument at 0.397 against an accuracy of 0.750, which reads as a broken model and is really an instrument narrower than the head. |
| `test_instruments.py` | 13 tests: vendored-normaliser parity against the repository, prefix filtering against brute force at four thresholds, the §10 power table against the code, and each guard **fired deliberately** — a leak injected, a row dropped, a manifest edited at an unchanged row count. | A guard nobody has seen fail is not known to work. Two of these tests exist because the pre-commit hooks, not a test, caught CRLF line endings and a double trailing newline — which changed file hashes *after* the manifest had pinned them. |

### Changed

| File | What changed | Why |
|---|---|---|
| `colab_distillation_stage1.py` · `colab_stage2_contrastive.py` | `run_head_retraining_and_eval` and `run_sanity_eval` renamed `run_memorisation_check`; output now prints `retrieval on seen-paraphrase split` and says it must not be quoted as accuracy. | Both split a corpus that is 44.7% near-duplicate within itself, so paraphrases of one sentence land on both sides. Stage 2 is worse: it trains the *encoder* on the whole file, then scores a split of it. Figures from these functions reached earlier revisions of the plan as quality claims. Kept, because they are the only cheap in-Colab signal that a run failed outright. |
| `README.md` | Added the instrument-contract section and its three rules. | Same reason as P0's README: an undocumented invariant is a temporary one. |
| `semantic_compression_plan.md` | Rev 5.2 → 5.4. B13–B15 added; B9 corrected to describe both drifts; near-duplicate share and `dev_hard` size corrected; three power-table cells corrected by 0.001; B15's impact corrected from 15 rows to 10; the measured baseline recorded under P1. | Two of those are corrections to numbers **this work got wrong first**. They are listed in §14 rather than quietly edited, because a plan that revises itself without saying so is worth less than one that is occasionally wrong out loud. |

---

## Generated, not authored

These are outputs. They are committed because they are the record of what was measured,
but nothing here should be edited by hand — regenerate instead.

| File | Produced by | Holds |
|---|---|---|
| `language_packs/en/dev_hard.csv` | `split_dev_sets.py` | 813 clean rows. **The primary decision instrument, frozen for P2–P8.** |
| `language_packs/en/dev_near.csv` | `split_dev_sets.py` | 657 near-duplicate rows. Regression detection only. |
| `language_packs/en/dev_split.manifest.json` | `split_dev_sets.py` | Method, threshold, input and output hashes, per-set MDE, intent coverage. |
| `language_packs/en/extras/holdout_honest.manifest.json` | `check_instruments.py --refreeze` | Re-pinned hashes, with the superseded ones and the reason for each drift preserved under `amendments[]`. |
| `INSTRUMENTS.md` | `inventory_instruments.py --write` | All nine evaluation sets: rows, intents, leak share, MDE, and what each may be asked. |
| `BASELINE.md` | `score_instruments.py` | The P1 baseline. `dev_hard` = **0.8327**. |
| `baseline_predictions.csv` | `score_instruments.py` | One row per scored utterance: instrument, text, gold, prediction, correct. |
| `holdout_results.md` | `test_distilled_holdout.py` | The P0 reproduction fingerprint, 0.8578. |

`baseline_predictions.csv` is the least obvious and the most important. McNemar's test
compares **which** rows two models get right, not how many — and that cannot be
recovered from two accuracy figures. Without this file, P2 could not be compared with
P1 without re-running P1.

---

## If you read one thing

| Question | File |
|---|---|
| What may I trust this encoder to be? | `artifact.py` |
| What may I ask this evaluation set? | `INSTRUMENTS.md` |
| What is the number to beat? | `BASELINE.md` |
| Did something move under me? | `check_instruments.py` |
| Why does this file exist at all? | this register |

---

## Commit map

| Commit | Covers |
|---|---|
| `18e34d8` | P0 — the artifact contract, its tests, the backfill, the gitignore, and the eight scripts moved onto it |
| `3f6a7b7` | P0 — the one consumer outside the directory, and the retired script |
| `f0666fc` | P1 — the instrument contract, the split, the guards, the inventory |
| `46d7d79` | P1 — the baseline: `score_instruments.py`, `BASELINE.md`, `baseline_predictions.csv`, and this register |
