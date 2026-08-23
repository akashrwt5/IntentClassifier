# Semantic compression

Distilling the on-device intent encoder down to a size that ships, without
losing the paraphrase generalisation the product depends on.

Plan of record: `docs/Prod-Work-Documentation/semantic_compression_plan.md`.

This directory is self-contained on purpose. Nothing here imports from the rest
of the repository at runtime, so it can be lifted into a separate project
unchanged. It reads evaluation and training data from the repo, but its own
logic depends only on `onnxruntime`, `transformers`, `numpy` and
`scikit-learn`.

---

## The artifact contract

An exported encoder is not one file. It is a model plus three facts that must
travel with it:

| Fact | Where it lives | What happens if it is wrong |
|---|---|---|
| **Tokenizer** | `tokenizer.json` beside the model | A 30,522-token tokenizer against a 10,000-row model corrupts ~7% of tokens — `mute`, `streaming`, `decrease` — and costs about 10 accuracy points |
| **Pooling** | `pooling.json` beside the model | Reading the wrong token scores a working encoder at **0.005**, below chance across 57 intents |
| **Vocabulary size** | `config.json` vs `tokenizer.json` vs `vocab.txt` | Python reads `tokenizer.json` and never notices a stale `vocab.txt`; a native Swift or Kotlin wordpiece reads `vocab.txt` and indexes out of range on device |
| **Classifier head** | `classifier_head.pkl` beside the model | Written one level up, two exports shared the path and overwrote each other — so one encoder got scored with the other one's head |

None of those failures crash. All three produce confident numbers computed from
the wrong rows, which is why they survived for months and were found only by
re-measuring from scratch.

The rule behind all four is the same: **everything an artifact needs to answer
a question lives in the artifact's own directory.** `artifact.py` applies it
once and **refuses to guess** when an artifact declares nothing.

```bash
# what does each export declare, and does it hold together?
python3 artifact.py output_models/*/model_quantized.onnx

# the guards, on synthetic fixtures -- no 9 MB artifacts needed
python3 test_artifact_contract.py
```

Every consumer here goes through `load_encoder()` / `encode()`. If you are
adding a script, use them rather than re-deriving tokenizer, pooling and id
range — six scripts did that independently and every one of them got at least
one wrong.

There is no pooling override and no default. Every encoder loaded here is one
this directory exported, and it declares its pooling in a file.

---

## What crosses the directory boundary

Nothing, in code — for the active scripts:

```bash
grep -rE "^\s*(from|import)\s+packages" *.py     # returns nothing
```

so this directory can be lifted into another project as it stands. The one
exception is `retired/interactive_baseline_only.py`, which still imports from
`packages.runtime`. That dependency is precisely why it is retired, its header
says so, and nothing else here uses it.

What does point outward is **data**, which any training script needs:

| Script | Reads |
|---|---|
| `train_experimental_head.py` | `language_packs/en/train.csv`, `extras/oos_2.csv`, `datasets/semantic_holdout_2.csv` |
| `test_distilled_holdout.py` | `language_packs/en/holdout_honest.csv`, and optionally the shipping TF-IDF model for context |
| `export_distilled_onnx.py` | `language_packs/en/train.csv`, `extras/oos_2.csv` |
| `build_pruned_l3.py` | `datasets/04_GENERATED_MASTER_training_data.csv` |
| `split_dev_sets.py` | reads `language_packs/en/{train,holdout_honest}.csv`, **writes** `dev_{near,hard}.csv` + `dev_split.manifest.json` beside them |
| `check_instruments.py` | the same files, plus `extras/holdout_honest.manifest.json` |
| `inventory_instruments.py` | every English evaluation set in `language_packs/en/` |

The three scripts above are the only ones here that **write** outside this
directory. That is deliberate: evaluation data belongs with the language pack,
not with the compression code, and putting a second copy inside this directory
would be the drift risk the artifact contract exists to prevent. Pass
`--out-dir` to place them elsewhere.

Each of those is grouped under an `# --- Outside this directory ---` header at
the top of its script. Lifting the directory out means repointing them.

`models/minilm-l6-v2.onnx` used to be loaded here as a size and quality
reference. It has been removed: it lives outside this directory, and the plan
replaces it with the real `bge-small` teacher baseline at P1.5 anyway.

`interactive_baseline_only.py` existed only to run that model, and has moved to
`retired/`. It still works — the mean pooling it hardcodes is the pooling
all-MiniLM-L6-v2 was trained with — but it is **not covered by the artifact
contract**, because a third-party model is not obliged to carry our metadata.
Its header says so. Do not extend it; if the external reference is needed
again, load it through `artifact.py` with a declared pooling.

This does not affect the suspended MiniLM **student** lineage. That is a
different thing, it is not retired until P1 decides against the expanded
paraphrase set, and its artifacts are untouched (now gitignored rather than
moved).

---

## The instrument contract

The artifact contract above answers "is this encoder what it says it is". The
instrument contract answers the other half: **is this ruler what it says it is.**

Every defect in the register produced a confident number rather than a crash,
and the longest-lived ones were the cases where a measuring instrument moved
underneath a measurement. B9 is the shape of it: the honest holdout's manifest
was frozen, then 77 training rows were added and every label in both files was
rewritten at an unchanged row count. Nothing failed. The manifest simply stopped
describing the files it named.

Three rules, enforced by `check_instruments.py` in CI:

1. **No instrument number without a script.** The 44% near-duplicate figure that
   steered earlier revisions of the plan was hand-measured and could not be
   reproduced when it was re-run. `inventory_instruments.py` regenerates
   `INSTRUMENTS.md` so no charter carries a number typed from memory.
2. **`dev_hard` is frozen for P2-P8.** Near-duplication is a relation between the
   holdout and `train.csv`, so growing the training set can silently un-harden it.
   The answer is not to re-derive it -- that retires every number measured on it --
   but to guard it: any training row that near-duplicates a `dev_hard` row fails
   the build, and the fix is to drop that training row.
3. **Every result is reported against its MDE.** A difference smaller than the
   instrument's minimum detectable effect is not a small win; it is a number the
   instrument cannot tell from zero.

```bash
python3 split_dev_sets.py --dry-run      # measure the split, write nothing
python3 check_instruments.py             # CI: has any ruler moved?
python3 inventory_instruments.py --write # regenerate INSTRUMENTS.md
python3 test_instruments.py              # the guards, on synthetic fixtures
```

`instruments.py` vendors `normalize_text` from
`packages/buildtime/nlu_training/leakage.py` rather than importing it, so this
directory stays liftable. A copy drifts, so `test_instruments.py` asserts the two
agree on adversarial cases whenever the repository is present, and skips when it
is not. That test is why the copy is allowed to exist.

---

## Layout

```
artifact.py                       the encoder contract: load, verify, embed
test_artifact_contract.py         one test per defect that actually occurred
backfill_artifact_metadata.py     one-time: teach pre-contract exports to describe themselves

instruments.py                    the instrument contract: normalise, near-duplicate, power
split_dev_sets.py                 derive dev_near / dev_hard from the honest holdout
check_instruments.py              CI guard: fails if a ruler moved under a measurement
inventory_instruments.py          measure every English eval set -> INSTRUMENTS.md
score_instruments.py              score the shipped encoder on all of them -> BASELINE.md
test_instruments.py               guards for the above, incl. vendored-normaliser parity
INSTRUMENTS.md                    generated: what each eval set may and may not be asked
BASELINE.md                       generated: the P1 baseline every later phase is judged on
baseline_predictions.csv          generated: per-row right/wrong, so P2 can run McNemar
FILE_REGISTER.md                  every file P0 and P1 touched, and the defect behind it

colab_distillation_stage1.py      Stage 1 -- distil the teacher into a shallow student
colab_stage2_contrastive.py       Stage 2 -- contrastive adaptation on the intent taxonomy
export_distilled_onnx.py          prune vocab, quantise to INT8, declare pooling, verify

build_pruned_l3.py                track1 experiment (vocab pruning, no distillation)
build_svd_compressed.py           track3 experiment (SVD factorisation)

train_experimental_head.py        fit a LogReg head on each encoder
test_distilled_holdout.py         score against holdout_honest.csv -> holdout_results.md
evaluate_compression.py           embedding sanity + latency across encoders
interactive_*.py                  type a sentence, see what each encoder does with it
retired/                          kept for provenance, outside the contract, not extended

dataset_generation/               the Super Dataset generator (separate pipeline)
output_models/                    exported artifacts (gitignored, regenerated)
```

## Training data is pinned, not pathed

The Colab scripts run where there is no repository checkout, so the training
file is uploaded by hand and a path cannot pin it. Three files here could be
uploaded as `train.csv` and two of them leak holdout rows into training:

| Candidate | Rows | Holdout rows present |
|---|---:|---:|
| `language_packs/en/train.csv` | 8,430 | **0** |
| `new_semantic/data/en/train.csv` | 23,989 | 585 (39.8%) |
| `complete_csv.csv` | 31,699 | 1,461 (99.4%) |

`language_packs/en/train.csv` is the training file. Both Colab scripts pin its
sha256 and row count and refuse to start on a mismatch. When the training data
legitimately changes, update `EXPECTED_SHA256` and `EXPECTED_ROWS` deliberately
and say why in the commit.

## One consumer lives outside this directory

`scripts/evaluate_models.py` benchmarks the distilled encoder against the
TF-IDF model, the student semantic head and MiniLM in one run, so it needs
`packages/runtime` and cannot live here. It loads the encoder through
`artifact.py` rather than by hand — it used to carry its own copy of the
hardcoded pooling, the id clamp and the old head path, and a second copy is a
second place for them to drift.

If this directory is lifted into another project, that script stays behind.

## Known-broken artifacts

`output_models/track1_pruned_l3` fails the contract and should keep failing
until it is re-exported or retired. `build_pruned_l3.py` rewrote `vocab.txt`
but not `tokenizer.json`, and a fast tokenizer reads the latter — so the
pruning silently never happened and every number that artifact produced had
~7% of its input replaced by `[UNK]`. The script is fixed; the artifact on
disk predates the fix.

`output_models/track3_svd_l6` fails only because it does not declare its
pooling. It is a superseded experiment; `backfill_artifact_metadata.py`
deliberately does not write a value it cannot source from a real config.
