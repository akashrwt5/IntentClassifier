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

Nothing, in code. `grep -E "^\s*(from|import)\s+packages" *.py` returns
nothing, so this directory can be lifted into another project as it stands.

What does point outward is **data**, which any training script needs:

| Script | Reads |
|---|---|
| `train_experimental_head.py` | `language_packs/en/train.csv`, `extras/oos_2.csv`, `datasets/semantic_holdout_2.csv` |
| `test_distilled_holdout.py` | `language_packs/en/holdout_honest.csv`, and optionally the shipping TF-IDF model for context |
| `export_distilled_onnx.py` | `language_packs/en/train.csv`, `extras/oos_2.csv` |
| `build_pruned_l3.py` | `datasets/04_GENERATED_MASTER_training_data.csv` |

Each of those is grouped under an `# --- Outside this directory ---` header at
the top of its script. Lifting the directory out means repointing them.

`models/minilm-l6-v2.onnx` used to be loaded here as a size and quality
reference. It has been removed: it lives outside this directory, and the plan
replaces it with the real `bge-small` teacher baseline at P1.5 anyway.
`interactive_baseline_only.py` existed only to run it and has moved to
`retired/`.

---

## Layout

```
artifact.py                       the contract: load, verify, embed
test_artifact_contract.py         one test per defect that actually occurred
backfill_artifact_metadata.py     one-time: teach pre-contract exports to describe themselves

colab_distillation_stage1.py      Stage 1 -- distil the teacher into a shallow student
colab_stage2_contrastive.py       Stage 2 -- contrastive adaptation on the intent taxonomy
export_distilled_onnx.py          prune vocab, quantise to INT8, declare pooling, verify

build_pruned_l3.py                track1 experiment (vocab pruning, no distillation)
build_svd_compressed.py           track3 experiment (SVD factorisation)

train_experimental_head.py        fit a LogReg head on each encoder
test_distilled_holdout.py         score against holdout_honest.csv -> holdout_results.md
evaluate_compression.py           embedding sanity + latency across encoders
interactive_*.py                  type a sentence, see what each encoder does with it
retired/                          scripts kept for provenance, not extended

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
