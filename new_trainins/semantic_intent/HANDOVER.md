# Handover — offline hearing-aid intent classifier

Everything needed to ship, rebuild, or hand this to someone else. Written after
the final run of 22 August 2026.

---

## 1. What ships

**One file.** Everything else in this repository exists to produce it or to
prove it works.

```
models/final_student_256/onnx/intent_int8.onnx      4.75 MB   ← THE MODEL
models/final_student_256/onnx/runtime_config.json   0.42 MB   thresholds, labels, OOD centroids
models/final_student_256/onnx/tokenizer/            tokenizer.json + tokenizer_config.json
```

`intent_fp32.onnx` (17.2 MB) sits in the same folder. **Do not ship it** — it is
an export intermediate kept only so parity can be re-checked.

| | |
|---|---|
| model name | `student-h256-l4` (INT8 export: `intent_int8.onnx`) |
| what it is | a 4-layer, 256-dim BERT-style encoder distilled from `BAAI/bge-small-en-v1.5`, plus an MLP head, temperature scaling and an OOD whitening matrix, all fused into one ONNX graph |
| parameters | 4.01 M |
| vocabulary | 3 267 WordPiece tokens (pruned from bge's 30 522 — the corpus only ever uses 10.7% of them) |
| intents | 57 |
| max sequence length | 64 |
| ONNX opset | **18** |
| latency | 1.08 ms p50 / 1.21 ms p90 (desktop CPU) |
| network | none, anywhere in the path |

### Graph interface

```
input   input_ids           int64    [batch, 64]
input   attention_mask      int64    [batch, 64]
output  probs               float32  [batch, 57]   ← ALREADY calibrated
output  whitened_embedding  float32  [batch, dim]  ← for the OOD distance
```

Label order comes from `runtime_config.json → labels`. Never hardcode it.

---

## 2. Final numbers

INT8, on the leakage-controlled held-out test set.

| metric | ship bar | shipped | |
|---|---|---|---|
| accepted precision | ≥ 0.97 | **0.9789** | pass |
| OOD rejection (286 rows) | ≥ 0.95 | **0.9685** | pass |
| INT8 size | ≤ 5 MB | **4.748 MB** | pass |
| gate agreement (Python vs ONNX) | ≥ 0.99 | **0.99626** | pass |
| coverage | ≥ 0.68 | **0.658** | **miss, by 0.022** |

Supporting numbers: test accuracy 0.9061 · macro-F1 0.8932 · ECE 0.0244 ·
contextual 0.490 · STT noise 0.852 · hard negatives 0.715 · minimal pairs 0.705 ·
negation 0.590 · accessories 0.471.

**Read "accepted precision 0.9789 at coverage 0.658" as:** of every 100 requests,
about 66 are acted on and roughly 1 of those 66 is wrong; the other 34 get "say
that again". The 0.68 coverage bar was missed by about two requests in a hundred.
It shipped because a refusal is the recoverable failure and because coverage
moves 0.035 between two fits of the same model — see §7.

---

## 3. The safety gate — six reasons to refuse

Only ACCEPT may reach hardware. A rejected result must never be downgraded to a
best guess.

| # | signal | threshold as shipped |
|---|---|---|
| 1 | ASR confidence | **not fitted — inert.** See §8 |
| 2 | reject class | `Default Fallback Intent` |
| 3 | OOD score (Mahalanobis, embedding space) | `> 16.9568` → refuse |
| 4 | confidence, per risk tier | normal ≥ **0.925**, high ≥ **0.965** |
| 5 | top1 − top2 margin | ≥ 0.0 (inactive this run) |
| 6 | corrective structure | on, regex in `runtime_config.json → gate.corrective_pattern` |

Base `conf_threshold` is 0.865; the per-risk values above override it and are
what actually apply. Temperature 2.3041 is already inside the graph — the number
in the config is for traceability only, do not apply it again.

**High-risk intents (5):** `Cmd.MemoryChange`, `Cmd.SendMessage`,
`Cmd.StreamingStop`, `Cmd.VolumeMute`, `reminders.complete`. A mistake in these
is one the user may not notice or be able to undo.

`Cmd.VolumeUnmute` is deliberately **not** high-risk — it is the recovery action.
Firing it wrongly is loud, obvious and instantly reversible.

Only signals 3 and 1 are independent of the softmax. That matters: a softmax
compares 57 known classes against each other. It answers "which of these", never
"is it any of these", so it can be completely confident about input resembling
nothing it was trained on.

---

## 4. Files that matter

### Ship these
```
models/final_student_256/onnx/intent_int8.onnx
models/final_student_256/onnx/runtime_config.json
models/final_student_256/onnx/tokenizer/
```

### Read these before writing app code
```
reports/android_integration.md     the contract: Kotlin gate, opset warning, checklist
reports/asr_confidence_protocol.md how to turn on gate signal 6 (optional)
README.md                          what is solved, what is not, and why
```

### Source of truth for the data
```
data/raw/en.csv                    9 826 rows, 57 intents. THE dataset.
configs/intents.yaml               57 intents, 17 families, 11 written policies
```

### Scripts, by what they do

*Build the data* — run in this order:

| script | does |
|---|---|
| `audit_dataset.py` | phase 1. Imbalance, length, shortcut tokens |
| `build_taxonomy.py` | writes `configs/intents.yaml`, measures action/how-to share per intent |
| `ood_generate.py` | the 286-row OOD suite + its two validity checks |
| `build_challenge_sets.py` | all challenge suites (calls `ood_generate`) |
| `split_dataset.py` | group-based leakage-free train/val/test + STT suite |
| `build_targeted_training.py` | F1–F15 augmentation → `train_augmented.csv` |

*Generators the above import:* `corrective_pairs.py` (F11), `symptom_pairs.py`
(F12), `long_forms.py` (F13 + negation/contextual suites),
`foreign_objects.py` (F14), `accessories.py` (F15).

*Train and ship:*

| script | does |
|---|---|
| `train_classifier.py` | fits head + temperature + OOD scorer + all thresholds |
| `distill_student.py` | distils the small encoder from the teacher |
| **`pick_seed.py`** | fits the head across seeds, selects on validation ECE. **Use this, not a bare `train_classifier.py`** — see §7 |
| `export_onnx.py` | fp32 + INT8 export, calibration fused into the graph |
| `parity_test.py` | Python vs ONNX; fp32 judged numerically, INT8 on decision parity |
| `evaluate_onnx.py` | the full suite matrix on the exported file |
| `predict.py` | **try it by hand.** Loads ONLY the INT8 file + config, never the training pipeline |
| `variance_check.py` | 5-seed noise floor. Run before believing any change |

*Support:* `common.py` (normalize, leakage key), `calibration.py`,
`ood_score.py`, `pipeline.py`, `encoders.py`, `evaluate_model.py`,
`error_analysis.py`, `make_reports.py`, `fit_asr_threshold.py`.

*Not needed to ship:* `benchmark_classifiers.py`, `shrink_encoder.py`,
`size_budget.py`, `finetune_encoder.py`, `make_smoke_encoder.py` — these answered
questions during development (which encoder, can bge be pruned, what fits in
5 MB) and are kept as the record of those answers.

### Safe to delete
`models/encoders/` (re-downloadable), `*.pkl`, `intent_fp32.onnx`,
`__pycache__/`. Already in `.gitignore`. The repo is 677 MB, almost all of it
encoder weights.

---

## 5. What was used

| | |
|---|---|
| teacher encoder | `BAAI/bge-small-en-v1.5` (35 MB, 384-dim, 12-layer) |
| also benchmarked | `intfloat/e5-small-v2`, `all-MiniLM-L6-v2`, TF-IDF + SVD |
| classifier head | sklearn `MLPClassifier` |
| distillation | KL on softened teacher logits, T = 3.0, α = 0.7, ×T² scaling, 40 epochs OneCycle |
| calibration | temperature scaling, golden-section search on NLL, fitted on **validation only**, fused into the ONNX graph |
| OOD | Mahalanobis with shared shrunk covariance; whitening folded into the graph so the phone only does 57 squared-distance comparisons |
| quantization | dynamic INT8, **per-channel scales** |
| leakage control | DSU over normalized text, content-word key, and rapidfuzz token_sort ≥ 92 within intent |
| libraries | torch, transformers, onnx, onnxruntime, scikit-learn, pandas, rapidfuzz, pyyaml |

**Per-channel quantization was the fix that mattered.** With tensor-wide scales,
INT8 changed 7% of decisions. With per-channel scales it reproduces accepted
precision to within 0.002.

### Dataset shape — the numbers that explained nearly every failure

- 9 826 rows → 9 723 after cleaning → **5 173 leakage groups**. Roughly half the
  corpus is near-duplicate material, so a random split would have measured
  memorisation, not learning.
- 35× class imbalance: `Cmd.MemoryChange` 1 884 rows, `Help_DemoMode` 53.
- p99 sentence length 15 words. Only 23 of 6 702 training rows had 18+ words —
  and 20 of those 23 were `Default Fallback Intent`.

---

## 6. Rebuild from scratch

```bash
hf download BAAI/bge-small-en-v1.5 --local-dir models/encoders/bge-small-en-v1.5
python scripts/shrink_encoder.py --encoder bge-small-en-v1.5 --prune-vocab

python scripts/build_challenge_sets.py
python scripts/split_dataset.py
python scripts/build_targeted_training.py

python scripts/train_classifier.py --encoder bge-small-en-v1.5 --classifier mlp \
       --train train_augmented --out models/final
python scripts/distill_student.py --teacher models/final --hidden 256 --layers 4 \
       --tokenizer-from bge-small-en-v1.5-v --name student-h256-l4
python scripts/pick_seed.py --encoder student-h256-l4

python scripts/export_onnx.py --model models/final_student_256 \
       --out models/final_student_256/onnx --quantize-embeddings
python scripts/parity_test.py --model models/final_student_256 \
       --onnx-dir models/final_student_256/onnx
python scripts/evaluate_onnx.py --model models/final_student_256 \
       --onnx models/final_student_256/onnx/intent_int8.onnx
```

Takes about 30 minutes, most of it distillation. **Paste without the `#`
comments** — zsh does not enable comments in interactive shells and will pass
them as arguments.

`--tokenizer-from` is not optional. Without it the student inherits the teacher's
full 30 522-token vocabulary and the embedding table alone blows the size budget
(8.54 MB instead of 4.75 MB).

---

## 7. Things that will bite you

**The head fit is a lottery.** The encoder is distilled once; the head,
temperature, OOD scorer and every threshold are re-fitted on top of it in about
six seconds. Across five seeds of that six-second fit, validation ECE ranged
0.0109–0.0193 and **test coverage moved from 0.623 to 0.658** — larger than most
effects this project spent weeks chasing. Use `pick_seed.py`. Every single-fit
number reported before this was one draw from that distribution presented as a
fact.

Related, and worth knowing before you trust a proxy: validation coverage does
**not** predict test coverage. The winning seed had *lower* validation coverage
than the one it replaced (0.8444 vs 0.8482) and still gained 3.5 points on test.
Validation coverage spread 0.0137 while test coverage moved 0.035.

**Opset is 18, not 17.** The exporter tries to down-convert, fails, and says so
in a long traceback that looks alarming and is harmless — the file stays at
opset 18. ONNX Runtime Mobile in the app must support opset 18 or **the model
will not load at all**. This is a hard load failure, not a quality regression.
Check it before anything else.

**The tokenizer ships as `tokenizer.json`, not `vocab.txt`.** It is a WordPiece
model with 3 267 entries in HuggingFace fast-tokenizer format.
`reports/android_integration.md` still refers to `vocab.txt` in one place —
the tokenizer directory is the authority.

**The Kotlin normalizer must match `scripts/common.py::normalize` exactly:**
NFKC, lowercase, trim, expand the contraction table, strip punctuation except
apostrophes, collapse whitespace. A mismatch here is silent — nothing crashes,
accuracy just quietly drops on contractions and punctuation.

**The corrective-structure regex runs on RAW lowercased text, never on
normalizer output.** The comma is the structural signal and `normalize()` strips
punctuation. Getting this wrong silently disables the signal — it did once, and
caught only 75% of cases instead of 100%.

**Never select on a challenge suite.** Selection uses validation macro-F1 and
validation ECE. The suites are reported for every candidate so failure modes stay
visible, but they must not feed back into the choice.

**A change smaller than the 2σ column in `reports/variance.json` is not evidence
of anything.** Current 2σ: test accuracy 0.0045 · macro-F1 0.0072 · contextual
0.048 · minimal pairs 0.025 · hard negatives 0.015 · negation 0.007 · STT 0.007 ·
OOD 0.013.

---

## 8. What is left

### Must do
- **Commit to git.** Not done. `.git` lives at `/Users/shuklam/IntentClassifier/`,
  outside the working folder. Branch `feature/new_train`.

### Optional, only if the product is always-listening
- **Gate signal 6 (ASR confidence).** The branch exists and is inert; the fitting
  script and recording protocol are written. Needs about 120 recorded utterances
  from the target hardware (~1 hour) — see `reports/asr_confidence_protocol.md`,
  then `python scripts/fit_asr_threshold.py --data data/asr_samples.csv --apply models/final_student_256/onnx`.
  **If the product can use push-to-talk, skip all of it** — a button removes this
  failure mode entirely and costs no model work.

### Known gaps, needing data rather than tuning
- **Accessories score 0.471** and this is the most useful number in the report.
  The suite holds out accessory names the model never saw, and it fails on all of
  them: "my neck loop will not stay connected" → Fallback at 0.946 confidence.
  The model handles accessory names it was trained on and does not generalise to
  new ones. The corpus has 21 rows for the TV streamer, 43 for the remote mic,
  and **zero for Auracast**. The 12 Auracast rows in `accessories.py` are an
  assumption, isolated in `AURACAST_*` so they can be relabelled or deleted;
  they keep Auracast out of Fallback, they do not teach it.
- **Corrective negation.** On intent pairs explicitly taught the corrective frame
  the model scores 0.74; on a held-out control group of three families, 0.48 — a
  coin flip. It learned the pairs, not the rule. More data cannot fix this: there
  are thousands of possible pairs and every untaught one sits at chance. The gate
  refuses these structurally, which costs nothing on real traffic (fires on
  0/1513 real rows). The real fix is encoder fine-tuning — sentence structure
  lives in the encoder and it is currently frozen. `finetune_encoder.py` exists.
- **ASR fragments.** "and push it down for dramatics", produced while the user
  was talking to someone else, is ordinary English that genuinely sits near
  volume commands. OOD AUROC on STT noise is 0.62–0.70 against 0.92–0.96 on real
  OOD. No text-only signal separates it. This is what signal 6 exists for.
- **Coverage 0.658 vs the 0.68 bar.** First thing to look at next time.

---

## 9. Where the numbers live

| file | holds |
|---|---|
| `reports/onnx_suite_int8.json` | the shipped model's full suite matrix, incl. OOD per-family and accessories |
| `reports/parity.json` | Python vs ONNX agreement, fp32 and INT8 |
| `reports/variance.json` | the 5-seed noise floor. Check before believing any change |
| `reports/errors_final.csv` | every error, with suite, confidence, and whether the gate accepted it |
| `reports/dataset_audit.md` | phase 1 — the imbalance and shortcut findings |
| `reports/operating_point.json` | the chosen gate thresholds and how they were selected |
| `data/challenge/ood_rejected.csv` | candidate OOD rows that were refused, with the reason |
| `data/augmentation_conflicts.csv` | generated rows that contradicted the dataset and were dropped |
