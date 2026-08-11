# Semantic Student Model — Principal Architect Documentation

> **Scope**: This document covers the _distilled semantic student_ (Stage 3 of the NLU cascade). It is written to be self-contained: a developer who has never seen the repo should be able to re-train, evaluate, calibrate, export, install, and integrate the model on Python, Android, and iOS from this document alone.

---

## Table of Contents

1. [What Problem This Solves](#1-what-problem-this-solves)
2. [Architecture Overview — The 3-Stage Cascade](#2-architecture-overview--the-3-stage-cascade)
3. [Model Architecture — TinyIntentClassifier](#3-model-architecture--tinyintentclassifier)
4. [Knowledge Distillation Approach](#4-knowledge-distillation-approach)
5. [Tokenizer Contract (CRITICAL)](#5-tokenizer-contract-critical)
6. [Temperature Calibration](#6-temperature-calibration)
7. [Shipped Artifacts — What's in models/semantic_student/en/](#7-shipped-artifacts--whats-in-modelssemantic_studenten)
8. [Full Pipeline: Train → Evaluate → Export → Install](#8-full-pipeline-train--evaluate--export--install)
9. [Test Harness and What Each Test Guards](#9-test-harness-and-what-each-test-guards)
10. [Runtime Integration — Python](#10-runtime-integration--python)
11. [Runtime Integration — Android (Kotlin/Java)](#11-runtime-integration--android-kotlinjava)
12. [Runtime Integration — iOS (Swift)](#12-runtime-integration--ios-swift)
13. [Known Pitfalls and Historical Bugs](#13-known-pitfalls-and-historical-bugs)
14. [Quick Reference Card](#14-quick-reference-card)

---

## 1. What Problem This Solves

The hearing-aid voice assistant needs to classify user utterances into **57 intents** (21 commands like `Cmd.VolumeIncrease`, 34 help topics like `Help_Battery`, plus `Default Fallback Intent` for out-of-domain queries).

The original Stage 3 used a **23 MB MiniLM-L6-v2** sentence-transformer encoder plus a logistic-regression head. That works, but:
- 23 MB is too large for constrained hearing-aid companion apps.
- The two-piece artifact (encoder ONNX + head `.npz`) complicates deployment.

The **Semantic Student** replaces that with a **single 1.2 MB ONNX file** that goes from text → token IDs → logits in one graph, with no external encoder.

---

## 2. Architecture Overview — The 3-Stage Cascade

```
User utterance
      │
      ▼
┌─────────────────────┐
│  Stage 2: TF-IDF    │  (sklearn LogReg, ~models/intent_model.onnx)
│  + LogReg            │
│  confidence >= 0.70  │──── YES ──▶ Return intent. DONE.
│                      │
└────────┬────────────┘
         │ NO (confidence < 0.70)
         ▼
┌─────────────────────┐
│  Stage 3: Student   │  (models/semantic_student/en/student.onnx)
│  Semantic Model      │
│  confidence >= 0.40  │──── YES ──▶ Return rescued intent. DONE.
│                      │
└────────┬────────────┘
         │ NO (confidence < 0.40, OR prediction is "Default Fallback Intent")
         ▼
┌─────────────────────┐
│  GenAI Fallback     │
│  (LLM / scripted)    │
└─────────────────────┘
```

> **IMPORTANT**: The student **never sees easy turns**. Stage 2 already handles ~58.6% of stress-test queries confidently. The student only processes the 41.4% that Stage 2 is unsure about. Always evaluate it on the **handover subset**, not the full test set.

### Key Gate Thresholds

| Gate | Value | Meaning |
|:-----|:------|:--------|
| Stage 2 gate | `0.70` | Stage 2 must be ≥70% confident to answer directly |
| Stage 3 gate | `0.40` | Student must be ≥40% confident to rescue the turn |
| Fallback class | `"Default Fallback Intent"` | If the student's argmax IS this class, route to GenAI regardless of confidence |

---

## 3. Model Architecture — TinyIntentClassifier

Defined in `new_semantic/scripts/train_en.py` (lines 84-110).

```
TinyIntentClassifier
├── Embedding(vocab_size=3000, dim=64, padding_idx=0)
├── TransformerEncoder
│   └── TransformerEncoderLayer × 2
│       ├── MultiHeadAttention(d_model=64, nhead=4)
│       └── FFN(64 → 128 → 64), dropout=0.10
├── LayerNorm(64)
└── Linear(64 → 57)    # 57 intents
```

### Forward Pass

```python
def forward(self, ids, mask):
    x = self.embedding(ids)                          # (1, max_len) → (1, max_len, 64)
    x = self.encoder(x, src_key_padding_mask=~mask)  # transformer encoder
    m = mask.unsqueeze(-1).float()
    pooled = (x * m).sum(1) / m.sum(1).clamp(min=1e-6)  # masked mean pooling
    return self.classifier(self.norm(pooled))         # (1, 57) raw logits
```

### Key Hyperparameters (from `new_semantic/config.py`)

| Parameter | Value | Notes |
|:----------|:------|:------|
| `EMBED_DIM` | 64 | Embedding and transformer hidden size |
| `NHEAD` | 4 | Must divide `EMBED_DIM` evenly |
| `FF_DIM` | 128 | Feed-forward inner dimension |
| `NUM_LAYERS` | 2 | Transformer encoder layers |
| `DROPOUT` | 0.10 | Applied in the transformer |
| `MAX_LEN` | 32 | Sequence length (subword mode) |
| `VOCAB_SIZE` | 3,000 | WordPiece-style subword vocabulary |
| Total params | ~300K | ~1.2 MB fp32 / ~0.6 MB INT8 |

---

## 4. Knowledge Distillation Approach

### Teacher: `intfloat/e5-small-v2`

A sentence-transformer that runs **offline during training only**. It is never shipped to the device.

### Distillation Pipeline

```
                     ┌───────────────────────────────────────┐
                     │  Teacher (E5-small-v2, offline)       │
                     │                                       │
Training CSVs ─────▶│  1. Encode ALL training phrases       │
  (labels)           │  2. Compute class prototypes          │
                     │     (mean of per-class embeddings)    │
                     │  3. Compute soft targets:             │
                     │     softmax(cosine(embed, protos) × 10)│
                     │                                       │
                     └───────────────┬───────────────────────┘
                                     │ soft targets
                                     ▼
                     ┌───────────────────────────────────────┐
                     │  Student (TinyIntentClassifier)       │
                     │                                       │
                     │  Loss = 0.70 × CE(labels)             │
                     │       + 0.30 × KL(student ‖ teacher,  │
                     │                          T=2)         │
                     │                                       │
                     │  • Inverse-frequency class weights    │
                     │  • Stratified train/val split (85/15) │
                     │  • Early stopping on macro-recall     │
                     │  • AdamW, lr=2e-3, weight_decay=1e-4  │
                     └───────────────────────────────────────┘
```

### Loss Function Breakdown

```python
# Hard label loss (weighted cross-entropy)
loss_ce = CrossEntropyLoss(weight=class_weights)(student_logits, true_labels)

# Soft label loss (KL divergence with temperature)
loss_kd = KL_div(
    log_softmax(student_logits / T, dim=-1),
    softmax(teacher_logits / T, dim=-1),
    reduction="batchmean"
) * (T * T)

# Combined loss
total_loss = 0.70 * loss_ce + 0.30 * loss_kd
```

| Component | Weight | Purpose |
|:----------|:-------|:--------|
| `CE_WEIGHT` | 0.70 | Weighted cross-entropy against ground-truth labels |
| `KD_WEIGHT` | 0.30 | KL divergence against teacher soft targets |
| `TEMPERATURE` | 2.0 | Distillation temperature (softens teacher's distribution) |

### Why Macro-Recall for Early Stopping

The dataset is heavily imbalanced (55× between the largest and smallest class). Accuracy hides this: a model can ignore 23 small classes and still hit 95%+ accuracy. **Macro-recall** gives each class equal weight, so the checkpoint selected is the one that serves ALL intents, not just the popular ones.

### Data Augmentation

The training script includes two augmentation mechanisms (controlled via CLI flags):

| Mechanism | Flag | What It Does |
|:----------|:-----|:-------------|
| UNK Augmentation | `--unk-aug 0.3` | Corrupts 70-100% of tokens in in-scope phrases → labels them as `Default Fallback Intent`. Teaches "mostly-unknown = junk". |
| UNK Robustness | `--unk-robust 0.3` | Corrupts 1-2 tokens in in-scope phrases → keeps TRUE label. Counter-example so the model doesn't learn "any UNK = junk". |
| Fallback Oversample | `--fallback-oversample 3` | Repeats real fallback rows N times (no synthetic text). |

> **WARNING**: `--unk-aug` and `--unk-robust` must be used **together** in roughly equal proportion. Using `--unk-aug` alone causes the model to learn "any UNK → fallback" and reject real commands that contain one unfamiliar word.

---

## 5. Tokenizer Contract (CRITICAL)

> **CAUTION**: The tokenizer is the model's **input contract**. If a single character of the regex changes, every token ID shifts silently, and the model gives confident wrong answers. Nothing crashes.

### The Regex

```python
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
```

This regex is defined identically in **three places** that must stay byte-identical:

| File | Purpose |
|:-----|:--------|
| `new_semantic/scripts/common.py` (line 22) | Training-time tokenizer |
| `packages/runtime/nlu_engine/semantic.py` (line 80) | Runtime `StudentSemantic._TOKEN_RE` |
| `packages/runtime/nlu_engine/test_student_semantic.py` (line 63) | Test that verifies they match |

### Pre-processing Steps (MUST be identical on every platform)

```python
# 1. Unicode normalize (NFKD)
text = unicodedata.normalize("NFKD", str(text))

# 2. Curly apostrophe → straight apostrophe
text = text.replace("\u2019", "'")    # ' → '

# 3. Lowercase
text = text.lower()

# 4. Extract tokens with _TOKEN_RE
tokens = _TOKEN_RE.findall(text)      # punctuation is discarded

# 5. Result: "volume up" == "volume up?!" (same tokens)
```

### Two Tokenizer Modes

| Mode | Behaviour | UNK Handling |
|:-----|:----------|:-------------|
| `word` | One ID per whole word | Unseen word → `[UNK]` (ID=1). Destroys all info. |
| `subword` ✅ | WordPiece greedy longest-match | `"quieter"` → `["quiet", "##er"]`. No word is ever UNK. |

**The shipped model uses `subword`** (mode recorded in `meta.json`). This was the key breakthrough: the word-level model couldn't distinguish `"quieter"` (real command, unseen word) from `"asdfghjkl"` (junk), because both were literally `[UNK]`.

### SubWord Vocabulary Construction (from `new_semantic/scripts/common.py`)

1. Start with `[PAD]` (0) and `[UNK]` (1).
2. Add **every character** seen in the corpus + its `##` prefixed form → guarantees zero UNK.
3. Score candidate pieces by `frequency × length` (prefer long, frequent pieces).
4. Greedily add top-scoring pieces until the vocab reaches 3,000 tokens.

### Encoding (ID Assignment)

```python
def encode(text, vocab, max_len=32, mode="subword"):
    tokens = tokenize(text)                                    # regex → word list
    pieces = [p for w in tokens for p in wordpiece(w, vocab)]  # split each word
    ids = [vocab.get(p, UNK_ID) for p in pieces][:max_len]     # to IDs, truncate
    ids += [PAD_ID] * (max_len - len(ids))                     # pad to static length
    return ids
```

### WordPiece Algorithm (greedy longest-match)

```python
def wordpiece(word, vocab):
    out, start = [], 0
    while start < len(word):
        end = len(word)
        cur = None
        while start < end:
            piece = word[start:end] if start == 0 else "##" + word[start:end]
            if piece in vocab:
                cur = piece
                break
            end -= 1
        if cur is None:
            out.append("[UNK]")
            start += 1
        else:
            out.append(cur)
            start = end
    return out
```

**Example**: `"quieter"` with subword vocab:
```
quieter → ["quiet", "##er"]     (2 pieces, both meaningful)
```

**Example**: `"asdfghjkl"` with subword vocab:
```
asdfghjkl → ["a", "##s", "##d", "##f", "##g", "##h", "##j", "##k", "##l"]
            (9 single-character pieces — the shattering IS the signal)
```

> **FOR ANDROID/iOS**: You must re-implement this exact tokenizer. The regex, Unicode normalization, apostrophe folding, and WordPiece algorithm must be **byte-identical** to the Python version. Test with the golden phrases in `test_student_semantic.py`.

---

## 6. Temperature Calibration

### The Problem

The raw student reported ECE (Expected Calibration Error) of 0.2029: when it said "90% confident", it was right only ~70% of the time. The gate at 0.40 reads that miscalibrated number, rejecting 27.4% of correct answers.

### The Fix

A single scalar **T = 0.68** (fitted by `new_semantic/scripts/calibrate.py`):

```python
# In classify():
logits = onnx_session.run(...)
z = logits / temperature         # T = 0.68 (sharpens)
z = z - z.max()
p = np.exp(z) / np.exp(z).sum()  # softmax
```

### Properties of Temperature Scaling

| Property | Explanation |
|:---------|:------------|
| **Rank-preserving** | T can **never** change which intent wins (argmax is invariant). Only the confidence value changes. |
| **ECE improvement** | 0.2029 → 0.0187 |
| **Gate accuracy** | 0.8374 → 0.8835 at the 0.40 gate (purely from better-shaped confidence) |
| **T < 1 sharpens** | Makes the model sound _more_ confident (which is correct here, because it was under-confident) |

### How Temperature is Fitted

`calibrate.py` performs a grid search over T values (0.05 to 5.0) minimizing negative log-likelihood (NLL) on a held-out dev set. It then reports ECE and gate accuracy on a separate test set to confirm improvement generalizes.

### Where T Lives in the Codebase

| Location | Role |
|:---------|:-----|
| `scripts/calibrate.py --tag ... --apply` | Fits T and writes it |
| `models/semantic_student/en/meta.json` → `"temperature": 0.68` | Stored value |
| `semantic.py` line 137 → `self.temperature` | Read at construction |
| `semantic.py` line 183 → `z = logits / self.temperature` | Applied at inference |

> **WARNING — FOR ANDROID/iOS**: You MUST apply `logits / T` **before** softmax. If you skip this (T=1.0), the accuracy doesn't change but the confidence gate becomes miscalibrated, silently rejecting ~27% of correct answers.

---

## 7. Shipped Artifacts — What's in `models/semantic_student/en/`

```
models/semantic_student/en/
├── student.onnx    (1.2 MB)  — the full model graph + weights, self-contained
├── vocab.json      (57 KB)   — {"mode": "subword", "vocab": {"[PAD]": 0, ...}}
├── labels.json     (1.3 KB)  — ["Cmd.ActivityAerobics", ..., "reminders.complete"]
└── meta.json       (0.4 KB)  — runtime configuration
```

### meta.json (current values)

```json
{
  "tag": "subw_vol5_s1",
  "max_len": 32,
  "threshold": 0.40,
  "tokenizer": "subword",
  "temperature": 0.68,
  "vocab_size": 3000,
  "seed": 1,
  "teacher": "intfloat/e5-small-v2",
  "init_embeddings": null,
  "freeze_embeddings": false,
  "synthetic_rows": 929,
  "synthetic_text": true,
  "source": "..."
}
```

### ONNX Model I/O Contract

| | Name | Shape | Type | Notes |
|:--|:-----|:------|:-----|:------|
| **Input 1** | `input_ids` | `[1, 32]` | `int64` | Token IDs (padded to max_len=32) |
| **Input 2** | `attention_mask` | `[1, 32]` | `bool` | `true` for real tokens, `false` for padding |
| **Output** | `logits` | `[1, 57]` | `float32` | Raw logits. Apply `÷T` then softmax yourself. |

> **CRITICAL — Static shape only.** The graph is exported at `[1, 32]` with **no dynamic axes**. The Apple Neural Engine (ANE) on iOS does not support dynamic sequence dimensions. Passing the wrong shape is a runtime crash, not a soft failure.

> **CRITICAL — No sidecar files.** If `student.onnx.data` exists alongside the ONNX, it means the weights are in a separate file and the `.onnx` is just a graph shell with no weights. The model will run but give random answers. The install script and test suite both check for this.

### labels.json (all 57 intents)

```
Cmd.ActivityAerobics, Cmd.ActivityCalories, Cmd.ActivityCycle,
Cmd.ActivityExercise, Cmd.ActivityRun, Cmd.ActivityStand,
Cmd.ActivityStep, Cmd.ActivityWalk, Cmd.BatteryLevel,
Cmd.FindMyPhone, Cmd.ListenMessage, Cmd.MemoryChange,
Cmd.SendMessage, Cmd.StreamingStart, Cmd.StreamingStop,
Cmd.TranscribeStart, Cmd.TranslationStart, Cmd.VolumeDecrease,
Cmd.VolumeIncrease, Cmd.VolumeMute, Cmd.VolumeUnmute,
Default Fallback Intent,
Help_Accessories, Help_AppSettings, Help_Battery,
Help_ChangingMemories, Help_CleanCare, Help_Customize,
Help_DemoMode, Help_DeviceSettings, Help_EdgeMode,
Help_FallAlert, Help_FindMyHearingAids, Help_Health,
Help_HearShare, Help_HearingCareAnywhereConnect,
Help_HeartRate, Help_HeartRateRecovery, Help_Home,
Help_InsertDevice, Help_IntelliVoice, Help_MaskMode,
Help_MemoryOptions, Help_Pairing, Help_Reminder,
Help_RemoteProgramming, Help_SelfCheck, Help_ThriveScore,
Help_Tinnitus, Help_Transcribe, Help_Translate,
Help_VoiceAssistant, Help_Volume, Help_WhatsNew,
Help_WiCROS,
reminders.add, reminders.complete
```

The order here **is** the logit column order. `labels[i]` corresponds to `logits[0][i]`.

---

## 8. Full Pipeline: Train → Evaluate → Export → Install

### Step 0: Prepare Data

```bash
cd new_semantic

# Merge training CSVs (if new data was added)
python scripts/build_merged_train.py

# Expand OOD test set (if needed)
python scripts/expand_ood_testset.py
```

### Step 1: Train

```bash
python scripts/train_en.py \
    --tag subw_vol5_s1 \
    --tokenizer subword \
    --vocab-size 3000 \
    --max-len 32 \
    --unk-aug 0.3 \
    --unk-robust 0.3 \
    --seed 1
```

**Outputs:**
- `models/en/student_subw_vol5_s1.pt` — PyTorch checkpoint
- `models/en/vocab_subw_vol5_s1.json` — vocabulary
- `models/en/labels_subw_vol5_s1.json` — label list
- `reports/train_subw_vol5_s1_summary.json` — training summary

### Step 2: Evaluate (standalone)

```bash
python scripts/evaluate.py --tag subw_vol5_s1 --threshold 0.40
```

Reports locked-test accuracy, macro-recall, stress accuracy, OOD fallback rate, and checks against the ship bar defined in `config.py`:

```python
SHIP_BAR = {
    "locked_accuracy_min_delta": -0.01,    # vs previous best, absolute
    "ood_fallback_rate_min_delta": 0.0,    # must NOT get worse
    "stress_accuracy_min_delta": -0.01,
    "onnx_parity_max_mismatches": 0,
    "int8_parity_max_argmax_flips": 0,
}
```

### Step 3: Calibrate Temperature

```bash
python scripts/calibrate.py --tag subw_vol5_s1 --apply
```

Fits T on dev data, writes it into the installed `meta.json`. Then re-pick the gate threshold on the new scale:

```bash
python scripts/select_policy.py --tags subw_vol5_s1 --reveal-test
```

### Step 4: Export to ONNX

```bash
python scripts/export_onnx.py \
    --tag subw_vol5_s1 \
    --threshold 0.40
```

**What this does:**
1. Exports the PyTorch model to `student_subw_vol5_s1.onnx` with static shape `[1, 32]`.
2. Folds any sidecar `.data` file back into the ONNX (self-contained).
3. Optionally quantizes to INT8 (`--skip-int8` to skip).
4. Runs **parity verification** against all test sets: FP32 ONNX must produce **0 argmax mismatches** and **0 gate disagreements** vs. the PyTorch model. If parity fails, the script refuses to write the artifact.

### Step 5: Install into Runtime

```bash
python scripts/install_student.py \
    --tag subw_vol5_s1 \
    --threshold 0.40 \
    --temperature 0.68
```

**What this does:**
1. Copies `student.onnx`, `vocab.json`, `labels.json`, `meta.json` into `models/semantic_student/en/`.
2. Checks that no `.onnx.data` sidecar exists (would mean broken weights).
3. Loads the installed copy through `nlu_engine.semantic.StudentSemantic` — the engine's own runtime class.
4. Runs every test utterance through both the source and installed models and verifies **0 label mismatches**.
5. If the installed copy disagrees with the source, the script **aborts** rather than shipping a broken artifact.

### Step 6: Evaluate the Cascade

```bash
python scripts/evaluate_cascade.py
```

This uses the **real Stage 2** artifact (`models/intent_model.onnx`) to simulate the production flow. Reports:
- Stage 2 handover rate (how often it defers to Stage 3)
- Stage 2 accuracy on what it keeps
- Student accuracy on the handover subset only (the only number that matters)
- End-to-end pipeline accuracy with and without the student
- OOD rejection rates at each stage

### Step 7: Enable in Production

Edit `language_packs/en/nlu_schema.json`:

```json
"semantic_rescue_enabled": true
```

> **IMPORTANT**: Do this AFTER `evaluate_cascade.py` confirms the student is safe, not before. It's a behaviour change that should be a deliberate, separate commit.

### Step 8: Run All Tests

```bash
pytest packages/runtime/nlu_engine/test_student_semantic.py -v
```

All 29 tests must pass.

---

## 9. Test Harness and What Each Test Guards

Every test in `test_student_semantic.py` exists because a specific failure mode was discovered (or nearly shipped):

### Tokenizer Tests

| Test | Guards Against |
|:-----|:---------------|
| `test_tokenizer_regex_matches_training` | Regex drift between training and runtime. A 1-char change silently shifts every ID. |
| `test_punctuation_is_discarded` | `"volume up"` and `"volume up?!"` must produce identical IDs. ASR output often has trailing punctuation. |
| `test_curly_apostrophe_normalises` | Keyboards emit `'` (U+2019); the model was trained on `'` (U+0027). Without normalization, `"it's"` becomes `[UNK]`. |
| `test_unknown_words_follow_the_installed_tokenizer_contract` | Subword mode must NOT produce `[UNK]` for unknown words (it shatters them into character pieces). |
| `test_subword_encoding_matches_the_training_contract` | IDs from the runtime tokenizer must be byte-identical to the training tokenizer for the same input. |
| `test_padding_and_truncation_hold_the_static_shape` | Shape must always be `[1, 32]`. A wrong length is a runtime crash on device. |

### Contract Tests

| Test | Guards Against |
|:-----|:---------------|
| `test_classify_returns_a_known_label_and_a_probability` | Basic sanity: output is a valid label and a probability in [0, 1]. |
| `test_label_count_matches_the_logit_width` | `labels.json` having a different count than the model's output dimension. Every prediction would be mislabelled. Nothing raises. |
| `test_fallback_intent_is_in_the_label_space` | The engine routes on the exact string `"Default Fallback Intent"`. If it's not a label, rejection breaks silently. |
| `test_it_is_deterministic` | Same input must always produce the same output. Non-determinism would make debugging impossible. |

### Calibration Tests

| Test | Guards Against |
|:-----|:---------------|
| `test_temperature_is_read_from_meta` | `meta.json` carried `temperature: 0.68` for a while before any code actually read it. The value existing is not the same as it being applied. |
| `test_temperature_actually_changes_the_confidence` | Reading T into an attribute and then never dividing by it. Every accuracy test still passes (T is rank-preserving), but the gate reads a miscalibrated scale. |
| `test_temperature_is_rank_preserving` | T changing WHICH intent wins (would mean softmax is applied in the wrong place). |
| `test_a_non_positive_temperature_is_refused` | T ≤ 0 → divide-by-zero or inverted distribution. Must fail at load, not serve nonsense. |

### Artifact Tests

| Test | Guards Against |
|:-----|:---------------|
| `test_installed_artifact_is_self_contained` | A `.onnx.data` sidecar sitting next to the graph (weights are missing, model gives random answers). **This nearly shipped.** |
| `test_meta_records_what_was_installed` | Meta must have tokenizer, max_len, threshold, temperature, and synthetic_text recorded. |
| `test_missing_artifacts_raise_rather_than_serve_nothing` | Missing files must raise `FileNotFoundError`, not silently return wrong answers. |
| `test_a_sidecar_is_refused` | Explicitly creating a sidecar file must be rejected at load time. |

### Engine Wiring Test

| Test | Guards Against |
|:-----|:---------------|
| `test_engine_prefers_the_student_over_minilm` | `_load_semantic` silently keeps using the 23 MB MiniLM stage when a student is installed. Every benchmark "with the new model" would actually be the old one. |

---

## 10. Runtime Integration — Python

The Python runtime is already implemented in `packages/runtime/nlu_engine/semantic.py`. Here's the exact flow for reference:

```python
import json
import re
import unicodedata
import numpy as np
import onnxruntime as ort

# 1. Load artifacts
model_dir = "models/semantic_student/en"
vocab_raw = json.load(open(f"{model_dir}/vocab.json"))
vocab = vocab_raw["vocab"]
labels = json.load(open(f"{model_dir}/labels.json"))
meta = json.load(open(f"{model_dir}/meta.json"))
max_len = meta["max_len"]          # 32
temperature = meta["temperature"]  # 0.68
threshold = meta["threshold"]      # 0.40

# 2. Create ONNX session
sess = ort.InferenceSession(f"{model_dir}/student.onnx",
                            providers=["CPUExecutionProvider"])

# 3. Tokenize (MUST match training — see Section 5)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
PAD_ID, UNK_ID = 0, 1

def wordpiece(word, vocab):
    out, start = [], 0
    while start < len(word):
        end = len(word)
        cur = None
        while start < end:
            piece = word[start:end] if start == 0 else "##" + word[start:end]
            if piece in vocab:
                cur = piece
                break
            end -= 1
        if cur is None:
            out.append("[UNK]")
            start += 1
        else:
            out.append(cur)
            start = end
    return out

def encode(text):
    t = unicodedata.normalize("NFKD", str(text)).replace("\u2019", "'")
    words = TOKEN_RE.findall(t.lower())
    pieces = [p for w in words for p in wordpiece(w, vocab)]
    ids = [vocab.get(p, UNK_ID) for p in pieces][:max_len]
    ids += [PAD_ID] * (max_len - len(ids))
    return np.array([ids], dtype=np.int64)

# 4. Classify
def classify(text):
    ids = encode(text)
    mask = ids != PAD_ID
    logits = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0][0]

    # Temperature scaling (MUST be applied before softmax)
    z = logits / temperature
    z = z - z.max()
    p = np.exp(z)
    p /= p.sum()

    top_idx = int(np.argmax(p))
    intent = labels[top_idx]
    confidence = float(p[top_idx])

    if confidence < threshold or intent == "Default Fallback Intent":
        return "Default Fallback Intent", confidence  # → route to GenAI
    return intent, confidence
```

---

## 11. Runtime Integration — Android (Kotlin/Java)

### Dependencies (build.gradle)

```kotlin
dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.16.3")
    implementation("com.google.code.gson:gson:2.10.1")
}
```

### Assets Required

Copy from `models/semantic_student/en/` into `app/src/main/assets/semantic_student/`:
```
assets/semantic_student/
├── student.onnx
├── labels.json
├── vocab.json
└── meta.json
```

### Full Kotlin Implementation

```kotlin
import ai.onnxruntime.*
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.text.Normalizer

class SemanticStudent(context: Context) {

    // ----- Constants (must match Python exactly) -----
    private val PAD_ID = 0L
    private val UNK_ID = 1L
    private val TOKEN_REGEX = Regex("[a-z0-9]+(?:'[a-z0-9]+)?")
    private val FALLBACK = "Default Fallback Intent"

    // ----- Loaded artifacts -----
    private val vocab: Map<String, Int>
    private val labels: List<String>
    private val maxLen: Int
    private val temperature: Float
    private val threshold: Float
    private val session: OrtSession

    init {
        val gson = Gson()
        val am = context.assets

        // vocab.json → {"mode": "subword", "vocab": {"[PAD]": 0, ...}}
        val vocabRaw = am.open("semantic_student/vocab.json")
            .bufferedReader().use { it.readText() }
        val vocabObj = gson.fromJson(vocabRaw,
            object : TypeToken<Map<String, Any>>() {}.type) as Map<String, Any>
        @Suppress("UNCHECKED_CAST")
        val vocabMap = vocabObj["vocab"] as Map<String, Double>
        vocab = vocabMap.mapValues { it.value.toInt() }

        // labels.json → ["Cmd.ActivityAerobics", ...]
        val labelsRaw = am.open("semantic_student/labels.json")
            .bufferedReader().use { it.readText() }
        labels = gson.fromJson(labelsRaw,
            object : TypeToken<List<String>>() {}.type)

        // meta.json → {"max_len": 32, "temperature": 0.68, ...}
        val metaRaw = am.open("semantic_student/meta.json")
            .bufferedReader().use { it.readText() }
        val meta = gson.fromJson(metaRaw,
            object : TypeToken<Map<String, Any>>() {}.type) as Map<String, Any>
        maxLen = (meta["max_len"] as Double).toInt()
        temperature = (meta["temperature"] as Double).toFloat()
        threshold = (meta["threshold"] as Double).toFloat()

        // ONNX session
        val env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions()
        val modelBytes = am.open("semantic_student/student.onnx").readBytes()
        session = env.createSession(modelBytes, opts)

        // Warm up (move ORT graph setup off the first real turn)
        classify("warm up")
    }

    // ----- WordPiece tokenizer (MUST match Python exactly) -----

    private fun wordpiece(word: String): List<String> {
        val out = mutableListOf<String>()
        var start = 0
        while (start < word.length) {
            var end = word.length
            var cur: String? = null
            while (start < end) {
                val piece = if (start == 0) word.substring(start, end)
                            else "##" + word.substring(start, end)
                if (vocab.containsKey(piece)) {
                    cur = piece
                    break
                }
                end--
            }
            if (cur == null) {
                out.add("[UNK]")
                start++
            } else {
                out.add(cur)
                start = end
            }
        }
        return out
    }

    private fun encode(text: String): Pair<LongArray, BooleanArray> {
        // 1. NFKD normalize + apostrophe fold + lowercase
        val normalized = Normalizer.normalize(text, Normalizer.Form.NFKD)
            .replace('\u2019', '\'')
            .lowercase()

        // 2. Extract tokens with the regex
        val words = TOKEN_REGEX.findAll(normalized).map { it.value }.toList()

        // 3. WordPiece each word
        val pieces = words.flatMap { wordpiece(it) }

        // 4. To IDs, truncate to max_len
        val ids = pieces
            .take(maxLen)
            .map { (vocab[it] ?: UNK_ID.toInt()).toLong() }
            .toLongArray()

        // 5. Pad to max_len (MUST be exactly maxLen for the static ONNX shape)
        val padded = LongArray(maxLen) { if (it < ids.size) ids[it] else PAD_ID }
        val mask = BooleanArray(maxLen) { padded[it] != PAD_ID }

        return Pair(padded, mask)
    }

    // ----- Public API -----

    data class Result(val intent: String, val confidence: Float)

    fun classify(text: String): Result {
        val (ids, mask) = encode(text)
        val env = OrtEnvironment.getEnvironment()

        val idsTensor = OnnxTensor.createTensor(env,
            arrayOf(ids), longArrayOf(1, maxLen.toLong()))
        val maskTensor = OnnxTensor.createTensor(env,
            arrayOf(mask), longArrayOf(1, maxLen.toLong()))

        val output = session.run(
            mapOf("input_ids" to idsTensor, "attention_mask" to maskTensor))
        val logits = (output[0].value as Array<FloatArray>)[0]

        // Temperature scaling (MUST happen before softmax)
        val z = FloatArray(logits.size) { logits[it] / temperature }
        val maxZ = z.max()!!
        for (i in z.indices) z[i] -= maxZ
        val exp = FloatArray(z.size) { kotlin.math.exp(z[it]) }
        val sumExp = exp.sum()
        val probs = FloatArray(exp.size) { exp[it] / sumExp }

        val topIdx = probs.indices.maxByOrNull { probs[it] }!!
        val intent = labels[topIdx]
        val confidence = probs[topIdx]

        return if (confidence < threshold || intent == FALLBACK) {
            Result(FALLBACK, confidence)
        } else {
            Result(intent, confidence)
        }
    }
}
```

### Usage in Activity

```kotlin
class MainActivity : AppCompatActivity() {
    private lateinit var student: SemanticStudent

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        student = SemanticStudent(this)

        val result = student.classify("make it louder")
        Log.d("NLU", "Intent: ${result.intent}, Confidence: ${result.confidence}")
    }
}
```

---

## 12. Runtime Integration — iOS (Swift)

### Dependencies (via SPM)

```swift
// Package.swift or Xcode → Add Package Dependency
.package(url: "https://github.com/nicklama/onnxruntime-swift-package-manager",
         from: "1.16.0")
```

Or via CocoaPods:
```ruby
pod 'onnxruntime-objc', '~> 1.16'
```

### Assets Required

Add the same 4 files from `models/semantic_student/en/` into your app bundle (drag into Xcode project, check "Copy items if needed").

### Full Swift Implementation

```swift
import Foundation
import OnnxRuntimeBindings  // or onnxruntime_objc

class SemanticStudent {

    private let vocab: [String: Int]
    private let labels: [String]
    private let maxLen: Int
    private let temperature: Float
    private let threshold: Float
    private let session: ORTSession

    private let PAD_ID: Int64 = 0
    private let UNK_ID: Int64 = 1
    private let FALLBACK = "Default Fallback Intent"

    // The regex — MUST match Python: r"[a-z0-9]+(?:'[a-z0-9]+)?"
    private let tokenRegex = try! NSRegularExpression(
        pattern: "[a-z0-9]+(?:'[a-z0-9]+)?", options: [])

    struct Result {
        let intent: String
        let confidence: Float
    }

    init(bundlePath: String) throws {
        // Load vocab.json
        let vocabData = try Data(contentsOf:
            URL(fileURLWithPath: "\(bundlePath)/vocab.json"))
        let vocabObj = try JSONSerialization.jsonObject(with: vocabData)
            as! [String: Any]
        let vocabDict = vocabObj["vocab"] as! [String: NSNumber]
        self.vocab = vocabDict.mapValues { $0.intValue }

        // Load labels.json
        let labelsData = try Data(contentsOf:
            URL(fileURLWithPath: "\(bundlePath)/labels.json"))
        self.labels = try JSONDecoder().decode([String].self, from: labelsData)

        // Load meta.json
        let metaData = try Data(contentsOf:
            URL(fileURLWithPath: "\(bundlePath)/meta.json"))
        let meta = try JSONSerialization.jsonObject(with: metaData)
            as! [String: Any]
        self.maxLen = (meta["max_len"] as! NSNumber).intValue
        self.temperature = (meta["temperature"] as! NSNumber).floatValue
        self.threshold = (meta["threshold"] as! NSNumber).floatValue

        // Create ONNX session
        let env = try ORTEnv(loggingLevel: .warning)
        let opts = try ORTSessionOptions()
        self.session = try ORTSession(
            env: env,
            modelPath: "\(bundlePath)/student.onnx",
            sessionOptions: opts)

        // Warm up
        _ = try? classify("warm up")
    }

    // ----- WordPiece tokenizer (MUST match Python exactly) -----

    private func wordpiece(_ word: String) -> [String] {
        var out: [String] = []
        var start = word.startIndex
        while start < word.endIndex {
            var end = word.endIndex
            var cur: String? = nil
            while start < end {
                let sub = String(word[start..<end])
                let piece = (start == word.startIndex) ? sub : "##\(sub)"
                if vocab[piece] != nil {
                    cur = piece
                    break
                }
                end = word.index(before: end)
            }
            if let found = cur {
                out.append(found)
                start = end
            } else {
                out.append("[UNK]")
                start = word.index(after: start)
            }
        }
        return out
    }

    private func encode(_ text: String) -> (ids: [Int64], mask: [Int64]) {
        // 1. NFKD normalize + apostrophe fold + lowercase
        let normalized = text
            .decomposedStringWithCompatibilityMapping
            .replacingOccurrences(of: "\u{2019}", with: "'")
            .lowercased()

        // 2. Extract tokens with the regex
        let range = NSRange(normalized.startIndex..., in: normalized)
        let matches = tokenRegex.matches(in: normalized, range: range)
        let words = matches.map {
            String(normalized[Range($0.range, in: normalized)!])
        }

        // 3. WordPiece each word
        let pieces = words.flatMap { wordpiece($0) }

        // 4. To IDs, truncate to max_len
        var ids = pieces.prefix(maxLen).map {
            Int64(vocab[$0] ?? Int(UNK_ID))
        }

        // 5. Pad to exactly max_len
        while ids.count < maxLen {
            ids.append(PAD_ID)
        }
        let mask = ids.map { $0 != PAD_ID ? Int64(1) : Int64(0) }
        return (ids, mask)
    }

    // ----- Public API -----

    func classify(_ text: String) throws -> Result {
        let (ids, mask) = encode(text)

        // Create input tensors — shape [1, maxLen]
        let idsData = NSMutableData(
            bytes: ids, length: ids.count * MemoryLayout<Int64>.size)
        let idsTensor = try ORTValue(
            tensorData: idsData,
            elementType: .int64,
            shape: [1, NSNumber(value: maxLen)])

        let maskData = NSMutableData(
            bytes: mask, length: mask.count * MemoryLayout<Int64>.size)
        let maskTensor = try ORTValue(
            tensorData: maskData,
            elementType: .int64,
            shape: [1, NSNumber(value: maxLen)])

        // Run inference
        let output = try session.run(
            withInputs: [
                "input_ids": idsTensor,
                "attention_mask": maskTensor
            ],
            outputNames: ["logits"],
            runOptions: nil)

        guard let logitsValue = output["logits"] else {
            throw NSError(domain: "SemanticStudent", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "No logits output"])
        }

        // Extract logits array
        let logitsData = try logitsValue.tensorData() as Data
        let logits: [Float] = logitsData.withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }

        // Temperature scaling (MUST be before softmax)
        var z = logits.map { $0 / temperature }
        let maxZ = z.max()!
        z = z.map { $0 - maxZ }
        let expZ = z.map { exp($0) }
        let sumExp = expZ.reduce(0, +)
        let probs = expZ.map { $0 / sumExp }

        let topIdx = probs.enumerated()
            .max(by: { $0.element < $1.element })!.offset
        let intent = labels[topIdx]
        let confidence = probs[topIdx]

        if confidence < threshold || intent == FALLBACK {
            return Result(intent: FALLBACK, confidence: confidence)
        }
        return Result(intent: intent, confidence: confidence)
    }
}
```

### Usage in SwiftUI / UIKit

```swift
// In your ViewModel or wherever NLU is invoked
let bundlePath = Bundle.main.path(forResource: "semantic_student",
                                   ofType: nil)!
let student = try SemanticStudent(bundlePath: bundlePath)

let result = try student.classify("turn up the volume")
print("Intent: \(result.intent), Confidence: \(result.confidence)")
```

> **ANE Compatibility**: The model is exported with static shapes and no dynamic axes specifically for ANE. Do NOT modify the ONNX graph to add dynamic axes.

---

## 13. Known Pitfalls and Historical Bugs

These are real bugs that either shipped or nearly shipped. They are the reason each test exists.

| # | Bug | Impact | How We Guard |
|:--|:----|:-------|:-------------|
| 1 | **Sidecar weights** | `torch.onnx.export` wrote weights to `student.onnx.data`. The shipped `.onnx` was graph-only (0.166 MB). Model ran but gave random answers. | `test_installed_artifact_is_self_contained`, `export_onnx.py` folds sidecar back, `StudentSemantic.__init__` refuses sidecar. |
| 2 | **Temperature present but not applied** | `meta.json` had `temperature: 0.68` for weeks. No code read it. All accuracy tests passed (T is rank-preserving). The gate read raw, miscalibrated confidence → rejected 27% of correct answers. | `test_temperature_actually_changes_the_confidence` |
| 3 | **Word-level UNK collapse** | `"quieter"` and `"asdfghjkl"` both became `[UNK]`. The model couldn't distinguish real commands with unseen words from junk. | Switched to subword tokenizer. `test_unknown_words_follow_the_installed_tokenizer_contract` |
| 4 | **INT8 size misreporting** | INT8 was declared "2x larger than FP32" because fp32 size was the graph-only file (0.166 MB) while INT8 was self-contained. Real ratio: INT8 is ~2.7x smaller. | `export_onnx.py` now reports self-contained sizes only. |
| 5 | **Engine loading wrong backend** | `_load_semantic` silently kept using the 23 MB MiniLM stage when a student was installed. Every benchmark "with the new model" was actually the old one. | `test_engine_prefers_the_student_over_minilm` |
| 6 | **Stage 2 "probabilities" are not probabilities** | The Stage 2 ONNX output named `"probabilities"` is actually `sklearn.decision_function` (raw scores, sum to ~0). Taking `.max()` makes everything look confident → handover rate drops to 0%. | `evaluate_cascade.py` applies softmax explicitly. |

---

## 14. Quick Reference Card

### Re-train with new data (complete sequence)

```bash
cd new_semantic

# 1. Prepare data
python scripts/build_merged_train.py
python scripts/expand_ood_testset.py

# 2. Train
python scripts/train_en.py \
    --tag my_new_tag \
    --tokenizer subword \
    --vocab-size 3000 \
    --max-len 32 \
    --unk-aug 0.3 \
    --unk-robust 0.3 \
    --seed 1

# 3. Evaluate standalone
python scripts/evaluate.py --tag my_new_tag --threshold 0.40

# 4. Calibrate temperature
python scripts/calibrate.py --tag my_new_tag --apply

# 5. Export to ONNX
python scripts/export_onnx.py --tag my_new_tag --threshold 0.40

# 6. Install into runtime location
python scripts/install_student.py \
    --tag my_new_tag \
    --threshold 0.40 \
    --temperature 0.68

# 7. Evaluate as a cascade (with real Stage 2)
python scripts/evaluate_cascade.py

# 8. Run all tests
pytest packages/runtime/nlu_engine/test_student_semantic.py -v

# 9. Enable in production (manual edit)
# language_packs/en/nlu_schema.json → "semantic_rescue_enabled": true
```

### Ship to mobile (checklist)

1. Copy `models/semantic_student/en/` → app assets
2. Verify `student.onnx.data` does NOT exist (weights must be inside the `.onnx`)
3. Read `meta.json` for `max_len`, `temperature`, `threshold`
4. Implement the tokenizer: NFKD normalize → fold `'` → lowercase → regex `[a-z0-9]+(?:'[a-z0-9]+)?` → WordPiece
5. Load `student.onnx` with ONNXRuntime
6. On each turn: `encode → session.run → logits / T → softmax → argmax`
7. If `confidence < threshold` OR `intent == "Default Fallback Intent"` → fallback to GenAI

### Verify on mobile (golden test phrases)

```
"turn it up"      == "turn it up?!"         (punctuation discarded)
"it's too quiet"  == "it\u2019s too quiet"  (apostrophe normalized)
"mute"            → shape must be [1, 32]   (padded)
"volume " × 200   → shape must be [1, 32]   (truncated, not overflow)
```

---

> **Document version**: 2026-08-11 | **Model tag**: `subw_vol5_s1` | **Author**: Principal Architect Review
