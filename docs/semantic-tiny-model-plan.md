# Tiny Semantic Model — Full Training Plan (English + Multilingual)

**Status:** Working plan / step-by-step runbook
**Language:** Hinglish (jaan-boojh kar — taaki steps follow karne mein aasani ho)
**Scope:** Stage 3 semantic rescue ke liye sabse powerful *tiny* model banana — English pehle, phir multilingual.

> **Padhne ka tareeka:** §1 se §3 tak "kya ho chuka hai aur kyun" hai. §4 se §9 tak
> actual steps hain jo aap chala sakte ho. §10 pitfalls hai — usko skip mat karna.

---

## 0. TL;DR — sabse important baat

**Aapka tiny model already ban chuka hai.** `semantic_project/57semanitc/` mein
`e5-distilled v3` student baitha hai:

| | value |
|---|---|
| Checkpoint | `v3_57intent_e5_distilled_v3_hard_negative/student_e5_distilled_v3_best_fp32.pt` |
| Size (fp32) | **0.75 MB** |
| Locked test accuracy (1,686 rows, 57 intents) | **98.87%** |
| Macro F1 | 0.9825 |
| Throughput | ~11,700 rows/sec |

Isi ke saamne **v6 E5-base** model hai jo **1.1 GB** ka hai aur **98.22%** deta hai
— yaani *tiny student bade model se behtar hai, 1/1400 size pe*.

**Toh ab kaam "naya model banana" nahi hai. Kaam ye hai:**

1. Us 0.75 MB student ko **ONNX + INT8 export** karke runtime mein wire karna (abhi
   `packages/runtime/nlu_engine/semantic.py` purane 22.9 MB MiniLM pe chal raha hai).
2. Uska **OOD / fallback gate** properly calibrate karna (abhi sirf 58 rows pe tested hai).
3. Phir **multilingual** version banana — wahi recipe, alag teacher.

Ye document teeno cover karta hai.

---

## 1. Aaj kahan khade hain — evidence table

> 🚨 **Warning — ye poori table invalid hai.** Locked test ki 100% rows `train.csv`
> mein maujood hain, aur ye sab models `train.csv` pe train hue the. Detail §10.1
> mein. Table historical record ke liye rakhi hai — inhe sach mat maano jab tak
> `reference_train.csv` pe retrain karke re-benchmark na ho jaaye.

Sab numbers repo ke JSON artifacts se liye gaye hain (source column dekho).
Sab ek hi **locked test = 1,686 rows, 57 intents** pe measure hue hain.

| Model | Size | Locked accuracy | Macro F1 | Source |
|---|---:|---:|---:|---|
| Tiny student v1 (TF-IDF/hybrid teacher) | 0.52 MB | 79.24% | 0.715 | `v3_57intent_locked_final_benchmark/` |
| Tiny student v2.2 (hard-negative) | 0.75 MB | 79.60% | 0.718 | `v3_57intent_v2_2_locked_benchmark/` |
| `e5-small-v2` frozen (no distillation) | ~130 MB *(est.)* | 89.09% | — | `v3_57intent_e5_small_v2_locked_benchmark/` |
| **e5-distilled v2** | **0.75 MB** | **98.75%** | 0.9816 | `v3_57intent_e5_distilled_v2_FINAL/final_summary.json` |
| **e5-distilled v3 (hard-negative)** | **0.75 MB** | **98.87%** | 0.9825 | `v3_57intent_e5_distilled_v3_locked_benchmark/` |
| v6 E5-base (English production vocab) | 1.1 GB | 98.22% | 0.9747 | `v6_locked_and_ood_benchmark/benchmark_summary_v6.json` |

### Isse kya seekha (ye plan ki poori foundation hai)

**Teacher ki quality hi sab kuch hai — student ka size nahi.**

Same student architecture (0.75 MB), sirf teacher badla:

- TF-IDF/hybrid teacher se distill → **79.6%**
- E5 embeddings se distill → **98.75%**

**19 percentage points ka jump, model size same.** Student chhota hone se accuracy
nahi girti; teacher kamzor hone se girti hai. Ye baat multilingual plan mein bhi
exactly wahi rahegi.

### Runtime mein abhi kya chal raha hai (gap)

| Artifact | Size | Note |
|---|---:|---:|
| `models/minilm-l6-v2.onnx` | 22.9 MB | `semantic.py` isi ko load karta hai |
| `models/semantic_head.npz` | 83 KB | LogReg head |
| `models/semantic_intent.onnx` | 24.0 MB | newer variant |

Yaani **best model (0.75 MB, 98.87%) production path mein wired nahi hai.** Ye
plan ka Phase 1 yahi hai.

---

## 2. Concepts — teacher, student, vocabulary (jaldi se)

### Teacher kya hai

Teacher = bada, accurate model jo **sirf training ke time chalta hai, ship kabhi
nahi hota**. Uska kaam hai har training sentence ke liye "sahi jawab" ya "sahi
embedding" produce karna, jise student copy kare.

**Isliye teacher chunte waqt size/latency dekhna hi nahi chahiye.** Sabse strong
teacher lo jo aapke laptop/GPU pe offline chal jaaye.

### Student kya hai

Student = chhota model jo actually device pe jaata hai. Aapka student:

```
TinyIntentClassifier
  nn.Embedding(vocab_size=895, embed_dim=64)
  TransformerEncoder(d_model=64, nhead=4, ff_dim=128, num_layers=2, dropout=0.10)
  LayerNorm(64)
  Linear(64 → 57)
```

`MAX_LEN = 24` tokens. Bas. Isi ne 98.87% diya.

### Vocabulary "kis level ka"

Ye sabse zyada confuse karne wali cheez hai, toh clear kar dete hain. Teen level hote hain:

| Level | Vocab size | Kaun use karta hai |
|---|---:|---|
| **Full pretrained** | 30,522 (BERT) / 250,002 (XLM-R) | Teacher models |
| **Domain-pruned subword** | ~5,500 (maine EN datasets pe measure kiya) | Agar aap pretrained tokenizer rakhna chahte ho |
| **Custom word-level (aapka current)** | **895** | Aapka student — `57semanitc/vocab.json` |

Aapka student **level 3** use karta hai: training text se hi vocabulary build hoti
hai, simple word-level tokenizer, `[PAD]/[UNK]/[CLS]/[SEP]` + top words. 895 tokens.

**Ye kaam kar gaya kyunki domain closed hai** — hearing-aid commands. Par ismein ek
built-in risk hai: koi bhi naya word `[UNK]` ban jaata hai. Ye §10 mein cover hai.

Reference ke liye, agar aap pretrained subword tokenizer rakhte:

| Vocab | d64 fp32 | d384 fp16 |
|---|---:|---:|
| 895 (current) | 0.23 MB | 0.69 MB |
| 5,557 (EN domain-pruned) | 1.4 MB | 4.3 MB |
| 30,522 (full BERT) | 7.8 MB | 23.4 MB |
| 250,002 (full XLM-R) | 64 MB | 192 MB |

895 × 64 wala embedding table sirf **229 KB** hai — isiliye poora model 0.75 MB mein
aa gaya.

---

## 3. Teacher aur student ka final selection

### English ke liye

| Role | Model | Kyun |
|---|---|---|
| **Teacher** | `intfloat/e5-small-v2` | Already proven — 98.75% student nikla ismein se |
| **Teacher (upgrade option)** | `intfloat/multilingual-e5-base` | Aapka v6 fine-tune, 98.22% khud; stronger teacher = stronger student |
| **Student** | `TinyIntentClassifier` (64-dim, 2-layer) | 0.75 MB, 98.87% |

> **Upgrade note:** v6 E5-base ko teacher banake dobara distill karna sabse saste
> improvements mein se hai — teacher 98.22% → student already 98.87% hai, toh
> headroom kam hai, par hard cases pe farak pad sakta hai. Ye optional hai.

### Multilingual ke liye

| Role | Model | Kyun |
|---|---|---|
| **Teacher** | `intfloat/multilingual-e5-base` | 100 languages, 768-dim, 12 layers. Repo mein already `v4_multilingual_e5_base_semantic/` mein fine-tuned hai |
| **Teacher (alternative)** | `paraphrase-multilingual-MiniLM-L12-v2` | `semantic_multilingual/download_models.py` isko already static-shape ONNX mein export karta hai |
| **Student** | Same `TinyIntentClassifier`, bada vocab | Architecture badalne ki zarurat nahi — sirf vocab aur data |

> **⚠️ Zaroori catch:** `v4_multilingual_e5_base_semantic/training_summary.json` mein
> `"multilingual_data": false` likha hai. Yaani multilingual backbone **English data
> pe hi** fine-tune hua. Multilingual capability latent hai, trained nahi. Multilingual
> student ke liye teacher ko pehle multilingual data pe fine-tune karna hoga.

---

## 4. Data — kya chahiye aur kis quality ka

### English (already available)

| File | Rows | Role |
|---|---:|---|
| `57semanitc/train.csv` | 8,430 | Training (57 intents) |
| `v3_57intent_locked_eval/locked_test_57intent.csv` | 1,686 | **Locked test — kabhi train/tune mat karna** |
| `v3_57intent_e5_distilled_v3_hard_negative/hard_negative_examples.csv` | 64 | Hard negatives |
| `unseen_semantic_stress_test.csv` | 595 | Stress test |
| `v3_57intent_e5_distilled_v3_negative_test/` | 58 | OOD / fallback test |

**Class balance problem** (`audit_summary.json` se):

- Largest: `Cmd.MemoryChange` — 1,601 samples
- Smallest: `Help_DemoMode` — 44 samples
- **36× imbalance**

P0 priority classes jo abhi patli hain: `Help_AppSettings` (46),
`Help_HearingCareAnywhereConnect` (47), `Cmd.TranscribeStart` (50),
`Help_Customize` (57), `Cmd.StreamingStop` (60).

**Action:** in P0 classes ko 150+ samples tak lao. Ye single sabse high-value data
kaam hai.

### OOD data — sabse badi weakness

Abhi OOD test sirf **58 rows** ka hai. Ek hearing-aid product ke liye ye bahut kam
hai. Target: **500+ OOD rows**, in categories mein:

- General knowledge ("what's the weather", "who won the match")
- Doosre apps ke commands ("play some music", "call mom")
- **Near-domain traps** — sabse important: "my ears hurt", "the hearing aid is
  broken", "how much did this cost" — ye domain ke kareeb hain par koi intent nahi.
- Adhoore utterances / ASR garbage

### Multilingual (partially available)

| Language | Rows | Intents | Min per intent |
|---|---:|---:|---:|
| en | 9,826 | 57 | 53 |
| fr | 9,805 | 57 | 53 |
| de | 6,900 | 57 | **16** |
| da | 9,387 | 57 | 39 |

Pending cleanup (ye pehle theek karo):

- `datasets/de_untranslated_placeholder_review.csv` — **2,926 rows**
- `datasets/da_untranslated_placeholder_review.csv` — 258 rows
- `datasets/da_label_conflicts_review.csv` — 181 rows
- `datasets/fr_label_conflicts_review.csv` — 21 rows

**Per-language OOD data abhi exist hi nahi karta.** Har language ke liye 200+ OOD
rows chahiye honge.

---

## 5. English pipeline — step by step

### Step 0 — Environment

```bash
cd /Users/shuklam/IntentClassifier
python -m venv .venv-semantic && source .venv-semantic/bin/activate
pip install torch transformers sentence-transformers scikit-learn pandas numpy onnx onnxruntime
```

> Ye **export-only deps** hain. `requirements.txt` mein mat daalo — runtime lean
> rehna chahiye (sirf `onnxruntime` + `numpy`). Repo rule hai
> (`.claude/memory/architecture.md` → "Dependency posture").

### Step 1 — Baseline reproduce karo

Kuch naya banane se pehle purana chala ke dekho ki numbers match karte hain:

```bash
cd semantic_project/57semanitc
python benchmark_e5_distilled_v3_locked_FINAL.py
```

Expected: locked accuracy ≈ **0.9887**. Agar match nahi hota, aage mat badho —
pehle reproducibility fix karo.

### Step 2 — Data improve karo (asli kaam yahi hai)

1. P0 classes ko 150+ samples tak expand karo (§4 ki list).
2. OOD set ko 58 → 500+ rows le jao.
3. `v6_english_vocab_review/candidate_vocab_review.csv` mein 133 rows review-pending
   hain — unhe review karke merge karo.

> **Rule:** `locked_test_57intent.csv` ko kabhi mat chhuo. Na train, na tune, na
> threshold selection. Wahi aapka ekmatra honest number hai.

### Step 3 — Teacher embeddings generate karo

```python
from sentence_transformers import SentenceTransformer
teacher = SentenceTransformer("intfloat/e5-small-v2")
emb = teacher.encode(texts, normalize_embeddings=True, batch_size=64)
np.save("teacher_train_embeddings.npy", emb)
```

Reference implementation: `e5_distilled_v2_FINAL_TRAIN_AND_TEST.py` (line ~316,
"E5 EMBEDDINGS" section).

### Step 4 — Student vocab build karo

Training text se hi vocab banti hai — pretrained tokenizer nahi:

```python
# reference: e5_distilled_v2_FINAL_TRAIN_AND_TEST.py line ~740
# "BUILD STUDENT VOCAB FROM TRAINING TEXT ONLY"
```

Output: `vocab.json` — abhi 895 tokens. Data badhane ke baad ye ~1,100–1,300 ho
jaayega. Chalega — 1,300 × 64 = 333 KB.

### Step 5 — Student train karo

Exact config jo 98.87% deta hai:

```python
MAX_LEN     = 24
EMBED_DIM   = 64
NHEAD       = 4
FF_DIM      = 128
NUM_LAYERS  = 2
DROPOUT     = 0.10

BATCH_SIZE  = 128
EPOCHS      = 40
PATIENCE    = 7
LR          = 2e-3
WEIGHT_DECAY = 1e-4

TEMPERATURE = 2.0      # KD softmax temperature
CE_WEIGHT   = 0.70     # true labels
KD_WEIGHT   = 0.30     # teacher distillation
VAL_SIZE    = 0.15
SEED        = 42
```

Loss = `0.70 × CrossEntropy(labels) + 0.30 × KL(student ‖ teacher, T=2.0)`

```bash
python e5_distilled_v2_FINAL_TRAIN_AND_TEST.py
```

Best epoch v2 mein 19 tha — 40 epochs se pehle hi early-stop ho jaata hai.

**Tuning karni ho toh sirf ye teen knobs chhuo:**

| Knob | Kab badlo |
|---|---|
| `KD_WEIGHT` 0.30 → 0.50 | Agar teacher strong hai aur student underfit kar raha hai |
| `NUM_LAYERS` 2 → 3 | Agar data 2× badha diya ho (size 0.75 → ~1.1 MB) |
| `EMBED_DIM` 64 → 96 | Sirf tab jab vocab bhi kaafi badh gaya ho |

Baaki sab waisa hi rakho — ye config already validated hai.

### Step 6 — Hard-negative refinement (v3 step)

v2 → v3 ne accuracy 98.75% → 98.87% ki, hard negatives mine karke:

```bash
python train_e5_distilled_v3_hard_negative_FINAL.py
```

Logic: v2 ke confident-but-wrong predictions nikaalo, unhe training set mein wapas
daalo, dobara train karo.

### Step 7 — Confidence gate calibrate karo

```bash
python v6_confidence_fallback_calibration_FIXED.py
```

Ye woh threshold deta hai jiske neeche student ka jawab reject karke GenAI pe jaana
hai. **Purana 0.55 blindly reuse mat karna** — naya model, naya threshold.

Calibration set: OOD data + holdout. **Locked test nahi.**

### Step 8 — ONNX export + INT8

```bash
python export_v3_57intent_to_onnx_FINAL.py
```

Constraints (non-negotiable, `.claude/memory/mobile.md` se):

- **Static batch size 1**, static `MAX_LEN=24`
- Output **logits**, temperature runtime pe apply hoti hai
- Dynamic axes bilkul nahi (ANE dynamic seq support nahi karta)

INT8:

```bash
python quantize_validate_tiny_student_int8.py
```

> **Reality check:** `v6_int8/int8_quantization_manifest.json` batata hai ki classifier
> head pe INT8 ne size **−0.04%** kiya (0.2096 → 0.2096 MB) — yaani kuch nahi. Kyunki
> woh linear head tha. Poore student pe (embedding + transformer) INT8 asli ~4× dega:
> **0.75 MB → ~0.19 MB**. Parity check zaroor karo — v6 mein 0 mismatches the.

### Step 9 — Runtime mein wire karo

Ye woh step hai jo abhi missing hai. `packages/runtime/nlu_engine/semantic.py`:

1. `ONNX_PATH` ko naye student ONNX pe point karo
2. `_tokenise()` ko WordPiece se **word-level `vocab.json`** tokenizer pe badlo
   (`MAX_LEN=24`, PAD=0, UNK=1)
3. `_embed()` ka mean-pool hata do — student directly logits deta hai
4. `DEFAULT_THRESHOLD` ko Step 7 wale naye number se replace karo
5. `models/manifest.json` update karo

> Ye behavior change hai. Repo rule ke hisaab se pehle reasoning likho aur approval
> lo (`CLAUDE.md` → "Repository rules").

---

## 6. Multilingual pipeline — step by step

English pipeline hi hai, teen changes ke saath: **teacher**, **vocab**, **data**.

### Step M1 — Data cleanup pehle (blocker)

Model ko haath lagane se pehle:

```
[ ] de_untranslated_placeholder_review.csv   — 2,926 rows resolve karo
[ ] da_untranslated_placeholder_review.csv   — 258 rows
[ ] da_label_conflicts_review.csv            — 181 rows
[ ] fr_label_conflicts_review.csv            — 21 rows
[ ] datasets/multilingual/pending/           — pva_intent_danish.csv, pva_intent_german.csv process karo
[ ] de ko 6,900 → 9,500+ rows tak lao (min/intent 16 → 50+)
```

Ye skip karoge toh model kharaab data seekhega aur baad mein debug karna namumkin hoga.

### Step M2 — Multilingual teacher fine-tune karo

Abhi ka `v4_multilingual_e5_base_semantic` **English-only** trained hai
(`"multilingual_data": false`). Use multilingual data pe dobara train karo:

```bash
python train_multilingual_e5_base_v4_semantic.py   # train.csv ko multilingual set pe point karke
```

Config jo v4 mein use hui:

```
backbone      = intfloat/multilingual-e5-base
epochs        = 3
batch_size    = 16
learning_rate = 2e-5
max_seq_length = 64
loss          = supervised contrastive
```

v4 (English data pe) validation accuracy **95.97%** thi. Multilingual data pe ye
thodi girega — 92–94% expect karo. Teacher ka kaam student se behtar hona hai, perfect
hona nahi.

### Step M3 — Multilingual student vocab

Yahan sabse bada design decision hai. Do options:

| Option | Vocab | Student size (fp32) | Trade-off |
|---|---:|---:|---|
| **A. Shared vocab, ek model** | ~9,000 words (4 lang) | ~2.9 MB | Ek artifact, code-switching handle karta hai |
| **B. Per-language vocab, alag models** | ~900–1,200 each | ~0.75 MB each | Har language chhoti, par 4 artifacts + language detection chahiye |

Maine `datasets/multilingual/*.csv` pe measure kiya:

| Language | unique words |
|---|---:|
| en | 2,071 |
| fr | 2,780 |
| de | 2,672 |
| da | 3,133 |
| **union** | **9,150** |

Overlap sirf ~14% hai (sum 10,656 vs union 9,150) — yaani languages aapas mein bahut
kam share karti hain.

**Recommendation: Option A (shared)**, kyunki:

- 9,150 × 64 = **2.3 MB** embedding table — abhi bhi chhota hai
- Code-switching real hai (`docs/adding-a-new-language.md` khud kehta hai ASR output
  mixed hota hai)
- Ek artifact = ek calibration, ek parity test, ek release

Agar 20 languages tak jaana hai toh Option B reconsider karna padega (§7 dekho).

### Step M4 — Student train karo

Same config jaisa §5 Step 5, sirf ye badlo:

```python
MAX_LEN = 32          # 24 → 32 (de/da compounds lambe hote hain)
EMBED_DIM = 96        # 64 → 96 (4× vocab, thodi zyada capacity)
NUM_LAYERS = 2        # same
```

Expected size: ~2.9 MB fp32, ~0.75 MB int8.

### Step M5 — Per-language evaluation

**Ek combined accuracy number mat dekhna.** Har language ka alag locked test chahiye.

Per-language holdouts **`legacy_research/test/`** mein padi hain (memory docs
`multilingual/test/` kehti hain — woh path stale hai, ND-2 ke move ke baad update
nahi hua):

```
legacy_research/test/en_holdout.csv
legacy_research/test/fr_holdout.csv
legacy_research/test/de_holdout.csv
legacy_research/test/da_holdout.csv
legacy_research/test/multilingual_holdout.csv
```

> **Pehla kaam:** in holdouts ko `legacy_research/` se nikaal ke ek proper eval
> location pe promote karo, aur `.claude/memory/datasets.md` ka path theek karo.
> `legacy_research/` naam hi batata hai ki koi inhe live eval nahi maan raha.

Per-language ship bar (TF-IDF stage ke existing gate ke hisaab se):

| Language | TF-IDF macro F1 (aaj) | Semantic student ka target |
|---|---:|---:|
| en | 0.90 | ≥ 0.95 |
| fr | 0.84 | ≥ 0.90 |
| de | 0.83 | ≥ 0.90 |
| da | **0.74** | ≥ 0.85 |

Danish already 0.75 ke gate se neeche hai — usko data se theek karo, model se nahi.

### Step M6 — Per-language threshold

Har language ka **apna** confidence threshold hoga. Ek global threshold multilingual
mein kaam nahi karta — kamzor language pe woh ya to bahut kuch reject karega ya bahut
kuch galat accept.

`config/calibration.json` mein per-language entry daalo, waise hi jaise TF-IDF stage
ke liye temperature hai.

---

## 7. Agar 20 languages tak jaana hai

Chhota summary — poori feasibility alag baat hai:

**Vocab growth (measured 4-lang trend se extrapolated):**

| Languages | union words (est.) | student vocab table (d96 fp32) |
|---:|---:|---:|
| 4 | 9,150 | 3.5 MB |
| 8 | ~18,000 | 6.9 MB |
| 20 | ~37,000–48,000 | **14–18 MB** |

Model abhi bhi 20 MB se neeche rahega — **size problem nahi hai.**

**Asli problem data hai.** Har nayi language ke liye chahiye:

| Artifact | Volume | Native speaker? |
|---|---|---|
| `nlu_lexicon.<lang>.json` | ~105 keys, 300–500 strings | **Haan** (sabse mushkil file) |
| `nlu_entities.<lang>.json` | 65 values / 109 synonyms | Haan |
| `nlu_schema.<lang>.json` | 56 fulfillment + prompts + triggers | Haan |
| Training CSV | 5,700–17,100 utterances | MT + human review |
| OOD set | 200+ rows | Haan |
| Datetime fixtures | 9+ golden rows | Haan |

Roughly **10–13 person-days per language**, aur **16 alag native speakers** — yaani
~8–10 person-months. Ye estimate hai, measurement nahi.

**Recommendation:** 20 languages ko tier karo, ek project mat banao:

- **Tier 1 (en, fr, de, da):** poora semantic student + OOD + calibration
- **Tier 2 (agli 4–6):** semantic student tabhi jab data quality bar clear kare
- **Tier 3 (baaki):** keyword triggers + TF-IDF only, semantic rescue ke bina

---

## 8. Export aur deployment

```
student .pt (fp32, 0.75 MB)
   ↓ export_v3_57intent_to_onnx_FINAL.py
student .onnx (static batch=1, seq=24, logits out)
   ↓ quantize_validate_tiny_student_int8.py
student_int8.onnx (~0.19 MB)
   ↓ multilingual/export_coreml_multilingual.py
Student.mlpackage (FP16, iOS)
   ↓ scripts/ci/assemble_pack.py
signed .nlu bundle
```

Har step pe parity check chahiye. Acceptance (`.claude/memory/mobile.md`):

> **accuracy Δ ≈ 0 aur 0 gate disagreements** vs golden fixtures.

---

## 9. Ship bar — non-negotiable

Koi bhi naya candidate (English ya multilingual) ye gate paas kare:

| # | Metric | Bar |
|---|---|---|
| 1 | Locked test accuracy | Δ ≥ **−1.0%** vs current best (98.87%) |
| 2 | **OOD fallback rate** | **worse nahi** (v6 pe 100%, par sirf 58 rows) |
| 3 | Stress test (595 unseen rows) | Δ ≥ −1.0% |
| 4 | Per-language macro F1 (multilingual) | §6 M5 ki table |
| 5 | ONNX ↔ PyTorch parity | 0 prediction mismatches |
| 6 | INT8 ↔ FP32 parity | 0 argmax flips |
| 7 | On-device peak RAM | budget ke andar |

> **Rule:** OOD rejection kabhi regress nahi honi chahiye, chahe accuracy kitni bhi
> badh jaaye. Hearing-aid product mein galat command execute karna, jawab na dene se
> zyada khatarnak hai.

---

## 10. Pitfalls — ye padhna zaroori hai

### 10.1 🚨 LOCKED TEST LEAK — saare accuracy numbers invalid hain

**Confirmed 2026-08-10.** `locked_test_57intent.csv` ki **1,686 mein se 1,686 rows
(100%)** `train.csv` mein verbatim maujood hain — text aur intent dono same.

```
locked rows whose TEXT appears in train.csv    : 1686/1686 = 100.0%
locked rows whose (TEXT,INTENT) pair is in train: 1686/1686 = 100.0%
genuinely unseen locked rows                    : 0
```

Isse §1 ki poori table ka matlab badal jaata hai. 98.87% **memorisation score** hai,
generalisation nahi. Yahi explain karta hai ki locked accuracy (98.75%) validation
accuracy (92.81%) se **zyada** kyun thi — held-out test train se aasan nahi ho sakta,
jab tak woh train ka hissa na ho.

**Ye pehle se documented tha.** `create_locked_57intent_split.py` ke apne docstring mein
likha hai:

> "The existing 57-intent V3/V2 checkpoints were trained using the original 8430-row
> train.csv. Therefore, an evaluation split made NOW from that same CSV is NOT an
> unseen benchmark for those already-trained checkpoints. **To obtain a scientifically
> valid V3-vs-V2 comparison, BOTH models must subsequently be retrained using ONLY
> reference_train.csv.**"

Woh script clean 80% split **already bana chuki hai**:

| File | Rows | Locked test se overlap |
|---|---:|---:|
| `train.csv` | 8,430 | **1,686 (100%)** |
| `v3_57intent_locked_eval/reference_train.csv` | 6,744 | **0** ✅ |

Par baad ki saari training scripts `train.csv` pe chalti rahin — `final_summary.json`
mein `"train_csv": ".../train.csv"` likha hai. Manifests mein
`"locked_test_used_for_training": false` flag hai, par woh sirf ye check karta hai ki
locked CSV file trainer ko pass hui ya nahi — ye nahi ki wahi rows train.csv ke andar
baithi hain.

**Isliye affected hain:** v2, v2.1, v2.2, v3, e5-distilled v2, e5-distilled v3, v6 —
yaani §1 ki poori ladder.

**Fix (Phase 1 ka pehla kaam):**

```
[ ] reference_train.csv (ya train_balanced.csv) pe dobara train karo
[ ] locked test pe dobara benchmark karo
[ ] §1 ki table ke saare numbers replace karo
[ ] tab decide karo ki kaunsa model actually best hai
```

**Realistic expectation:** asli accuracy validation number ke kareeb hogi — **~92–93%**,
98.87% nahi. Ye bura nahi hai; 0.75 MB model ke liye 92% bahut acchi baat hai. Bas ab
woh number *sach* hoga.

### 10.2 895-token vocab ka `[UNK]` risk

Student ki vocab training text se hi bani hai. Koi bhi word jo training mein nahi tha
→ `[UNK]`. Aur semantic rescue ka poora point hi **unseen phrasing** handle karna hai.

Mitigation:

- Vocab mein character-level fallback pieces rakho
- `[UNK]` rate ko OOD set pe measure karo — agar >5% hai toh vocab badhao
- Vocab badhana sasta hai: +400 tokens = +25 KB

### 10.3 OOD test sirf 58 rows ka hai

100% fallback rate impressive dikhta hai, par 58 rows pe confidence interval bahut
chaudi hai. 500+ rows tak le jao — warna ye number kuch prove nahi karta.

### 10.4 `semantic_project/` repo se disconnected hai

`semantic_project/` mein 200+ files hain — v1 se v7 tak, `_FINAL`, `_FIXED`,
`_FINAL_v2` naming ke saath. Ye production pipeline (`packages/`) se juda hua nahi hai.

Best model wire karne se pehle:

1. Winning artifacts ko `packages/buildtime/nlu_training/` mein promote karo
2. Ek reproducible training script rakho, 20 variants nahi
3. Baaki ko `legacy_research/` mein le jao (`legacy_research/SemanticSupport/` pattern already hai)

### 10.5 Stage 3 abhi English-only hai

`docs/adding-a-new-language.md` ka "Deliberately deferred capabilities" #1 —
semantic rescue non-English ke liye wired hi nahi hai. Multilingual student banane se
pehle ye runtime plumbing karni padegi.

### 10.6 Teacher badalne ka matlab sab kuch dobara

Teacher change = naya embedding space = **head retrain + threshold re-tune + fixtures
regenerate**. Purana threshold kabhi carry-forward mat karna.

---

## 11. Checklist — order mein follow karo

### Phase 1 — English production (highest value, sabse kam risk)

```
[ ] 1.0  🚨 LEAK FIX — train_balanced.csv / reference_train.csv pe retrain karo,
         locked test pe re-benchmark, §1 ki table replace karo (§10.1)
[ ] 1.1  Naya honest baseline record karo (~92-93% expect, 98.87% nahi)
[ ] 1.2  Har training script ka source CSV audit karo (train.csv leaky hai)
[ ] 1.3  OOD set 58 → 500+ rows
[ ] 1.4  P0 classes 150+ samples tak
[ ] 1.5  candidate_vocab_review.csv ke 133 rows review karo
[ ] 1.6  Student retrain (same config, better data)
[ ] 1.7  Hard-negative refinement pass
[ ] 1.8  Confidence threshold calibrate (naye OOD set pe)
[ ] 1.9  ONNX export (static batch=1, seq=24) + parity
[ ] 1.10 INT8 quantize + parity (0 argmax flips)
[ ] 1.11 semantic.py ko naye student pe wire karo
[ ] 1.12 Ship bar (§9) chalao
```

### Phase 2 — Repo hygiene

```
[ ] 2.1  Winning scripts packages/buildtime/ mein promote karo
[ ] 2.2  semantic_project/ ko legacy_research/ mein archive karo
[ ] 2.3  .claude/memory/inference.md + mobile.md update karo
[ ] 2.4  ADR likho: kyun tiny student ne MiniLM ko replace kiya
```

### Phase 3 — Multilingual (en/fr/de/da)

```
[ ] 3.1  Data cleanup — 3,386 pending review rows resolve karo
[ ] 3.2  German data 6,900 → 9,500+
[ ] 3.3  Per-language OOD sets (200+ rows each)
[ ] 3.4  Multilingual teacher fine-tune (asli multilingual data pe)
[ ] 3.5  Shared vocab build (~9,150 words)
[ ] 3.6  Student train (MAX_LEN=32, EMBED_DIM=96)
[ ] 3.7  legacy_research/test/*_holdout.csv ko proper eval location pe promote karo
[ ] 3.8  Per-language evaluation (§6 M5)
[ ] 3.9  Per-language thresholds → config/calibration.json
[ ] 3.10 Stage 3 runtime ko multilingual ke liye wire karo
[ ] 3.11 Ship bar per language
```

### Phase 4 — Scale (agar Phase 3 clean gaya)

```
[ ] 4.1  Tier 2 languages choose karo (data feasibility ke basis pe)
[ ] 4.2  Per-language cost measure karo (estimate nahi)
[ ] 4.3  Tab decide karo ki 20 realistic hai ya nahi
```

---

## Appendix A — File index

| Purpose | Path |
|---|---|
| Best student checkpoint | `semantic_project/57semanitc/v3_57intent_e5_distilled_v3_hard_negative/student_e5_distilled_v3_best_fp32.pt` |
| Training script (reference) | `semantic_project/57semanitc/e5_distilled_v2_FINAL_TRAIN_AND_TEST.py` |
| Hard-negative refinement | `semantic_project/57semanitc/train_e5_distilled_v3_hard_negative_FINAL.py` |
| ONNX export | `semantic_project/57semanitc/export_v3_57intent_to_onnx_FINAL.py` |
| INT8 quantize | `semantic_project/quantize_validate_tiny_student_int8.py` |
| Threshold calibration | `semantic_project/57semanitc/v6_confidence_fallback_calibration_FIXED.py` |
| Multilingual teacher | `semantic_project/57semanitc/train_multilingual_e5_base_v4_semantic.py` |
| Student vocab | `semantic_project/57semanitc/vocab.json` (895 tokens) |
| Locked test | `semantic_project/57semanitc/v3_57intent_locked_eval/locked_test_57intent.csv` |
| Runtime semantic path | `packages/runtime/nlu_engine/semantic.py` |
| Multilingual ONNX export | `packages/buildtime/nlu_training/semantic_multilingual/download_models.py` |

## Appendix B — Size math (reference)

```
Student = embedding table + transformer + head

embedding : vocab × embed_dim
transformer (per layer, d=64, ff=128):
    attention  4 × (64×64)      = 16,384
    ffn        64×128 + 128×64  = 16,384
    ≈ 33k params/layer × 2 layers ≈ 66k
head      : 64 × 57 = 3,648

English (vocab 895, d64):
    embedding  895 × 64   =  57,280
    transformer            ≈  66,000
    head                   =   3,648
    total                  ≈ 127k params → fp32 ≈ 0.51 MB
    (actual checkpoint 0.75 MB — optimizer/LayerNorm/buffers included)

Multilingual (vocab 9,150, d96):
    embedding  9,150 × 96 = 878,400
    transformer (d96)      ≈ 148,000
    head       96 × 57     =   5,472
    total                  ≈ 1.03M params → fp32 ≈ 4.1 MB, int8 ≈ 1.0 MB
```

---

**Last updated:** 2026-08-10
**Numbers ka source:** repo artifacts (`semantic_project/**/[*.json]`, `datasets/`,
`models/`). Estimates ko clearly "est." mark kiya gaya hai.
