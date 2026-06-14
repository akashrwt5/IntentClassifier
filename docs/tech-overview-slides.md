# On-Device Intent Classifier — Technical Overview
### Slide deck for engineering audiences

---

## Slide 1 — Title

**On-Device, Offline-Capable Intent Classification**
*Low-latency natural language understanding for mobile hearing-aid companion apps*

- Target audiences: iOS, Android
- Total model footprint: ~25 MB
- End-to-end latency: < 20 ms on modern ARM
- No network required for inference

---

## Slide 2 — The Problem

**What we need to solve:**

> Users speak naturally. Intent classifiers expect scripted phrases.

- "I can't hear very well" ≠ "volume up"
- "remind me to take my pills tomorrow morning for breakfast" has a date, a time, a topic, and a carrier phrase all mixed together
- Misclassifications on a hearing-aid app aren't just inconvenient — they erode user trust in the device

**Hard constraints:**
- Must work **fully offline** — no cloud round-trip
- Must run on **low-power mobile hardware** (A-series chip, Snapdragon 6xx)
- Must achieve **≥ 88/100** on a held-out benchmark of real user utterances
- Must **never log raw speech** by default (medical device privacy)

---

## Slide 3 — Architecture: The 3-Stage Cascade

```
User utterance
      │
      ▼
┌─────────────────────────────┐
│ Stage 1: Keyword Pre-filter │  ~0 ms
│ Schema-driven exact/regex   │
│ match against nlu_schema.json│
└─────────────┬───────────────┘
              │ hit (exact/regex)
              │                 miss
              ▼                  ▼
      Return intent        ┌─────────────────────────────┐
      (conf 0.90–0.97)     │ Stage 2: TF-IDF + LogReg    │  ~1 ms
                           │ ONNX model, 2.2 MB           │
                           │ 60 intent classes            │
                           └─────────────┬───────────────┘
                                         │ conf ≥ 0.70
                                         │                miss
                                         ▼                 ▼
                                 Return intent       ┌─────────────────────────────┐
                                                     │ Stage 3: MiniLM + Head      │  ~10–15 ms
                                                     │ MiniLM-L6-v2 ONNX (22 MB)  │
                                                     │ 384-dim sentence embedding  │
                                                     │ Logistic head (59 × 384)    │
                                                     └─────────────┬───────────────┘
                                                                   │ conf ≥ 0.55
                                                                   │                miss
                                                                   ▼                 ▼
                                                           Return intent       GenAI fallback
                                                           (semantic_rescue)   (network required)
```

**Design principle:** Each stage is cheaper than the next. Fast paths handle the common cases; the heavy model only runs when needed.

---

## Slide 4 — Stage 1: Keyword Pre-filter

**What:** Declarative rule engine, zero ML.

**How:**
- Rules defined in `data/nlu_schema.json` under `keyword_triggers`
- Three match types, each with a calibrated confidence:

| Type | Confidence | Example |
|---|---|---|
| `exact` | 0.97 | `"mute"` → `Cmd.VolumeMute` |
| `regex_guarded` | 0.90 | pattern + exclusion regex |
| `regex` | 0.75 | bare pattern match |
| `contains` | 0.85 | substring anywhere in utterance |

**Negation guard:** "I don't want to translate this" — the word "translate" is a `contains` keyword, but `_is_negated()` checks a 30-char window before the match for negation cues ("not", "don't", "without", etc.) and suppresses the hit.

**Why not hardcode intents in Python?** The schema drives everything — changing a trigger phrase means editing one JSON file, not redeploying code.

---

## Slide 5 — Stage 2: TF-IDF + Logistic Regression (ONNX)

**Model:** scikit-learn `Pipeline(TfidfVectorizer, CalibratedClassifierCV(LogisticRegression))`

**Key hyperparameters:**
- `min_df=2` (prunes noise tokens)
- `C=15.0` (strong regularisation for 60-class problem)
- `class_weight='balanced'` (handles unequal class sizes)
- Isotonic calibration via `CalibratedClassifierCV`

**Export path:** `skl2onnx` → `intent_model.onnx` (2.2 MB)
- Input: raw text string
- Output: probability vector over 60 intent classes
- Entire vectoriser is baked into the ONNX graph — no pre-processing on the mobile side

**Threshold:** Accept if `max(prob) ≥ 0.70`. Slot-filling intents use a lower threshold (0.60) to avoid stalling flows on borderline-confident inputs.

**Training gate:** `train.py` exits non-zero if test-split accuracy < 0.85. Prevents silently shipping a degraded model.

---

## Slide 6 — Stage 3: MiniLM-L6-v2 + Logistic Head

**Why a second model?**

TF-IDF is a bag-of-words model — it can't handle vocabulary gaps. "Everything sounds too quiet" shares zero vocabulary with "turn up the volume", but semantically they're identical.

**Encoder: MiniLM-L6-v2**
- 6-layer transformer, INT8 quantized ONNX (22 MB)
- Input: WordPiece token IDs + attention mask + token type IDs
- Output: `last_hidden_state` `[1, seq_len, 384]`
- Post-processing: mean-pool over non-padding tokens → L2-normalise → 384-dim unit vector
- Tokenizer: hand-rolled WordPiece from `minilm-vocab.txt` (no external library needed on mobile)

**Head: Logistic Regression**
- Trained on MiniLM embeddings of all 1 876 training phrases (SetFit-style, encoder frozen)
- 59 × 384 weight matrix + 59-dim bias (stored as `semantic_head.json`, 232 KB)
- Explicit "Default Fallback Intent" class trained on 156 curated out-of-scope phrases
- `class_weight='balanced'`, `C=10.0`, `max_iter=2000`

**Why a learned OOS class instead of a threshold?**
A threshold on max-probability always predicts the nearest class — it can't generalise to "this topic doesn't belong here". Teaching an explicit OOS class lets the head learn what rejection looks like.

---

## Slide 7 — Multi-Turn Slot Filling

**What:** Some intents require parameters ("reminder" needs a time and topic). The engine prompts for missing slots across turns.

```
User:   "set a reminder"
Engine: "What would you like to be reminded about?"
User:   "dinner with mum"
Engine: "When should I remind you?"
User:   "tomorrow at 7pm"
Engine: ✅ Reminder set for [tomorrow 19:00 UTC+01:00] — dinner with mum
```

**Key design decisions:**

| Feature | Why |
|---|---|
| `MAX_SLOT_ATTEMPTS = 3` | Prevents infinite loops if user gives unrecognisable answers |
| Context TTL = 90 s | Stale context from a previous conversation doesn't re-fire accidentally |
| Session TTL = 600 s | Idle sessions are reset cleanly |
| Injectable clock | Tests run deterministically — no `time.sleep` |
| Interrupt threshold = 0.85 | A strong new command during slot fill cancels it and starts fresh |
| `contains`-tier can't interrupt | "Ask about the translate feature" mid-reminder doesn't abandon the reminder |

**Slot schema in `nlu_schema.json`:**
```json
{
  "intent": "reminders.add",
  "slots": [
    { "name": "topic", "question": "What would you like to be reminded about?" },
    { "name": "datetime", "type": "datetime", "question": "When should I remind you?" }
  ]
}
```

---

## Slide 8 — Entity Extraction

**Datetime parsing:**
- Natural-language patterns: "tomorrow at 7pm", "next Monday morning", "in 2 hours"
- All datetimes stored as UTC ISO 8601 with offset: `"2025-06-15T18:00:00+00:00"`
- Edge guards: "quarter to 13" (invalid hour), "0 to 3" (minute = 60) → safe `None` return

**Enum matching (memory modes):**
- Exact match first: "Restaurant" → `Restaurant`
- Fuzzy match second (Levenshtein ratio ≥ 0.3): "restraunt" → `Restaurant`
- Length gate: words < 5 characters bypass fuzzy — prevents "cab"→"Car", "gum"→"Gym" collisions
- One-shot scan (initial utterance): fuzzy disabled entirely

**Back-reference resolution:**
- "Change back" / "Remind me again" resolve from session state via declarative `back_reference` field in schema

---

## Slide 9 — Artifact Integrity

**SHA-256 manifest (`models/manifest.json`):**

All 8 model artifacts are hashed at training time and verified at startup:

```
models/intent_model.onnx         sha256: a3f2...
models/intent_labels.json        sha256: 9b1c...
models/minilm-l6-v2.onnx        sha256: 7e44...
models/minilm-vocab.txt          sha256: c8d0...
models/semantic_head.json        sha256: 2fa1...
... (3 more)
```

- `verify_manifest()` called on `IntentClassifier.__init__` — startup fails fast on mismatch
- `generate_manifest()` called at the end of every training script — no manual step needed
- `_meta` block stores `holdout_accuracy` from the last run

**Why this matters:** A corrupted or swapped model file produces wrong classifications silently. The manifest catches it immediately.

---

## Slide 10 — Holdout Accuracy Gate

**End-to-end pipeline gate (not just unit tests):**

```bash
python scripts/test_holdout.py --strict
```

**Dual budget (both must pass):**
- Total correct ≥ 88 / 100
- Wrong-action misses ≤ 5 / 100

A wrong-action miss (confidently fired wrong intent) is penalised more than a GenAI fallback (safe miss).

**Gate is provably real:** `test_sprint3_hardening.py` sets `MIN_HOLDOUT_TOTAL=99` and asserts `returncode == 1` — the gate fails at impossible floors, so it can't be a no-op.

**Current score: 90/100, 4 wrong-action misses.**

| Stage | Contribution |
|---|---|
| Stage 1 (keyword) | ~15 clean exact/regex classifications |
| Stage 2 (TF-IDF) | ~62 correct classifications |
| Stage 3 (semantic) | ~13 rescues (utterances TF-IDF misses) |
| GenAI fallback | 6 genuine out-of-scope queries |
| Misses | 4 wrong-action, 2 safe misses |

---

## Slide 11 — Tech Stack

| Layer | Technology |
|---|---|
| Training | Python 3.11, scikit-learn, pandas, numpy |
| TF-IDF export | skl2onnx → ONNX 1.x |
| Semantic encoder | MiniLM-L6-v2 (Hugging Face), ONNX Runtime, INT8 quantized |
| Semantic head | scikit-learn LogisticRegression, exported to JSON |
| Runtime (Python) | ONNX Runtime (`onnxruntime`) |
| Runtime (Android) | ONNX Runtime Mobile (`onnxruntime-android`) |
| Runtime (iOS) | Core ML (converted from ONNX via `coremltools`) |
| Fuzzy matching | `rapidfuzz` (Levenshtein ratio) |
| Testing | pytest + custom runner |
| Integrity | SHA-256 manifest (stdlib `hashlib`) |
| Slot schemas | JSON (`data/nlu_schema.json`) |

---

## Slide 12 — Mobile Deployment Plan

**Android:**
- All 6 artifacts in `app/src/main/assets/nlu/`
- `NLUEngine.kt` — single class mirroring the Python cascade
- ONNX Runtime Mobile handles both TF-IDF and MiniLM ONNX models
- WordPiece tokenizer ported to Kotlin (~60 lines)
- Logistic head: load `semantic_head.json`, matrix multiply via Android NDK or Kotlin arrays

**iOS:**
- Models converted to `.mlpackage` via `coremltools` (one-time, offline, on a Mac)
- `NLUEngine.swift` — same 3-stage logic
- MiniLM runs on the Apple Neural Engine on A12+ → ~3–5 ms (vs 15 ms CPU)
- Logistic head: load `semantic_head.json`, matrix multiply via `Accelerate.vDSP`

**Bundle size:** ~25 MB total (MiniLM dominates at 22 MB; compresses to ~19 MB in the App Store)

**Load strategy:** MiniLM loaded lazily (first Stage 3 call). Cold start uses only Stage 1 + Stage 2.

---

## Slide 13 — Key Decisions Summary

| Decision | Alternative considered | Why we chose this |
|---|---|---|
| Frozen MiniLM + trained head | Fine-tune full model | Keeps training to <2 min; generalises well without GPU |
| JSON head export | ONNX or Core ML native | Trivially parseable on both platforms; <250 KB |
| Explicit OOS class | Threshold-only rejection | Learned rejection generalises; threshold is fragile |
| Hand-rolled WordPiece | Python `tokenizers` lib | No C++ native dependency needed on Android/iOS |
| SHA-256 manifest | Checksum in training logs | Verified at runtime, not just at training time |
| Dual holdout budget (total + wrong-action) | Single accuracy metric | Wrong-action misses are not equal to safe fallbacks |
| `contains` interrupt demotion | No change | Substring keyword in mid-sentence is weak signal; shouldn't cancel multi-turn flows |

---

## Slide 14 — Results & Next Steps

**What we have today:**
- ✅ 90/100 holdout accuracy (target: ≥ 88)
- ✅ 4 wrong-action misses (target: ≤ 5)
- ✅ Full offline operation, no network dependency
- ✅ 150 automated test assertions across 5 test files
- ✅ Mobile-ready artifacts (ONNX + JSON head)
- ✅ SHA-256 artifact integrity enforced at startup

**Recommended next steps (in priority order):**
1. Wire Stage 2 on Android — validate common commands end-to-end
2. Wire Stage 3 on Android — validate semantic rescues ("i can barely hear")
3. iOS Core ML conversion (`scripts/export_coreml.py`)
4. Wire Stage 2 + 3 on iOS
5. Run `data/semantic_holdout_100.csv` against each native implementation — target ≥ 75/100
6. Fix known `regex`-tier interrupt limitation (standalone-command detection)

---

## Appendix — File Map

```
IntentClassifier/
├── data/
│   ├── nlu_schema.json           # Intent config, slot definitions, keyword triggers, thresholds
│   ├── intent_data_new.csv       # Training phrases (~1 876 rows)
│   ├── semantic_oos.csv          # 156 curated out-of-scope phrases
│   └── semantic_holdout_100.csv  # 100-utterance held-out benchmark
├── models/
│   ├── intent_model.onnx         # TF-IDF + LogReg (2.2 MB)
│   ├── intent_labels.json        # 60 intent label array
│   ├── minilm-l6-v2.onnx        # MiniLM encoder (22 MB, INT8)
│   ├── minilm-vocab.txt          # WordPiece vocabulary (30 522 tokens)
│   ├── semantic_head.json        # Logistic head weights (232 KB)
│   └── manifest.json             # SHA-256 integrity record
├── scripts/
│   ├── nlu/
│   │   ├── engine.py             # Orchestrator, slot filling, session management
│   │   ├── classifier.py         # Stage 1 + 2 (keyword + TF-IDF)
│   │   ├── semantic.py           # Stage 3 (MiniLM + head)
│   │   ├── entities.py           # DateTime + enum entity extraction
│   │   ├── context.py            # Session + context TTL management
│   │   └── manifest.py           # SHA-256 manifest generation + verification
│   ├── train.py                  # TF-IDF training pipeline
│   ├── train_semantic_head.py    # Semantic head training
│   ├── test_holdout.py           # End-to-end accuracy gate
│   └── test_sprint[1-3]_hardening.py  # Regression pin tests
└── docs/
    ├── development-history.md    # This project's full development chronicle
    └── tech-overview-slides.md   # This document
```
