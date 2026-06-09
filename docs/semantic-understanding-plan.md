# Semantic Understanding for On-Device NLU
## Design Document & Implementation Plan

**Project:** Hearing Aid App — On-Device NLU (Dialogflow Replacement)  
**Branch:** `feature/stt-intent-integration-adv`  
**Status:** Proposed — ready to implement  

---

## 1. The Problem We Are Solving

Our current NLU pipeline uses TF-IDF + Logistic Regression to classify user intent.
This works well when users say something close to what exists in training data.
It fails when users express the same idea using different words.

**Measured failure rate on novel phrasings (tested against current model):**

| User says | Expected intent | TF-IDF result | Confidence |
|---|---|---|---|
| "kill the sound" | Cmd.VolumeMute | Cmd.VolumeDecrease | 0.26 ❌ |
| "dim the audio" | Cmd.VolumeDecrease | Default Fallback | 0.21 ❌ |
| "log a jog" | Cmd.ActivityRun | Default Fallback | 0.17 ❌ |
| "I need some quiet" | Cmd.VolumeMute | Default Fallback | 0.44 ❌ |
| "switch listening profile" | Cmd.MemoryChange | Default Fallback | 0.32 ❌ |
| "cant hear well" | Cmd.VolumeIncrease | Default Fallback | 0.28 ❌ |
| "aid keeps cutting out" | Help_DeviceSettings | Default Fallback | 0.26 ❌ |

All of these are real, valid requests. All currently fall to GenAI fallback, which
requires a network connection. For a hearing aid that must work offline, this is a
direct failure of the product promise.

**Root cause:** TF-IDF is a bag-of-words model. It has no concept of meaning.
It learned that "turn it down" maps to VolumeDecrease because those exact words
appeared in training data. It does not know that "dim the audio" means the same thing,
because "dim" and "audio" never appeared together in training data for that intent.

No amount of tuning the TF-IDF threshold or retraining fixes this. It is a fundamental
architectural limitation.

---

## 2. Why Not Just Add More Training Phrases to TF-IDF?

This is the first question everyone asks. The answer is: it helps at the margins but
does not solve the problem.

**What adding more phrases to TF-IDF gives you:**
- Better coverage of specific phrasings you explicitly added
- Slightly higher accuracy on those exact phrases
- Marginal improvement on closely related variations

**What it does not give you:**
- Understanding of semantically similar phrases you did not think of
- Generalisation to future novel phrasings from real users in production
- Any ability to handle ASR output variations (when speech-to-text returns unexpected words)

**The mathematics of the problem:**

TF-IDF represents each training phrase as a sparse vector over a vocabulary of ~6,229
tokens. If a user says a word that is not in the vocabulary — "dim", "kill", "ping",
"jog" — the model assigns that word a weight of zero. The phrase becomes invisible
to the model regardless of how many other phrases you add.

You cannot add enough phrases to cover all the ways real users will speak. The English
language has ~171,000 words. A hearing aid user might say any of them. You have 7,720
phrases covering 1,508 unique words. That is 0.88% of the language.

**With more data you go from 0.88% to perhaps 2%. The remaining 98% still fails.**

Semantic understanding solves this at the architectural level. A sentence embedding
model trained on 1 billion sentence pairs already knows that "dim" is related to
"reduce" and "lower". You do not need to teach it this — it already knows.

---

## 3. What Semantic Understanding Means

A semantic embedding model converts any sentence into a fixed-size vector of numbers
(384 dimensions) that represents the *meaning* of the sentence, not the words.

Sentences with similar meaning produce vectors that are close together in this 384-
dimensional space, even if they share no words:

```
embed("turn it down")        → [0.12, -0.45, 0.33, ...]  ←─┐ close
embed("dim the audio")       → [0.14, -0.43, 0.31, ...]  ←─┘ (cosine sim: 0.91)

embed("how do I pair aids")  → [0.67,  0.21, -0.14, ...]  ←─┐ close
embed("help with pairing")   → [0.65,  0.19, -0.12, ...]  ←─┘ (cosine sim: 0.94)

embed("turn it down")        → [0.12, -0.45, 0.33, ...]  ←─┐ far apart
embed("how do I pair aids")  → [0.67,  0.21, -0.14, ...]  ←─┘ (cosine sim: 0.11)
```

This is not rule-based. The model learned the structure of language from 1 billion
sentence pairs. It knows "dim" and "reduce" are related because it has seen them used
in similar contexts billions of times.

---

## 4. The Two-Stage Architecture

We do not replace the TF-IDF model. We add semantic understanding as a safety net
that only activates when TF-IDF fails:

```
User utterance
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Keyword Pre-filter  (~0ms)                    │
│  Deterministic rules for exact phrases (mute, unmute)   │
└──────┬──────────────────────────────────────────────────┘
       │ no match
       ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2: TF-IDF + Logistic Regression  (~1ms)          │
│  Fast, handles 90%+ of known phrasings                  │
│                                                         │
│  confidence >= threshold ──────────────────────────────►│ FULFILL
│  confidence < threshold  ──────────────────────────────►│
└──────────────────────────────┬──────────────────────────┘
                               │ ~10% of turns
                               ▼
              ┌────────────────────────────────────────────┐
              │  Stage 3: Semantic Fallback  (~8-15ms)     │
              │                                            │
              │  1. Embed the utterance with bge-small     │
              │  2. Find nearest neighbour in the index    │
              │     of 7,720 pre-embedded training phrases │
              │  3. Return the intent of the nearest match │
              │                                            │
              │  similarity >= 0.82 ──────────────────────►│ FULFILL
              │  similarity < 0.82  ──────────────────────►│ GENAI fallback
              └────────────────────────────────────────────┘
```

**Key design decisions:**

- Stage 3 only runs when Stage 1 and 2 both fail (~10% of turns)
- 90% of turns pay zero extra cost — Stage 2 is fast enough
- The semantic model never replaces the classifier; it rescues it
- A second threshold (0.82) filters low-quality semantic matches — genuinely
  out-of-scope queries still fall to GenAI correctly

---

## 5. Model Selection: Why Not BERT, and Why bge-small

### Why BERT, RoBERTa, DistilBERT are wrong for this task

These models were designed for token-level tasks: text classification, question
answering, named entity recognition. They were pre-trained on masked language modelling
and produce one embedding per token, not per sentence.

To use them for semantic similarity you must:
1. Mean-pool all token embeddings to get a sentence vector
2. Fine-tune specifically on sentence-pair similarity

Without step 2, BERT cosine similarity is barely better than random for semantic
search — proven in the original SBERT paper (Reimers & Gurevych, 2019). This is the
reason the Sentence Transformers library exists.

Additionally, these models are too large for mobile:

| Model | Size | Inference (mobile) | Suitable for on-device |
|---|---|---|---|
| BERT-base | 440 MB | ~200ms | ❌ |
| RoBERTa-base | 500 MB | ~220ms | ❌ |
| DistilBERT | 265 MB | ~80ms | ❌ |
| MobileBERT | 100 MB | ~40ms | ❌ (wrong task) |

### The correct model family: Sentence Transformers

Models specifically trained via contrastive learning on sentence pairs. They optimise
the embedding space so that similar sentences land close together, dissimilar sentences
land far apart. This is exactly the property needed for nearest-neighbour intent search.

### Model comparison (MTEB benchmark — higher = better semantic quality)

| Model | Size (float32) | Size (INT8) | MTEB Score | Latency (iPhone 12) | Notes |
|---|---|---|---|---|---|
| paraphrase-MiniLM-L3-v2 | 17 MB | 5 MB | 48.1 | ~4ms | Ultra-fast, lower quality |
| **all-MiniLM-L6-v2** | **22 MB** | **6 MB** | **56.3** | **~8ms** | Good balance |
| all-MiniLM-L12-v2 | 33 MB | 9 MB | 59.8 | ~15ms | Better quality, same size as bge |
| **bge-small-en-v1.5** | **33 MB** | **9 MB** | **62.2** | **~12ms** | **Best quality at mobile size** |
| e5-small-v2 | 33 MB | 9 MB | 62.1 | ~12ms | Competitive with bge-small |
| all-mpnet-base-v2 | 438 MB | 110 MB | 69.6 | ~180ms | Best quality, desktop only |

MTEB scores are measured on general text (news, web, Wikipedia). For a **closed-domain
system** like a hearing aid app, the difference between MiniLM-L6 (56.3) and bge-small
(62.2) narrows — both models understand the domain well enough. The practical difference
in this use case is small.

### Why we recommend all-MiniLM-L6-v2 (INT8) over bge-small for this project

The recommendation changed after measuring the actual index size:

```
Embedding index: 7,720 phrases × 384 dimensions × 4 bytes = 11.3 MB
                 (or 5.7 MB with float16 storage)
```

Total on-device footprint:

| Model choice | Model | Index (float16) | Total |
|---|---|---|---|
| MiniLM-L6 INT8 | 6 MB | 5.7 MB | **~12 MB** |
| bge-small INT8 | 9 MB | 5.7 MB | **~15 MB** |
| MiniLM-L3 INT8 | 5 MB | 5.7 MB | **~11 MB** |

For a hearing aid app, 12 MB total for a semantically-aware offline NLU that handles
novel phrasings is a good tradeoff. bge-small gives slightly better quality at 3 MB
more. The team should make this call based on the app's binary size budget.

**Default recommendation: all-MiniLM-L6-v2 (INT8 quantised) at ~12 MB total.**
If binary size is not a constraint: bge-small-en-v1.5 (INT8) at ~15 MB.

---

## 6. Why We Have Enough Data Right Now

Earlier in this project we mentioned waiting for more data from the Dialogflow console.
That was incorrect for the semantic layer. Here is why:

**The embedding model is pre-trained.** It already understands language. We do not train
it. We use it as a feature extractor.

**The index is built from existing phrases.** We embed all 7,720 existing training
phrases once, offline, and store the result as a `.npz` file. This is not training.
It is pre-computation.

**The nearest-neighbour search at runtime** compares the query embedding against the
7,720 stored embeddings and returns the closest one. It needs no additional data.

**Our current dataset is sufficient:**
- 7,720 phrases across 60 intents
- 0 intents with fewer than 18 phrases
- Diverse phrasings already present (Dialogflow export contains crowd-sourced variants)

More data improves the **TF-IDF Stage 2** model (wider vocabulary coverage). It does
not change how the semantic layer works. If the team receives more phrases from the
Dialogflow console, we retrain Stage 2 — the semantic layer picks it up automatically
since it reads from the same training CSV.

---

## 7. What Changes in the Codebase

### New files

```
scripts/
├── build_semantic_index.py     # Run once after training — embeds all phrases
└── nlu/
    └── semantic.py             # SemanticFallback class

models/
└── semantic_index.npz          # Pre-computed embeddings (5.7 MB, float16)

# MiniLM ONNX model downloaded once:
models/
└── minilm-l6-v2.onnx           # 6 MB (INT8 quantised)
```

### Modified files

**`scripts/nlu/engine.py`** — 5 lines added to `_handle_new_intent()`:

```python
# Before (current):
if intent == "Default Fallback Intent" or conf < effective_threshold:
    return NLUResult(type="FALLBACK", ...)

# After:
if intent == "Default Fallback Intent" or conf < effective_threshold:
    sem_intent, sem_conf = self.semantic.classify(text)
    if sem_conf >= SEMANTIC_THRESHOLD:
        intent, conf = sem_intent, sem_conf
        cfg = self.intents.get(intent)
        if cfg:
            # continue to normal slot-filling / fulfillment path
            ...
    else:
        return NLUResult(type="FALLBACK", ...)
```

**`data/nlu_schema.json`** — no changes required.  
**`scripts/train.py`** — add one line: call `build_semantic_index.py` after training.

### No changes to:
- Entity extraction
- Slot filling
- Context management
- Session management
- iOS weights export

The semantic layer is a pure addition. It does not modify any existing behaviour.

---

## 8. Expected Improvement

Based on the novel phrasing test against the current model:

| Metric | Current (TF-IDF only) | After semantic layer |
|---|---|---|
| Known phrasings accuracy | 98.8% | 98.8% (unchanged) |
| Novel phrasing accuracy | ~30% (estimated) | ~85% (estimated) |
| Offline coverage | ~90% of intents | ~97% of intents |
| Avg latency (normal turn) | ~1ms | ~1ms (Stage 2 sufficient) |
| Avg latency (fallback turn) | ~1ms (→ GenAI) | ~9ms (→ semantic rescue) |
| Network dependency | Required for novel phrasings | Eliminated for novel phrasings |

The last row is the most important for a hearing aid app. Today, any novel phrasing
requires a network call to GenAI. After the semantic layer, the system handles it
on-device, silently, in 9ms.

---

## 9. Implementation Plan

### Phase 1 — Infrastructure (no ML, 1 day)
1. Download `all-MiniLM-L6-v2` ONNX INT8 from Hugging Face
2. Write `scripts/nlu/semantic.py` — `SemanticFallback` class with:
   - ONNX inference for embedding generation
   - Mean-pooling + L2 normalisation (same as the model's training)
   - Nearest-neighbour search over the index using cosine similarity
3. Write `scripts/build_semantic_index.py` — embed all training phrases, save `.npz`
4. Unit test: verify `cos_sim(embed(A), embed(B))` is high for paraphrase pairs

### Phase 2 — Integration (half day)
1. Wire `SemanticFallback` into `NLUEngine.__init__()`
2. Add Stage 3 rescue logic in `_handle_new_intent()` (5 lines)
3. Add `SEMANTIC_THRESHOLD = 0.82` to schema or engine constants
4. Add `NLUResult.semantic_rescue: bool` field for observability

### Phase 3 — Threshold Tuning (half day)
1. Run the novel phrasing test suite against the semantic layer
2. Test boundary cases: out-of-scope queries should still fall to GenAI
3. Adjust threshold if needed (0.80–0.85 range expected)
4. Add to test suite: verify known GenAI queries do not get rescued incorrectly

### Phase 4 — Mobile Export (1 day)
1. Verify ONNX tokeniser works in ONNX Runtime on Android
2. For iOS: verify Core ML conversion or ONNX Runtime CPU provider
3. Document the index format so KMM/iOS team can implement the search natively

### Future (after new data from team, if received)
- Retrain TF-IDF Stage 2 with additional phrases (improves Stage 1 coverage)
- Optional: contrastive fine-tune MiniLM on hearing aid phrase pairs
  (improves semantic quality for domain-specific language like "aids", "RIC", "ITC")

---

## 10. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Semantic threshold too low — rescues wrong intents | Medium | Test with out-of-scope queries; threshold at 0.82 is conservative |
| Index size too large for app binary budget | Low | Use float16 storage (5.7 MB); further reduce with phrase deduplication |
| ONNX tokeniser mismatch between Python and mobile | Medium | Validate on device early; use same tokeniser config (WordPiece, max 128 tokens) |
| MiniLM latency too high on older devices | Low | L3 variant (4ms) available as fallback; semantic only fires on ~10% of turns |
| Contrastive fine-tuning degrades general quality | Low | Only pursue fine-tuning after measuring base model quality in production |

---

## 11. Definition of Done

- [ ] `semantic.py` implemented and unit tested
- [ ] `build_semantic_index.py` generates valid `.npz` from training CSV
- [ ] Novel phrasing test suite: 8/10 previously-failing cases now correct
- [ ] Out-of-scope queries (weather, general knowledge) still fall to GenAI
- [ ] Latency on normal turns unchanged (<2ms)
- [ ] Latency on semantic rescue turns <20ms on target device
- [ ] All 21 existing NLU engine tests continue to pass
- [ ] Index and model size documented for mobile team
