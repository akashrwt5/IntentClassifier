# Production-Readiness Architecture Review
## On-Device NLU Engine — Dialogflow Replacement
**Reviewer:** Principal Conversational AI Architect (Dialogflow / Google NLU, 10+ years)  
**Date:** 2026-06-14  
**Branch:** `feature/Adv2/AddSemanticUnderstanding-2`  
**System:** Hearing-aid companion app, English-only, iOS + Android, ~60 intents, on-device cascade

---

## System Architecture Overview

```
User Utterance (text / ASR output)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ NLUEngine.handle(session_id, text)                  │
│  [A] Active yes/no CONFIRMATION context?            │
│  [B] Active slot-filling (pending_intent)?          │
│  [C] Back-reference pre-pass                        │
│  [D] Fresh intent classification                    │
└─────────────────────────────────────────────────────┘
         │ [D] fresh turn
         ▼
┌──────────────────────────────┐
│ Stage 1: Keyword Pre-filter  │  ~0ms, deterministic
│  schema "keyword_triggers"   │  returns intent @ 1.0
└──────────────────────────────┘
         │ no match
         ▼
┌──────────────────────────────┐
│ Stage 2: TF-IDF + LogReg     │  ~2-4ms, ONNX 4.1MB
│  ONNX + isotonic calibration │  conf ≥ 0.70 → FULFILL
└──────────────────────────────┘
         │ conf < threshold
         ▼
┌──────────────────────────────┐
│ Stage 3: MiniLM Semantic     │  ~8-15ms, ONNX 22MB
│  Frozen encoder + LR head   │  conf ≥ 0.55 → FULFILL
└──────────────────────────────┘
         │ conf < 0.55
         ▼
┌──────────────────────────────┐
│ Stage 4: GenAI Fallback      │  Network required
│  External HTTPS redirect     │
└──────────────────────────────┘
```

```
Slot Filling State Machine:
  ┌──────────────┐    intent classified    ┌────────────┐
  │   CLASSIFY   │──────────────────────►  │   PROMPT   │
  └──────────────┘                         └────────────┘
         ▲                                       │
         │ interrupt (conf ≥ 0.75)               │ user answers slot
         │                                       ▼
  ┌──────────────┐    all slots filled    ┌────────────┐
  │  INTERRUPT   │◄──────────────────────  │    FILL    │
  └──────────────┘                         └────────────┘
                                                 │
                                           ┌─────▼──────┐
                                           │   FULFILL  │
                                           └────────────┘
```

---

## 1. Semantic Understanding

### 1.1 Duplicated Semantic Threshold — Dual-Gate Logic `[High]`

`SEMANTIC_THRESHOLD = 0.55` exists in both `engine.py` and `semantic.py`. The threshold is applied
**twice**: once internally in `SemanticFallback.classify()`, once in `engine.py` after the call
returns. If the two values diverge during a hotfix, the engine silently behaves incorrectly with
no log or error.

**Why it's a problem:** Configuration drift between two files causes silent accuracy degradation.
Neither a test nor a log will reveal which threshold rejected a rescue.

**Impact:** Medium — manifests as an intermittent accuracy drop that is invisible without observability.

**Fix:** Remove `SEMANTIC_THRESHOLD` from `engine.py`. Pass it from the schema:
`SemanticFallback(threshold=schema.get("semantic_threshold", 0.55))`. Schema becomes single source
of truth.

---

### 1.2 Legacy 1-NN Path Silently Reactivates on Stale Artifact `[Medium]`

`semantic.py` contains a near-match fallback via `semantic_index.npz`. If that file exists on disk
from a prior training run with different data composition, the 1-NN path reactivates silently and
can produce mispredictions with cosine scores that bypass the main confidence gate.

**Why it's a problem:** An old index from a different training corpus produces silent mispredictions
with no warning. The file is checked-in to the repo and will always be present.

**Impact:** Low-Medium — depends on how different the stale index is from current data.

**Fix:** Remove the 1-NN path entirely (documented as deprecated). If retained, emit a startup
warning whenever the index is loaded. Add an explicit config flag to enable it.

---

### 1.3 No Out-of-Scope Class in Semantic Head `[High]`

`train_semantic_head.py` explicitly excludes `Default Fallback Intent` from training. The head
therefore has no learned rejection boundary. It will always return one of the 59 in-scope intents
regardless of how nonsensical the input is — rejection relies entirely on the softmax threshold.

**Why it's a problem:** True out-of-scope queries ("what is the weather", "call my doctor") can
breach the 0.55 threshold and trigger wrong hearing-aid actions. This was already observed: during
early training without `class_weight="balanced"`, "give me lottery numbers" → `reminders.add` at
confidence 1.00.

**Impact:** High — wrong actions on a medical device are user-visible and trust-damaging.

**Fix:** Curate a synthetic OOS corpus from SNIPS OOS, CLINC-150 OOS partition, plus domain-adjacent
queries. Train the head with this as an explicit `Default Fallback Intent` class.

---

### 1.4 Custom WordPiece Tokenizer Diverges from HuggingFace `[Medium]`

The hand-rolled WordPiece tokenizer in `semantic.py` has two concrete gaps vs the reference
implementation:

1. **No NFD/NFKD unicode normalization** — accented characters from ASR ("café", "naïve") produce
   unexpected tokenization
2. **CLS token included in mean-pool** — HuggingFace excludes special tokens from mean-pool.
   The ONNX path and the ST fallback path produce vectors in different subspaces

**Why it's a problem:** Subtle accuracy degradation on non-ASCII inputs. The ST fallback path
mismatch is a correctness issue if ever exercised.

**Impact:** Medium for non-ASCII; Low for typical English-only hearing-aid commands.

**Fix:** Remove the ST fallback path (not installable on mobile). Add a unit test that verifies
tokenization of 5 known inputs against the reference HuggingFace output.

---

## 2. Intent Classification

### 2.1 Keyword Pre-filter Returns Hardcoded 1.0 Confidence `[High]`

Every keyword match returns `1.0` regardless of match quality. Two concrete failure modes:

- `contains: ["translate"]` matches "I don't need to translate anything right now" (negation) at 1.0
- During slot-filling, any keyword match at 1.0 exceeds `INTERRUPT_THRESHOLD = 0.75` and abandons
  the active flow — even a negated or contextually irrelevant match

**Why it's a problem:** False positive commands executed with fabricated maximum confidence.
A user saying "the meeting sounds quiet" during reminder slot-filling triggers `Cmd.VolumeIncrease`
at 1.0 and loses all collected slots.

**Impact:** High — user-visible wrong actions and abandoned conversation flows.

**Fix:** Return calibrated confidence by match type:
- `exact` → 0.97
- `contains` → 0.85
- `regex` with `not_regex` → 0.90
- bare `regex` → 0.75

Add negation detection to `contains` rules.

---

### 2.2 Isotonic Calibration Unstable for Small Classes `[Medium]`

`CalibratedClassifierCV(method="isotonic", cv=3)` on classes with fewer than 60 samples (e.g.,
`Help_Battery: 18`, `Help_EdgeMode: 19`) will overfit the isotonic function, producing flat or
step-function probabilities near the decision boundary. This was already observed — `INTERRUPT_THRESHOLD`
was lowered from 0.85 → 0.75 "because calibrated probabilities are more moderate." That is
calibration failing, not working.

**Impact:** Medium — thresholds tuned against flattened calibrated outputs become wrong after any
retrain that changes class distribution.

**Fix:** Switch to `method="sigmoid"` (Platt scaling) which is stable with small samples. Add a
reliability diagram to training output.

---

### 2.3 Reported Holdout Accuracy Belongs to a Discarded Model `[Medium]`

`train.py` evaluates accuracy on the `X_train` split model, reports the number, then retrains on
all data `X` and exports that different model. The exported model's true holdout accuracy is never
measured. The quoted 83-86% figure may differ from the actual deployed model.

**Impact:** Medium — the production model's true accuracy is unknown.

**Fix:** After final `pipeline.fit(X, y)`, re-run holdout evaluation and log as
`FINAL MODEL HOLDOUT ACCURACY`. Record in `manifest.json`.

---

### 2.4 No Threshold Regression Gating `[Medium]`

No automated test asserts "at threshold T, precision on the holdout set is at least P."
Thresholds are hardcoded and manually tuned. They are never re-validated after a retrain.

**Impact:** Medium — a retrain that shifts calibration could silently push many utterances
across the wrong side of a threshold.

---

## 3. Entity Extraction

### 3.1 Datetime Parser is Timezone-Naive `[High]`

All datetime outputs are local naive `datetime` ISO strings. `datetime.now()` returns local time
with no `tzinfo`. A traveling user would get reminders that fire at the wrong time.

Two crashable edge cases:
- `"quarter to thirteen"` — hour out of range 1-12, no guard, produces nonsensical result
- `"0 to 3"` → `minute = 60` → `datetime.replace(minute=60)` raises uncaught `ValueError`
  that propagates to the caller

**Impact:** High for timezone (wrong reminder times for travelers); Medium for edge cases
(crashes on malformed ASR output).

**Fix:** Use `datetime.now(tz=timezone.utc)` and store UTC ISO 8601 with `+00:00` suffix.
Wrap `datetime.replace()` in `try/except ValueError`. Validate hour is in range 1-12 before
computing quarter/ten-to logic.

---

### 3.2 Fuzzy Match Fires on Short Common Words `[Medium]`

For `syn = "pub"` (length 3), `limit = max(1, round(0.9)) = 1`. This means "pub" fuzzy-matches
any 3-letter token with edit distance ≤ 1: "hub", "sub", "rub", "pun", "pug". A user saying
"I need to rub my eyes" during a memory-change flow would extract `Memory=Pub`.

**Impact:** Medium — incorrect memory selection on common English words that happen to be
close to a memory name.

**Fix:** Set minimum token length for fuzzy matching to 5 characters. Or restrict fuzzy matching
to explicitly prompted slots only (not full-sentence entity scanning).

---

### 3.3 `strip_datetime` Over-Strips Prepositions from Reminder Topics `[Low]`

`strip_datetime` unconditionally removes `\bat\b`. "Remind me to be at the dentist" strips "at"
and produces topic "be the dentist."

**Impact:** Low — affects reminder name quality, not correctness of intent or action.

**Fix:** Only strip `at`/`on` when followed by a time/date expression: use `\bat\s+\d` instead
of `\bat\b`.

---

## 4. Dialogue Management

### 4.1 Context Lifespan is Turn-Based Only — No Time Expiry `[High]`

`SessionStore` has no `created_at` timestamp. A session stored from hours or days ago still has
active contexts with full lifespan. A confirmation context alive from a previous conversation
could fire `Cmd.SendMessage - yes` on any affirmative utterance in a new session — sending a
message the user never intended.

**Why it's a problem:** Stale context causing silent wrong actions in a new conversation.
The `Cmd.SendMessage` confirmation case is particularly dangerous.

**Impact:** High — user-visible wrong action with no indication of why it happened.

**Fix:** Add `created_at: float` (Unix timestamp) and `ttl_seconds: int` to each context.
Expire contexts older than TTL regardless of lifespan count. A reasonable TTL for confirmation
is 30 seconds. Auto-reset sessions idle more than N minutes.

---

### 4.2 Keyword 1.0 Confidence Always Triggers Slot Interrupt `[High]`

Compound with Issue 2.1. Interrupt detection re-uses `classifier.classify()` which includes the
keyword pre-filter. Any keyword match during slot-filling exceeds `INTERRUPT_THRESHOLD = 0.75`
at hardcoded 1.0, abandoning the active flow and discarding collected slots.

**Impact:** High — data loss of collected slots mid-conversation.

---

### 4.3 No Maximum Slot-Fill Attempt Limit `[High]`

If datetime parsing fails repeatedly (ASR clarity challenge, unusual phrasing), the engine asks
"When should I remind you?" indefinitely. There is no escape hatch other than the user triggering
an interrupt intent.

**Impact:** High — for hearing-aid users with ASR clarity challenges this is a real trap.

**Fix:** Add a `slot_fill_attempts` counter to session state. After 3 failed extractions for the
same slot, abandon the flow gracefully and route to GenAI.

---

### 4.4 "yes, no worries" Parsed as Negative `[Medium]`

`_yes_no()` applies "neg wins if both neg and pos." The word "no" in "no worries" matches `\bno\b`
→ `neg=True`. Combined with "yes" → `pos=True`, the neg-wins rule returns `False`. The user
expresses agreement; the engine cancels.

**Impact:** Medium — common English agreement idiom misclassified as cancellation.

**Fix:** Maintain an idiom exclusion list: "no worries", "no problem", "no doubt", "no way"
should not contribute negative polarity.

---

### 4.5 `_active_confirmation` is an O(n) Linear Scan `[Low]`

On every turn, `_active_confirmation` iterates all ~60 intents checking for a matching followup
context. Not a performance issue today, but duplicate context names in the schema (copy-paste
error) produce non-deterministic behaviour — whichever intent is first in dict iteration order wins.

**Fix:** Build a reverse index `{context_name: (intent_name, followup_cfg)}` at startup.
Assert uniqueness of context names during schema loading.

---

## 5. Offline Constraints

### 5.1 GenAI URL Hardcoded Placeholder, User Query Appended `[Critical]`

```python
self.genai_url = "https://genai.yourcompany.com/chat?query="
```

The user's raw utterance is URL-encoded and appended to this string, then returned in
`NLUResult.url`. This means:
- Any debug logging of `NLUResult.to_dict()` logs user utterances in plaintext
- HTTP server logs contain user utterances as URL query parameters
- The placeholder will fail in production unless replaced
- No graceful offline message when device has no network

**Impact:** Critical — GDPR and medical data risk. Hearing-aid users' medical-context utterances
("remind me to take my medication", "my doctor says") captured in logs.

**Fix:** Remove `url` from `NLUResult`. Return `action="genai.fallback"` only. Let the application
layer construct the GenAI navigation URL internally. Never pass raw utterance text outside the
NLU layer.

---

### 5.2 Synchronous 26MB Model Load; OOM Silently Disables Semantic Stage `[High]`

`_load_semantic()` is wrapped in `try/except Exception: return None`. On low-memory devices, the
MiniLM OOM failure is swallowed silently — the user gets no indication and accuracy degrades to
TF-IDF-only with no log or metric emitted.

**Impact:** High — silent accuracy degradation on entry-level devices with no diagnostic.

**Fix:** Load models on a background thread. Expose a `ready: bool` property. Emit a diagnostic
event when semantic stage is disabled. Measure and log cold-start latency.

---

### 5.3 No ONNX Warm-Up Run `[Medium]`

ONNX Runtime requires a warm-up inference (JIT compilation, memory allocation) before reaching
peak performance. The first user utterance experiences a 3-5x latency spike — every cold-start
of the app.

**Fix:** Run one embed of a dummy string in `SemanticFallback.__init__()` after loading the
ONNX session.

---

### 5.4 Vocab Cache is Class-Level, Not Instance-Scoped `[Medium]`

`SemanticFallback._vocab_cache` is a class variable. Two instances with different vocab paths
share the same cache — the first-loaded vocab wins silently. The `_VOCAB_PATH` is also a class
constant, ignoring the `model_path` parameter passed to `__init__`.

**Fix:** Move `_vocab_cache` and `_VOCAB_PATH` to instance scope in `__init__`. Accept a
`vocab_path` parameter.

---

## 6. Production Readiness

### 6.1 Zero Observability — No Logs, Metrics, or Tracing `[Critical]`

The entire codebase contains zero `import logging`, zero metrics emission, zero tracing.
In production there is no way to:
- Measure real-world keyword / TF-IDF / semantic / GenAI hit rates
- Detect threshold drift after a retrain
- Reconstruct a user bug report
- Measure per-stage latency in the field

This is the single largest gap vs Dialogflow, which provides full conversation analytics,
intent hit rates, and fallback analysis out of the box.

**Fix:**
```python
logger.info("nlu.classify", extra={
    "session_id": session_id,
    "stage": "keyword|tfidf|semantic|genai",
    "intent": intent,
    "confidence": conf,
    "latency_ms": elapsed,
    "semantic_rescue": bool,
})
```
Add structured logging to every stage decision in `engine.py`.

---

### 6.2 Manual Training Scripts, No CI/CD, No Accuracy Gate `[High]`

Training completes and exports even if holdout accuracy drops to 0%. There is no:
- Automated training pipeline triggered by data commits
- Accuracy regression gate
- Model versioning or artifact lineage
- Blue/green deployment capability

**Fix:** Add to `train.py`:
```python
if holdout_acc < MINIMUM_HOLDOUT_ACCURACY:
    raise RuntimeError(f"Accuracy {holdout_acc:.2f} below minimum {MINIMUM_HOLDOUT_ACCURACY}")
```
Record holdout accuracy in `manifest.json`. Add a CI workflow triggered on data commits.

---

### 6.3 Semantic Artifacts Excluded from Manifest `[High]`

`TRACKED_FILES` in `manifest.py` covers only `intent_model.onnx`, `intent_labels.*`,
`intent_pipeline.pkl`. The 22MB `minilm-l6-v2.onnx`, `semantic_head.json`, `semantic_head.npz`,
and `minilm-vocab.txt` are not integrity-verified at startup. A tampered or corrupted MiniLM
model is undetected.

**Fix:** Add all four semantic artifacts to `TRACKED_FILES`. Have `train_semantic_head.py` call
`generate_manifest()` at the end of its run.

---

### 6.4 Missing Labels File Silently Skips Parity Check `[Medium]`

```python
if not labels_path.exists():
    return  # skip during development
```

In production, if `intent_labels.json` is missing (update failure, storage permission error),
the engine starts with no label-schema parity check. Intent index mismatches produce wrong
intents silently with no error surfaced.

**Fix:** In production builds (via env var or build flag), raise `RuntimeError` on missing labels
file rather than silently returning.

---

## 7. Failure Analysis

### 7.1 Slot-Fill Has No Maximum Attempt Limit — Users Can Be Trapped `[High]`

Already covered in 4.3. Worth calling out separately as a failure mode: a user mid-slot-fill
for `reminders.add` whose ASR consistently produces garbled datetime text will receive
"When should I remind you?" indefinitely with no recovery path.

---

### 7.2 Stale Back-Reference Resolves Without Recency Check `[Medium]`

`_try_back_reference` uses `session.prev_memory` from a potentially day-old session. "Change back"
could silently switch to yesterday's memory preset without confirmation or recency warning.

---

### 7.3 Session State Not Multi-Process Safe `[Low/High]`

`SessionStore` is in-process. On multi-worker deployment (Gunicorn, isolates), sessions written
by worker A are invisible to worker B. Benign on-device; critical if ever deployed as a shared
service.

---

### 7.4 GenAI Fallback Provides No Offline Message `[Medium]`

When the device is offline, the GenAI URL navigates to an error page with no explanation.
There is no `offline: bool` field in `NLUResult` and no graceful "I'm offline" response.

**Fix:** Check network reachability before returning a FALLBACK result. If offline, return a
canned response: "I'm not sure how to help with that right now."

---

## 8. Comparison Against Dialogflow

```
Capability                       Dialogflow CX         This System
──────────────────────────────────────────────────────────────────────
Intent resolution                ★★★★★ (LLM+ML)       ★★★☆☆ (TF-IDF+MiniLM)
Entity extraction                ★★★★☆ (system+custom) ★★★☆☆ (regex+fuzzy)
Dialogue state management        ★★★★★ (state machine) ★★☆☆☆ (no time-expiry)
Observability / analytics        ★★★★★ (built-in)      ★☆☆☆☆ (none)
Out-of-scope rejection           ★★★★☆ (trained OOS)   ★★☆☆☆ (threshold only)
Multilingual                     ★★★★★ (40+ langs)     ★☆☆☆☆ (EN only)
CI/CD & model lifecycle          ★★★★★ (managed)       ★☆☆☆☆ (manual scripts)
Security & privacy               ★★★★☆ (SOC2, GDPR)   ★★☆☆☆ (URL exposure)
Offline capability               ☆☆☆☆☆ (cloud-only)   ★★★★☆ (on-device)
Latency (on-device)              ★☆☆☆☆ (round-trip)   ★★★★☆ (~15ms)
Cost at scale                    ★★☆☆☆ (API cost)      ★★★★★ (zero marginal)
Customization                    ★★★★☆ (UI+API)        ★★★★★ (full code access)
Schema-driven keyword rules      ★★★☆☆ (synonym ents.) ★★★★☆ (declarative JSON)
```

**Features that match Dialogflow:** Intent classification, slot filling, back-reference, entity
extraction, affirmative/negative context, schema-driven configuration.

**Features missing vs Dialogflow:** Time-based context expiry, observability, CI/CD lifecycle,
out-of-scope training, multilingual, reliability diagrams, A/B testing, rollback.

**Features better than Dialogflow:** Fully offline operation, zero marginal cost at scale,
full code access, sub-20ms on-device latency, privacy (no utterances sent to cloud for
classification), schema-driven keyword pre-filter.

---

## 9. Security and Privacy

### 9.1 User Utterances in NLUResult Logged in Debug Builds `[Critical — GDPR/Medical]`

`NLUResult.url` contains the full user utterance URL-encoded. `to_dict()` includes `url` whenever
it is non-empty. Any debug logging of the result dict captures medical-context hearing-aid
utterances ("remind me to take my hearing aid medication", "my audiologist said") in plaintext
logs on the device and on any server receiving the GenAI request.

**Impact:** Critical — GDPR violation risk. Potential HIPAA adjacency for medical device context.

**Fix:** Remove `url` from `NLUResult` entirely (covered in 5.1). Never serialize raw user
utterances in NLU results.

---

### 9.2 Manifest Not Tamper-Proof `[Medium]`

`manifest.json` is written to the same `models/` directory as the artifacts it protects.
An attacker who can write a model file can also rewrite the manifest. Provides integrity against
accidental corruption only — not adversarial tampering. For a hearing-aid application, a tampered
NLU model that routes all commands to a malicious action is a supply-chain attack vector.

**Impact:** Medium for typical mobile threat model; High for medical device context.

**Fix:** Sign the manifest with a private key; embed the public key in the app binary.
Note: OTA model updates bypass app signing and need independent verification.

---

### 9.3 No Input Length Limit Before ONNX Inference `[Low]`

A 10,000-character input (adversarial or from a buggy ASR) iterates the full WordPiece loop
before truncation to 64 tokens. The tokenization loop itself runs the full string length.

**Fix:** Add `text = text[:500]` as the first line of `NLUEngine.handle()`. Log an audit event
if input is truncated.

---

## 10. Final Verdict

### Issue Severity Summary

| Severity | Count | Key Items |
|----------|-------|-----------|
| **Critical** | 3 | User data in URL/logs (×2 angles), zero observability |
| **High** | 11 | OOS class missing, keyword 1.0 confidence, session time-expiry, slot max attempts, semantic artifacts in manifest, datetime timezone-naive, OOM silent failure, slot interrupt via keyword |
| **Medium** | 12 | Duplicate threshold, calibration instability, fuzzy match short tokens, negation polarity, ONNX warmup, stale 1-NN, offline message, vocab cache scope |
| **Low** | 3 | strip_datetime prepositions, input length limit, O(n) context scan |

### Dimension Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Semantic Understanding | 6/10 | Cascade architecture is right. OOS class missing. Tokenizer gap vs HuggingFace. |
| Intent Classification | 5/10 | TF-IDF solid. Keyword 1.0 is dishonest. Calibration shaky for small classes. |
| Entity Extraction | 5/10 | Datetime parser impressive but timezone-naive and crashable on edge cases. Fuzzy match over-fires. |
| Dialogue Management | 4/10 | Slot filling functional. No time-based expiry. No max attempts. |
| Offline Constraints | 7/10 | Strongest area. All inference on-device. Correct fallback strategy. |
| Production Readiness | **2/10** | Zero observability. Manual training. Incomplete manifest. No CI. No accuracy gate. |
| Security & Privacy | **2/10** | Raw utterances in URL. No manifest signing. No input sanitization. |
| **Overall** | **4.5/10** | |

---

### Recommended Sprint Plan

#### Sprint 1 — Blockers before any beta release

| # | Action | Severity |
|---|--------|----------|
| 1 | Remove raw user utterance from `NLUResult.url` — never pass outside NLU layer | Critical |
| 2 | Add structured logging to all stage decisions in `engine.py` | Critical |
| 3 | Add time-based session/context expiry (TTL per context type) | High |
| 4 | Fix keyword pre-filter: negation guards on `contains` rules, honest confidence per match type | High |
| 5 | Add max slot-fill attempt limit (3 attempts → graceful abandon) | High |
| 6 | Add semantic artifacts to `TRACKED_FILES` in `manifest.py` | High |

#### Sprint 2 — Before general availability

| # | Action | Severity |
|---|--------|----------|
| 7 | Add OOS training class to semantic head (SNIPS/CLINC OOS corpus) | High |
| 8 | Fix datetime: UTC ISO 8601 throughout, guard edge cases | High |
| 9 | Add accuracy regression gate to `train.py` (`raise` if below minimum) | High |
| 10 | Async model loading with OOM diagnostic event | High |
| 11 | Single semantic threshold from schema; remove duplicate in `engine.py` | Medium |
| 12 | ONNX warm-up run in `SemanticFallback.__init__()` | Medium |
| 13 | Fix "yes, no worries" negative polarity via idiom exclusion list | Medium |
| 14 | Add graceful offline message when GenAI is unreachable | Medium |

#### Sprint 3 — Quality hardening

| # | Action |
|---|--------|
| 15 | CI/CD pipeline: training triggered by data commits, accuracy gate in CI |
| 16 | Calibration validation: reliability diagram, ECE metric logged post-train |
| 17 | Remove legacy 1-NN path from `semantic.py` or make it opt-in via config |
| 18 | Instance-scoped vocab cache in `SemanticFallback` |
| 19 | Record final holdout accuracy in `manifest.json` alongside checksums |
| 20 | Manifest signing with private key for OTA model update integrity |

---

### Summary

This is a technically sophisticated prototype that correctly solves the core problem: replacing a
cloud NLU service with an on-device cascade that runs at under 20ms with acceptable accuracy for
the defined 60-intent domain. The three-stage cascade architecture is sound. The schema-driven
keyword pre-filter is an elegant declarative approach. The MiniLM SetFit-style head is the right
tool for semantic rescue on a constrained device.

**The system is not production-ready in its current state.** The three Critical issues (user data
exposure via URL, zero observability) must be resolved before any user-facing release. The High
issues around keyword confidence inflation, stale sessions, missing OOS class, and semantic
artifacts outside the manifest represent correctness and reliability risks that would generate
user-visible failures at real-world usage rates.

The 83–86% holdout accuracy is a legitimate benchmark for the current corpus but is likely
overstated for the exported model (retrained on a superset) and would be lower on true
out-of-scope inputs where the semantic head has no learned rejection boundary.

With 4–6 weeks of focused engineering on Sprint 1 and Sprint 2 items above, this system would
be a credible, privacy-respecting, low-latency replacement for the defined domain.

**Go / No-Go: NO-GO for production. CONDITIONAL GO for closed beta with Sprint 1 items completed.**
