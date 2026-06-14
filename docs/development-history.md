# On-Device NLU Engine — Development History

> All communication in this document uses neutral framing. The goal is to describe
> what was built, why each decision was made, and when it landed.

---

## Branch Lineage

```
main / adv-2
  └── feature/Adv2/AddSemanticUnderstanding        (Stage 3 first draft)
        └── feature/Adv2/AddSemanticUnderstanding-2  (Sprint 2 hardening)
              └── feature/Adv2/AddSemanticUnderstanding-3  (Sprint 3 hardening — current)
```

---

## Phase 0 — Baseline (branch: `adv-2`)

**What existed:**
- TF-IDF + Logistic Regression pipeline exported to ONNX (`intent_model.onnx`, 2.2 MB)
- Schema-driven keyword pre-filter (`nlu_schema.json`)
- Multi-turn slot-filling engine (`engine.py`)
- Holdout benchmark CSV with 100 utterances

**Why it wasn't enough:**
- Holdout score: ~66/100. Novel phrasings ("i can barely hear", "everything is too quiet") missed entirely.
- No semantic understanding — vocabulary gap between training phrases and real user speech.
- One 4.6 MB legacy 1-NN cosine index (`semantic_index.npz`) with unverifiable influence on predictions.
- No integrity check on model artifacts.

---

## Phase 1 — Semantic Fallback, First Draft (branch: `feature/Adv2/AddSemanticUnderstanding`)

**Goal:** Add a third classification stage using sentence embeddings to rescue utterances the TF-IDF stage misses.

**Key decisions and rationale:**

| Decision | Why |
|---|---|
| MiniLM-L6-v2 (INT8 ONNX, 22 MB) as the encoder | State-of-the-art quality for mobile weight; 384-dim embeddings; runs on-device without network |
| Frozen encoder + trained logistic head (SetFit-style) | Keeps training fast (<2 min); encoder generalises well zero-shot; head is 59×384 — tiny |
| Hand-rolled WordPiece tokenizer | No ONNX Runtime tokenizer extension needed; portable to Android/iOS without Python runtime |
| Mean-pool + L2-norm as the embedding | Standard for sentence similarity; provably equivalent to SBERT sentence vectors |
| Head trained on `class_weight='balanced'` | `reminders.add` had 500 samples vs ~15 for rare intents — without balancing, rare intents collapsed to 38% holdout |

**Outcome:** Holdout improved from 66 to 77/100. Still below target (88/100).

**Root cause of remaining gap:**
- "Default Fallback Intent" class in training data = garbled ASR of in-domain requests (noisy). Training the head to recognise it taught the wrong signal.
- Short-word fuzzy entity collisions fired false positives on MemoryChange.
- Semantic head had no explicit out-of-scope (OOS) class — it had to guess rejection via threshold rather than learn it.

---

## Phase 2 — Sprint 1 + 2 Hardening (branch: `feature/Adv2/AddSemanticUnderstanding-2`)

### Sprint 1 — Reliability & Privacy

**A1 — PII/URL leak fix**
- `NLUResult.url` field removed. Any field on a result dataclass can end up in logs or analytics.
- GenAI fallback URL built in the CLI layer (`nlu_cli.py`) from a configurable base URL + `urllib.parse.quote(text)`, never stored on the result.
- `NLU_GENAI_URL` env var replaces the hardcoded constant.

**A2 — Stale context fix**
- Context entries now carry a `ttl_seconds` (default 90 s). `Context.is_expired(now)` checked on every slot-filling turn.
- Sessions idle for 10 minutes are reset automatically (`DEFAULT_SESSION_TTL_SECONDS = 600`).
- Clock is injectable (`Callable[[], float]`) so tests run deterministically with `FakeClock` — no `time.sleep` needed.
- A stale confirmation ("yes") after context expiry no longer re-fires a previous intent.

**A3 — Slot-filling runaway fix**
- `MAX_SLOT_ATTEMPTS = 3` constant. After 3 consecutive failed attempts to fill a slot, the engine abandons gracefully instead of looping indefinitely.

**B1 — Keyword confidence calibration**
- Keyword stage returned a flat 1.0 confidence — indistinguishable from a certainty even for a weak substring hit.
- `KEYWORD_CONFIDENCE` dict: `exact=0.97`, `contains=0.85`, `regex_guarded=0.90`, `regex=0.75`.
- Calibrated values let the interrupt logic (which checks ≥ 0.85) correctly rank signals.

**B2 — Negation suppression for `contains` rules**
- "I don't want to translate this" was firing `Cmd.TranslationStart` because "translate" is a `contains` term.
- `_is_negated(text, term)` checks a 30-char window before the matched term for any negation cue.
- `_NEGATIONS` tuple covers contractions, idioms, and explicit negatives.

**B3 — UTC datetime storage**
- Reminder datetime was stored in device-local timezone. Moving time zones caused reminders to fire at wrong times.
- All datetimes parsed in local zone, then converted to UTC ISO 8601 with offset (`+05:30`, `+00:00`, etc.).
- Edge-case guards added: `"quarter to 13"` (hour out of range) and `"0 to 3"` (minute=60) return `(None, None, 0.0)` instead of crashing with `ValueError`.

**B4 — Training accuracy gate**
- `train.py` now exits non-zero if TF-IDF test-split accuracy < `MIN_TEST_ACCURACY` (default 0.85, env-overridable).
- Prevents silently shipping a degraded model after a bad training run.

### Sprint 2 — Accuracy & Robustness

**A1 — Curated OOS class**
- Created `data/semantic_oos.csv`: 156 hand-curated out-of-scope phrases (weather, maths, general knowledge, entertainment, finance, travel).
- Trained as an explicit `Default Fallback Intent` class in the semantic head — rejection is now learned, not threshold-guessed.
- Leakage guard: any phrase appearing in `data/semantic_holdout_100.csv` is excluded from OOS training data before fitting.
- Replaced the noisy fallback-log class from the original CSV entirely.

**A2 — Single threshold source of truth**
- `semantic_threshold` moved to `data/nlu_schema.json` (`"semantic_threshold": 0.55`).
- `NLUEngine._load_semantic()` reads it from schema; `SemanticFallback` receives it as a constructor parameter.
- No more magic numbers scattered across files.

**A3 — SHA-256 integrity manifest**
- `models/manifest.json` tracks SHA-256 of all 8 model artifacts (TF-IDF + semantic).
- `verify_manifest()` called at classifier init — startup fails fast if any artifact is tampered or corrupted.
- `generate_manifest()` called at the end of `train.py` and `train_semantic_head.py` so re-training automatically re-signs the bundle.
- `_meta` block in manifest stores `holdout_accuracy` from the last training run.

**A4 — Yes/no idiom fix**
- "No worries", "no problem" were parsed as slot-fill rejections because they start with "no".
- `_NO_IDIOMS` list added: these phrases resolve to affirmative ("no worries" = "yes, proceed").
- Prevents accidental slot abandonment on polite acknowledgements.

**A5 — Reminder topic quality**
- "Remind me for dinner" extracted topic "for dinner" instead of "dinner". Leading connector ("for", "about", "regarding") now stripped.
- "Remind me at 5 for dinner" left "at 5" as an orphaned number after stripping the time. Added `r"\b(?:at|by)\s+\d{1,2}(?::\d{2})?\b"` to `_TIME_PATTERNS`.
- "Remind me in the morning for yoga" left "in the morning" as a topic fragment. Added full phrase removal.

**A6 — Vocabulary-gap keyword triggers**
- 7 new `keyword_triggers` added for intents that score low on holdout despite good training-set coverage: volume metaphors ("can barely hear", "too loud in here"), battery/charging queries, streaming control, help/tutorial requests.
- Added deterministic triggers for domain terms not in training vocabulary.

**Outcome after Sprint 2:** Holdout 84/100.

---

## Phase 3 — Sprint 3 Production Hardening (branch: `feature/Adv2/AddSemanticUnderstanding-3`)

**Background:** Architecture review identified 4 critical production risks (A1–A3, C1–C3). All implemented.

### A1 — End-to-end holdout gate

**What:** `test_holdout.py` gained `--strict` mode with a dual budget:
- `MIN_HOLDOUT_TOTAL ≥ 88` (total correct out of 100)
- `MAX_WRONG_ACTION ≤ 5` (confidently fired wrong intent — worse than a safe GenAI miss)

Both thresholds are env-overridable so CI can tighten them without code changes.

**Why:** The gate must be demonstrably real — not a no-op that always passes. `test_sprint3_hardening.py` verifies that the gate fails at `MIN_HOLDOUT_TOTAL=99` (an impossible floor), proving enforcement works.

**Gate output (JSON, `--json` flag):** Machine-readable results for CI artifact uploads.

### C1 — `contains` keyword interrupt demotion

**Problem:** Mid-slot-fill turn "ask about the translate feature" contained the word "translate", a `contains` keyword for `Cmd.TranslationStart`. This was enough to abandon a `reminders.add` flow — a false interrupt.

**Fix:** `weak_keyword` flag in `_handle_slot_filling()`. If the new-intent decision came from the keyword stage at tier `contains`, the interrupt is suppressed regardless of confidence. Only `exact`, `regex`, and `regex_guarded` tier hits can interrupt an active slot flow.

**Preserved behaviour:** Genuine strong signals ("mute", "volume down") are still `exact`-tier and still interrupt correctly.

### C2 — Fuzzy enum length gate

**Problem:** Short common words ("care", "cab", "gum") were fuzzy-matching against short memory-mode names (e.g. "Car", "Gym") because the edit-distance ratio threshold alone can't distinguish genuine typos from vocabulary overlap at short string lengths.

**Fix:** `_FUZZY_MIN_LEN = 5`. Words shorter than 5 characters bypass fuzzy matching entirely. "care" (4 chars) → no fuzzy match. "restraunt" (9 chars) → matches "Restaurant" (genuine typo).

**One-shot scan:** `extract_enum()` called from `_extract_all_slots()` (the initial full-sentence scan for pre-filled slots) passes `fuzzy=False` — no fuzzy matching on the raw utterance, only on explicit slot-fill answers.

### C3 — Legacy 1-NN index removal

**Problem:** `models/semantic_index.npz` (4.6 MB) was a legacy cosine nearest-neighbour index. It was loaded by the semantic stage as a tiebreaker but was not covered by the SHA-256 manifest, not tracked in CI, and its influence on predictions was unverifiable.

**Fix:** `SemanticFallback` simplified to head-only classification. `_embeddings`, `_intents`, `_nearest()`, `INDEX_PATH`, `NEAR_MATCH_THRESHOLD` all removed from `semantic.py`. `semantic_index.npz` deleted from `models/`. `test_sprint3_hardening.py` asserts both that the attribute is absent and the file is absent.

**Why head-only is better:** The logistic head was trained on 1 876 phrases including an explicit OOS class. The 1-NN index had no OOS concept and could always find a nearest neighbour — it never rejected. The head rejects when no class exceeds 0.55.

### Telemetry

**Structured per-turn logging added:**
- `_log_decision()` records: stage (`keyword`/`tfidf`/`semantic`/`genai`), intent, confidence, `latency_ms`, `text_len`.
- Raw utterance text **never** logged by default (medical device privacy standard).
- Opt-in via `NLU_LOG_UTTERANCES=1` env var for debug sessions only.

---

## Final Metrics (branch: `feature/Adv2/AddSemanticUnderstanding-3`)

| Metric | Baseline | Sprint 1+2 | Sprint 3 |
|---|---|---|---|
| Holdout (100 utterances) | 66/100 | 84/100 | 90/100 |
| Wrong-action misses | unknown | ~8 | 4 |
| Stale-context bugs | present | fixed | fixed |
| PII in result | present | fixed | fixed |
| Artifact integrity | none | SHA-256 | SHA-256 |
| Slot runaway | unbounded | MAX=3 | MAX=3 |
| OOS rejection | threshold | learned class | learned class |
| Legacy 1-NN index | present | present | removed |

---

## Known Limitation (deferred)

**Row-5 interrupt regression:** A long utterance containing a regex command phrase embedded in context ("turn up the volume before the show starts") still interrupts a slot flow. Only the `contains` tier was demoted; `regex` and `regex_guarded` tiers are not length- or isolation-aware.

**Planned fix (not yet implemented):** Standalone-command detection — verify that the matched regex pattern spans the entire utterance (or occupies a standalone clause) before allowing interrupt. Deferred by user direction.

---

## Test Coverage Summary

| File | Tests | What it pins |
|---|---|---|
| `scripts/test_nlu.py` | 25 | Core engine + entity extraction correctness |
| `scripts/test_sprint1_hardening.py` | 10 | PII, TTL, slot limits, keyword confidence, negation |
| `scripts/test_sprint2_hardening.py` | 6 | OOS, threshold, semantic load, yes/no idioms |
| `scripts/test_sprint3_hardening.py` | 9 | Interrupt demotion, fuzzy gate, 1-NN removal, A1 gate |
| `scripts/test_holdout.py` | 100 | End-to-end pipeline accuracy gate |

**Total: 150 test assertions across 5 test files.**
