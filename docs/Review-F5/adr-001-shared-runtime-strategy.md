# ADR-001: Shared Runtime Strategy for the On-Device NLU Platform
## Python + Swift + Kotlin vs. Configuration-Driven Native vs. Shared C++ vs. Shared Rust

**Status:** Proposed — for Technical Architecture Review Board decision
**Date:** 2026-07-13
**Deciders:** Platform lead, iOS lead, Android lead, ML lead
**Supersedes:** R2 in `docs/production-architecture-review-and-roadmap.md` (which recommended a Rust core without full justification — this document provides the analysis and *revises the timing* of that recommendation)

---

## Decision (summary, for readers who stop here)

1. **Do not introduce Rust or C++ today.** The duplication that justifies a shared runtime — the dialogue engine on two mobile platforms — has not happened yet. Introducing a fourth language before the third platform exists buys risk without retiring any.
2. **Adopt Option B immediately** (configuration-compiled logic + shrink the code that could ever need sharing). This work is prerequisite to *every* option, including staying native, and it halves the surface of the eventual decision.
3. **Commit to a decision trigger, not a technology:** the day the team commits to shipping **multi-turn dialogue on Android**, build the shared core rather than starting a third hand-port. At that trigger, **Rust with UniFFI is the recommended implementation** over C++, for reasons argued in Part 3. Per the current roadmap this trigger is roughly two quarters away — so this ADR is a *pre-decision* the team should ratify now and execute then.
4. **If Android is descoped, or mobile dialogue remains single-turn command execution, do not build a shared runtime at all.** Option B + parity testing is sufficient for a classification-only mobile surface indefinitely.

The primary benefit of a shared runtime is **correctness and maintainability, not performance**. Anyone selling this migration on speed is selling it wrong — Part 8 shows the runtime deltas are noise for this workload.

---

## Context

The platform is an on-device Dialogflow replacement for a hearing-aid companion app: a four-stage cascade (keyword rules → TF-IDF/LogReg → MiniLM semantic rescue → GenAI fallback) plus a dialogue engine (slot filling, confirmations, interruptions, back-references, datetime resolution). Reference implementation is Python (~3,100 lines across `scripts/nlu/`). iOS re-implements the classification path by hand in Swift from a JSON weights export. Android does not exist yet but is committed on the roadmap. The team is small (single-digit engineers), Python- and Swift-centric, shipping a medical-adjacent product where a confidently wrong action changes a hearing aid's state.

Forces in play: cross-platform behavioral consistency is a *product safety* property here, not an aesthetic one; the team cannot absorb a large new-technology tax; the dialogue engine is about to grow (universal verbs, corrections, timeout prompts, confirmation policies per the roadmap); and every line added to the Python engine today is a line someone must eventually port twice.

---

# Part 1 — Current Architecture Analysis

## 1.1 Component inventory by location

**Exists only in Python (reference implementation):**

| Component | File | Lines | Nature |
|---|---|---|---|
| Dialogue orchestration (routing, confirmation, slot filling, interrupts, back-refs) | `engine.py` | 706 | Pure business logic |
| Entity extraction (enum, fuzzy Levenshtein, open text) | `entities.py` | ~290 of 791 | Pure business logic |
| Date/time parsing + multilingual lexicon grammar | `entities.py` | ~500 of 791 | Pure business logic |
| Session/context store (TTLs, lifespans, fulfillment memory) | `context.py` | 137 | Pure business logic |
| Cascade orchestration + keyword rules + negation + calibrated thresholds | `classifier.py` | 188 | Business logic wrapping inference |
| WordPiece tokenizer + mean-pool + L2 norm | `semantic.py` | ~120 of 213 | Pure business logic (pre/post-processing) |
| Manifest verification | `manifest.py` | 89 | Business logic |

**Duplicated on iOS today (Swift, in the STT repo):**

| Component | Source of truth | iOS copy |
|---|---|---|
| Text normalization (lowercase/strip) | `train.py` / `classifier.py` | Swift scorer |
| Tokenization + bigram construction | sklearn `TfidfVectorizer` internals | Swift scorer |
| Sublinear TF + IDF scaling + L2 normalization | sklearn internals | Swift scorer (from `intent_classifier_weights.json`: vocab 1,370 entries, idf, coef, intercept) |
| Temperature softmax + confidence/gap thresholds | `classifier.py` | Swift scorer |
| WordPiece tokenizer + pooling for MiniLM | `semantic.py` | Swift (feeding the CoreML embedder) |
| Golden-fixture parity harness | `test_ios_conformance.py` + CoreML fixtures | XCTest side |

**Will need duplication on Android (currently zero code):** everything in the iOS list, plus — the moment Android ships multi-turn — the entire "exists only in Python" table above.

**Pure ML inference (correctly platform-specific):** TF-IDF LogReg matmul (ONNX / CoreML / hand-rolled), MiniLM encoder forward pass (ONNX Runtime / CoreML-ANE). These are matrix operations behind stable tensor interfaces; platform runtimes are the *right* place for them.

**Correctly platform-specific (never share):** speech recognition (SFSpeechRecognizer / SpeechAnalyzer vs. Android SpeechRecognizer), audio session management, Bluetooth/hearing-aid control, UI, app lifecycle, model download plumbing, keychain/keystore.

## 1.2 Duplication today vs. duplication at maturity

```
                     TODAY                          AT PLATFORM MATURITY (roadmap items shipped)
              ┌──────────────────┐            ┌────────────────────────────────────────────┐
 Python       │ ████████████████ │ 3,100 ln   │ ██████████████████████████ │ ~5,000 ln     │
              │ (everything)     │            │ (everything, grown: verbs, corrections,    │
              │                  │            │  timeouts, workflow interpreter, policies) │
 Swift        │ ███              │ ~400 ln    │ ██████████████████████████ │ ~4,000 ln     │
              │ (classifier path)│            │ (full dialogue engine port)                │
 Kotlin       │                  │ 0 ln       │ ██████████████████████████ │ ~4,000 ln     │
              └──────────────────┘            └────────────────────────────────────────────┘
 Duplication factor: ~1.13×                    Duplication factor: ~2.6× — every behavior
 (one small hot spot)                          exists three times, evolving in three repos
```

The honest reading of this picture cuts both ways:

- **Against acting now:** today's duplicated surface is small (~400 Swift lines) and covered by parity fixtures. The catastrophe is *projected*, not present.
- **For deciding now:** the cheapest moment to choose a shared runtime is *before* the Swift dialogue port begins. Once 4,000 lines of Swift dialogue code exist, the shared runtime must beat not just "port to Kotlin" but "throw away working Swift" — a much worse trade. The decision window is now; the *execution* window is at the Android trigger.

## 1.3 Evidence that duplication already bites — within a single language

`scripts/export_ios_weights.py` contains an embedded fallback trainer (used when `intent_pipeline.pkl` is absent) whose hyperparameters have **already drifted** from `train.py`: `min_df=1` vs. `min_df=2`, and random per-class sampling (`g.sample(..., random_state=42)`) vs. the deterministic keep-last cap `train.py` adopted after review finding P2-2. Two copies of the same logic, same repo, same language, maintained by the same people, diverged within weeks. This is the base rate for logic duplication; three languages and three repos will not beat it.

---

# Part 2 — The Architectural Problem, Concretely

For each behavior below: what triple-implementation looks like, the bug class it breeds, and why parity testing does not save you.

### 2.1 Keyword matching (32 rules, 4 tiers, `not_regex` guards)

Rules like `\btranslat(e|es|ing|ion)\b` guarded by a 200-character `not_regex` must execute identically in Python `re`, `NSRegularExpression`/Swift `Regex`, and `java.util.regex`. These are **three regex dialects**: different Unicode word-boundary semantics (`\b` against `é` differs between ICU and Python), different default case-folding, different handling of `(?i)` inline flags. *Bug class:* a French utterance matches the guard on Android but not iOS; one platform launches translation, the other asks a clarifying question. *Why parity tests miss it:* the fixture corpus contains the utterances someone thought of; regex-dialect divergence lives precisely in the inputs nobody thought of (accented characters, emoji-adjacent tokens, unusual whitespace).

### 2.2 Carrier-phrase stripping and topic derivation

`_derive_topic()` applies 6 ordered carrier regexes, then datetime-span stripping, then leading-connector removal — *order matters*. "Remind me at 9pm for dinner" → strip carrier → strip time → strip "for" → "dinner". *Bug class:* a platform that strips connectors before datetime produces reminder topics like "at 9pm dinner" — user-visible data corruption stored on the device. *Why parity tests miss it:* the pipeline is order-sensitive across three transformation stages; the combinatorial input space (carrier × time-format × connector × language) is far larger than any fixture set, and failures are silent (a slightly wrong reminder title throws no error).

### 2.3 Negation detection

`_is_negated()` scans a 30-**character** window before a matched term for 9 negation cues. Three implementations must agree on: character vs. UTF-8-byte vs. UTF-16-code-unit windows (Swift `String` indexing and Java `String.substring` disagree with Python slicing on non-ASCII text), substring vs. word-boundary matching of cues. *Bug class:* "don't translate this" fires translation on exactly one platform for German text where the 30-unit window lands differently. *Why parity misses it:* the window boundary bug only triggers when a multibyte character straddles position 30 — a fixture would need to be constructed adversarially by someone who already knew the bug existed.

### 2.4 Confidence thresholding and calibration policy

Not one threshold — an interacting policy web: base 0.70, slot-intent 0.60, semantic 0.55, agreement-gate 0.50, interrupt 0.75, keyword-tier confidences (0.97/0.90/0.85/0.75), temperature division before softmax, confidence-gap check (weights JSON carries `conf_gap_threshold: 0.20` — a rule that exists on iOS and **does not exist in the Python engine at all**: divergence already shipped). *Bug class:* platforms disagree about whether 0.68 executes or falls back; A/B metrics become uninterpretable because "fallback rate" measures different policies per platform. *Why parity misses it:* parity fixtures assert (intent, probability) pairs; they do not assert *policy decisions* across the full decision tree, and the tree's edge cases (exactly-at-threshold, agreement-gate-but-different-intents) are unlikely to be sampled.

### 2.5 Dialogue state machine

Routing priority (confirmation → slot filling → back-reference → fresh classify), context TTL sweep before routing, lifespan decrement on fresh turns only. *Bug class:* one platform decrements lifespans on every turn including slot answers; contexts die one turn early; a confirmation asked after a slot answer silently stops working — the nastiest kind of bug: stateful, timing-dependent, unreproducible from a single-utterance report. *Why parity misses it:* single-turn parity fixtures cannot express this at all; conversation-level parity suites help but must enumerate state × transition × timing combinations in three test harnesses, and the harnesses themselves (fake clocks, session injection) are triplicated infrastructure.

### 2.6 Slot filling

Interrupt check with weak-keyword demotion → awaited-slot resolution (fuzzy on, open-entity verbatim capture) → opportunistic other-slot fill (fuzzy **off**, skip-awaited to avoid datetime double-advance) → attempt accounting → budget abandonment. The fuzzy-on/off asymmetry and the skip-awaited rule are exactly the subtleties a porter flattens. *Bug class:* Android enables fuzzy matching in the full-sentence scan; "care" matches memory "Car"; a hearing-aid memory change fires from an unrelated sentence — the wrong-action class the whole calibration effort exists to prevent. *Why parity misses it:* the behavior difference appears only for near-miss vocabulary under a specific scan mode; fixtures written from the spec test the spec, not the porter's misreading of it.

### 2.7 Context management

TTL expiry (90s contexts, 600s sessions), injectable clocks, partial-datetime parking. *Bug class:* one platform checks TTL at context *use* rather than turn *start*; a stale "yes" fires a confirmed action minutes after the user walked away. This exact bug was already found and fixed once in Python (Sprint-1 A2). Under triplication, it gets refound and refixed per platform, on production users. *Why parity misses it:* time-dependent behavior requires clock-injection test infrastructure per platform; drift in the *test harnesses* hides drift in the logic.

### 2.8 Entity extraction and fuzzy matching

Levenshtein with `_FUZZY_MIN_LEN = 5` gate, accent-aware word boundaries (`(?<![0-9A-Za-zÀ-ÿ])`), synonym canonicalization, open-entity verbatim rule (store what the user said, not the English canonical — the fr/de/da correctness fix). *Bug class:* Android ports miss the verbatim rule; French users get reminders titled "Take Medication" instead of "prendre des médicaments". Already-fixed bug, re-introduced by porting. *Why parity misses it:* per-language, per-entity fixtures at the needed density constitute thousands of cases × three harnesses; maintaining that costs more than sharing the code.

### 2.9 Date/time parsing — the worst offender

~500 lines: future-hour disambiguation, UTC conversion with offsets, day-parking + later-time anchoring ("tomorrow" … "3pm" → tomorrow 15:00), explicit-day-wins anti-double-advance logic, per-language lexicons, out-of-range guards ("quarter to 13"). The team *already* maintains cross-language parity CSVs (`tests/datetime_parity/`) because they know this drifts. *Bug class:* reminders 12 hours or one day off — silent, user-facing, trust-destroying. *Why parity misses it:* the input space is a natural-language grammar × clock-state product; the parking/anchoring logic depends on *session state and wall-clock time*, which fixture CSVs freeze artificially. Three implementations of a stateful grammar cannot be exhaustively cross-checked; they can only be replaced by one implementation.

### 2.10 Interruption, confirmation, workflow execution

Interrupt = classifier tier + threshold + weak-keyword demotion + schema lookup, mid-slot-flow. Confirmation = yes/no lexicons + `_NO_IDIOMS` ("no worries" is a *yes*) + uncertainty phrases → re-prompt. Workflow = the growing schema interpreter (validation, clarification, completion, confirmation policies). Each is a policy engine whose rules the roadmap actively grows. *Bug class:* every schema feature added in Python is a feature Android doesn't have until someone ports it — platforms drift not only in bugs but in *capabilities*, and the content team can no longer write one intent definition that behaves identically everywhere, which breaks the "schema is the platform" strategy at its root.

### 2.11 Why parity testing alone is insufficient — the general argument

1. **Parity tests sample; implementations diverge in the unsampled space.** Fixtures encode known inputs. Dialect differences (regex, string indexing, locale case-folding) live in unknown inputs by construction.
2. **State explodes the fixture space.** Single-turn fixtures cannot cover TTL × interrupt × parking × attempt-budget interactions. Conversation fixtures across three harnesses are themselves triplicated code that drifts.
3. **Parity tests detect divergence after it is written, at CI time — the cost is already paid.** The engineer wrote the wrong port, CI caught it (if sampled), someone debugs across two languages. A shared runtime prevents the write, not just the ship. And divergence *between* CI runs (iOS parity currently blocked on a missing PAT secret — it has not been running) demonstrates that parity infrastructure itself decays.
4. **Parity requires an oracle that is itself moving.** Python is the oracle; the roadmap changes Python weekly. Every oracle change invalidates fixtures across two other repos with their own release cadences. The synchronization cost grows with feature velocity — precisely when the team can least afford it.
5. **Empirical base rate:** two copies in one Python repo drifted (§1.3). `conf_gap_threshold` exists on iOS and not in the Python engine (§2.4). The `Help.X` vs `Help_X` duplicate content in `Engage.zip` drifted. This team — a careful one — has a measured drift rate above zero at N=2 implementations. N=3 with a bigger surface will be worse.

---

# Part 3 — Options Evaluation

### Option A — Continue Python + Swift + Kotlin (three hand-written implementations)

| Dimension | Assessment |
|---|---|
| Development complexity | Low per-platform, familiar; **O(3×) per feature** forever |
| Learning curve | None |
| Performance | Best-possible per platform (native, no FFI) — though Part 8 shows this doesn't matter here |
| Battery | Equivalent to any option |
| Maintainability | Poor at scale: every dialogue feature ×3, every bug fix ×3, three test harnesses |
| Debugging | Best-in-class per platform (Xcode/Android Studio native); but cross-platform *divergence* debugging is the worst of any option — comparing Swift and Kotlin implementations line-by-line |
| Testing | Parity suites required forever, growing superlinearly with features (§2.11) |
| CI/CD | Simple per repo; complex in aggregate (fixture synchronization across 3 repos, cross-repo PATs — already failing) |
| Long-term scalability | The team becomes a porting organization; feature velocity divides by 3 |
| Mobile integration | Trivial (it *is* the mobile code) |
| Developer productivity | High initially, decaying as parity surface grows |
| Future extensibility | Every roadmap feature (universal verbs, corrections, workflow interpreter) triples in cost |

**Honest case for A:** if mobile scope stays *classification-only* (single-turn commands, no on-device dialogue), the duplicated surface stays ~400 lines/platform and A is genuinely fine — golden fixtures cover a small pure-function surface well. A is not a strawman; it is the right answer for a smaller ambition than the roadmap states.

### Option B — Configuration-driven architecture; generate artifacts; keep native runtimes

Move every rule that *can* be data into compiled bundle tables: keyword rules compiled to a portable matcher spec (or RE2-subset with a conformance-tested feature whitelist), thresholds/policies as data, yes/no + idiom lexicons as data, carrier phrases as ordered pattern tables, datetime as declarative grammar tables, dialogue behaviors as schema flags. Native platforms implement *interpreters* for these tables.

| Dimension | Assessment |
|---|---|
| Development complexity | Medium: compiler + table formats + N interpreters |
| Learning curve | None (existing languages) |
| Performance | Native; table interpretation overhead negligible |
| Maintainability | Much better than A for *rules*; unchanged for *algorithms* (state machine, fuzzy match, datetime resolution engine, slot-fill orchestration are not expressible as data without inventing a DSL — see below) |
| Testing | Table semantics testable once against the spec; interpreter parity still needed but over a **much smaller, stable** surface |
| CI/CD | Bundle pipeline (already planned); moderate |
| Long-term scalability | Good for content velocity; the interpreters themselves still triplicate and drift |
| Future extensibility | New *rules* free; new *behaviors* still ×3 |

**The DSL ceiling:** pushed to its limit, Option B turns the dialogue engine into an interpreter for an ever-richer behavior DSL — at which point you have invented a programming language executed by three independently written interpreters, which is the original problem wearing a costume. Config-driven design is *necessary* (it shrinks and stabilizes the shareable core) but *not sufficient* (interpreters for stateful orchestration still drift).

**Verdict:** do Option B unconditionally — it is prerequisite work for every other option and pure win for the content pipeline. It is not, alone, a complete answer if multi-turn dialogue ships to two mobile platforms.

### Option C — Shared C++ runtime

One C++17/20 core (logic from Part 1's "Python-only" table), exposed via C ABI; Swift interops directly (Swift/C++ interop or ObjC++ shim), Kotlin via JNI; Python via pybind11 for the reference path.

| Dimension | Assessment |
|---|---|
| Development complexity | High: manual binding layers both sides (JNI is notoriously error-prone boilerplate), CMake/NDK toolchains, dependency management (no standard package manager) |
| Learning curve | Significant for a Python/Swift team; C++ proficiency ≠ C++ *safety* proficiency |
| Performance | Excellent; identical to Rust for this workload |
| Battery | Equivalent |
| Maintainability | Core is single-source (good); binding code is hand-maintained hazard surface; UB and memory bugs are a new *class* of defect the team doesn't currently have |
| Debugging | LLDB works on iOS; Android NDK debugging is workable but clunky; memory corruption bugs debug in days, not hours |
| Testing | One core suite (good); needs ASan/UBSan/valgrind discipline in CI; fuzzing possible (libFuzzer) |
| CI/CD | Cross-compilation matrix (arm64-ios, arm64/x86_64-android, host) + sanitizer jobs |
| Long-term scalability | Good; hiring for "careful C++" is harder than the market suggests |
| Mobile integration | Mature — this is how SQLite, ICU, Realm, and most codecs ship; the most proven path in existence |
| Developer productivity | Lower than Rust for a greenfield core: manual memory discipline, manual bindings |
| Future extensibility | Fine; ecosystem for NLP-adjacent needs (ICU, ONNX Runtime C API) is first-class |

**When C is right:** the team already has strong C++ engineers, or a hard dependency forces it (e.g., deep ICU integration, an existing C++ codebase to join). Neither holds here.

### Option D — Shared Rust runtime

One Rust core; **UniFFI** generates Swift and Kotlin bindings from one interface definition; PyO3 (or the same UniFFI Python target) for the reference/test path; models stay in platform runtimes (CoreML / ONNX Runtime Mobile) behind an inference-callback trait.

| Dimension | Assessment |
|---|---|
| Development complexity | Medium-high initially: cargo + cross-compilation (`cargo-ndk`, Xcode framework packaging) is genuinely simpler than the C++ equivalent; UniFFI removes the binding-authorship problem that dominates Option C |
| Learning curve | **The main cost.** Real: ownership/borrowing takes weeks to months. Mitigated: the core is small (~4–5k lines), synchronous, no lifetimes-heavy async, mostly string processing + state machines — the easy 60% of Rust |
| Performance | Excellent; indistinguishable from C++ here |
| Battery | Equivalent |
| Maintainability | Best of all options: single source, memory-safe by construction (no new UB defect class), exhaustive-match on state machines catches missed transitions at compile time — directly relevant to §2.5's bug class |
| Debugging | Good on iOS (LLDB understands Rust); adequate via NDK on Android; *vastly* better than A for cross-platform issues because there is no cross-platform divergence to debug |
| Testing | One suite tests the shipped logic (not a reference approximation of it); `cargo test` + built-in fuzzing (`cargo-fuzz`) for tokenizer/datetime grammar; property-based testing (proptest) fits the parsing surface unusually well |
| CI/CD | Cross-compile matrix; simpler than C++ (cargo is deterministic; no sanitizer matrix needed for memory safety) |
| Long-term scalability | Strong; growing mobile-core precedent (major cross-platform apps ship Rust cores; UniFFI is production infrastructure at Mozilla and others) |
| Mobile integration | Good and improving; less battle-tested than C++'s decades, more than adequate for a logic library with no exotic platform calls |
| Developer productivity | Lower for ~1 quarter, then *higher* than A (one implementation) and higher than C (compiler catches what code review would have to) |
| Future extensibility | Excellent: candle/ort crates if inference ever moves in-core; wasm target gives the Training Studio the *shipped* engine in-process for free (a concrete synergy: the Studio's testing console runs the real runtime, not a Python approximation) |

### Option E — Kotlin Multiplatform (KMP), for completeness

Share Kotlin logic; compile to Android natively and iOS via Kotlin/Native framework. Pros: Android team's language; no FFI authorship; good iOS interop story and maturing fast. Cons for *this* team: the reference/training world is Python (KMP has no useful Python bridge, so the Python engine either remains a divergent oracle — reintroducing the drift problem at the reference layer — or is deleted and the training/eval pipeline loses its in-process engine); the team is currently Python+Swift-centric, so KMP is as much a new language as Rust for most members; Studio reuse (wasm) is weaker. KMP is the right answer for Kotlin-centric teams whose source of truth can *be* Kotlin. That is not this team. Not recommended here, but recorded so the Board sees it was weighed.

### Recommendation from Part 3

**B now; D at the Android-dialogue trigger; C only if the team's hiring reality turns C++-shaped; A permanently if mobile scope stays classification-only.** Rust beats C++ here on the two dimensions that dominate for a small team: binding generation (UniFFI eliminates the largest hand-maintained hazard surface of Option C) and the absence of a memory-unsafety defect class the team has no institutional muscle to manage. C++'s maturity advantage is real but buys nothing specific for a self-contained logic library with no exotic dependencies.

---

# Part 4 — Shared Runtime Architecture (executed at the trigger)

```
┌────────────────────────────── AUTHORING / TRAINING (Python — unchanged) ─────────────────┐
│  content/*.yaml ─► nlu_compiler ─► NLU Bundle (schema tables, entity tables, datetime     │
│  datasets (DVC) ─► trainers    ─►   grammar, models, calibration, manifest, signature)    │
│                     evaluate ──► report card                                               │
│  Python engine role CHANGES: from "reference implementation" to thin PyO3 wrapper over    │
│  nlu-core — training/eval/Studio call the SAME logic that ships. The oracle problem ends. │
└───────────────────────────────────────┬───────────────────────────────────────────────────┘
                                        │ bundle.nlu (signed)
                                        ▼
┌────────────────────────────── SHARED RUNTIME: nlu-core (Rust) ────────────────────────────┐
│                                                                                            │
│  BundleLoader (verify sig+manifest, mmap tables)                                           │
│  TextPipeline (normalize, tokenize, TF-IDF featurize, WordPiece)                           │
│  RuleEngine   (keyword tiers, negation, carrier stripping — table-driven from bundle)      │
│  Cascade      (stage orchestration, temperature calibration, policy thresholds)            │
│  EntityEngine (enum/synonym/fuzzy, datetime grammar interpreter)                           │
│  DialogueCore (state machine, slot filling, confirmations, interrupts, contexts, TTLs)     │
│  WorkflowInterpreter (schema-driven validation/clarification/completion)                   │
│  TelemetryEmitter (structured events, no raw text)                                         │
│                                                                                            │
│  trait InferenceBackend {  tfidf_logits(features) -> Vec<f32>;                             │
│                            embed(token_ids) -> Vec<f32>;  }        ◄── injected by host    │
└────────┬──────────────────────────────┬───────────────────────────────┬───────────────────┘
         │ UniFFI-generated Swift        │ UniFFI-generated Kotlin       │ PyO3 / UniFFI-Py
         ▼                               ▼                               ▼
┌──────────────────┐          ┌──────────────────────┐        ┌──────────────────────────┐
│ iOS shell        │          │ Android shell        │        │ Python shell             │
│ InferenceBackend │          │ InferenceBackend     │        │ InferenceBackend = ORT   │
│  = CoreML (ANE)  │          │  = ORT Mobile        │        │ (CI, eval, Studio dev,   │
│ SFSpeech / UI /  │          │    (+NNAPI/XNNPACK)  │        │  server-side if ever)    │
│ BT / lifecycle   │          │ SpeechRec / Compose /│        └──────────────────────────┘
│ BundleManager DL │          │ BT / lifecycle / DL  │
└──────────────────┘          └──────────────────────┘
```

The inference inversion is the load-bearing design choice: **the core never links an ML runtime.** Platforms keep their optimal engines (ANE via CoreML on iOS; ORT/NNAPI on Android; ORT in Python), the core stays a small dependency-free logic library, and model-format evolution never forces a core release.

---

# Part 5 — Responsibility Boundaries

| Responsibility | Layer | Why it lives there |
|---|---|---|
| Dataset prep, augmentation, training, calibration fitting, evaluation, export, bundle compile+sign | **Python** | ML ecosystem (sklearn, ONNX, coremltools) is Python-native; runs on dev machines/CI, never on device; velocity matters more than portability here |
| Studio train-runner, registry tooling | **Python** | Same ecosystem; desktop context |
| Intent cascade orchestration; keyword/negation/carrier rules; thresholds & policy engine | **Shared runtime** | Pure deterministic logic whose cross-platform *identity* is a product-safety requirement (Part 2) |
| Dialogue state machine, slot filling, confirmations, interrupts, context/TTL, conversation state | **Shared runtime** | Stateful business logic — the highest-drift-risk surface (§2.5–2.7); exhaustive-match state machines benefit most from a single compiled implementation |
| Entity extraction, datetime grammar, validation engine | **Shared runtime** | Grammar interpretation over bundle tables; §2.9 shows why one implementation |
| Workflow interpreter (schema execution) | **Shared runtime** | The schema contract must mean exactly one thing on every platform, or the content team's single-definition promise breaks |
| Telemetry *event generation* (schema, redaction guarantees) | **Shared runtime** | Privacy invariants (never embed raw text) enforced in one audited place |
| Telemetry *transport* (batching, upload, consent gating) | **Native** | Network stacks, background execution, consent UI are platform concerns |
| Bundle *verification & loading* (signature, manifest, mmap) | **Shared runtime** | Security check must be identical everywhere; one audit surface |
| Bundle *download & storage* | **Native** | URLSession/WorkManager, storage quotas, connectivity awareness |
| TF-IDF logit computation, MiniLM forward pass | **Native (behind trait)** | CoreML/ANE and ORT/NNAPI are platform-optimal; tensor-in/tensor-out is a stable, easily parity-tested contract (unlike the logic around it) |
| Speech recognition, audio session | **Native** | OS frameworks, permissions, hardware |
| Bluetooth / hearing-aid control, executed actions | **Native** | The runtime returns `action` keys; device control is the app's domain |
| UI (SwiftUI/Compose), app lifecycle, memory-pressure response | **Native** | Obviously |
| Session *persistence* encryption | **Native (keys) + core (serialization)** | Core defines the state blob + TTL semantics; platform keystores own key material |

Rule of thumb applied throughout: **if two platforms must agree on it, share it; if the OS owns it, keep it native; if it needs the ML ecosystem, keep it Python.**

---

# Part 6 — Runtime Interaction Flow

```
 1  Mic → platform ASR (SFSpeechRecognizer / Android SpeechRecognizer)     [native]
 2  ASR final text → NluSession.handle(text)                               [FFI call in]
 3  Core: session rehydrate, TTL sweep, input bounding (strip, 500 chars)  [core]
 4  Core: dialogue router — universal verbs? confirmation? slot fill?      [core]
 5  Core: RuleEngine keyword pass (bundle tables)                          [core]
 6  Core → host: tfidf_logits(sparse features)                             [FFI callback out]
 7  Native: CoreML / ORT executes; returns logits                          [native]
 8  Core: temperature softmax, policy thresholds; if low-conf:
      core → host: embed(token_ids) → native runs MiniLM → vector back;
      core: head matmul, OOS/agreement gates                               [core+callback]
 9  Core: workflow interpreter — extract entities, validate, decide
      PROMPT / CONFIRM / FULFILL / FALLBACK; update session state          [core]
10  Core: TelemetryEmitter.record(turn_event)                              [core, buffered]
11  Return NLUResult{type, intent, action, params, prompt, trace}          [FFI return]
12  Native: execute `action` (volume, BT command, navigation), speak/show
      `prompt`, TelemetryAgent drains event buffer on its own schedule     [native]
```

Interaction contract notes: calls **in** are synchronous from the host's worker thread (the host owns async, §7); calls **out** are the two inference callbacks plus a monotonic-clock provider (so the core never reads wall-clock directly — keeps TTL logic deterministic and testable on all platforms); the core is otherwise side-effect-free — no I/O, no network, no threads — which is what keeps the FFI surface small and the library trivially embeddable.

---

# Part 7 — Communication Layer

## 7.1 Mechanism per platform

```
                    ┌───────────────────────────────┐
                    │        nlu-core (Rust)        │
                    │   #[uniffi::export] interface │
                    └──────┬───────────┬────────────┘
              C ABI (cdylib/staticlib) │  scaffolding generated by UniFFI
         ┌────────────────┐  ┌─────────────────┐  ┌──────────────────┐
         │ Swift bindings │  │ Kotlin bindings │  │ Python bindings  │
         │ (generated)    │  │ (generated, JNA │  │ (PyO3 module or  │
         │ XCFramework:   │  │  over JNI; .so  │  │  UniFFI-py)      │
         │ arm64 device + │  │  per ABI in AAR)│  │  wheel for CI/   │
         │ simulator      │  │                 │  │  Studio/eval     │
         └────────────────┘  └─────────────────┘  └──────────────────┘
```

- **UniFFI over hand-written FFI/JNI:** one interface definition generates Swift + Kotlin + Python bindings; the entire §Option-C binding-maintenance hazard disappears. Hand-rolled `extern "C"` + JNI is reserved for hot paths only if profiling ever demands it (it will not, for text-command rates).
- **C ABI remains the substrate** underneath UniFFI — meaning if UniFFI were ever abandoned, the fallback is ordinary C-ABI bindings, not a rewrite. No lock-in.

## 7.2 Memory ownership

Rule: **each side owns what it allocates; the boundary passes owned copies, not borrows.** Strings/results crossing the boundary are serialized into UniFFI's buffer format and freed by generated destructors (Swift deinit / Kotlin `Cleaner`). Session objects live in core memory behind opaque handles (`Arc<Mutex<Session>>`); hosts hold a reference-counted handle, never a raw pointer. Inference callbacks receive borrowed input buffers valid only for the call and return owned vectors. No shared mutable memory crosses the boundary — copying a <1KB result per turn is nanoseconds; the simplicity is worth infinitely more than zero-copy cleverness at an FFI boundary.

## 7.3 Error propagation

Core errors are a closed Rust enum (`BundleInvalid{reason}`, `SignatureRejected`, `InferenceFailed{stage}`, `SessionExpired`, …) that UniFFI surfaces as Swift `Error` conformances and Kotlin sealed exceptions — idiomatic on both sides, exhaustively matchable. Rust panics are caught at the boundary (UniFFI wraps entry points) and surfaced as a distinct `InternalError` plus a telemetry event — a panic must degrade the turn, never crash the app. Contract: core code is written panic-free (no unwrap on input-derived data); the catch is a belt-and-suspenders, monitored so any occurrence is a P1.

## 7.4 Threading and async

```
 Main thread (UI)          Host worker (serial queue / single-thread dispatcher)
      │  handle(text) ──────────►│
      │                          │── FFI in ──► core (sync, no threads inside)
      │                          │◄─ callback ─ tfidf_logits → CoreML/ORT (may use
      │                          │              its own internal thread pool — opaque)
      │◄── NLUResult (hop) ──────│
```

The core is **synchronous and internally single-threaded**; the host owns concurrency: iOS wraps calls in an `actor`/serial `DispatchQueue`, Android in a single-thread `CoroutineDispatcher`. One session = one serial execution context (rapid ASR partials queue rather than race — the §RK8 fix falls out of the design). Core objects are `Send` and internally locked, so a *misuse* from two threads is safe, merely serialized. Async/await appears only in host-language wrappers (`async func handle(_:)`) around the worker hop; the FFI itself stays synchronous — turn latency is tens of milliseconds and an async FFI would add complexity for nothing.

---

# Part 8 — Performance

Headline first: **for this workload, the shared runtime is a maintainability decision, not a performance decision.** Turn latency is dominated by model inference (2–4ms TF-IDF, 8–15ms MiniLM) which stays exactly where it is today. Everything the core replaces is string processing measured in microseconds in any of the four languages.

| Aspect | Assessment |
|---|---|
| Cold start | Core library load is negligible (static-linked on iOS; one `System.loadLibrary` on Android, single-digit ms). Bundle load dominated by mmap + signature verify (~10–30ms) — same work regardless of language. Model warm-up (~hundreds of ms, CoreML/ORT) unchanged and remains the real cold-start cost. Net: **neutral.** |
| Warm start | Identical to today: models resident, core state in memory. Neutral. |
| Memory | Core adds ~2–5MB code + <1MB state. Rust regex/table structures comparable to NSRegularExpression/java.util.regex footprints. Models (~23MB embedder) dominate unchanged. Slight *win* available: one shared implementation makes mmap'd table sharing easier than three ad-hoc ones. Net: **neutral, ±5MB.** |
| Battery | Inference and radio dominate energy; string logic is noise. FFI copies per turn are nanojoules. Net: **neutral.** |
| CPU | Rust/C++ string processing is faster than Kotlin/Swift equivalents by small constant factors on a path that takes <1ms total. Unmeasurable in product. |
| Binary size | The honest cost: **+1.5–3MB per platform** (Rust std, panic machinery, UniFFI scaffolding; post-strip, LTO, `panic=abort`). Against a ~23MB embedder already shipped, acceptable — but it must be stated to the Board. C++ would be marginally smaller (~1MB). |
| Build time | New cross-compilation stage (~1–3 min CI per target, cacheable). Local iterative `cargo build` of a 5k-line crate: seconds. Mobile app builds consume a prebuilt XCFramework/AAR — **app-developer build time unchanged.** |
| Model loading | Unchanged — native side owns it (Part 5). |
| Thread safety | Genuine improvement: today's plan (§RK8) relies on per-platform discipline; the core's compiler-enforced `Send`/lock design makes the session-race bug class unrepresentable. |
| Caching | Compiled regex/table caches built once per bundle load in one place, instead of three independently invented caching layers (one of which would inevitably cache stale tables across a bundle swap — a bug class Option A gets for free). |

Conclusion for the Board: expect **zero user-visible performance change**, a ~2–3MB binary cost, and the elimination of two latent defect classes (session races, stale-cache-across-swap). The ROI is engineering-hours and correctness, and it should be approved or rejected on that basis alone.

---

# Part 9 — Migration Strategy (no big bang; system keeps working throughout)

Preconditions (independent, already-planned work): Option B config compilation, bundle format v1, pytest tree, parity corpus consolidated. These proceed regardless of this ADR's outcome.

**Phase 1 — Leaf extraction (lowest risk, highest drift-pain first).**
Build `nlu-core` containing only the *stateless* pipeline: normalization, TF-IDF featurization, WordPiece, keyword/negation/carrier rule engine, temperature+policy math. Ship it inside the **Python** engine first (PyO3): `classifier.py` delegates to the core while `engine.py` stays pure Python. The full existing test suite now runs against core-backed classification — Python CI becomes the core's regression harness. Then swap the iOS Swift scorer for the core XCFramework behind a feature flag, A/B'd against the hand-rolled scorer on the existing golden fixtures *on device*. Rollback: flip the flag; the Swift scorer remains in-tree until Phase 3.

**Phase 2 — Stateful core.**
Port EntityEngine + datetime grammar (parity CSVs are the acceptance suite), then DialogueCore + WorkflowInterpreter. Python `engine.py` becomes a thin wrapper (or is retired to executable-spec status with cross-checking CI — decide by whether training/eval needs in-process dialogue). Android integrates for the first time — **its first NLU is the shared core; Android never has a hand-written implementation to migrate.** iOS adopts multi-turn dialogue *from the core directly*, skipping the Swift port that Option A would have required — this is the payoff that funds the whole migration.

**Phase 3 — Consolidation.**
Delete the Swift scorer and the Python duplicate paths; parity suite shrinks from "three implementations agree" to "one implementation × three inference backends agree" (a far smaller contract: tensor-in/tensor-out). Studio testing console embeds the core (wasm or PyO3) so authors test the shipped engine.

**Validation strategy throughout:** (1) the frozen pre-migration Python engine acts as oracle for a one-time exhaustive corpus replay per phase (every training utterance + holdouts + conversation scripts through old and new, diff must be empty or explained); (2) device A/B flags compare stage-distribution/fallback/latency telemetry between old and new paths on real traffic before each cutover; (3) fuzzing on tokenizer + datetime grammar from day one of Phase 1.

**Rollback strategy:** each phase ships behind a flag with the previous implementation intact; bundle format is implementation-agnostic (both paths read the same bundle), so content releases never couple to the migration; a phase rolls back by flag-flip within a release, or remote-config within hours if flags are server-controlled. Point of no return is deliberately late (Phase 3 deletion), and only after two release cycles of core-path telemetry at 100%.

---

# Part 10 — Final Recommendation

*Addressed to a small engineering team building an on-device conversational AI platform.*

**Would I recommend introducing Rust or C++ today?**
No — not today, and this is a revision of my earlier review's emphasis. Today you have one small, fixture-covered duplication hot spot (the Swift scorer) and zero Android code. Introducing a fourth language now would spend your scarcest resource — small-team focus — ahead of the problem it solves. Today's priorities remain the ones already agreed: bundle format, config compilation, CI, label-space cleanup, telemetry.

**If not now — when?**
At a precise, observable trigger: **the sprint you commit to multi-turn dialogue on Android** (or, equivalently, the sprint you'd otherwise start hand-porting the dialogue engine to a second mobile platform). At that moment the math flips: a Kotlin port costs a full engine implementation *plus* permanent 3-way parity, while the shared core costs roughly one engine implementation *once* and retires the parity problem. Do not wait until after the Swift dialogue port begins — that is the expensive path where working native code must later be thrown away. Ratify this ADR now; execute at the trigger. On the current roadmap that is approximately two quarters out — close enough that Phase-1 leaf extraction could reasonably start next quarter if capacity allows.

**Which technology at the trigger?**
Rust with UniFFI, per Part 3 — chosen over C++ specifically because a small team without systems-programming institutional muscle should not adopt a memory-unsafe language plus hand-written JNI as its correctness foundation, and over KMP because your source-of-truth ecosystem (training, evaluation, Studio) is Python and PyO3 keeps it first-class. If between now and the trigger you hire a strong C++ bench or the roadmap turns Kotlin-centric, rerun Part 3 with those facts; the architecture (Parts 4–7) is deliberately identical either way.

**What indicators tell us the current architecture has reached its limits?**
Watch for these; any two together mean the trigger has effectively fired even if the roadmap hasn't said so: (1) a cross-platform behavior bug ships to users that parity fixtures missed; (2) the iOS/Android feature gap for dialogue behaviors exceeds one release cycle; (3) >20% of NLU engineering time goes to porting or parity maintenance rather than features; (4) the parity CI itself is chronically red or disabled (note: it is disabled *today* — the missing PAT — which is a yellow flag already); (5) the content team writes intents that behave differently per platform and starts maintaining per-platform workarounds.

**What technical debt are we accepting by postponing?**
Named and priced: the Swift scorer remains a hand-maintained duplicate (~400 lines, fixture-covered — the acceptable kind of debt); every dialogue feature added in Python deepens the eventual Phase-2 port by that much; the `conf_gap_threshold` class of quiet policy divergence persists until unification; and if the trigger fires late, some Swift dialogue code may be written and discarded. We are explicitly *not* accepting unbounded 3-way drift — Option B plus a repaired parity CI caps the exposure in the interim.

**What would I personally do as the Principal Engineer owning this platform for five years?**
Spend this quarter on Option B and the bundle/CI foundations — they pay off under every future. Fix the parity CI this week; it is the only alarm on the existing debt and it is currently unplugged. Write the `InferenceBackend` and session-state contracts *now*, in the schema and in prose, so both the Python engine and any future core implement the same seams — that costs nothing and makes the migration mechanical. Start Phase-1 leaf extraction one quarter before the Android dialogue commitment (small, reversible, and it de-risks the FFI toolchain while stakes are low). Then execute Phases 2–3 exactly as Part 9 lays out, letting Android's first NLU be the shared core so the platform never pays for a third hand-written engine. Five years out, the asset I'd want to own is one audited, fuzzed, compiler-checked conversation engine that training, the Studio, and every device run identically — and the discipline to have built it no earlier than the moment it was needed.

---

## Consequences

**Easier:** Android NLU arrives as integration work, not implementation; dialogue features ship simultaneously on all platforms; the Studio tests the real engine; parity testing collapses to a small tensor-level contract; session-race and stale-cache bug classes become unrepresentable.

**Harder:** hiring/onboarding adds a Rust-basics requirement for core work; two engineers must become FFI-toolchain-literate; debugging a turn now crosses one language boundary (mitigated by the trace output and the panic-catch telemetry); binary +2–3MB.

**Revisit if:** Android is descoped (→ stay Option A/B); team composition turns C++- or Kotlin-heavy (→ rerun Part 3); UniFFI stalls as a project (→ fall back to C-ABI bindings, same core).

## Action Items

1. [ ] Board ratifies: Option B now, Rust core at the Android-dialogue trigger (this ADR → Accepted).
2. [ ] Repair iOS parity CI (PAT secret) — this week; it is the interim safety net.
3. [ ] Define `InferenceBackend` + session-state contracts in the bundle/engine spec (pre-work, no Rust).
4. [ ] Land Option B: compiled rule tables, policy-as-data, datetime grammar tables.
5. [ ] One-sprint Rust/UniFFI spike (tokenizer + keyword engine on iOS device, XCFramework pipeline) one quarter before the trigger — validates toolchain assumptions in Part 7 before committing.
6. [ ] Re-review this ADR at the Android commitment milestone with spike results attached.
