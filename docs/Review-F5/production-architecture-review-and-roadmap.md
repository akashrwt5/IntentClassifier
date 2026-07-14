# Production Architecture Review & Engineering Roadmap
## On-Device Conversational AI Platform — IntentClassifier

**Reviewer role:** Principal Engineer, Conversational AI Platforms
**Scope:** Entire repository (data, training, NLU engine, exports, multilingual, CI, docs) + mobile integration surface
**Date:** 2026-07-13
**Verdict:** Strong proof of concept with unusually good hardening for its stage — but it is a *pipeline*, not yet a *platform*. The gap to production is less about model quality and more about system boundaries, artifact lifecycle, mobile runtime unification, and tooling for non-engineers.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture Review](#2-current-architecture-review)
3. [Production Readiness Assessment](#3-production-readiness-assessment)
4. [Gap Analysis](#4-gap-analysis)
5. [Risks](#5-risks)
6. [Proposed Architecture](#6-proposed-architecture)
7. [Detailed Component Design](#7-detailed-component-design)
8. [Data Flow Diagrams](#8-data-flow-diagrams)
9. [Training Pipeline Architecture](#9-training-pipeline-architecture)
10. [Runtime Architecture (Mobile)](#10-runtime-architecture-mobile)
11. [Dialogue Architecture](#11-dialogue-architecture)
12. [GUI / Training Studio Architecture](#12-gui--training-studio-architecture)
13. [Folder Structure Recommendations](#13-folder-structure-recommendations)
14. [Deployment Architecture](#14-deployment-architecture)
15. [Testing Strategy](#15-testing-strategy)
16. [MLOps Strategy](#16-mlops-strategy)
17. [Security Review](#17-security-review)
18. [Performance Recommendations](#18-performance-recommendations)
19. [Phased Implementation Plan](#19-phased-implementation-plan)
20. [Long-Term Vision](#20-long-term-vision)

---

## 1. Executive Summary

### What this system is today

A four-stage on-device NLU cascade replacing Dialogflow for a hearing-aid companion app:

```
Utterance → [1] Keyword pre-filter (32 declarative rules, ~0ms)
          → [2] TF-IDF + LogReg ONNX (59 intents, ~2–4ms, temperature-calibrated)
          → [3] MiniLM-L6-v2 + LR head semantic rescue (learned OOS class, ~8–15ms)
          → [4] GenAI fallback (network)
```

Plus a Python dialogue engine (`scripts/nlu/engine.py`) implementing confirmations, slot filling, interruption, back-references, context TTLs, and structured telemetry; a 4-stage data pipeline (base → corrections → augmentation → master); multilingual models (en/fr/de/da) with per-language temperature calibration; ONNX, JSON-weights, and CoreML FP16 exports; and conformance/holdout/parity test gates.

### What is genuinely good

The team has already fixed problems most PoCs never discover: confidence calibration (temperature scaling, ECE measured per language), holdout leakage guards, SHA-256 artifact manifests verified at startup, label/schema parity asserts, learned OOS rejection instead of threshold guessing, context TTLs against stale-confirmation bugs, slot-attempt budgets, keyword-tier-aware interrupt demotion, and privacy-conscious telemetry (no raw utterances by default). Two prior internal architecture reviews (`docs/architecture-review.md`, `docs/nlu-architecture-review.md`) were largely implemented. This is disciplined work.

### Why it is not production-ready

1. **The platform has three NLU implementations and no single source of truth.** Python/ONNX Runtime is the reference; iOS runs a *hand-rolled* Swift TF-IDF scorer from a JSON weights export plus CoreML for the embedder; Android exists only in documentation. Conformance tests mitigate but do not eliminate the N-runtimes problem — every feature (negation guard, keyword tiers, datetime anchoring, carrier-phrase stripping) must be re-implemented per platform and *will* drift.
2. **The dialogue/workflow layer is only ~15% declarative.** 57 of 59 intents are fire-and-forget; only 2 have slots; confirmations exist as *training labels* (`Cmd.SendMessage - yes` / `- no` are classifier classes — a dialogue act leaked into model label space). Validation rules, clarifications, and completion conditions are code, not configuration.
3. **No model lifecycle.** Artifacts are gitignored binaries regenerated locally; there is no registry, no signed releases, no OTA update path, no rollback, no A/B mechanism, no staged rollout. "Deploy" = manually run `copy_artifacts_to_stt.py` into a sibling repo.
4. **Repository is a working directory, not a codebase.** 39-entry `scripts/` grab bag, three overlapping module trees (`scripts/nlu`, `scripts/SemanticSupport`, `multilingual/SemanticSupport`), `sys.path` hacks instead of a package, a 1MB `Engage.zip` and a training `checkpoints/` dir committed at root, stale tooling (`auto_label.py` still targets a retired label taxonomy — running it would poison training data), README describing a system three generations old.
5. **No tooling for non-engineers.** Every intent, utterance, synonym, and regex requires a code-adjacent edit. The keyword-trigger `not_regex` guards are already at the edge of human maintainability.
6. **Observability stops at Python logging.** There is no on-device telemetry event schema, no aggregation plan, no dashboards, no fleet-level fallback-rate or latency monitoring.

### Top recommendations (detail in later sections)

| # | Recommendation | Effort | Impact |
|---|---|---|---|
| R1 | Define a versioned, signed **NLU Bundle** (models + schema + entities + thresholds + manifest) as *the* deployment unit | M | Unblocks OTA, rollback, A/B, GUI |
| R2 | Unify runtimes: one shared native core (Rust/C++ w/ Swift+Kotlin bindings) or ORT-everywhere + generated-not-hand-written preprocessing | L | Kills the drift class of bugs permanently |
| R3 | Make the intent workflow fully configuration-driven (slots, validation, confirmation, completion, handler binding in schema) | M | 57 intents become extensible without engine edits |
| R4 | Restructure repo into installable packages with pyproject, real test tree, and CI on every PR (not just one branch workflow) | M | Multi-team scalability |
| R5 | Lightweight MLOps: DVC for datasets, MLflow (local) for experiments, git-tag-driven bundle releases as the model registry | M | Reproducibility + auditability |
| R6 | Remove dialogue-act labels from classifier label space; fix taxonomy (`Cmd.*` vs `Help_*` vs `reminders.add`) | S | Cleaner decision boundaries, honest metrics |
| R7 | Telemetry event schema + on-device aggregation (privacy-preserving), incl. the currently-unspecified `unknown_data` capture path | M | Production visibility |
| R8 | Training Studio desktop app for PM/QA/linguists (Section 12) | L | Content velocity without engineers |

---

## 2. Current Architecture Review

### 2.1 Component inventory (as found)

| Layer | Component | Location | State |
|---|---|---|---|
| Data | 4-stage CSV pipeline (01→04), ~10k rows, 59 intents | `data/` | Working; imbalance 1884:53 |
| Data | Entity definitions (enum+synonyms, fuzzy flags) | `data/nlu_entities.json` | Good design, English canonical |
| Data | Dialogue schema (intents, slots, triggers, thresholds, yes/no lexicons) | `data/nlu_schema.json` | v2; only 2 slotted intents |
| Data | Localization overlays (schema/entities/lexicon × fr/de/da) | `data/localization/` | Sound overlay-merge design |
| Training | TF-IDF+LR trainer w/ CV, gates, manifest | `scripts/train.py` | Solid single-model script |
| Training | Semantic head trainer (SetFit-style, OOS class) | `scripts/train_semantic_head.py` | Solid |
| Training | Multilingual trainer + temperature fitting + ECE | `multilingual/train_multilingual.py` | Well-engineered; per-lang cap before concat |
| Runtime (ref) | NLUEngine: routing, slot filling, confirmation, interrupts, back-refs | `scripts/nlu/engine.py` (706 lines) | Feature-rich; emerging God class |
| Runtime (ref) | IntentClassifier (ONNX + keyword tiers + negation) | `scripts/nlu/classifier.py` | Clean |
| Runtime (ref) | EntityExtractor (enum/fuzzy/datetime/open) | `scripts/nlu/entities.py` (791 lines) | Powerful; hand-rolled datetime = liability |
| Runtime (ref) | SessionStore w/ TTLs, injectable clock | `scripts/nlu/context.py` | Clean, testable |
| Runtime (ref) | SemanticFallback (ONNX MiniLM + hand-rolled WordPiece) | `scripts/nlu/semantic.py` | Works; tokenizer is 3rd implementation risk |
| Integrity | SHA-256 manifest gen/verify | `scripts/nlu/manifest.py` | Good; integrity ≠ authenticity |
| Export | ONNX, iOS JSON weights, CoreML FP16 mlprogram, INT8/palettized embedder | `scripts/export_*`, `multilingual/export_coreml_multilingual.py` | Works; deployment is manual copy |
| Mobile | iOS: Swift hand-rolled TF-IDF scorer + CoreML embedder + softmax(logits/T) | separate `STT` repo | Parity-tested via golden fixtures |
| Mobile | Android | — | **Documentation only. No code.** |
| QA | Conformance (ONNX vs iOS weights), holdout `--strict` gate, datetime parity, CoreML Tier-A/B | `scripts/test_*`, `tests/`, `multilingual/test/` | Real gates, wrong location/structure |
| CI | One workflow, one branch (`coreml-macos.yml` on `claude/coreml-export`) | `.github/workflows/` | No PR CI, no training CI |
| Ops | `unknown_data.csv` low-confidence capture, `auto_label.py` | `data/`, `scripts/` | auto_label is **stale/dangerous** |

### 2.2 Architectural assessment against the review criteria

**Is it modular?** Partially. `scripts/nlu/` has clean seams (classifier / entities / context / semantic / engine). But module boundaries stop at Python: the schema knows about the engine's behaviors implicitly, the engine hardcodes policy constants (`INTERRUPT_THRESHOLD`, `AGREEMENT_THRESHOLD`, carrier regexes, `_NO_IDIOMS`), and multilingual support was bolted on as parallel trees rather than a language dimension of one tree. Three semantic-support module trees exist.

**Can it scale?** The *model* scales (59→~200 intents is fine for this architecture). The *organization* does not: adding a language touches ≥6 places; adding a slotted intent touches schema + engine assumptions + tests; adding a platform means re-implementing preprocessing a third time.

**Can multiple teams work on it?** Not yet. There is no ownership boundary between "content" (intents, utterances, synonyms — should be PM/linguist territory) and "engine" (developer territory). Everything routes through Python edits and a single shared `data/` directory. No CODEOWNERS, no PR CI to protect main.

**Is it future-proof?** The cascade design is — stage 3 already proves you can slot in better encoders, and the GenAI stage-4 seam is exactly where an LLM/hybrid path lands. The *packaging* is not: nothing about the current artifact layout supports two model versions coexisting, per-user rollout cohorts, or downloadable language packs.

### 2.3 Design decisions I challenge

1. **Hand-rolled iOS scorer (JSON weights) instead of a real runtime.** The prior review's P0-1 chose option (b) (conformance tests). That was right for a PoC; it is wrong at production scale. Every preprocessing nuance (lowercasing, token pattern, sublinear TF, L2 norm, bigram construction) is now security-critical parity surface across three codebases. See R2.
2. **`Cmd.SendMessage - yes` / `Cmd.SendMessage - no` as classifier labels.** Yes/no is a dialogue act resolved by the engine's confirmation lexicons. Training them as intents (78–82 samples each) steals probability mass from real intents, corrupts calibration (bare "yes" is only valid in a confirmation context the classifier cannot see), and double-implements a function the engine already owns. Remove from label space; keep utterances as yes/no lexicon data.
3. **`Default Fallback Intent` as a TF-IDF training class (1,308 rows).** Legitimate for the semantic head (learned OOS is the right pattern). For the TF-IDF stage it makes the largest "intent" a garbage class whose contents define, implicitly and invisibly, the domain boundary. Prefer: train TF-IDF on in-domain only; let calibrated confidence + stage-3 OOS do rejection. At minimum, curate the class deliberately (it currently mixes true OOS with garbled ASR of in-domain requests — the exact failure Phase-1 history documented).
4. **Intent taxonomy inconsistency.** `Cmd.VolumeIncrease` (dotted PascalCase), `Help_Pairing` (underscored), `reminders.add` (lowercase dotted) coexist; `Engage.zip` contains both `Help.X` and `Help_X` duplicates of the same content. Naming is API surface — mobile handlers, analytics, and the future GUI all key on it. Fix once, now, with a migration map (Section 7.1).
5. **Hand-rolled datetime parsing (~500 lines of `entities.py`) + per-language lexicons.** Impressive, and now a permanent maintenance tax with a known-fragile AM/PM heuristic. The parity CSVs (`tests/datetime_parity/`) prove the team already fears drift. Long-term: isolate behind an interface, adopt a table-driven grammar shared across platforms (Section 7.3).
6. **Keyword triggers with 200-char `not_regex` guards.** These are un-reviewable and un-localizable ("what languages does translate support" guards written in English only). Keyword stage should be a thin, ordered, testable rule table owned by content tooling — not regex golf. Each rule needs provenance and test coverage requirements (Section 12).
7. **`auto_label.py` self-training loop.** It appends to the *base* training file keyed on a label taxonomy that no longer exists (`VOLUME`, `REMINDER`, `WEATHER_FORECAST`). If run today it would corrupt the dataset. Delete or rewrite behind the Studio's review queue. Never auto-append to source data.
8. **Repo hygiene.** `Engage.zip` (1MB binary), `checkpoints/checkpoint-800`, `.DS_Store`, two committed venvs' pycache traces, `multilingual_intent/` skeleton dirs, README that documents a 9-intent system that no longer exists. Each is small; together they signal the absence of a "what ships is what's in the repo" discipline that production requires.

---

## 3. Production Readiness Assessment

Scored against what I would require to approve a launch to millions of devices. ✅ ready · 🟡 partial · ❌ missing.

| Area | Status | Evidence / gap |
|---|---|---|
| Core classification quality (en) | ✅ | Holdout 90/100 e2e; macro-F1 0.90; ECE 0.018; wrong-action ≤4 |
| Classification quality (fr/de/da) | 🟡 | Machine-translated training data; da macro-F1 0.745 — below shippable bar |
| Confidence calibration | ✅ | Temperature scaling per language, ECE measured, thresholds re-derived |
| OOS / fallback behavior | 🟡 | Learned OOS in stage 3; TF-IDF garbage class concern (§2.3-3); OOS sets small (156+368) |
| Dialogue management | 🟡 | Confirmations/slots/interrupts work; only 2 intents exercise it; no timeout prompts, no universal cancel/correct verbs (Section 11) |
| Entity extraction | 🟡 | Enum+fuzzy solid; datetime fragile edges (AM/PM); no numeric range validation; no composite entities |
| Multilingual architecture | 🟡 | Overlay design is right; language addition = 6+ manual steps; no language-pack packaging |
| iOS runtime | 🟡 | Working, parity-tested; hand-rolled scorer, no OTA, no A/B, single model slot |
| Android runtime | ❌ | Does not exist |
| Model lifecycle (version/publish/rollback/A-B) | ❌ | Nothing beyond git + manual copy script |
| Update security (signing) | ❌ | SHA-256 manifest = integrity only; no signature, no trust root |
| Observability | ❌ | Python logging only; no device event schema, no aggregation, no dashboards |
| Testing | 🟡 | Real gates exist; not a pytest tree; CI covers one branch only; no perf/memory/device tests |
| MLOps reproducibility | 🟡 | Deterministic caps + seeds + manifests; no data versioning tool, no experiment tracking, no lockfile (`requirements.txt` uses `>=`) |
| Privacy | 🟡 | No-utterance-logging default is right; `unknown_data.csv` captures raw text+timestamps with no documented consent/retention path |
| Non-engineer tooling | ❌ | None |
| Documentation | 🟡 | Deep design docs (excellent); top-level README materially wrong |

**Overall: not approvable for production launch.** Approvable trajectory: yes — the hard ML problems are largely solved; the remaining work is platform engineering with known patterns.

---

## 4. Gap Analysis

Grouped by the theme that closes them.

### G1 — Deployment unit (closes: lifecycle, OTA, A/B, GUI foundation)
- No single artifact that captures "everything the runtime needs" (model(s) + labels + schema + entities + lexicons + thresholds + calibration + manifest). Pieces live in `models/`, `data/`, `config/`, and env vars.
- No bundle versioning or compatibility contract (which engine versions can load which bundle schema versions).
- No signature over the bundle.

### G2 — Runtime unification (closes: platform drift, Android)
- Three preprocessing implementations (Python, Swift, planned Kotlin) of TF-IDF, WordPiece, negation windows, carrier stripping, datetime.
- Dialogue engine exists only in Python — iOS currently gets classification but the full conversation state machine would need a Swift rewrite; Android needs everything.

### G3 — Configuration-driven workflows (closes: intent design, dialogue coverage)
- No per-intent: validation rules, optional-slot semantics beyond `required: false`, clarification prompts distinct from re-prompts, confirmation policies, completion conditions, platform/language availability flags, execution-handler metadata beyond a bare `action` string.
- Engine policy constants (interrupt threshold, slot attempts, TTLs, carrier phrases, idioms) are code, per-deployment untunable, and untranslated.

### G4 — Data & training operations (closes: MLOps)
- Dataset versioning by filename convention; no lineage from a model back to exact data hash (manifest hashes artifacts, not inputs).
- No experiment tracking; comparisons live in markdown files written by hand.
- No pinned environment (`>=` requirements); training not containerized.
- Class imbalance handled only by `class_weight=balanced` + 500-cap; no per-intent sample floor alerting; no confusion-driven data acquisition loop.
- Augmentation is a hardcoded phrase bank inside a script.

### G5 — Quality engineering (closes: testing)
- Tests are runnable scripts, not a suite; no coverage; no fixtures isolation; CI absent on PRs.
- No performance/memory/battery benchmarks on real devices; no cold/warm-start budget enforcement.
- No conversation-level (multi-turn) golden tests as a first-class corpus.

### G6 — Production feedback loop (closes: observability, retraining)
- No telemetry schema, transport, sampling, or aggregation design.
- The retraining loop (`unknown_data.csv` → review → retrain) has no privacy contract, no tooling, and a broken automation script.

### G7 — Human tooling (closes: GUI)
- Nothing exists between "edit CSV/JSON by hand" and "run five scripts in the right order."

---

## 5. Risks

Ranked by (likelihood × blast radius) for a millions-of-users launch.

| # | Risk | L×I | Notes / trigger |
|---|---|---|---|
| RK1 | **Platform drift**: iOS/Android/Python disagree on a preprocessing nuance; a class of users silently gets wrong intents | High×High | Any retrain that changes vocab or any Swift/Kotlin refactor. Conformance tests only cover sampled utterances. |
| RK2 | **Unsigned model updates**: any future OTA path built on the current manifest is trivially spoofable (attacker who can write files can rewrite manifest) | Med×High | Becomes critical the day OTA ships; design it in now. |
| RK3 | **Wrong-action on a medical device**: miscalibrated confident error changes hearing-aid state unexpectedly (already tracked: ≤4/100 budget) | Med×High | Regression via data drift; budget not enforced per-intent (a 4-error budget concentrated on VolumeIncrease is worse than spread). |
| RK4 | **Danish (and future language) quality shipped on translated data** | High×Med | da macro-F1 0.745. Machine-translated utterances don't reflect how Danes actually phrase commands. |
| RK5 | **Data poisoning via stale `auto_label.py` / unreviewed appends** | Med×Med | Script exists, is documented in README, and is wrong. |
| RK6 | **Privacy incident**: `unknown_data.csv`-style raw-utterance capture reaching production without consent/retention policy (medical context) | Med×High | GDPR/HIPAA-adjacent exposure; current code logs raw text + timestamp. |
| RK7 | **Single-maintainer bus factor on hand-rolled datetime + regex triggers** | High×Med | 500-line parser + 32 regex rules with no owner-facing tooling. |
| RK8 | **Dialogue-state bugs at scale**: in-memory session store semantics unspecified for app lifecycle (backgrounding, process death, multi-device) | Med×Med | Engine is Python-reference only; mobile behavior undefined. |
| RK9 | **No rollback**: a bad bundle ships inside an app release; recovery = full app store release cycle | Med×High | Direct consequence of G1. |
| RK10 | **Repo entropy**: three module trees + stale docs cause a new team to build feature 4 on the wrong tree | High×Low-Med | Already visible (`multilingual_intent/` skeletons, README drift). |

---

## 6. Proposed Architecture

### 6.1 North star

**One content model, one bundle format, one portable runtime core, N thin platform shells.**

```
┌────────────────────────── AUTHORING PLANE ──────────────────────────┐
│  Training Studio (desktop GUI)      CLI (power users / CI)          │
│        │  edits                          │                          │
│        ▼                                 ▼                          │
│  Intent & Entity Definitions (YAML/JSON, git-versioned, schema-     │
│  validated)  +  Utterance datasets (DVC-versioned)                  │
└───────────────┬─────────────────────────────────────────────────────┘
                │  build (CI)
                ▼
┌────────────────────────── BUILD PLANE ──────────────────────────────┐
│  Data compiler → Trainers (tfidf / semantic-head / per-language)    │
│  → Evaluation gates (holdout, OOS, conversation, parity)            │
│  → Bundle assembler → SIGN → NLU Bundle v{X}  (registry = releases) │
└───────────────┬─────────────────────────────────────────────────────┘
                │  publish (staged rollout / A-B via remote config)
                ▼
┌────────────────────────── RUNTIME PLANE ────────────────────────────┐
│           ┌── nlu-core (portable: Rust or C++) ──────────────┐      │
│           │ preprocessing · tokenizers · cascade orchestration│      │
│           │ dialogue state machine · entities · telemetry emit│      │
│           └───────┬───────────────┬───────────────┬──────────┘      │
│             Swift shell      Kotlin shell     Python shell          │
│             (CoreML/ORT)     (ORT/NNAPI)      (ORT, ref+server)     │
│  BundleManager: verify sig → load → warm → swap/rollback            │
│  TelemetryAgent: anonymous counters → batched upload                │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Key decisions and trade-offs

**D1 — NLU Bundle as the deployment unit.**
A single versioned archive (`bundle.nlu`, a zip):

```
bundle.nlu/
  bundle.json            # bundle format version, content version, engine compat range,
                         # language, created_at, training data hashes, metrics report card
  manifest.json          # SHA-256 per file (existing pattern, kept)
  signature.sig          # Ed25519 over manifest.json (new)
  schema/                # compiled intent workflows + thresholds + lexicons (per language)
  entities/              # compiled entity tables + datetime grammar tables
  models/tfidf/          # intent_model.onnx OR coreml equivalent + labels + calibration
  models/semantic/       # embedder ref + head weights + vocab
```

*Trade-off:* an extra compile step between "editable source" and "runtime artifact." Worth it: it decouples authoring format (human-friendly YAML, one file per intent) from runtime format (compact, validated), enables signing, and gives the GUI a stable contract.

**D2 — Runtime core: recommend Rust core with FFI shells; fallback option ORT-everywhere.**
- *Option A (recommended): `nlu-core` in Rust*, compiled to iOS (static lib via UniFFI/swift-bridge) and Android (JNI/UniFFI), also usable from Python for the reference implementation. Model inference stays in platform runtimes (CoreML on iOS for ANE, ONNX Runtime Mobile on Android), but **everything around the model** — normalization, TF-IDF featurization, WordPiece, keyword rules, negation, datetime, dialogue state machine — lives once in Rust.
  - Pros: eliminates RK1 permanently; dialogue engine ships to Android for free; single test suite tests the real shipped logic.
  - Cons: new language in the stack; FFI build complexity; team ramp-up. Mitigation: the core is small (~3–5k lines), pure logic, no async, no platform APIs.
- *Option B: ONNX Runtime everywhere + tokenization inside the graph* (ONNX Runtime Extensions provides string normalization/TF-IDF/BERT tokenizer ops). Keeps everything in the model file; Swift/Kotlin shrink to "run session, apply thresholds."
  - Pros: no new language.
  - Cons: dialogue engine still needs per-platform ports (the bigger drift surface); ORT-Extensions ops constrain feature evolution; loses CoreML/ANE for the embedder unless you keep a second path.
- *Rejected: keep hand-rolled per-platform implementations + more conformance tests.* Conformance testing scales linearly with features; drift risk compounds. Acceptable only until Phase 2 lands.

**D3 — Schema-first workflows (Section 7.1/11):** every dialogue behavior currently hardcoded becomes schema: interrupt policy, slot re-prompt vs clarification, confirmation requirements, universal verbs (cancel/repeat/correct), completion conditions, handler binding. Engine becomes an interpreter.

**D4 — Registry = signed GitHub Releases (Phase 1) → artifact service (later).** A model registry does not require infrastructure on day one: a release per bundle version, with metrics report card in `bundle.json`, satisfies versioning/approval/rollback/audit. Move to a real service only when rollout targeting demands it.

**D5 — Language as data, not as code forks.** One trainer, one engine, `LANGUAGES` registry (already exists in `train_multilingual.py` — good), per-language bundle outputs. Delete the parallel `multilingual_intent/` and merge `scripts/SemanticSupport` vs `multilingual/SemanticSupport` into one module.

---

## 7. Detailed Component Design

### 7.1 Intent definitions (authoring format)

One YAML file per intent, compiled into the bundle. This is the contract the GUI edits, reviewers review, and the engine interprets.

```yaml
# intents/device/volume_increase.yaml
id: cmd.volume.increase          # canonical id: domain.object.action, lowercase dotted
version: 3                        # bumped on breaking change to slots/handler contract
domain: device_control            # grouping for ownership, metrics, GUI filtering
status: active                    # draft | active | deprecated (deprecated intents still
                                  # classified, mapped to replacement via `superseded_by`)
description: Raise hearing-aid volume.
platforms: [ios, android]
languages: [en, fr, de, da]
handler:
  action: volume.increase         # stable execution key the app binds to
  params_contract: {}             # typed contract; app CI validates it implements this
classification:
  training_data: dvc://datasets/volume_increase   # linked, not embedded
  keyword_triggers:               # thin, testable, per-language
    en: [{type: regex, pattern: '\bturn (it )?up\b', tier: guarded, guard_tests: 4}]
  confidence:
    min_execute: default          # inherits global; overridable per intent
    wrong_action_cost: high       # high-cost intents get stricter thresholds + confirm
dialogue:
  slots: []                       # see reminders example below
  confirmation:
    required: false               # true forces yes/no before execute (use for high-cost)
  responses:
    fulfillment: {en: "Volume increased.", fr: "...", ...}
```

Slotted example capturing everything Section 6 of the task list requires:

```yaml
# intents/reminders/add.yaml
id: reminders.add
dialogue:
  slots:
    - name: name
      entity: sys.open_text
      required: true
      prompt: {en: "What should I remind you about?"}
      clarification: {en: "Sorry — just the reminder topic, like 'take medication'."}
      validation: {max_len: 80}
    - name: datetime
      entity: sys.datetime
      required: true
      prompt: {en: "When should I remind you?"}
      validation: {must_be_future: true, max_horizon_days: 365}
      on_ambiguous: confirm       # AM/PM ambiguity → ask, don't guess (fixes P1-4)
    - name: recurrence
      entity: recurrence
      required: false
      default: none
  completion:
    condition: all_required_filled
    confirmation:
      required: true
      prompt: {en: "Set '{name}' for {datetime|speakable}?"}
  policies:
    max_slot_attempts: 3
    interruptible: true
    context_ttl_s: 90
  back_reference: {pattern_key: again, source: last_fulfilled}
```

**Compiler responsibilities:** JSON-Schema validation, cross-checks (every slot entity exists; every language listed has prompts; every keyword rule has ≥N guard tests; label set == trained label set), taxonomy lint, and generation of the runtime `schema/` tables plus a migration map for renamed intents (`cmd.volume.increase ← Cmd.VolumeIncrease`) so telemetry stays continuous.

**Taxonomy convention:** `domain.object.action` lowercase dotted. Hierarchy gives you domain-level fallbacks ("something about volume" → clarify within domain), domain-level metrics, and GUI grouping. Reserved intents: `sys.fallback`, `sys.out_of_scope`, `sys.cancel`, `sys.repeat`, `sys.correct` — engine-level, never trained as app intents.

### 7.2 Entity subsystem

Keep the current enum/synonym/fuzzy design (it is good) and extend:

- **Reusable entity registry** (already de facto exists in `nlu_entities.json`): add `normalize` (output canonicalization rules), `validation` (regex/range), and `contextual: true` (entity only resolvable inside a given intent's slot context — prevents global fuzzy false-positives like the "care"→"Car" class of bugs, which today is handled by a blunt length gate).
- **System entities** namespaced `sys.*`: datetime, duration, number, ordinal, open_text. Versioned independently of app entities.
- **Datetime:** freeze the current parser behind `DatetimeParser` interface; port to table-driven grammar in `nlu-core` (Phase 2); ambiguity returns a structured `AmbiguousTime(candidates=[...])` that the dialogue layer turns into a confirm — never a silent guess. The parity CSV corpus becomes the cross-platform contract test.
- **Synonym governance:** synonyms get provenance (`added_by`, `source: manual|mined|import`) so the Studio can bulk-review mined synonyms before activation.

### 7.3 Cascade orchestrator

Current stage logic is right. Changes:

1. **Policy table, not constants:** thresholds (execute, slot, semantic, agreement, interrupt), per `wrong_action_cost` class, live in the bundle. The engine reads; the Studio tunes; evaluation re-derives after each calibration run (this already happens manually — make it output the table).
2. **Explain-mode output:** every classification returns a structured trace (stage hits, top-N per stage, threshold comparisons, rule ids fired). Powers the Studio testing console and device debug screens; zero cost when disabled.
3. **Stage-3 embedder as a swappable interface** — the multilingual work already proved two embedders; formalize `Embedder{embed(text)->vec, id, dim}` so E5/newer encoders drop in.

### 7.4 Session & dialogue state

- Extract the state machine (Section 11) from `NLUEngine`; `engine.py` becomes: route(turn) → state machine → cascade → workflow interpreter.
- **Persistence contract:** session state must survive app backgrounding but not app restarts (fresh conversation on relaunch is the correct UX for voice control); serialize to platform storage with TTL enforcement on rehydrate. Define this in the core so both platforms behave identically.
- Session store gains an explicit `abandon(reason)` API and emits telemetry on every state transition.

### 7.5 BundleManager (mobile)

Responsibilities: locate active bundle (shipped asset or downloaded), verify signature + manifest, load with warm-up, expose `swap(newBundle)` (atomic pointer swap after successful warm-up + smoke inference), `rollback()` to last-known-good, report bundle id in all telemetry. Two-slot storage (active + candidate) bounds disk usage.

---

## 8. Data Flow Diagrams

### 8.1 Runtime turn (target architecture)

```
ASR text ──► nlu-core.handle(session, text)
  │
  ├─ session rehydrate/TTL sweep
  ├─ DIALOGUE ROUTER
  │    ├─ universal verbs? (cancel/repeat/correct) ──► state machine action
  │    ├─ active confirmation? ──► yes/no resolver ──► branch
  │    ├─ active slot fill? ──► interrupt check ──► slot resolver
  │    └─ fresh turn ▼
  ├─ CASCADE
  │    keyword table → tfidf(platform model) → calibrate(T) → threshold(policy)
  │        └─ low conf → embedder(platform model) → head → OOS/agreement gates
  ├─ WORKFLOW INTERPRETER (schema-driven)
  │    slots → validation → clarification → confirmation → completion
  ├─ RESULT {type, intent, action, params, prompt, trace}
  └─ TelemetryAgent.emit(turn_event)          App executes `action`
```

### 8.2 Content-to-production flow

```
Studio/CLI edit ─► intents/*.yaml, entities/*, datasets (DVC)
   ─► PR (schema validation + lint + leakage guard in CI)
   ─► merge ─► CI build: compile → train → evaluate gates → assemble → sign
   ─► bundle vN release (registry)
   ─► remote config: {cohort: 5%, bundle: vN}
   ─► devices download → verify → warm → swap    (failure → auto-rollback)
   ─► telemetry: fallback rate, wrong-action proxies, latency by bundle id
   ─► promote 5%→50%→100% or halt
```

### 8.3 Feedback loop

```
device low-confidence event (NO raw text by default; opt-in program only)
   ─► anonymous counters: {stage, top_intent, conf_bucket, lang}
   ─► weekly review in Studio: confusion drift, fallback clusters
   ─► targeted data acquisition (opt-in transcripts / synthetic / linguist-authored)
   ─► dataset PR ─► retrain ─► bundle vN+1
```

---

## 9. Training Pipeline Architecture

### 9.1 Keep

Per-language capping before concat; stratified split + CV; accuracy gate that blocks export; holdout leakage guard; final-model holdout recorded in manifest; temperature fitting with ECE; deterministic keep-last capping; explicit "test-split ≠ pipeline metric" honesty in comments.

### 9.2 Fix

| Issue | Fix |
|---|---|
| Trainer scripts share no code (3 copies of load/clean/cap/export) | `training/` package: `datasets.py`, `gates.py`, `calibrate.py`, `export/{onnx,coreml,weights}.py`; scripts become thin CLIs |
| `requirements.txt` uses `>=` | Lock with `uv`/pip-tools; pin in CI; record env hash in bundle.json |
| Dataset versioning by filename | DVC (or git-lfs at minimum) for `datasets/`; bundle.json records data hash → full lineage model↔data |
| Imbalance 1884:53 | Per-intent floor alert (fail build if any active intent <40 samples); report per-intent F1 vs global; cap stays |
| Augmentation phrase banks inside `build_augmented_data.py` | Move to `datasets/augmentation/*.yaml` with provenance; add paraphrase-LLM generation as an *offline, human-reviewed* source (never auto-merged) |
| Dialogue-act labels in data (`- yes`/`- no`) | Migrate rows into yes/no lexicons + confirmation tests; drop labels |
| OOS coverage thin | Grow `semantic_oos` to ≥1k across domains incl. adversarial near-domain ("turn up the heating"); track OOS-recall as a first-class gate next to holdout accuracy |
| Cross-validation only on train split of one file | Add k-fold report per intent to the evaluation artifact; gate on macro-F1 floor and per-intent F1 floor, not accuracy alone (imbalance makes accuracy flattering) |
| Evaluation scattered across scripts | Single `evaluate` command producing one JSON report: test-split, holdout e2e, OOS recall, wrong-action count, calibration (ECE + reliability data), confusion matrix, per-language — this JSON is the Studio's dashboard input and the CI gate input |

### 9.3 Multilingual data strategy

Machine-translated masters bootstrap a language but must not be its steady state. Pipeline per language: MT bootstrap → native-speaker review pass in Studio (flag unnatural utterances) → holdout authored by native speakers only (never translated — translated holdouts measure translation quality, not user reality) → ship gate: macro-F1 ≥0.85 on the native holdout. Danish stays behind a flag until it passes.

---

## 10. Runtime Architecture (Mobile)

### 10.1 Current state honestly assessed

iOS: hand-rolled Swift TF-IDF scorer (JSON weights) + CoreML FP16 embedder + temperature softmax, golden-fixture parity tests in CI (needs a PAT secret to run — currently blocked). No dialogue engine, no bundle management, no telemetry. Android: nothing. "Offline behavior" is trivially satisfied (everything is offline except stage 4) but model *updates* are impossible without app releases.

### 10.2 Target mobile design (both platforms)

```
App layer          VoiceAssistantViewModel / composables
                        │ NLUResult(action, params, prompt)
SDK layer          NluSdk (Swift/Kotlin, thin)
                     ├─ BundleManager   (verify/load/swap/rollback, §7.5)
                     ├─ nlu-core FFI    (dialogue + cascade orchestration)
                     ├─ InferenceBackend
                     │     iOS: CoreML (embedder, ANE) + tfidf in core
                     │     Android: ORT Mobile (+NNAPI EP where it wins)
                     └─ TelemetryAgent  (event schema §on-device, batching)
```

Budgets to enforce in CI (device farm, per model of record):

| Metric | Budget | Notes |
|---|---|---|
| Cold start (bundle load + warm-up) | <400 ms, off main thread | warm-up inference included (pattern already in `semantic.py` — keep) |
| Warm inference p50 / p95 (stage ≤2) | <10 / 25 ms | |
| Warm inference p95 (with stage 3) | <60 ms | |
| Peak RSS delta for NLU | <60 MB | INT8 embedder ~23 MB dominant; mmap where possible |
| Battery | no wakeups; inference on-demand only | no background execution needed for NLU itself |
| Threading | single owner thread/actor for session state; inference on worker; results to main | prevents session races with rapid ASR partials |

Model loading: lazy-load stage 3 (first low-confidence turn) *or* eager on app-foreground — measure both; hearing-aid users likely tolerate 300ms warm-up better than a slow first rescue. Keep the models memory-mapped; drop stage 3 under memory pressure and degrade gracefully (already the Python behavior — replicate and telemeter it).

Offline: stages 1–3 fully offline (assert in tests: no network permission needed by SDK). Stage 4 GenAI gated on connectivity with an offline apology prompt defined in schema.

---

## 11. Dialogue Architecture

Formalize the implicit state machine. States: `IDLE`, `SLOT_FILLING(intent, awaiting, attempts)`, `CONFIRMING(intent, branch_map)`, plus terminal emissions `FULFILL/PROMPT/CONFIRM/FALLBACK`.

Universal verbs handled *before* classification in every non-idle state (schema-defined lexicons, localized):

| Verb | Behavior |
|---|---|
| cancel ("never mind", "stop", "forget it") | abandon flow, confirm abandonment briefly |
| correct ("no, not that", "I meant X") | in CONFIRMING → treat as no + re-extract from remainder; in SLOT_FILLING → clear awaiting slot value and re-resolve current utterance |
| repeat ("what did you say") | re-emit last prompt, no state change |

Already-good behaviors to preserve as spec, not code: interrupt threshold with weak-keyword demotion; slot-attempt budget → graceful exit; context TTL 90s / session TTL 10min; partial-datetime parking (day without time anchors later bare time); opportunistic multi-slot fill with skip-awaited; back-references from last-fulfilled params; "no worries" idiom handling.

Add (currently missing):

1. **Timeout prompts:** state machine emits an optional `on_timeout` event (schema-configurable) so the app can say "Still there? I was setting a reminder" instead of silently expiring.
2. **Correction of filled slots:** "actually make it 6pm" mid-flow — resolve against filled slots, not just awaiting slot.
3. **Confirmation for high-cost intents:** driven by `wrong_action_cost: high` (e.g., anything that changes device state during a call). This converts residual wrong-action risk (RK3) into a UX cost knob.
4. **Dialogue history ring buffer** (last N turns, in-memory, never logged) to support "repeat", corrections, and future LLM context.
5. **Multi-intent utterances** ("turn it up and start streaming") — out of scope for classifier v1; detect via conjunction heuristic and clarify ("One at a time — volume first?") rather than mis-executing.

Conversation-level golden tests (Section 15) become the executable spec for all of the above.

---

## 12. GUI / Training Studio Architecture

### 12.1 Product shape

Desktop app for PM/QA/linguists/AI engineers. Recommended stack: **Tauri (Rust core + web UI)** — reuses `nlu-core` in-process for the live testing console (real engine, not a reimplementation), small binaries, no server dependency. Alternative: Electron+Python-sidecar if the team stays Python-only (heavier, but lower ramp). The Studio edits *git working copies* of the content repo — git remains the source of truth and review mechanism; the Studio shells out to git (branch/commit/PR) so non-engineers never see it.

```
┌ Studio (Tauri) ─────────────────────────────────────────────┐
│ UI: Dataset Mgr │ Intent Designer │ Entity Designer │ Flow  │
│     Training Dashboard │ Testing Console │ Model Mgmt       │
├─ core services ─────────────────────────────────────────────┤
│ content-store (YAML/CSV ⇄ UI models, validation via the     │
│   same compiler as CI — one validator, never two)           │
│ engine-host (nlu-core in-proc: live classify + full trace)  │
│ train-runner (spawns python trainers, streams progress)     │
│ registry-client (list/compare/publish bundles)              │
└──────────────────────────────────────────────────────────────┘
```

### 12.2 Modules mapped to requirements

**Dataset Management:** intent CRUD (create/rename with automatic migration-map entry/merge/split with utterance reassignment UI); CSV/Dialogflow import (the `dialogflowData/` and `Engage.zip` corpora become one-time imports through this path); near-duplicate detection (embedding cosine over the existing MiniLM — reuse stage-3 infra); validation panel (compiler diagnostics inline); full-text + semantic search; bulk edit with preview; every mutation = a git commit with the operator's name.

**Intent Designer:** form view of the YAML in §7.1 — description, utterances (with per-utterance provenance and holdout/train membership badges), slots with entity pickers, prompts/clarifications/confirmations per language with missing-translation indicators, platform/language availability toggles, handler action key with copy-to-clipboard contract stub, keyword triggers with a live regex tester that *requires* adding guard tests before save.

**Entity Designer:** synonym tables with bulk paste, normalization preview ("input → canonical"), regex entities with test corpus, fuzzy on/off + min-length per entity, usage panel ("used by 3 intents") blocking unsafe deletes.

**Conversation Flow Designer:** node-graph editor (React Flow) rendering the *schema*, not free-form drawing — nodes are Intent → Slot(s) → Validation → Clarification → Confirmation → Execution → Response, i.e., the workflow interpreter's actual semantics. Edits write schema fields; invalid graphs are unrepresentable. Include a "simulate from here" action that seeds the testing console at any node.

**Training Dashboard:** start/monitor runs (train-runner streams stdout + parsed metrics), experiment list from MLflow (params, data hash, metrics), side-by-side model comparison (delta confusion matrices, per-intent F1 diff, calibration curves), failed-prediction browser (holdout misses with stage traces), one-click "promote to candidate bundle".

**Testing Console:** live text input against any loaded bundle; displays predicted intent, calibrated confidence, top-N per stage, entities with spans, processing time per stage, session context inspector, tokenization view; conversation mode (scripted multi-turn with state visualization); "save as regression test" button that appends to the conversation golden corpus. Embedding visualization: 2-D projection of the utterance vs. intent centroids — genuinely useful for diagnosing OOS/near-miss cases; include it.

**Model Management:** bundle list from registry with report cards; publish (requires green gates + a human approval checkbox recorded in bundle.json), archive, rollback (re-points remote config), version compare, release notes (pre-filled from data/schema diff), deployment history with cohort percentages.

### 12.3 Why not a web app

The corpus, models, and training all live locally/in-repo; a desktop app avoids standing up auth/hosting for v1 and keeps utterance data off shared infrastructure (privacy posture). Revisit when multiple concurrent editors need locking — then the content repo moves behind a lightweight service.

---

## 13. Folder Structure Recommendations

Monorepo (mobile SDKs may live with app repos initially; pull them in when `nlu-core` lands):

```
intent-platform/
├── pyproject.toml                  # single installable workspace (uv)
├── content/                        # ── owned by content team ──
│   ├── intents/{domain}/*.yaml
│   ├── entities/*.yaml
│   ├── lexicons/{lang}.yaml        # yes/no, universal verbs, carriers, idioms
│   └── policies.yaml               # thresholds, TTLs, budgets
├── datasets/                       # ── DVC-tracked ──
│   ├── train/{lang}/*.csv
│   ├── holdout/{lang}/*.csv        # native-authored, never trained, never translated
│   ├── oos/*.csv
│   ├── conversations/*.yaml        # multi-turn golden corpus
│   └── augmentation/*.yaml
├── packages/
│   ├── nlu_compiler/               # content → runtime bundle schema; the ONE validator
│   ├── nlu_engine/                 # Python reference runtime (today's scripts/nlu)
│   ├── nlu_training/               # datasets, trainers, calibrate, gates, evaluate
│   ├── nlu_export/                 # onnx, coreml, weights, bundle assembly, signing
│   └── nlu_core/                   # (Phase 2) Rust portable core + ffi/
├── apps/
│   ├── studio/                     # Tauri app
│   └── cli/                        # `nlu train|evaluate|build-bundle|publish|test`
├── bundles/                        # local build output (gitignored)
├── tests/                          # pytest tree: unit/ integration/ conversation/ parity/ perf/
├── .github/workflows/              # pr.yml, train-and-gate.yml, release-bundle.yml, macos-parity.yml
└── docs/                           # ADRs + current docs (prune stale; fix README first)
```

Migration notes: `scripts/` dissolves into `packages/` + `apps/cli`; delete `multilingual_intent/`, `checkpoints/`, `Engage.zip` (import via Studio, keep in DVC if needed); merge the two `SemanticSupport` trees; `main/README.md` content merges into root README.

---

## 14. Deployment Architecture

### 14.1 Channels

1. **Baked bundle** in the app binary — the floor version; app always works offline out of the box.
2. **OTA bundle** via CDN + remote config (Firebase Remote Config or equivalent already in a hearing-aid app stack): config maps `{app_version, platform, language, cohort} → bundle_url + version + hash`.

### 14.2 Rollout state machine

```
build → sign → publish(candidate) → internal (dogfood) → 5% cohort
     → watch 48h: fallback_rate, interrupt_rate, latency_p95, crash-free
     → 50% → 100%          any tripwire → halt + rollback (config repoint; devices
                            revert to last-known-good local bundle instantly)
```

A/B: remote config assigns bundle variants per cohort; telemetry is keyed by `bundle_id` so comparison is a query, not a special mechanism. Keep experiments to threshold/policy changes and model swaps — schema-breaking changes ride app releases (engine compat range in `bundle.json` enforces this).

### 14.3 Compatibility contract

`bundle.json: {format: 3, engine_min: "2.1", engine_max: "3.x"}`. Devices refuse incompatible bundles and report it. Bundle format changes require a deprecation window of one app release.

---

## 15. Testing Strategy

Pyramid, all in `tests/`, all in PR CI:

| Layer | Content | Gate |
|---|---|---|
| Unit | tokenizers, datetime grammar, entity extraction, yes/no resolver, policy math, compiler validation | 100% pass; coverage floor on nlu_engine/core |
| Component | cascade with stubbed models; state machine transitions incl. TTL/interrupt/attempt-budget (FakeClock pattern — keep it) | pass |
| Golden — classification | per-language native holdouts + OOS sets, run e2e; per-intent F1 floors, wrong-action budget **per domain** (not just global ≤5) | block release |
| Golden — conversation | multi-turn YAML corpus: slot flows, interruptions, corrections, cancellations, timeouts, back-references, partial datetimes | block release |
| Parity | same utterance corpus through Python ref / Rust core / iOS / Android; top-1 + prob ±0.01 (existing conformance + CoreML Tier-B + datetime parity CSVs fold in here) | block release |
| Perf | device-farm cold/warm latency, RSS, model-load; budgets of §10.2 | block release |
| Adversarial/edge | empty/500-char inputs, emoji, ASR garble corpus (mine `unknown_data`'s `geg`/`ajkshdjkhdkhd` style), negations, near-domain OOS | tracked, ratchet |
| Fuzz (Phase 2, Rust core) | tokenizer + datetime grammar fuzzing | crash-free |

CI matrix: `pr.yml` (lint, unit, component, compiler validation, leakage guard) on every PR; `train-and-gate.yml` on dataset/content changes (retrain → full golden + parity → artifact upload); `macos-parity.yml` generalizes the existing CoreML workflow to run on main, not a feature branch; unblock the STT-repo PAT so iOS parity actually runs.

---

## 16. MLOps Strategy

Deliberately lightweight — this is a mobile model measured in MB, not a serving fleet.

| Concern | Tool | Notes |
|---|---|---|
| Dataset versioning | DVC over the existing repo remote (or S3) | `bundle.json` records dataset hash; leakage guard runs in CI on every data PR |
| Experiment tracking | MLflow, local file backend committed as CI artifacts | trainers log params/metrics/artifacts; Studio reads it; no server to run |
| Model registry | Signed GitHub Releases per bundle (Phase 1) | promotion = release tag + approval recorded in bundle.json; upgrade path to an artifact service later |
| Reproducible training | uv lockfile + Dockerfile for the training env + seeds already fixed | `nlu build-bundle --from-scratch` must be bit-stable for weights JSON (document known ONNX nondeterminism if any) |
| Evaluation reports | single `evaluate` JSON (§9.2) attached to every release | the report card *is* the approval artifact |
| Deployment automation | `release-bundle.yml`: tag → build → sign (key in CI secrets/KMS) → attach → update remote-config staging | prod promotion stays a human action in Phase 1 |
| Retraining cadence | on data change, not on schedule | telemetry review (Studio) drives data changes |

---

## 17. Security Review

| Area | Finding | Action |
|---|---|---|
| Offline processing | Genuine strength: stages 1–3 need no network | Assert in SDK tests (no network dep); market it |
| PII | Engine deliberately never embeds utterances in results/logs (good); **but** `unknown_data.csv` pattern captures raw text + timestamps — in a medical context this is sensitive | Define the capture path: default = counters only; raw text only under explicit opt-in program with retention limit + on-device redaction (digits, names via platform NER) before any upload |
| Model confidentiality | TF-IDF weights JSON is human-readable IP in the app bundle | Accept (weak IP) or ship weights inside the compiled model only; encryption adds little vs. determined reverse engineering — don't cargo-cult it. CoreML/ONNX binaries are sufficient obfuscation for this asset class |
| Update integrity | SHA-256 manifest detects corruption, not tampering | Ed25519 signature over manifest (§6.2-D1); public key pinned in app; reject unsigned bundles; key rotation procedure documented |
| Tampering detection | Startup manifest verify exists in Python; must exist identically on device | BundleManager verifies signature+hashes before *every* load, not just download |
| Supply chain | `>=` deps, no lockfile, pip on CI | Lock + hash-pin (`--require-hashes`); Dependabot; pin GitHub Actions by SHA |
| GenAI fallback | Default placeholder URL `genai.yourcompany.com` could ship accidentally; utterance leaves device here | Startup assert placeholder ≠ prod; document that stage 4 is the privacy boundary and require explicit user consent setting for it |
| Session data | In-memory + proposed persisted sessions contain reminder topics (potentially health-related) | Encrypt persisted session blob with platform keystore; short TTL already limits exposure |

---

## 18. Performance Recommendations

1. **Measure before optimizing further** — current numbers (2–4ms TF-IDF, 8–15ms MiniLM CPU, ~23MB INT8 embedder) are already well within voice-UX budgets. The missing work is *enforcement* (CI budgets, §10.2) not speed.
2. Embedder: pursue the ANE compute-plan work already scoped (STATUS.md S5) — FP16 mlprogram likely runs CPU/BNNS today; ANE placement could cut stage-3 latency 2–3× and battery per inference. On Android, benchmark NNAPI/XNNPACK EPs; XNNPACK is usually the safe default for BERT-tiny class models.
3. Lazy stage-3 load with warm-up on first idle (not first miss); telemeter time-to-first-rescue.
4. Cap WordPiece `max_len=64` is right for commands; assert ASR upstream truncation so p99 latency is bounded.
5. Keyword stage: precompile regexes once per bundle load (Python does; ensure core does); 32 rules is nothing — resist premature trie work.
6. Memory: mmap ONNX/CoreML files; drop stage 3 under `didReceiveMemoryWarning`/`onTrimMemory` and telemeter the degradation rate — this decides whether the 23MB is a real fleet problem.
7. Bundle download: bundles are <30MB; ship deltas only if telemetry shows update-completion problems on cellular.

---

## 19. Phased Implementation Plan

### Phase 0 — Stop the bleeding (1–2 weeks, parallel with anything)
Delete/quarantine `auto_label.py`; fix root README; remove `Engage.zip`, `checkpoints/`, skeleton dirs; add `pr.yml` CI (lint + existing test scripts + leakage guard); lock dependencies; add CODEOWNERS separating `data/` from `scripts/nlu/`; document the `unknown_data` privacy stance.

### Phase 1 — Platform foundations (4–6 weeks)
Repo restructure to §13 (mechanical, no behavior change; parity tests prove it). Label-space cleanup (remove `- yes/- no`, taxonomy migration map). Bundle format v1 + compiler + signing; `train.py`/multilingual trainers emit bundles; pytest tree; `evaluate` unified report; DVC + MLflow wiring. iOS switches to consuming signed bundles (still baked into the app — no OTA yet). **Exit gate:** one command builds a signed bundle from clean checkout; CI green on every PR.

### Phase 2 — Runtime unification + Android (6–10 weeks)
`nlu-core` in Rust: normalization, TF-IDF featurizer, WordPiece, keyword engine, policy/threshold logic, dialogue state machine, entity/datetime (port grammar tables; parity CSVs as contract tests). Swift + Kotlin shells; Android app integration (its first working NLU). BundleManager both platforms. Python engine becomes a thin wrapper over the same core (or stays as executable spec with parity CI — decide by team skill mix). **Exit gate:** identical outputs across Python/iOS/Android on the full parity corpus; Android ships behind a flag.

### Phase 3 — Lifecycle & observability (4–6 weeks, overlaps P2)
OTA channel + remote config + staged rollout + rollback; telemetry event schema + on-device aggregation + dashboard (fallback rate, stage distribution, latency, model-load, per-bundle); wrong-action budget monitoring per domain; A/B via cohorts. **Exit gate:** ship a bundle update to 5% and roll it back without an app release.

### Phase 4 — Training Studio (8–12 weeks, can start UI once compiler API is stable in P1)
MVP: Dataset Manager + Intent Designer + Testing Console (highest leverage: unblocks linguists for the Danish data problem). Then Training Dashboard + Model Management + Flow Designer. **Exit gate:** a PM adds an intent, trains, evaluates, and produces a candidate bundle without engineering help.

### Phase 5 — Language & workflow scale-out (ongoing)
Native-data program for da (and next languages) through the Studio; migrate all 59 intents to full workflow schema (confirmations for high-cost intents, universal verbs, timeout prompts); conversation golden corpus to 200+ scripts.

Dependencies: P1 blocks everything; P2 and P3 are parallelizable; P4 MVP needs only P1.

---

## 20. Long-Term Vision

The architecture is deliberately shaped so each future capability is an extension point, not a rewrite:

- **More languages:** language = content pack + trained bundle + lexicon tables. Downloadable language packs ride the OTA channel (bundle per language already implied by design).
- **Voice assistants / STT coupling:** the cascade consumes text; keep it that way. ASR n-best/lattice input is a stage-0 enhancement (classify top-3 hypotheses, agreement-gate them) that slots in front of the keyword stage.
- **LLMs:** stage 4 is the seam. Near-term: on-device small LM (Gemma/Phi class via CoreML/ORT GenAI) for paraphrase-robust intent selection over the *schema's own intent descriptions* (constrained decoding to the intent list — hallucination-safe). The workflow interpreter is model-agnostic: an LLM that emits `{intent, slots}` drives the same dialogue machinery.
- **Hybrid cloud + on-device:** BundleManager + policy table generalize to a router: `{cost, privacy_tier, connectivity} → local cascade | cloud model`. Privacy default remains local-first; cloud requires the consent flag already specced for GenAI.
- **RAG / knowledge base:** the `Help_*` domain (~30 intents mapping to static articles) is the natural first RAG surface — replace N help intents with one retrieval intent over embedded articles (the MiniLM infrastructure embeds documents too). This *reduces* classifier label count and improves both.
- **Tool/function calling:** the `handler.params_contract` in the intent schema is already a function signature. Exporting the intent registry as a tool manifest makes the app's capabilities callable by any future LLM planner for multi-step requests.
- **Personalization:** per-user thresholds and synonym boosts (e.g., user's own memory-preset names) as a small on-device overlay to the bundle — never uploaded. Session store already holds the state pattern.
- **Federated learning:** realistic scope = federated *analytics* first (confusion statistics, threshold tuning) via secure aggregation; federated fine-tuning of the tiny LR heads (59×384 floats) is feasible later precisely because the heads are small — another payoff of the frozen-encoder design.
- **Multimodal:** intents gain `input_modalities`; the workflow interpreter doesn't care where slots come from — a tap on a memory-preset UI can fill the same slot as speech.

The consistent thread: **the schema is the platform.** Everything — classifiers, LLMs, GUIs, cloud escalation, personalization — plugs into the intent/entity/workflow contract. Invest in that contract first (Phase 1), and every later phase gets cheaper.

---

## Appendix A — Immediate defect list (independent of the roadmap)

1. `auto_label.py` writes retired labels (`VOLUME`, `REMINDER`, `WEATHER_FORECAST`) into `01_source_base_training_data.csv` — delete or rewrite. **P0 data-integrity.**
2. Root `README.md` documents a 9-intent architecture that no longer exists; `main/README.md` is the real guide — merge them. **P1 onboarding.**
3. `Cmd.SendMessage - yes` / `- no` labels in training data and schema. **P1 model quality.**
4. iOS parity workflow blocked on missing `INTENTCLASSIFIER_PAT` secret (per STATUS.md) — the P0-1 conformance gate is not actually running against the iOS code. **P0 process.**
5. `DEFAULT_GENAI_URL` placeholder can reach production paths; add a startup guard. **P1 safety.**
6. Only one CI workflow, scoped to a feature branch — main is unprotected. **P0 process.**
7. Danish model (macro-F1 0.745) below any reasonable ship bar — flag off. **P0 product.**
8. `requirements.txt >=` ranges — non-reproducible training environment. **P1 reproducibility.**
9. Duplicate module trees (`scripts/SemanticSupport` vs `multilingual/SemanticSupport`; `multilingual_intent/` skeletons) — consolidation before anyone builds on the wrong one. **P1 maintainability.**
10. `unknown_data.csv` raw-utterance capture has no documented consent/retention story. **P0 privacy (if the capture path ships).**
