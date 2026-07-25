# Implementation Plan — Production-Ready On-Device NLU (Python + Language Packs)

**Author:** Principal Architect (project technical owner)
**Date:** 2026-07-24
**Grounded in:** `docs/Review-F5/` (ADR-001…005 + production roadmap)
**Scope decision:** We adopt the **Language-Pack architecture in Python now** and
**defer the shared Rust core** (ADR-001 Part 4) to a later, optional phase.

---

## 0. What this plan commits to

Three requirements you set, treated as non-negotiable acceptance criteria:

1. **Stage 3 (semantic / MiniLM) is a plugin, disabled by default.** The engine
   runs and ships without it; it is loaded only when a Language Pack declares it
   and the runtime is explicitly allowed to.
2. **The engine is a language-agnostic execution framework.** It contains *zero*
   language-specific rules. Every language ships as a self-contained **Language
   Pack**. Adding a language = adding a pack, never editing engine code.
3. **Fully automated model build & release.** Merging to `main` triggers GitHub
   Actions to validate → train → evaluate/gate → export (ONNX/CoreML/metadata) →
   cut a **versioned GitHub Release** with the artifacts attached.

Everything below serves those three, plus the minimum surrounding work that makes
them genuinely production-grade rather than a demo.

---

## 1. The one honest tradeoff of "no Rust"

ADR-001's central recommendation was a shared Rust core to permanently kill
**platform drift** (RK1): the risk that Python, Swift, and Android disagree on a
preprocessing nuance and a slice of users silently gets wrong intents.

Deferring Rust means we keep that risk and manage it the way the project already
does: **generated (not hand-written) artifacts + conformance/parity tests**. This
is coherent and is explicitly ADR-001's "Option B, acceptable until the trigger."
The key discipline that makes it safe:

- The Language Pack carries **data tables, not code** (regexes, lexicons, grammar
  tables, model weights). Each platform interprets the *same* tables.
- Anything a platform must compute (TF-IDF featurization, WordPiece) is driven by
  pack tables and covered by a **parity corpus** run in CI across every runtime.

**What we explicitly accept by deferring Rust:** the dialogue state machine and
entity/datetime logic still have to be reimplemented per platform (Python now,
Swift on iOS, Kotlin later). We contain drift with tests; we do not eliminate it.
The architecture below is deliberately shaped so that *if* we later adopt a shared
core, the Language-Pack boundary and interfaces do not change — only the
implementation behind them does. **No decision here blocks that future.**

---

## 2. Target architecture (Python, no Rust)

```
        AUTHORING / BUILD (Python, dev machine + CI)
   content/<lang>/…  ──►  compiler  ──►  Language Pack (versioned, signed)
   datasets/<lang>/… ──►  trainers  ──►    models + tables + manifest
                          evaluate  ──►    report card (gate)
                                     │
                                     ▼  pack.<lang>.v{N}  (GitHub Release asset)
   ───────────────────────────────────────────────────────────────────────────
        RUNTIME  (identical pipeline for every language)
   text ─► Engine (execution framework) ─► loads Language Pack ─► NLUResult
```

### 2.1 The engine — an execution framework, nothing else

The engine owns the *process*, never the *language*. Its responsibilities:

- Execute the NLU pipeline (stage orchestration / cascade)
- Manage inference (call the pack's models through an interface)
- Run slot filling, entity resolution, confirmation, interruption, back-reference
- Execute workflows (schema-driven), produce deterministic `NLUResult`
- Load and validate Language Packs

The engine must **never** contain `if language == "en"`, `if text contains
"turn up"`, hardcoded carrier regexes, yes/no words, idioms, or a datetime
grammar. Today it contains all of these — see §4 for the exact eviction list.

### 2.2 The Language Pack — everything language-specific

A pack is a self-describing directory (zipped for distribution) implementing a
fixed interface:

```
LanguagePack
 ├── pack.json           # manifest: lang, version, engine compat, capabilities, hashes
 ├── config.json         # per-language runtime config (thresholds, flags, feature toggles)
 ├── normalizer.json     # normalization rules (casing, unicode NFC, digit/diacritic policy)
 ├── tokenizer/          # tokenizer spec + vocab (WordPiece vocab, token pattern)
 ├── intent_model/       # TF-IDF (or equivalent) ONNX + labels + calibration(temperature)
 ├── entities/           # enum entity tables + synonyms + fuzzy flags
 ├── datetime/           # datetime grammar tables (data, not code)
 ├── lexicons.json       # yes/no, uncertainty, idioms, universal verbs, carrier phrases, negation cues
 ├── keywords.json       # keyword pre-filter rules (portable regex subset), tiers, guards
 ├── schema.json         # intent workflows: slots, prompts, validation, confirmation, completion
 └── plugins/
     └── semantic/       # OPTIONAL Stage-3: encoder ref + head + vocab (present only if declared)
```

The runtime loads a pack and runs the **same** code path regardless of language.
This is the ADR-005 "bundle" idea, scoped to one language and kept in Python-native
form for now (JSON tables + ONNX models), without the full signing/OTA machinery
until Phase 3.

### 2.3 Engine ↔ Pack interfaces (the contracts)

The engine depends on abstractions, the pack provides implementations:

| Interface | Engine expects | Pack provides |
|---|---|---|
| `Normalizer` | `normalize(text) -> text` | rules table (casing, NFC, digits) |
| `Tokenizer` | `tokenize(text) -> tokens` | vocab + token pattern |
| `IntentModel` | `classify(features) -> [(intent, score)]` | ONNX model + labels + temperature |
| `EntityExtractor` | `extract(type, text) -> value` | enum tables + datetime grammar tables |
| `KeywordRules` | ordered rule eval | compiled keyword rules |
| `Lexicon` | yes/no, verbs, carriers, idioms | per-language word tables |
| `Workflow` | slots/validation/completion | compiled `schema.json` |
| `SemanticPlugin` (optional) | `classify(text) -> (intent, score)` | encoder + head + vocab, or **absent** |

The engine calls only these. It cannot tell English from Japanese; it only sees
`Normalizer`, `Tokenizer`, `IntentModel`, etc.

### 2.4 The decision rule (apply to every new component)

> "Does this behavior depend on a specific language?"
> If yes → Language Pack. If it is generic for all languages → Engine.

Example applications:
- Slot-filling state machine → **engine** (generic).
- The prompt text asked during slot filling → **pack** (language).
- "How to decide two models agree" (agreement gate) → **engine** (generic math).
- The word "yes" → **pack** (language).
- Temperature-scaled softmax → **engine** (generic); the temperature value → **pack**.

---

## 3. Requirement 1 in detail — pluggable, off-by-default semantic stage

**Design:** the semantic stage is a plugin the engine discovers, not a hardcoded
branch. Three layers of control, first wins:

1. Runtime override — `NLUEngine(enable_semantic=True/False)`
2. Environment — `NLU_ENABLE_SEMANTIC`
3. Pack declaration — `config.json: { "semantic_enabled": false }` (default)

**Off means not loaded.** When disabled, the ~23 MB encoder and its warm-up are
never paid for; the engine degrades cleanly to keyword → TF-IDF → GenAI fallback.
This matches the earlier `Restructuring/` proof-of-concept, now generalized: the
plugin lives *inside the pack* (`plugins/semantic/`), so a pack can ship without a
semantic stage at all, and a memory-constrained device profile can force it off.

**Why off by default is correct (not just lighter):** the review flags the
semantic head as having no reliable out-of-scope rejection — on a medical device it
can breach threshold and fire a *wrong action*. Opt-in keeps the shipped default
deterministic and lightweight, which is the stated design goal.

**Engine change:** the cascade reads `stages` from pack `config.json`. A pack that
omits/disables the semantic stage produces an engine with `self.semantic = None`.
No engine code names MiniLM.

---

## 4. Requirement 2 in detail — evict language from the current engine

This is the core refactor. Today's `scripts/nlu` mixes engine and language. The
table below is the **eviction list**: what moves out of code into the pack.

| Currently in code | File | Moves to pack as | Why |
|---|---|---|---|
| Carrier-phrase regexes (`_CARRIER`) | `engine.py` | `lexicons.json: carriers` | English-only patterns; already partially lexicon-driven |
| Yes/no / negative word sets | `engine.py` (`affirmative`,`negative`) | `lexicons.json: yes/no` | Pure language data |
| Uncertainty + "no" idioms (`_UNCERTAIN`, `_NO_IDIOMS`) | `engine.py` | `lexicons.json: idioms` | English idioms hardcoded |
| Leading-connector strip regex | `engine.py` | `lexicons.json: connectors` | English grammar |
| Keyword triggers + `not_regex` guards | `schema.json` / `classifier.py` | `keywords.json` | English regexes; must be per-language |
| Datetime grammar (~500 lines) | `entities.py` | `datetime/*.json` tables | The worst offender; interpret tables, don't hardcode |
| Enum entities + synonyms | `data/nlu_entities.json` | `entities/*.json` | Already data; just relocate into pack |
| WordPiece vocab + tokenizer | `semantic.py` | `tokenizer/` + `plugins/semantic/` | Language/model specific |
| Intent labels + temperature | `models/` | `intent_model/` | Per-language artifacts |
| Policy constants (`INTERRUPT_THRESHOLD`, `AGREEMENT_THRESHOLD`, `MAX_SLOT_ATTEMPTS`, TTLs) | `engine.py`/`context.py` | `config.json` (defaults) | Tunable per pack/deployment; not code |

**What stays in the engine (correctly generic):** the turn priority (confirm →
slot-fill → back-ref → classify), the slot-filling state machine, interruption
logic, back-reference resolution, temperature-softmax math, agreement-gate math,
cascade orchestration, session/TTL mechanics, telemetry emission.

**Acceptance test for requirement 2:** `grep -rn 'if language' packages/nlu_engine`
returns nothing, and a new language is added with **zero** edits under
`packages/nlu_engine/` — only a new pack directory. We enforce this with a CI
check (§5, `pr.yml`) and a "hostile language" test: load a fake `zz` pack and
confirm the engine runs end-to-end without code changes.

---

## 5. Requirement 3 in detail — automated model build & release (GitHub Actions)

Pipeline triggered on merge to `main` (and manually via `workflow_dispatch`):

```
 validate ─► train ─► evaluate/gate ─► export ─► assemble pack ─► release
   │           │          │              │            │             │
 schema+     per-lang   holdout/OOS/   ONNX +      pack.<lang>    GitHub Release
 dataset     TF-IDF +   parity gates   CoreML +    .zip + hashes  vX.Y.Z + assets
 lint +      (semantic  (block on      metadata    + manifest
 leakage     head if     regression)
 guard       declared)
```

Concretely, three workflows (roadmap §15 CI matrix, minus the Rust job):

- **`pr.yml`** (every PR): lint, unit + component tests, compiler/schema
  validation, dataset leakage guard, "no `if language` in engine" check, hostile-
  pack test. Fast; protects `main`.
- **`train-and-gate.yml`** (on dataset/content change or manual): retrain affected
  language(s), run the full evaluation gate (per-intent F1 floors, wrong-action
  budget per domain, OOS recall, ECE, datetime parity), upload the per-run gating
  reports as **CI artifacts** (transient, for debugging a run). **Blocks** if any
  gate regresses.
- **`release-pack.yml`** (on merge to `main`, gated by the above): export ONNX +
  CoreML + metadata, assemble the versioned Language Pack(s), generate SHA-256
  manifest, create a **GitHub Release** tagged `pack-<lang>-vX.Y.Z`, and attach the
  pack archives + the **final report card** for the shipped pack as release assets.

**Hard rule — every generated model is accuracy-tested before it can ship.** A
model is never released on the strength of having trained successfully.
`release-pack.yml` runs only if `train-and-gate.yml` passed, so the
accuracy/regression gate (per-intent F1 floors, wrong-action budget per domain,
OOS recall, ECE, datetime parity, and the golden + native-holdout suites) is a
mandatory precondition for every release. A model that trains but fails the gate
is blocked, not shipped — this is a safety requirement on a medical device, not a
nicety.

The distinction between transient CI reports and the published report card is the
model-registry policy — see the *Model registry policy* subsection below.

**Reproducibility prerequisites (must land with this):** pin `requirements.txt`
(currently `>=`) to a lockfile, fix seeds (already done in trainers), and record
dataset hashes + git SHA + env-lock hash into `pack.json` lineage. Without these,
"automated" produces non-reproducible artifacts.

**Signing:** Phase 1 ships SHA-256 manifests (integrity). Ed25519 signing (trust)
is added in Phase 3 with OTA — designed for now, not implemented yet, so the
manifest layout already reserves a `signature` slot.

### Model registry policy (Version 1)

**GitHub Releases are the production model registry for Version 1.** Every
successful release publishes a versioned Language Pack and its associated
deployment artifacts — ONNX, CoreML, metadata, and the final evaluation report
card — as **GitHub Release assets**. Client applications retrieve production
models **exclusively from GitHub Releases**.

**GitHub Actions Artifacts are for temporary CI outputs only** — logs,
intermediate files, and per-run gating reports — and **must never** be used as the
production distribution mechanism. They expire, are not versioned as a release,
and carry no traceability contract; treating them as a source of production models
is prohibited.

This gives us, with no infrastructure beyond GitHub:

- **Versioned model distribution** — one release per pack version (`pack-<lang>-vX.Y.Z`).
- **Reproducible releases** — pack lineage records dataset hashes, env lock, and git SHA.
- **Simple rollback** — repoint clients to a previous release; prior versions stay published.
- **Traceability** — every deployed model maps back to the exact Git commit that built it.
- **No added infrastructure** — the registry is the release list itself.

`release-pack.yml` is therefore the *only* producer of production artifacts, and
it produces them as Release assets. The client-side updater resolves models by
querying GitHub Releases (by tag, or "latest per language/channel"), never by
reading CI artifacts. Migrating to a dedicated artifact service is deferred until
rollout targeting or download scale demands it (ADR roadmap D4); until then this
policy is binding.

---

## 6. Adding a new language (end-to-end)

This is the payoff of requirements 1–3 working together: adding a language is a
**content + data** task that the pipeline turns into a released model, with **no
engine code change**.

### The flow

```
author pack source  (content/<lang>/  +  datasets/<lang>/)
        │  open PR
        ▼
pr.yml:   schema/compiler validation · leakage guard · "no if language" check · hostile-pack test
        │  merge to main
        ▼
train-and-gate.yml:  train intent model ─► fit calibration ─► ACCURACY GATE
        │                                     (F1 floors, wrong-action budget,
        │                                      OOS recall, ECE, datetime parity,
        │                                      golden + native-holdout suites)
        ├── gate fails ──►  BLOCKED  (no release; report card shows why)
        │
        └── gate passes ─►  release-pack.yml:  export ONNX/CoreML + assemble pack
                                               + manifest ─► GitHub Release pack-<lang>-vX.Y.Z
```

The engine is never touched. `pr.yml` fails the PR if anyone edits
`packages/nlu_engine/` to make a language work (§4 acceptance test). And per the
§5 hard rule, the model is **accuracy-tested before it can be released**.

### What you must supply vs. what is automated

| Pack content | Who supplies it | Automated from dataset? |
|---|---|---|
| Training utterances (`datasets/<lang>/train`) | data / linguist | — (this *is* the dataset) |
| Native holdout set (`datasets/<lang>/holdout`) | native speaker | ❌ must be native-authored; never machine-translated |
| Intent model (ONNX) + labels | pipeline | ✅ trained |
| Calibration / temperature | pipeline | ✅ fit from a val split |
| Tokenizer + normalizer config | platform default or override | ✅ shared multilingual default; override only if needed |
| Enum entities + synonyms (`entities/`) | native speaker | ❌ authored |
| Datetime grammar tables (`datetime/`) | native speaker | ❌ authored (largest single effort) |
| Lexicons: yes/no, carriers, idioms, verbs | native speaker | ❌ authored |
| Keyword rules (`keywords.json`) | content team | ❌ authored (portable-regex subset) |
| Response strings: prompts, fulfillment | linguist | ❌ translated / authored |
| `pack.json` manifest + `config.json` | compiler + defaults | ✅ generated / defaulted |

Rule of thumb: **the model is generated; the language resources are authored.** A
CSV of utterances alone produces a classifier but not a usable assistant — the
prompts, entities, datetime grammar, and lexicons are what make a turn actually
work in that language.

### "Add a dataset → get a model" — true, but not sufficient

- It *is* true that no engine edits are needed — that is requirement 2 delivered.
- It is *not* true that a dataset alone yields a **shippable** language: the
  accuracy gate (§5) trains the model and then **refuses to release it** if quality
  is below bar. Danish today (macro-F1 0.745 on translated data) is exactly this
  case — the pipeline builds it and blocks the release until native data lifts it.

### Minimum checklist to add language `<lang>`

1. `datasets/<lang>/train.csv` — utterances → intent ids (existing taxonomy).
2. `datasets/<lang>/holdout.csv` — native-authored, never trained, never translated.
3. `content/<lang>/` — entities, datetime grammar, lexicons, keywords, responses.
4. `config.json` — thresholds/flags (or inherit defaults); semantic stage off unless declared.
5. Open PR → green `pr.yml` → merge → automated train, **accuracy gate**, and (if it passes) release.

No step touches `packages/nlu_engine/`.

---

## 7. Repository restructure (Python packages, no Rust core)

Grounded in roadmap §13, with the Rust core row removed:

```
intent-platform/
├── pyproject.toml                 # single installable workspace
├── content/<lang>/                # ── content team ── (intents, entities, lexicons, keywords, policies)
├── datasets/<lang>/               # ── DVC-tracked ── (train / holdout / oos / conversations)
├── packages/
│   ├── nlu_engine/                # language-INDEPENDENT runtime (today's scripts/nlu, de-languaged)
│   ├── nlu_langpack/              # LanguagePack loader + interfaces + validation
│   ├── nlu_compiler/              # content → pack; the ONE validator
│   ├── nlu_training/              # trainers, calibration, gates, evaluate
│   └── nlu_export/                # onnx, coreml, pack assembly, (later) signing
├── apps/
│   └── cli/                       # nlu train | evaluate | build-pack | test
├── packs/                         # build output (gitignored); release artifacts
├── tests/                         # pytest tree: unit / component / golden / parity / perf
├── .github/workflows/             # pr.yml, train-and-gate.yml, release-pack.yml
└── docs/                          # ADRs + this plan (prune stale; fix root README)
```

Consolidation debt to clear en route (roadmap Appendix A): delete `auto_label.py`
(writes retired labels — data-integrity P0), remove `Engage.zip` / `checkpoints/` /
skeleton `multilingual_intent/`, merge the two `SemanticSupport` trees, fix the root
README, add CODEOWNERS separating `content/` from `packages/nlu_engine/`.

---

## 8. Phased plan (no Rust) with exit gates

**Phase 0 — Stop the bleeding (1–2 weeks, parallel).**
Delete/quarantine `auto_label.py`; fix root README; remove committed binaries and
skeleton dirs; add `pr.yml`; lock dependencies; CODEOWNERS.
*Exit:* `main` is protected by CI; repo contains only what ships.

**Phase 1 — Language Pack architecture + pluggable semantic (4–6 weeks).**
Define the `LanguagePack` interface and loader (`nlu_langpack`). Evict all language
data from `nlu_engine` per §4. Make the semantic stage a pack-declared, off-by-
default plugin. Build the `nlu_compiler` (content → pack) as the single validator.
Port `en` to a pack; keep behavior identical (parity replay against the frozen
current engine is the acceptance oracle).
*Exit:* engine has zero `if language`; `en` runs entirely from a pack; hostile-pack
test passes; semantic off by default and loads only when declared+enabled.

**Phase 2 — Automated build & release (3–4 weeks, overlaps P1 tail).**
Stand up `train-and-gate.yml` and `release-pack.yml`; unify the evaluation gate
into one report card; wire DVC dataset hashes + lineage into `pack.json`; emit
versioned GitHub Releases with pack + CoreML + ONNX + report card.
*Exit:* merging to `main` produces a signed-manifest, versioned pack release with
no manual steps; a clean checkout rebuilds a byte-stable pack.

**Phase 3 — Lifecycle & multi-language scale-out (ongoing).**
Add fr/de/da packs through the same pipeline (flag Danish off until native data
lifts macro-F1 above bar — roadmap RK4). Add Ed25519 signing + OTA + rollback when
on-device update is needed. iOS consumes packs (baked first, OTA later).
*Exit:* a new language ships as a pack with no engine change; a bad pack rolls back
without an app release.

**Deferred (explicitly out of scope now):** the shared Rust core (ADR-001 Part 4).
Revisit at the "Android needs the full dialogue engine" trigger. The Language-Pack
boundary defined here is what makes that migration a swap-behind-interfaces rather
than a rewrite.

---

## 9. Immediate next steps (what I'd do first, in order)

1. **Lock the `LanguagePack` interface** (`nlu_langpack`) — the contract everything
   else depends on. One PR, no behavior change.
2. **Evict language data from `engine.py`** into an `en` pack (carriers, yes/no,
   idioms, connectors) — behavior-preserving, parity-tested.
3. **Move keyword rules + datetime grammar** into pack tables (the two biggest
   language couplings).
4. **Make the semantic stage pack-declared + off by default** (generalize the
   `Restructuring/` PoC).
5. **Stand up `pr.yml`** with the "no `if language`" and hostile-pack checks so the
   architecture can't regress.
6. **Then** `release-pack.yml` for the automated versioned releases.

Steps 1–4 are the architecture; 5–6 make it enforceable and automated. I can start
on step 1 (the interface + loader) whenever you approve this direction.

---

## 10. Open decisions for you

1. **Where do packs live at rest** — attached to GitHub Releases only, or also a
   CDN bucket for future OTA? (Recommend: Releases now, CDN at Phase 3.)
2. **One pack per language, or one multi-language pack?** (Recommend: one per
   language — matches downloadable language packs and per-language rollout.)
3. **Danish:** ship flagged-off with translated data, or hold until native data?
   (Recommend: hold; 0.745 macro-F1 is below a medical-device bar.)
4. **iOS consumption timing** — do we update the Swift side to load packs in
   Phase 1, or keep it on the current export and switch at Phase 3? (Recommend:
   Phase 3, to avoid a second moving target during the refactor.)
```
