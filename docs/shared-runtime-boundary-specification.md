# Shared Runtime Responsibility & Boundary Specification
## ADR-001.1 — Defining What Lives Where

**Status:** Proposed — companion to ADR-001 (accepted direction: shared runtime deferred to the Android-dialogue trigger; this document fixes its boundaries in advance)
**Date:** 2026-07-13
**Purpose:** The official reference answering one question for every future feature: **"Where should this piece of functionality live?"**
**Audience:** Every engineer adding functionality to the platform, before they write it.

---

## The One-Paragraph Rule

> **Share the *decisions*, never the *devices*.** Anything that determines *what the assistant decides* (what was said, what it means, what to do next, what to ask) must produce identical results on every platform and therefore lives in the shared runtime or in configuration it interprets. Anything that touches *how the device does things* (microphones, radios, screens, networks, files, keys, chips) lives native. Anything that *produces* the decision-making artifacts (models, tables, bundles) lives in Python. When in doubt, apply the four-question test in §10.

---

## Quick Reference — Decision Flowchart

```
New functionality proposed
        │
        ▼
Q1: Does the OS or hardware own it (sensor, radio, screen,
    file system, keystore, network stack, app lifecycle)?
        │ yes ──────────────────────────────► NATIVE (iOS / Android)     §5, §8
        ▼ no
Q2: Is it needed only at build/training time
    (never executes on a user's device)?
        │ yes ──────────────────────────────► PYTHON                     §4
        ▼ no
Q3: Would two platforms disagreeing about it change what
    the user experiences (intent, slots, prompts, actions)?
        │ no ───────────────────────────────► NATIVE (per-platform choice OK)
        ▼ yes
Q4: Is it a *behavior* (algorithm, state, control flow)
    or a *fact* (value, pattern, mapping, text)?
        │ fact ─────────────────────────────► SHARED CONFIGURATION       §7
        ▼ behavior
                    ──────────────────────► SHARED RUNTIME               §3
   (…and if it is a behavior that only exists to interpret
    facts, put the facts in configuration and only the
    interpreter in the runtime — smallest possible runtime.)
```

---

# Part 1 — Review of ADR-001's Proposed Runtime Modules

ADR-001 (Part 4/5) placed eight modules in the shared runtime. Re-examined here, each classified **mandatory** (the runtime is pointless without it) or **optional** (defensible either way), with changes where my earlier position doesn't survive scrutiny.

| # | Module (ADR-001) | Problem it solves | Verdict on re-review |
|---|---|---|---|
| 1 | TextPipeline (normalize, tokenize, TF-IDF featurize, WordPiece) | The proven drift hot spot — the Swift scorer re-implements sklearn semantics today; `min_df` drift already occurred within Python itself | **Mandatory.** This is the module with *measured* drift, not projected drift. It is also stateless and small — the ideal first extraction. |
| 2 | RuleEngine (keyword tiers, negation, carrier stripping) | Three regex dialects, ordered transformations, character-window semantics (ADR-001 §2.1–2.3) | **Mandatory, narrowed.** The *engine* (matcher semantics, tier logic, negation windows) is runtime; the *rules* are configuration (§7). ADR-001 blurred this; this spec fixes it: the runtime must contain zero intent-specific patterns. |
| 3 | Cascade + policy thresholds (calibration math, gates) | Policy web divergence — `conf_gap_threshold` exists on iOS but not in the Python engine *today* | **Mandatory, narrowed.** Threshold *values* and gate *parameters* are configuration; only the evaluation order and gate semantics are runtime. |
| 4 | EntityEngine (enum/synonym/fuzzy + datetime grammar) | Datetime is the worst single drift risk in the codebase (~500 lines, stateful, already has cross-language parity CSVs because the team fears it) | **Mandatory.** The entity *tables* and datetime *grammar tables* are configuration; the interpreters (fuzzy matcher with length gates, grammar resolver with parking/anchoring) are runtime. |
| 5 | DialogueCore (state machine, slot filling, confirmations, interrupts, contexts, TTLs) | The module whose duplication *is* the ADR-001 trigger; stateful logic where fixture parity is structurally insufficient (§2.5–2.7) | **Mandatory.** This is the reason the runtime exists. If DialogueCore stays native, do not build the runtime at all. |
| 6 | WorkflowInterpreter (schema-driven validation/clarification/completion) | The schema contract must mean exactly one thing everywhere, or the content team's single-definition promise breaks | **Mandatory,** with the anti-DSL constraint of §7.3: the interpreter executes a *closed* vocabulary of behaviors; new behavior kinds require a runtime release, deliberately. |
| 7 | TelemetryEmitter | Privacy invariant (no raw utterances) enforced in one audited place | **CHANGED — narrowed to "turn-event assembly + redaction."** ADR-001 implied the runtime owns telemetry generally. Wrong: device metrics (memory, battery, model-load timing, download success) originate natively and must stay native. The runtime assembles only *decision events* (stage, intent, confidence, latency of its own work) because only it knows them and because the redaction guarantee must sit where the raw text is. Buffering, sampling, consent gating, transport: native (§5). |
| 8 | BundleLoader (signature verify, manifest verify, mmap tables) | Identical accept/reject decisions on every platform; one audited security check | **CHANGED — split.** Verification logic (signature check, manifest hashing, compatibility gate) stays runtime: a bundle rejected on iOS must be rejected on Android, byte-for-byte the same rule. But *file discovery, download, storage, and the mmap/open syscalls* move native behind the Bundle Interface (§9.2): the runtime receives an opened, read-only byte view and never touches the file system itself. ADR-001's "mmap tables" phrasing leaked an OS concern into the runtime. |

Also re-examined and **confirmed excluded** (ADR-001 got these right): model inference (stays native behind the Inference Interface — CoreML/ANE and ORT/NNAPI are platform-optimal and the tensor contract is small and parity-testable); semantic-head matmul stays *in* the runtime (it is 59×384 deterministic math welded to the OOS/agreement gates — splitting it across the boundary would put half a decision on each side).

**Net change from ADR-001:** two narrowings (7, 8), two clarified runtime/config splits (2, 3). No module moves in or out wholesale. The runtime gets *smaller* under review, which is the direction a boundary review should push.

---

# Part 2 — Responsibility Matrix

Every component appears exactly once. ⭐ = exists today; ○ = planned.

### PYTHON (build-time plane — never executes on a device)

| Component | Note |
|---|---|
| ⭐ Dataset preparation & augmentation (`build_augmented_data.py`, phrase banks) | |
| ⭐ Training: TF-IDF/LogReg (`train.py`), semantic head, multilingual trainers | |
| ⭐ Calibration fitting (temperature, ECE) & threshold derivation | Values it derives ship as *configuration* |
| ⭐ Evaluation & gates (holdout, OOS, wrong-action budget, confusion) | |
| ○ Experiment tracking integration (MLflow), dataset versioning (DVC) | |
| ⭐ Model export (ONNX, CoreML, weights JSON) | |
| ○ Bundle compiler & assembler (content YAML → runtime tables) + **signing** | Signing is build-time crypto; *verification* is runtime |
| ○ Content validation (the single validator the Studio and CI share) | |
| ⭐ Data quality tooling (leakage guard, mislabel scan, dedup) | |
| ○ Studio train-runner & registry tooling | |
| ⭐→○ Reference NLU engine (`scripts/nlu/`) | Role transitions per Part 4 — production logic migrates to runtime; Python keeps the experimentation shell |

### SHARED RUNTIME (decision plane — identical on every device)

| Component | Mandatory? |
|---|---|
| Text pipeline: normalization, tokenization, TF-IDF featurization, WordPiece encode, pooling/L2 | Mandatory |
| Rule engine: keyword-tier matcher, negation detector, carrier stripper (interpreting config tables) | Mandatory |
| Cascade orchestrator: stage sequencing, temperature application, threshold/gate evaluation (config values) | Mandatory |
| Semantic head classification (matmul + softmax + OOS/agreement gates) | Mandatory |
| Entity engine: enum/synonym lookup, fuzzy matcher (+ length gate), open-entity capture, normalization application | Mandatory |
| Datetime resolver: grammar interpretation, future-hour policy, day-parking/anchoring, UTC conversion | Mandatory |
| Dialogue state machine: routing priority, confirmation handling, yes/no + idiom resolution, universal verbs | Mandatory |
| Slot-filling engine: awaited-slot resolution, opportunistic fill, attempt budget, interrupt evaluation | Mandatory |
| Context/session manager: contexts, lifespans, TTL sweeps, fulfillment memory, back-references; state serialization *format* | Mandatory |
| Workflow interpreter: validation-rule execution, clarification/confirmation/completion per schema | Mandatory |
| Validation engine: closed predicate set (range, regex, length, must_be_future, …) parameterized by config | Mandatory |
| Bundle verification & table access: signature check, manifest hashing, compat gate, table parsing over host-provided bytes | Mandatory |
| Turn-event assembly + redaction guarantee (decision telemetry only) | Mandatory |
| Explain-trace generation (per-stage decisions for Studio/debug) | Optional (strongly recommended — near-free, transformative for tooling) |

### NATIVE iOS

| Component |
|---|
| ⭐ Speech recognition (SFSpeechRecognizer / SpeechAnalyzer), audio session |
| ⭐ CoreML model execution (TF-IDF logits, MiniLM embedding) behind the Inference Interface |
| Bluetooth / hearing-aid control; execution of all `action` keys |
| SwiftUI, app lifecycle, memory-pressure response (may instruct runtime to drop stage 3) |
| Bundle download, storage, two-slot management, file open/mmap; providing bytes to runtime |
| Session-state persistence: keychain-backed encryption of the runtime's serialized blob |
| Telemetry transport: buffering, sampling, consent gating, upload; native device metrics (memory/battery/load-time) |
| Permissions (mic, speech, notifications); push notifications; analytics SDK |
| Threading/async host: serial executor owning runtime calls |

### NATIVE ANDROID

| Component |
|---|
| ○ Speech recognition (Android SpeechRecognizer, offline mode), audio focus |
| ○ ONNX Runtime Mobile execution (+ NNAPI/XNNPACK selection) behind the Inference Interface |
| ○ Bluetooth / hearing-aid control; execution of all `action` keys |
| ○ Jetpack Compose UI, lifecycle, `onTrimMemory` response |
| ○ Bundle download (WorkManager), storage, file provision to runtime |
| ○ Session-state persistence: Keystore-backed encryption |
| ○ Telemetry transport + native device metrics |
| ○ Permissions; push; analytics SDK |
| ○ Coroutine dispatcher host for runtime calls |

### SHARED CONFIGURATION (fact plane — compiled into the signed bundle)

| Component |
|---|
| ⭐ Intent definitions (id, domain, status, handler action key, platform/language availability) |
| ⭐ Slot definitions (entity ref, required/optional, defaults) |
| ⭐ Entity definitions: enum values, synonym tables, fuzzy flags, normalization maps |
| ⭐ Keyword rules (patterns, tiers, guards) — data interpreted by the rule engine |
| ⭐ Thresholds & policies: confidence floors, gaps, interrupt threshold, attempt budgets, TTLs, keyword-tier confidences |
| ⭐ Calibration values (temperature per model/language) |
| ⭐ Prompts, clarifications, confirmations, fulfillment text — per language |
| ⭐ Yes/no lexicons, idioms, uncertainty phrases, universal-verb lexicons, carrier-phrase tables — per language |
| ○ Conversation flows / workflow definitions (completion conditions, confirmation requirements, back-reference declarations) |
| ○ Validation rule *instances* (parameters binding the runtime's closed predicate set) |
| ○ Datetime grammar tables (per-language lexicons, format patterns) |
| ⭐ Model artifacts + labels + manifest + signature (carried by the bundle; executed natively; verified by runtime) |

---

# Part 3 — Justification per Runtime Component

Format per component: why not native / cost if native / benefit if shared / complexity justified?

**Text pipeline.** Can't stay native: it defines the model's input contract — sklearn-equivalent featurization has *already* drifted (`min_df`, sampling strategy) between two Python files; three languages guarantee worse. Cost if native: silent per-platform accuracy divergence invisible until user reports. Benefit: benchmark numbers become fleet-true. Complexity: lowest of any module (pure functions) — **justified, first to move.**

**Rule engine.** Can't stay native: three regex dialects + ordered-transformation semantics (§ADR-001 2.1–2.3). Cost if native: per-locale matching divergence; every new rule tier re-implemented ×3. Benefit: one matcher semantics; rules become safely editable by the content team because behavior is identical everywhere. Complexity: moderate (needs a defined regex subset) — **justified.**

**Cascade + gates.** Can't stay native: the policy web (base/slot/semantic/agreement/interrupt thresholds interacting) is exactly where iOS already diverged (`conf_gap_threshold`). Cost if native: fallback-rate metrics incomparable across platforms; A/B analysis invalid. Benefit: policy tuning becomes a config change with identical effect everywhere. Complexity: low — **justified.**

**Entity engine + datetime resolver.** Can't stay native: stateful grammar with session interaction (day-parking) — the fixture space is unbounded (§ADR-001 2.9); the team already maintains parity CSVs as a confession. Cost if native: reminders 12h/1-day wrong per platform; the verbatim-capture rule (French reminder titles) class of already-fixed bugs re-introduced by porting. Benefit: retire the parity CSVs as *cross-implementation* tests (they become single-implementation regression tests). Complexity: the highest of any module (port of ~500 lines of subtle logic) — **justified precisely because the alternative is doing that port twice, natively, without a compiler.**

**Dialogue state machine + slot filling + context manager.** Can't stay native: stateful, timing-dependent, order-sensitive logic where parity testing is structurally weakest; every already-fixed bug (stale-context "yes", fuzzy-scan wrong-action, TTL semantics) would be re-discovered per platform on production users. Cost if native: 2 × ~4,000-line ports plus permanent feature lag between platforms. Benefit: dialogue features ship simultaneously everywhere; the state-machine bug class becomes compile-checked. Complexity: high — **justified; this is the trigger condition itself. If this module is not moving, cancel the runtime.**

**Workflow interpreter + validation engine.** Can't stay native: the schema is the platform's central contract; three interpreters = three slightly different meanings of the same YAML, which silently breaks the Studio's promise. Cost if native: content-team definitions behave differently per platform; per-platform workaround forks in *content*. Benefit: "write once, behaves identically" becomes literally true. Complexity: moderate, bounded by the closed-vocabulary rule (§7.3) — **justified.**

**Bundle verification.** Can't stay native: accept/reject must be one rule — a bundle that iOS accepts and Android rejects (or worse, the reverse under attack) is a security incident. Cost if native: two crypto-adjacent implementations to audit. Benefit: one audited gate. Complexity: low (verify-only; signing stays in Python) — **justified.** *(Narrowed: file I/O stays native — §1.8.)*

**Turn-event assembly + redaction.** Can't stay native: the "no raw utterances" invariant must be enforced where the raw text lives — inside the decision path. Cost if native: three redaction implementations, one privacy bug away from a medical-context incident. Benefit: single audit point for the strongest privacy claim the product makes. Complexity: trivial — **justified.** *(Narrowed: everything after event creation is native.)*

**Explain-trace.** Optional. Stays only because it is a byproduct of the decision path (the runtime already knows every gate it evaluated) and powers the Studio testing console with the *real* engine. If runtime size ever matters, it compiles out. **Justified as optional.**

**Components evaluated and REJECTED for the runtime** (the "answer is no" cases Part 3 demands): model execution (platform runtimes are optimal; tensor contract is small and testable — moving it would forfeit ANE/NNAPI and bloat the runtime with ML dependencies); telemetry transport (network is an OS concern; consent is a UI concern); bundle download/storage (ditto); session-blob encryption (keys belong in hardware-backed platform keystores); ASR post-processing hooks (tempting — "shouldn't ASR-garble normalization be shared?" — no: ASR output quirks are per-platform-engine facts; normalize them natively before the boundary, or the runtime accretes platform-specific patches, violating its own definition).

---

# Part 4 — Python's Future Role (challenging ADR-001)

ADR-001 asserted Python "becomes a thin PyO3 wrapper — the oracle problem ends." Challenged properly:

### Option A — Python as thin wrapper over the shared runtime

| Dimension | Assessment |
|---|---|
| Maintainability | One implementation; drift extinct by construction |
| Experimentation | **Worse.** Changing engine *behavior* now requires Rust. A researcher testing "what if interrupts used entropy instead of a threshold" must modify the core or fork it |
| Training | Unaffected (training never used the engine in-process for gradient work; it uses it for evaluation) |
| Testing | Strongest possible: CI, Studio, and eval exercise the shipped logic bit-for-bit |
| Research | Friction for behavior research; none for model research (models are Python-side regardless) |
| Debugging | Rust debugging from Python contexts is adequate (traces, logs) but worse than pdb on pure Python |
| Productivity | Higher for the platform team (no dual maintenance); lower for the one engineer doing dialogue-behavior research |
| Long-term evolution | Single evolution path; behavior changes gated through one codebase |

### Option B — Python as independent reference implementation consuming the same bundle

| Dimension | Assessment |
|---|---|
| Maintainability | **Permanent N=2 duplication of the entire engine** — the exact disease ADR-001 documents, kept alive deliberately at the reference layer |
| Experimentation | Best: full Python hackability |
| Training | Same as A |
| Testing | Weaker than it looks: "reference agrees with runtime" is the same parity treadmill, now self-imposed; the oracle drifts every sprint |
| Research | Best |
| Debugging | pdb-native; but *divergence* debugging (why does reference disagree with runtime?) returns as a permanent tax |
| Productivity | Every engine change costs 2× forever; the reference silently rots the day it stops being anyone's priority (see: `main/README.md` vs `README.md` today) |
| Long-term evolution | Two evolution paths that must be reconciled — a standing committee where a decision should be |

### Ruling

**Option A, with an explicit experimentation escape hatch — and one honest concession.**

Option B's benefits are real but they accrue to *research velocity*, and there is a cheaper way to buy that: the runtime exposes its seams (stage interface, policy table, inference callbacks — the same interfaces §9 defines anyway), so experimental behavior is prototyped **in Python above or beside the core** — an experimental stage injected via the stage interface, a policy variant via config, a new entity resolver as a Python callback in dev builds. Nothing about experimentation requires re-implementing the *stable* engine; it requires the stable engine to be *open at the joints*. The joints are already required for testing and for the Inference Interface, so the escape hatch is nearly free.

The concession: pre-migration, the current Python engine serves as the one-time migration oracle (ADR-001 Part 9 replay). After Phase 3 it is **deleted, not demoted** — a "reference implementation" that no longer gates anything is documentation that compiles, and it will lie within a quarter. Keep the *conversation corpus* (language-agnostic YAML) as the permanent executable spec instead; corpora don't drift, implementations do.

---

# Part 5 — Permanently Native Components

These are never candidates, regardless of how the runtime evolves:

| Component | Why never |
|---|---|
| Speech recognition | OS-integrated engines (SFSpeechRecognizer/SpeechAnalyzer, Android SpeechRecognizer) with permissions, on-device model management, and language packs owned by the OS. A shared abstraction would be the union of two moving APIs — pure liability. Output (text) is already the platform-neutral contract. |
| CoreML / ONNX Runtime / NNAPI execution | Hardware acceleration (ANE, NNAPI delegates) is *the* reason inference is fast and battery-cheap; abstracting it into the runtime forfeits vendor optimization and welds ML-runtime dependencies into a logic library. The tensor-level Inference Interface is the correct seam. |
| Bluetooth / hearing-aid control | Safety-critical device control with platform stacks, pairing, and regulatory surface. The runtime emits `action` keys; it must never know what a radio is. |
| UI (SwiftUI / Compose) | Self-evident; also the accessibility surface, which is deeply platform-specific and critical for this user base. |
| Audio session / focus management | OS arbitration between the app, the hearing aids, and other apps. |
| App lifecycle & memory pressure | The OS tells *the app*; the app may instruct the runtime (e.g., drop stage 3), never the reverse. |
| Networking (bundle download, telemetry upload, GenAI calls) | Connectivity awareness, background transfer, TLS, proxies, certificate handling — OS stacks do this correctly; a runtime HTTP client would do it worse and double the security surface. |
| Storage & file system | Sandbox rules, backup exclusion flags, storage quotas are platform semantics. Runtime consumes byte views. |
| Keys, keystores, biometrics, permissions | Hardware-backed security must use platform APIs (Keychain/Keystore) to exist at all. |
| Push notifications, analytics SDKs, crash reporting | Vendor SDKs are platform artifacts; also consent-gated UI concerns. |
| Model download & bundle slot management | Networking + storage + retry policy = native by composition; the runtime only *verifies* what it is handed. |

---

# Part 6 — Configuration vs Code, per Subsystem

Categories: **Hardcoded** (in some codebase, changed via release) · **Configuration** (bundle data, changed via content release) · **Generated code** (derived from config at build time) · **Runtime logic** (shared runtime behavior).

| Subsystem | Category | Reasoning |
|---|---|---|
| Intent definitions | **Configuration** | The platform's core content; must be editable by the Studio without engineering. |
| Slot definitions | **Configuration** | Part of intent contracts; same lifecycle. |
| Entities & synonyms | **Configuration** | Vocabulary facts; linguist-owned; per-language. |
| Keyword rules | **Configuration** (patterns, tiers, guards) interpreted by **runtime logic** (matcher) | Rules change weekly with content; matcher semantics change yearly with releases. Splitting them puts each on its correct cadence. |
| Conversation flows / workflow definitions | **Configuration** interpreted by **runtime logic** | Flows are per-intent facts; execution semantics must be single-sourced. |
| Policies & thresholds | **Configuration** | Tuning knobs derived by calibration and adjusted by rollout evidence; must never require an app release. |
| Validation rules | **Configuration** *instances* over a **runtime-logic** closed predicate set | The anti-DSL line (§7.3): config says `{type: range, min: 0, max: 100}`; the runtime implements `range`. Config may not express arbitrary logic — the day validation config contains expressions, three interpreters of a new programming language exist again. New predicate *kinds* are deliberate runtime releases. |
| Prompts / clarifications / confirmations | **Configuration** | Localized text; content-team owned; A/B-able. |
| Yes/no lexicons, idioms, universal verbs, carrier phrases | **Configuration** | Language facts, not algorithms. (Today several are hardcoded in `engine.py` — `_NO_IDIOMS`, `_CARRIER` — this spec mandates their extraction.) |
| Datetime grammar | **Configuration** tables interpreted by **runtime logic** resolver | Lexicons/formats per language are facts; parking/anchoring/rollover policy is behavior. |
| Calibration temperatures | **Configuration** | Model-coupled values produced by training; travel with the model in the bundle. |
| Action keys → app handlers | **Generated code** | Generate typed constants (Swift enum / Kotlin sealed class) from the schema so an intent whose handler the app hasn't implemented is a *compile* error, not a runtime silent no-op. The one place codegen clearly beats both config and hand-written code. |
| FFI bindings | **Generated code** (UniFFI) | Hand-written bindings are Option-C's rejected hazard. |
| Golden fixtures / parity corpora | **Generated code** (from datasets + recorded runtime outputs) | Hand-maintained fixtures rot; generated ones regenerate. |
| Cascade order, state-machine transitions, fuzzy algorithm, tokenizers | **Runtime logic** | Behaviors whose identity across platforms is the product requirement. Not config: making control flow configurable is the DSL trap. |
| Engine-internal constants with no tuning meaning (e.g., stable-softmax epsilon, buffer sizes) | **Hardcoded** (runtime) | Not everything deserves a knob. A knob nobody should turn is an incident waiting for a hand. |
| Build/CI settings, signing keys, registry endpoints | **Hardcoded** (build system / CI secrets) | Build-plane concerns; never ship in bundles. |

**Litmus test:** *if the content team changed it, would you want a retrain or re-release to be unnecessary?* → Configuration. *If it changed, would every platform need to behave differently in lockstep?* → Runtime logic. *Is it a mapping the compiler could check?* → Generated code.

---

# Part 7 — Runtime Interfaces (architecture, not code)

Seven interfaces define the entire boundary. Everything crosses through one of these; anything that can't is in the wrong layer.

### 9.1 Inference Interface (native → provides; runtime → consumes)
- **In:** sparse TF-IDF feature vector (index/value pairs) *or* token-id sequence + attention mask.
- **Out:** logit vector / embedding vector (fixed dims declared by the bundle).
- **Responsibilities:** native executes the model referenced by the active bundle on its optimal backend; numerical contract (dtype, dimension, ordering) is declared in the bundle manifest and asserted by the runtime.
- **Lifecycle:** registered at session-factory construction; re-registered on bundle swap (model handles change).
- **Errors:** native reports `InferenceFailed{stage, cause}`; runtime degrades per policy (stage-2 failure → keyword-only + fallback; stage-3 failure → skip rescue) and emits a telemetry event. The runtime never retries — retry policy is a host concern.
- **Ownership:** input buffers borrowed for the call; output vectors owned by the runtime after return.

### 9.2 Bundle Interface (native → provides bytes; runtime → verifies & serves)
- **In:** read-only byte view of a candidate bundle + its origin tag (baked | downloaded).
- **Out:** `Verified(bundle_id, compat, report_card)` or a typed rejection (`BadSignature`, `HashMismatch`, `Incompatible{needs}`).
- **Responsibilities:** runtime owns every accept/reject rule; native owns acquisition, storage, the two-slot scheme, and deciding *when* to attempt a swap (foreground idle, charging, etc.).
- **Lifecycle:** verify → warm (runtime primes tables; native primes models) → atomic activate → old bundle released. Rollback = re-activate previous slot; runtime is stateless across swaps except live sessions, which complete on the old bundle.
- **Ownership:** bytes owned by native (it did the I/O); runtime holds the view only while the bundle is active.

### 9.3 Session Interface (the primary API; native → calls; runtime → decides)
- **In:** `handle(session_id, text)`; also `reset(session_id)`, `serialize(session_id)`, `restore(blob)`.
- **Out:** `NLUResult{type, intent, action, params, prompt, confidence, trace?}` — the *complete* decision; native never post-processes meaning.
- **Responsibilities:** runtime owns all conversational state and its TTL semantics; native owns *when* to serialize (backgrounding) and where the encrypted blob rests.
- **Lifecycle:** sessions created lazily, expired by TTL on the runtime's injected clock (native supplies a monotonic clock provider — the runtime never reads wall time directly, keeping state logic deterministic and testable).
- **Errors:** malformed/oversized input is bounded and handled, never thrown; internal panics surface as `InternalError` results (degrade the turn, never the app).
- **Ownership:** results owned by native on return (copied across the boundary); session state owned by the runtime behind opaque handles.

### 9.4 Action Interface (runtime → emits; native → executes)
- **In (to native):** `action` key + typed params from the fulfilled intent (`volume.increase`, `reminders.add{name, datetime, recurrence}`).
- **Out (back to runtime):** optional execution acknowledgment (`ok | failed{reason}`) so failed executions can drive an apologetic prompt and telemetry, without the runtime knowing *why* radios fail.
- **Responsibilities:** the generated action-key constants (§7) make the contract compile-checked; the runtime guarantees params passed validation before emission.
- **Lifecycle:** fire-per-fulfillment; acknowledgment optional and asynchronous.
- **Ownership:** the semantic decision is the runtime's; the physical act is native's. This is the load-bearing line of the whole spec.

### 9.5 Telemetry Interface (runtime → emits events; native → transports)
- **Out (from runtime):** structured turn events (stage, intent, confidence bucket, latency, flags — schema-versioned, redaction-guaranteed, never raw text).
- **Responsibilities:** runtime guarantees content; native owns buffering, sampling, consent gating, batching, upload, and merging with native-origin metrics (memory, battery, model-load).
- **Lifecycle:** synchronous handoff of an owned event object per turn; native drains on its own schedule.
- **Errors:** telemetry must never affect a turn — the handoff is infallible by design (native buffers or drops).

### 9.6 Workflow/Content Interface (configuration → runtime, at bundle load)
- Not a call interface — a *format contract*: the compiled schema tables (intents, slots, flows, policies, lexicons, grammars) with a declared format version. The runtime's compat gate refuses future formats; the compiler refuses to emit tables for behaviors the target runtime version lacks. This is the interface that lets content release independently of code on both planes.

### 9.7 Validation Interface (internal, but named because the Studio shares it)
- The closed predicate vocabulary (§7) with declared parameter schemas. Exposed read-only so the Studio can render rule editors and the compiler can reject unknown predicates — one vocabulary, three consumers (runtime, compiler, Studio), zero re-implementations.

```
        ┌ Native (iOS/Android) ────────────────────────────────┐
        │  ASR   UI   BT/actions   download   storage   telem  │
        └──┬───────▲──────▲───────────┬──────────┬────────▲────┘
   handle()│       │      │           │bytes     │blob    │events
      (9.3)│  NLUResult  (9.4)   (9.2)│     (9.3)│   (9.5)│
        ┌──▼───────┴──────┴───────────▼──────────▼────────┴────┐
        │                    SHARED RUNTIME                     │
        │   cascade · dialogue · entities · workflow · verify   │
        └──────────────┬────────────────────────────▲───────────┘
                 (9.1) │ features/tokens            │ logits/vectors
        ┌──────────────▼────────────────────────────┴───────────┐
        │        Native inference (CoreML / ORT / NNAPI)        │
        └────────────────────────────────────────────────────────┘
```

---

# Part 8 — What Should NEVER Move Into the Shared Runtime

Beyond Part 5's list, the standing prohibitions with reasons — these hold even under future pressure:

1. **Any OS API call.** The runtime is a pure library: no syscalls beyond memory, no threads of its own, no files, no sockets, no clocks (injected). This single property is what makes it embeddable in three hosts, fuzzable, and deterministic. Every exception ever granted erodes all three.
2. **Networking of any kind** — including "just the GenAI call" (stage 4). The runtime *decides* to fall back; native *makes* the request. A runtime with an HTTP client acquires TLS, proxies, consent, and background-transfer semantics — three platforms' worth of the exact problems it exists to escape.
3. **Cryptographic key custody.** Verification with a baked public key: yes. Private keys, key generation, secure storage: never — hardware-backed keystores are unreachable from a portable library *by definition*.
4. **UI or UX policy** — including prompt *timing*, TTS voice choice, haptics. The runtime says *what* to ask; native decides *how and when* it reaches the human.
5. **Analytics/crash vendor SDKs.** Vendor lock inside the shared core would hold every platform hostage to one SDK's platform support matrix.
6. **App lifecycle awareness.** The runtime must not know if it is foregrounded; hosts translate lifecycle into explicit calls (serialize, drop-stage-3). Inverted knowledge here creates the untestable "library that behaves differently when backgrounded" class of bug.
7. **Model execution** (restated for permanence): the day someone proposes "just embed a tiny ORT in the core for consistency," the answer is that consistency of *tensors* is already tested at a 20-line interface, and the cost is every future hardware accelerator.
8. **Platform-specific ASR cleanup.** Each engine's quirks are normalized natively, before the boundary, or the "shared" runtime silently forks per platform from the inside.
9. **Anything the content team changes weekly.** If it changes at content cadence, it is configuration; runtime code changes at release cadence. (The inverse of the §7 litmus test, as a prohibition.)

---

# Part 9 — Migration Readiness: Decisions to Make Today

Everything below is cheap now, pays regardless of whether Rust ever lands, and converts the future migration from archaeology into mechanics:

1. **Adopt this document's interfaces in the Python engine now.** Restructure `scripts/nlu/` so `engine.py` consumes an `InferenceBackend`, an injected clock (already done — keep), a `BundleView`, and emits telemetry events through one function. The Python engine becomes the first *host* of the future runtime's shape. Migration then means replacing an implementation behind interfaces that already exist, with tests that already target them.
2. **Make the bundle format implementation-agnostic and versioned from v1.** No pickles (Python-only — `intent_labels.pkl` violates this today; JSON twin already exists, make it primary), no sklearn-coupled table layouts, declared format version + compat rules. The bundle is the contract both the current Python engine and any future runtime read — if both read the same bytes, cutover is a swap, not a translation.
3. **Extract every hardcoded behavior-fact into configuration now** (`_CARRIER`, `_NO_IDIOMS`, `_UNCERTAIN`, thresholds, keyword confidences per §7). Each extraction shrinks the future ported surface and is independently valuable for localization this quarter.
4. **Freeze the closed validation-predicate vocabulary and the workflow behavior vocabulary in the schema spec** — documented semantics, not implementation-defined ones. The migration's hardest part is reverse-engineering intent from code; a written semantic spec (with the conversation corpus as executable examples) eliminates it.
5. **Grow the conversation corpus (YAML, language-agnostic) as the primary spec.** Every dialogue bug fixed gets a corpus entry. At migration time this corpus — not the Python code — is the acceptance suite; it is the one artifact guaranteed not to rot, because CI runs it forever.
6. **Keep the parity CSVs and golden fixtures generated, not hand-edited,** with a single generator script — so at migration they regenerate against the new runtime instead of being hand-audited.
7. **Enforce the "no OS calls in engine code" rule in Python immediately** (lint: no `requests`, no `open()` outside the bundle loader, no `time.time()` outside the injected clock). A Python engine that already obeys Part 8 is a port; one that doesn't is a rewrite.
8. **Document decision semantics as behavior tables** (routing priority, gate order, TTL rules) in the schema spec, reviewed like code. These tables are the future runtime's requirements document, written while the knowledge is fresh.
9. **Run the one-sprint FFI toolchain spike** (ADR-001 action 5) *before* it blocks anything, so the packaging pipeline (XCFramework/AAR/wheel) is boring by the time it matters.
10. **CI gate on the interfaces:** any PR that adds engine logic reaching around an interface (e.g., classifier reading a file directly) fails review by policy, citing this document.

Each decision reduces migration risk the same way: it moves knowledge out of implementations and into contracts, formats, and corpora — the three things a migration can consume mechanically.

---

# Part 10 — The Four-Question Test (restated for daily use)

Before implementing any new functionality, answer in order:

1. **Does the OS/hardware own it?** → Native. (§5, §8)
2. **Does it run only at build time?** → Python. (§4 matrix)
3. **Would platform disagreement change what the user experiences?** If no → Native, platforms free to differ.
4. **Is it a fact or a behavior?** Fact → Configuration. Behavior → Shared Runtime — and extract its facts into configuration first, so the runtime gains only the interpreter.

Tie-breakers: prefer configuration over runtime (data over code), runtime over native (once over thrice), native over runtime *whenever Q1 or Q3 gives permission* (the runtime stays minimal by default, not maximal). If a component seems to need splitting, split it along the fact/behavior line (§7) or the decide/execute line (§9.4) — those two seams are load-bearing everywhere in this platform.

---

## Consequences

**Easier:** every "where does this go?" debate resolves by flowchart; the Python engine can begin conforming to the target shape this quarter; content velocity decouples from release cadence on both planes; the future migration consumes contracts instead of reverse-engineering code.

**Harder:** discipline costs — extracting hardcoded lexicons, maintaining the corpus, honoring the no-OS-calls lint — land now, before the runtime exists to enforce them; the closed predicate vocabulary will occasionally frustrate someone who wants one clever expression in config.

**Revisit when:** the Android trigger fires (execution review); any component resists classification by the four-question test twice (the test is wrong, amend this document); the workflow vocabulary grows past ~20 behavior kinds (DSL-creep audit).

## Action Items

1. [ ] Ratify this specification as the boundary reference (link from CONTRIBUTING).
2. [ ] Refactor Python engine to the §9 interfaces; add the no-OS-calls lint.
3. [ ] Extract hardcoded lexicons/policies to configuration (§7 table, "hardcoded today" items).
4. [ ] Make JSON labels primary; deprecate pickle artifacts from the bundle path.
5. [ ] Establish the conversation corpus as a first-class dataset with CI execution.
6. [ ] Add the four-question test to the PR template for engine-adjacent changes.
