# ADR-002: Capability & Action Execution Architecture
## How Features Are Organized and Registered, and How the Shared Runtime's Decisions Become Platform Actions

**Status:** Proposed — covers both follow-up requirements from the ADR-001.1 approval
**Date:** 2026-07-13
**Depends on:** ADR-001 (shared runtime direction), ADR-001.1 (boundary specification — extends its Action Interface §9.4)

This ADR has two parts, deliberately in one document because they are two halves of one seam:
**Part A — Capability Architecture:** what a feature *is*, how it is packaged, registered, and made available.
**Part B — Action Execution & SDK Architecture:** how the shared runtime communicates with those feature modules.

---

# Part A — Capability Architecture

## A1. Problem

ADR-001.1 fixed where logic lives, but the platform still has no answer to "what is a *feature*?" Today the answer is implicit and scattered: 59 intents in one flat schema, action keys (`volume.increase`, `genai.fallback`) with no owner, handler code somewhere in each app, help content in a zip, and no mechanism for the obvious product reality that **feature availability varies** — by platform, by app version, by hearing-aid model (WiCROS, fall alert, heart rate are hardware-dependent), by connection state (most device commands require paired aids), and by region/consent (telehealth, GenAI).

Without a feature-organizing principle: teams cannot own slices end-to-end; the classifier happily recognizes "start streaming" on a device that cannot stream and the app silently no-ops; adding a feature touches the schema, both apps, and the training set with no single artifact tying them together; and the Studio has nothing to show a PM except a flat list of 59 intents.

## A2. Decision

Organize the platform around **Capabilities**: versioned vertical slices that bundle everything one feature needs — content, models' training data references, native handlers, and availability rules — under one manifest and one owner. Capabilities are the unit of ownership, packaging, availability, and (eventually) independent content release.

> **Definition.** A Capability is a named, versioned unit consisting of: (a) a **content package** (intents, entities, prompts, flows, keyword rules, training-data references), (b) an **action contract** (the action keys it emits, with typed parameter schemas), (c) **native modules** per platform implementing that contract, and (d) an **availability specification** (conditions under which it is usable). The NLU runtime knows capabilities only through (a), (b), and (d) — never (c).

## A3. Capability Map (initial decomposition of the current 59 intents)

| Capability | Intents (today's names) | Native dependency profile |
|---|---|---|
| `device.volume` | Cmd.Volume{Increase, Decrease, Mute, Unmute} | BT connection to aids |
| `device.memory` | Cmd.MemoryChange (+ back-reference) | BT connection; memory list is *device-provided* (see §A6.3) |
| `device.status` | Cmd.BatteryLevel | BT connection |
| `streaming` | Cmd.Streaming{Start, Stop} | BT + aid model supports streaming |
| `reminders` | reminders.add, reminders.complete | Local notifications permission |
| `messaging.ptt` | Cmd.SendMessage, Cmd.ListenMessage *(the `- yes/- no` labels dissolve into confirmation flows per the earlier review)* | Mic permission, account |
| `translation` | Cmd.TranslationStart | Mic; on-device or cloud translation stack |
| `transcription` | Cmd.TranscribeStart | Mic |
| `find` | Cmd.FindMyPhone (+ future FindMyAids) | Platform find APIs / BT ranging |
| `activity` | Cmd.Activity{Step, Run, Walk, Stand, Cycle, Exercise, Aerobics, Calories} | Health data permission; aid sensors |
| `diagnostics` | SelfCheck | BT; aid model support |
| `telehealth` | TeleHearAI, RemoteProgramming-adjacent | Account, region, clinician link |
| `help` | all ~30 `Help_*` intents | None (static content) — future RAG surface |
| `sys` (reserved) | fallback, OOS, cancel/repeat/correct verbs, GenAI routing | Owned by the platform team, not a product feature |

Decomposition rule: a capability boundary follows the **native dependency profile and product ownership**, not linguistic similarity. `device.volume` and `streaming` both talk BT but differ in hardware gating and ownership; `help` is one capability despite 30 intents because it has one dependency profile (none) and one owner.

## A4. Capability Anatomy

```
content/capabilities/reminders/
├── capability.yaml            # THE manifest (below)
├── intents/
│   ├── add.yaml               # full workflow definition (ADR-001.1 §7 format)
│   └── complete.yaml
├── entities/
│   └── recurrence.yaml        # capability-scoped entity (global entities live in content/entities/)
├── prompts/{en,fr,de,da}.yaml # localized prompt/clarification/confirmation text
└── datasets → dvc reference   # training utterances tagged to this capability
```

The manifest is the single source of truth the compiler, the apps, the Studio, and CI all read:

```yaml
# capability.yaml (illustrative structure, not code)
id: reminders
version: 2                       # bumped on action-contract or intent-contract breaking change
owner: team-companion-app
status: active                   # draft | active | deprecated
platforms: [ios, android]
languages: [en, fr, de, da]
actions:                         # the contract Part B dispatches against
  - key: reminders.add
    params: {name: string, datetime: iso8601, recurrence: enum?}
    execution: {requires: [notifications_permission], mode: sync, cost: low}
  - key: reminders.complete
    params: {name: string?}
    execution: {requires: [], mode: sync, cost: low}
availability:                    # evaluated natively, snapshot passed to runtime (§A6)
  requires_permissions: [notifications]
  requires_device_features: []   # e.g. streaming lists [aid.streaming]
  requires_connection: none      # none | aids_paired | aids_connected
  feature_flag: reminders_v2     # remote-config kill switch
unavailable_responses:           # localized, per condition — content, not code
  notifications_permission: {en: "I need notification permission to set reminders — you can enable it in Settings."}
```

## A5. Registration Model

Two registration planes, deliberately separate:

**Build-time (static) registration.** Each app declares its installed capability modules through ordinary dependency injection at composition root. From the compiled bundle's capability manifests, codegen (ADR-001.1 §7, "generated code") produces per-platform **action-key constants and typed parameter structs per capability**; app CI fails if a capability the app claims to install has action keys with no registered handler — *a feature gap is a compile/CI error, not a runtime silence.* Conversely, a handler registered for an action absent from the bundle is a CI warning (dead code or version skew).

**Runtime (dynamic) availability.** Static registration says "this app *can* do reminders"; availability says "this phone, right now, *may*." The native **CapabilityRegistry** (owned by the SDK, §B4) evaluates each installed capability's `availability` spec against live state — permissions, paired-aid feature set, connection state, feature flags, region — and maintains an **availability snapshot**: `{capability_id → available | unavailable(condition)}`. The snapshot is pushed to the shared runtime on session start and on state change (aid disconnects mid-conversation).

## A6. Availability Semantics in the NLU (the key design decision)

**Unavailable capabilities remain in the classifier's label space and are still recognized.** The runtime routes a recognized-but-unavailable intent to the capability's declared `unavailable_response` instead of fulfillment.

Rationale, because the alternative (removing unavailable intents from classification) looks attractive and is wrong:

1. **Recognition ≠ execution.** A user who says "start streaming" on non-streaming aids deserves "your hearing aids don't support streaming," not a GenAI fallback or a misclassification into the nearest available intent. Removing labels *redistributes* probability mass onto wrong intents — the wrong-action class of failure, now caused by configuration.
2. **One model per language, not per device matrix.** Per-availability models would explode the training/calibration matrix (platform × aid model × permission state). One label space, filtered at fulfillment, keeps the ML surface flat.
3. **Telemetry sees demand.** "Recognized but unavailable" events are a product signal (how many users ask for streaming on non-streaming aids?) that label removal would destroy.

Mechanics: availability filtering is a **fulfillment-stage gate in the shared runtime** (it is a decision affecting what the user experiences — Q3 of the boundary test), driven entirely by the snapshot + manifest data (facts — configuration). Slot-filling never *starts* for an unavailable intent (don't collect a reminder's details and then refuse). Mid-conversation loss (aids disconnect during a volume flow) is an interruption event: the runtime abandons the flow with the capability's `connection_lost` response — this rule lives in the runtime because it is dialogue behavior, per ADR-001.1 Part 6.

### A6.3 Device-provided vocabulary

`device.memory`'s enum values (the user's actual configured memory presets) are **device state, not bundle content**. The manifest may declare an entity as `source: dynamic`; the SDK feeds current values into the runtime's entity engine via the availability/context snapshot. Same mechanism later serves contact names for messaging and personalization overlays (roadmap). This is the one sanctioned path for runtime entity injection — everything else in the entity space stays bundle-compiled.

## A7. Versioning & Compatibility

- **Capability version** gates action-contract changes: the bundle records each capability's version; the generated native constants embed it; the SDK refuses to dispatch to a handler whose registered contract version mismatches the bundle's (fail-fast at bundle activation with a typed error, not per-turn).
- **Bundle ⊇ app tolerance:** a bundle may contain capabilities an app build doesn't install (older app, staged feature) — those behave as `unavailable(app_version)`. An app may install modules for capabilities absent from the bundle — handlers idle harmlessly. This bidirectional tolerance is what lets content and app releases stay decoupled (ADR-001.1 §9.6).
- **Deprecation:** `status: deprecated` + `superseded_by` keeps recognition alive while routing to the successor, preserving telemetry continuity during intent-taxonomy migrations (directly needed for the `Cmd.*`/`Help_*` rename already planned).

## A8. Ownership & Studio Alignment

Capabilities are the Conway boundary: a team owns its capability's content package, native modules on both platforms, datasets, and quality metrics (per-capability F1, wrong-action budget, fallback rate — the per-domain budgets the earlier review demanded now have a natural home). CODEOWNERS maps `content/capabilities/<id>/` and the native module paths to the owning team. The Studio's navigation *is* the capability map; per-capability dashboards replace the flat 59-intent list. The `sys` capability (fallback, universal verbs, GenAI routing) is owned by the platform team and is the only capability without native handlers.

## A9. What a Capability May NOT Do

Guardrails preventing the abstraction from leaking:

1. May not reference another capability's intents, entities, or handlers. Cross-feature conversations ("remind me to do a self-check") are the *dialogue layer's* composition (an open-text reminder mentioning self-check is still just a reminder), never inter-capability calls.
2. May not ship runtime *logic* — a capability is content + contracts + native handlers; any "custom matching code" a capability wants is a new runtime predicate/behavior proposal (ADR-001.1 anti-DSL rule) through platform-team review.
3. May not bypass the SDK dispatcher to observe NLU internals (no capability sees another's turns, confidence values, or raw text).
4. May not declare availability conditions the registry cannot evaluate from declared state (no arbitrary native availability callbacks in v1 — conditions are a closed vocabulary: permissions, device features, connection, flag, region, app version — same closed-vocabulary discipline as validation predicates).

---

# Part B — Action Execution & SDK Architecture

## B1. Problem

Part A defined the feature modules; ADR-001.1 drew the load-bearing line — *the runtime decides, native executes*. The unspecified middle is everything between `NLUResult{action: "volume.increase"}` and a Bluetooth packet reaching a hearing aid: who routes the action, on what thread, with what timeout, what happens when execution fails mid-conversation, how long-running actions (streaming, translation sessions) report back, how the app embeds all of this without knowing the runtime exists, and how any of it is testable. Left implicit, each app will invent its own dispatch layer — re-creating per-platform behavioral drift one layer above the one we just eliminated.

## B2. Decision

Ship a thin **NLU SDK** per platform (Swift package / Android library) that owns the entire seam: runtime hosting, capability dispatch, execution-outcome feedback into dialogue, and telemetry stitching. Apps interact with two surfaces only — a **façade** (turns in, results out) and the **capability handler contract** (actions in, outcomes out). Everything between is SDK-internal and identical in design on both platforms.

## B3. SDK Layering

```
┌─ APP ────────────────────────────────────────────────────────────────┐
│  UI / ViewModels          Capability modules (Part A)                │
│      │ speak/show              ▲ typed ActionRequest → ActionOutcome │
└──────┼─────────────────────────┼─────────────────────────────────────┘
       ▼                         │
┌─ NLU SDK ────────────────────────────────────────────────────────────┐
│  NluFacade            — start/stop, handle(text), session control,   │
│                          conversation events stream                  │
│  ├─ RuntimeHost       — owns nlu-core instance, serial executor,     │
│  │                       clock provider, availability snapshot push  │
│  ├─ ActionDispatcher  — action key → handler routing, timeout,       │
│  │                       dedup, outcome → runtime feedback           │
│  ├─ CapabilityRegistry— static registration + dynamic availability   │
│  │                       evaluation (§A5)                            │
│  ├─ InferenceBackend  — CoreML / ORT adapters (ADR-001.1 §9.1)       │
│  ├─ BundleManager     — download/store/verify-via-runtime/swap       │
│  └─ TelemetryAgent    — merges runtime turn events + execution       │
│                          events + device metrics; consent gate;      │
│                          batch upload                                │
└──────┬───────────────────────────▲───────────────────────────────────┘
       ▼ FFI (UniFFI)              │ inference & clock callbacks
┌─ SHARED RUNTIME (nlu-core) ──────┴───────────────────────────────────┐
│  cascade · dialogue · entities · workflow · bundle verify · events   │
└──────────────────────────────────────────────────────────────────────┘
```

Design rule: **the SDK contains no feature logic and no dialogue logic** — it is plumbing with strong opinions about threading, failure, and observability. Feature logic is in capabilities; dialogue logic is in the runtime. If a piece of SDK code mentions a specific capability, it is in the wrong layer.

## B4. The Action Contract (extends ADR-001.1 §9.4)

Every action key declared in a capability manifest (§A4) carries an execution descriptor the dispatcher enforces:

| Field | Values | Dispatcher behavior |
|---|---|---|
| `mode` | `sync` \| `async` \| `session` | §B6 |
| `cost` | `low` \| `high` | `high` requires the runtime to have obtained confirmation (the dispatcher *asserts* the result carries `confirmed=true` — defense in depth; the schema should already have forced a CONFIRM turn) |
| `requires` | closed vocabulary: permissions, `aids_connected`, `foreground`, `network` | precondition-checked *before* handler invocation; failure short-circuits to a typed outcome without touching the handler |
| `timeout` | duration (default 3s sync / 15s async) | expiry produces `Failed(timeout)` — a handler may never hang a conversation |
| `idempotency` | `safe_to_repeat` \| `dedup_window(ms)` | identical action+params within the window is dropped with `Duplicate` outcome (double-fire from ASR echo / rapid partials) |

**Typed dispatch, not stringly dispatch.** Handlers receive generated parameter structs (ADR-001.1 §7 codegen), so `reminders.add` params arrive as `{name, datetime, recurrence?}` — a bundle/handler contract mismatch is caught at bundle activation (version gate, §A7), never as a runtime cast failure.

**Outcome taxonomy** (closed, exhaustive — every handler returns exactly one):

```
Success(payload?)                       — optional data for dialogue ("battery is at 70%")
Failed(reason: transient | device_unreachable | permission_denied |
       precondition_failed | unsupported | timeout | internal)
NeedsUser(kind: permission_flow | os_settings | pairing_flow)
Duplicate                               — dedup window hit; dialogue treats as Success
```

## B5. Execution Flow (normative sequence)

```
 1. App: facade.handle(text)                          [app → SDK, async]
 2. RuntimeHost: serial-executor hop → runtime.handle  [SDK → FFI]
 3. Runtime returns NLUResult                          [decision complete]
 4. type == PROMPT/CONFIRM → facade emits Prompt event; STOP (no dispatch)
    type == FALLBACK(GENAI) → facade emits GenAI event; app owns the network call; STOP
    type == FULFILL ▼
 5. Dispatcher: registry lookup (action → handler)
     ├─ capability unavailable now? → runtime.notifyExecution(unavailable(condition))
     └─ preconditions (requires) → fail fast if unmet
 6. Handler invoked on dispatcher's executor (handler declares main-thread
    need via contract; dispatcher hops accordingly)
 7. Outcome (or timeout) returned
 8. Dispatcher → runtime.notifyExecution(outcome)      [feedback, §B7]
 9. Runtime may emit a follow-up NLUResult (failure prompt, success prompt
    with payload interpolation: "It's at {battery_level}%")
10. TelemetryAgent stitches turn event ⊕ execution event under one turn_id
11. Facade emits final ConversationEvent to the app (speak/show)
```

Steps 5–9 are invisible to the app; the app experiences a stream of conversation events (`Prompt`, `Confirm`, `Executed`, `ExecutionFailed`, `GenAIRequested`) and never routes actions itself.

## B6. Execution Modes

**`sync`** (volume, memory change, battery query): handler completes within timeout; outcome feeds dialogue immediately. The common case; the whole flow is one round trip.

**`async`** (find-my-phone ring, self-check): handler acknowledges initiation quickly (`Success(initiated)`); completion arrives later as an **out-of-band execution event** the dispatcher forwards via `notifyExecution` with the originating `turn_id`. Dialogue policy for late results is schema-declared per action: `announce` ("Self-check complete — everything looks good"), or `silent` (result surfaces in UI only). The runtime holds no timer — expiry of the *conversation's* interest is the standard context TTL; a late result after context expiry downgrades to `silent` automatically.

**`session`** (streaming, translation session, PTT listen): the action opens a long-lived native session owned entirely by the capability; the conversation's job ends at successful start. Stop/status arrive as *new turns* ("stop streaming") — the dialogue layer never holds a handle to a live session. The capability pushes session-state changes into the **availability/context snapshot** (§A6.3 mechanism), which is how "stop streaming" can be recognized as relevant and how a dropped stream can, per schema, generate a proactive prompt.

This trichotomy is the whole model — a new action must fit one of the three or be redesigned; no bespoke lifecycles.

## B7. Failure Feedback Is Dialogue, Not Plumbing

The design's key inversion: **execution outcomes flow back into the runtime and produce the user-facing response from schema-configured content**, exactly like any other dialogue turn. `Failed(device_unreachable)` on `volume.increase` yields the capability's configured response ("I can't reach your hearing aids — are they in the case?") selected by the *runtime*, localized from the *bundle*, identical on both platforms. The SDK maps only the closed outcome taxonomy; it never composes user-facing text. Consequences:

- Retry semantics are conversational ("try again" is a back-reference the runtime already supports), not silent SDK retries — a medical-adjacent device command must never re-fire invisibly.
- `NeedsUser(permission_flow)` produces a runtime prompt *plus* a structured event the app uses to open the OS flow; when the flow completes, availability snapshot updates and the user simply repeats the command (v1 deliberately does **not** auto-resume the pending intent — resumption-after-permission is a v2 dialogue feature, noted in §B10).
- Every failure is a telemetry pair (decision event + execution event, one `turn_id`) — fleet-level "intent recognized correctly but execution failed" becomes a first-class metric, distinguishing NLU quality from device/BT quality. Today these are indistinguishable, which corrupts any accuracy claim made from production data.

## B8. Threading & Reentrancy

One serial executor owns all runtime calls (ADR-001 Part 7 contract): turns, `notifyExecution`, snapshot pushes — totally ordered, no runtime-internal locking ever contended. Handlers run off that executor (dispatcher hops), so a slow handler can never block the next turn's *classification* — but note the dialogue consequence: a turn arriving while its predecessor's `sync` action is in flight is queued *behind the notifyExecution* (order preserved: decide → execute → feedback → next decision). For `async`/`session` modes, new turns interleave freely by design. Reentrancy rule for capabilities: a handler may not call the façade (no self-injected turns); proactive capability speech (dropped stream) goes through the declared snapshot/prompt mechanism, keeping the runtime the sole author of conversation.

## B9. Testing Architecture

| Layer | Instrument |
|---|---|
| Capability handlers | Contract tests generated from the manifest: every declared action × {each outcome} — handler proves it returns only taxonomy outcomes within timeout |
| Dispatcher | Fake registry + scripted handlers: precondition short-circuits, timeout, dedup window, mode lifecycles, outcome ordering |
| Runtime feedback loop | Conversation corpus (ADR-001.1 §9.5) extended with **execution stanzas**: a YAML turn may declare `execution: Failed(device_unreachable)` and assert the follow-up prompt — failure dialogue becomes corpus-testable without hardware |
| App integration | The SDK ships a `FakeNluFacade` (scripted events) so UI tests never load models, and a `RecordingRegistry` so E2E tests assert dispatched actions without touching BT |
| CI completeness gate | Generated check: installed capabilities' handler registrations ⊇ bundle action keys for that platform (§A5) |

## B10. Deliberate v1 Exclusions

Named so their absence is a decision, not an oversight: auto-resume of a pending intent after a `NeedsUser` flow completes (v2 dialogue feature); cross-action transactions (no multi-action atomicity — multi-intent utterances are clarified, not batched, per the platform review §11); SDK-level silent retries (conversational retries only, §B7); capability-to-capability messaging (prohibited, §A9); streaming partial-ASR turns into the runtime (endpoint-final text only in v1 — partials change cascade semantics and are a runtime feature proposal, not an SDK option).

---

## Consequences

**Easier:** feature teams ship end-to-end slices with declarative, testable availability; per-capability quality budgets and telemetry come for free; the Studio gets its information architecture; the intent-taxonomy migration gains a safe mechanism (deprecation routing); apps shrink to UI + capability modules with no routing logic; failure UX becomes localized shared content; NLU quality and execution quality separate cleanly in telemetry; failure dialogue is corpus-testable without hardware.

**Harder:** the flat schema must be repartitioned into capability packages (mechanical but nontrivial); the availability snapshot adds a native↔runtime contract that must stay small; three closed vocabularies (availability conditions, execution requirements, outcome taxonomy) need governance — extending any of them is a platform-team change versioned with the bundle format; total ordering through one executor must be protected against "just this once" direct handler calls.

**Revisit when:** a capability legitimately needs cross-capability composition (likely first: reminders ↔ activity); dynamic-vocabulary needs exceed the memory-preset pattern; capability count exceeds ~25; auto-resume demand appears in telemetry (`NeedsUser` followed by repeat-command within 60s); any capability needs a fourth execution mode; partial-ASR latency work begins.

## Action Items

1. [ ] Ratify the capability map (§A3) — product + platform sign-off on boundaries and owners.
2. [ ] Ratify outcome taxonomy + execution descriptor vocabulary (platform team, both app teams).
3. [ ] Add `capability.yaml` to the bundle compiler's inputs; emit capability tables + availability schema into the bundle.
4. [ ] Repartition `content/` by capability; tag datasets with capability ids in DVC.
5. [ ] Specify `notifyExecution` and the availability-snapshot push in the runtime interface set (extends ADR-001.1 §9).
6. [ ] Build codegen: manifests → typed params, action constants, contract-test skeletons, completeness gate.
7. [ ] Add execution stanzas to the conversation-corpus format.
8. [ ] Wire per-capability metrics into the evaluation report and telemetry schema.
9. [ ] Prototype the dispatcher + fake registry in the current Python engine host first (per ADR-001.1 Part 9: Python adopts target interfaces before the runtime exists).
