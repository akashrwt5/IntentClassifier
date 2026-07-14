# ADR-003: Conversation Orchestration Platform
## How the Assistant Reasons, Plans, and Manages Conversations

**Status:** Proposed — revised per Board review (rev. 2, 2026-07-14): Planner/Workflow boundary defined (§3), Policy Engine lifecycle expanded (§8), state machine declared closed (§4), Planner extensibility contract (§3), context storage model (§5), component interface contracts (Part 12), package boundaries (Part 13)
**Date:** 2026-07-13 (rev. 2: 2026-07-14)
**Depends on:** ADR-001 (shared runtime), ADR-001.1 (boundary specification), ADR-002 (capabilities + action execution/SDK)
**Placement:** Everything in this document is **shared-runtime logic + configuration** by the ADR-001.1 four-question test — orchestration decisions change what the user experiences (Q3) and are behaviors (Q4). Nothing here moves native; nothing here alters approved boundaries.

---

## Executive Summary

The approved stack handles `utterance → intent → workflow → action` — one request, one workflow, one action. The Board's motivating examples break that assumption in **two different ways**, and recognizing the difference is the central design judgment of this ADR:

- *"Actually, never mind." / "Do that again." / "No, I meant the left one." / "Turn the volume back to what it was." / "Cancel that reminder."* — these are **not planning problems**. They are conversational acts over history: universal verbs, back-references, referent repair, undo. They need a richer **Dialogue Manager** and a real **conversation history model**, both already sketched in prior ADRs and completed here.
- *"Increase my volume and then start streaming." / "Set a reminder and send a message." / "Remind me to charge if the battery drops below 20%."* — these are planning problems, but **small, deterministic ones**: 2–3 step sequences and simple conditions. They need a **Planner**, but a plan *compiler*, not an AI reasoner.

So the design is deliberately modest where the industry giants are grand: a thin **Conversation Orchestrator** that owns conversation lifecycle and arbitration; a deterministic **Planner** that compiles an interpretation into a small plan DAG; the existing **Dialogue Manager** extended with a frame stack and repair acts; the existing **Workflow Engine** unchanged as the per-intent executor; a **Policy Engine** that makes every cross-cutting rule configuration; and a layered **Context model** feeding them all. No component becomes a God object because each owns exactly one noun: the Orchestrator owns *the conversation*, the Planner owns *plans*, the Dialogue Manager owns *turns*, the Workflow Engine owns *intent workflows*, Capabilities own *execution*, Policies own *rules*.

Alexa/Google-scale mechanisms this ADR deliberately rejects for this product: ML-based plan synthesis, open-ended multi-agent routing, cross-session goal memory, and speculative parallel execution — a hearing-aid assistant with high-cost device actions needs *predictable* orchestration, and a small team needs *debuggable* orchestration.

---

# Part 1 — Review of Existing Architecture

**Already solved (build on, do not touch):** cascade classification + calibration (ADR-001); dialogue primitives — confirmation contexts, slot filling, attempt budgets, TTLs, interruption with weak-keyword demotion, back-references (`prev_memory`, `last_fulfilled` — note: today's engine already implements primitive undo, "change back", proving the pattern); boundary placement of all decision logic in the shared runtime (ADR-001.1); workflow interpretation over closed vocabularies (ADR-001.1 Part 6); capability packaging, availability snapshots, device-provided vocabulary (ADR-002 A); action dispatch, outcome taxonomy, execution modes, `notifyExecution` feedback, execution-aware telemetry (ADR-002 B).

**Missing (this ADR's scope):** a conversation model above single-turn routing — today `engine.handle()`'s priority ladder (confirmation → slot-fill → back-ref → classify) *is* the conversation model, which collapses under multi-step requests; any notion of a plan spanning >1 workflow; conversational scope resolution ("cancel *that*" — which that?); repair acts ("no, I meant the left one"); suspension/resumption (nested conversations); an action history enabling generalized undo; conditional requests; a policy layer (today's policies are scattered constants + per-intent schema fields with no arbitration rules); explicit conversation lifecycle states the app/UI can subscribe to.

**Gaps acknowledged but out of scope here:** long-term memory/personalization (explicitly excluded by the Board), proactive/assistant-initiated conversations beyond ADR-002 B6's session-drop prompt, partial-ASR streaming turns (ADR-002 B10 exclusion stands).

---

# Part 2 — Who Owns the Conversation?

| Candidate | Evaluation |
|---|---|
| **Dialogue Manager** | Natural for turns, wrong for lifecycle: making it own plans and recovery regrows today's `engine.py` God-object trajectory (706 lines and climbing). Turn interpretation and conversation arbitration are different cadences and different state. |
| **Planner** | A planner that owns the conversation becomes an AI-architecture cliché: everything routes through planning, so everything becomes a plan, including "yes." Plans are *artifacts*; ownership of a mutable long-lived process by a producer of immutable artifacts inverts the dependency. |
| **Workflow Engine** | Owns one intent's micro-flow; promoting it to conversation owner forces cross-intent semantics (suspension, plan sequencing) into a component whose contract is deliberately intent-local. |
| **A Capability** | Violates ADR-002 A9 (no capability observes NLU internals or other capabilities); the conversation is inherently cross-capability. |
| **Conversation Orchestrator (new)** | Correct — *if* strictly scoped. Risk: "orchestrator" is the traditional name of the God object. Mitigation: the Orchestrator owns **lifecycle and arbitration only** — the state machine (Part 4), the frame stack, plan admission, and recovery *selection*. It contains no linguistic logic (Dialogue Manager's), no plan construction (Planner's), no workflow semantics (Workflow Engine's), no policy values (Policy Engine's config), no execution (Dispatcher's). It is small because everything it coordinates has somewhere else to live. |

**Decision: the Conversation Orchestrator owns the conversation.** Ownership means: it is the single writer of conversation state, the single authority on state transitions, and the arbiter when components disagree (e.g., Dialogue Manager reports an interrupt while a plan step is executing). Everything else is a service it consults.

The anti-God-object enforcement is structural, not aspirational: the Orchestrator may only (a) call the five services through their interfaces, (b) mutate ConversationContext, (c) emit lifecycle events. A PR adding domain logic to the Orchestrator fails review by citing this section — the same discipline as ADR-001.1's four-question test.

```
                       ┌────────────────────────────┐
    turn text ───────► │  CONVERSATION ORCHESTRATOR │ ───► NLUResult / events
                       │  (state machine, frames,   │
                       │   arbitration, recovery    │
                       │   selection)               │
                       └──┬────┬────┬────┬────┬─────┘
              consults ▼    ▼    ▼    ▼    ▼
                 Dialogue Planner Workflow Policy  Context
                 Manager          Engine   Engine  Store
                 (turns)  (plans) (intent  (rules) (state)
                                  flows)
```

---

# Part 3 — Planner Architecture

**Is a Planner required?** Yes — but the honest analysis first. Of the seven motivating examples, five need *no planner* (they are dialogue acts over history). The planner earns its existence from exactly two request classes: **compound requests** (N intents in one utterance, possibly ordered) and **conditional requests**. Both are bounded, enumerable, and deterministic. Therefore:

> **The Planner is a pure, stateless plan compiler**: `(interpretation, context, policies) → Plan | ClarificationRequest | Rejection`. It runs synchronously within a turn, produces an immutable artifact, and is never consulted again about a plan it produced. It contains no ML in v1, holds no state, performs no I/O, and never talks to the user (it *requests* clarification; the Dialogue Manager *renders* it).

### Plan model

A Plan is a small DAG (v1 cap: 3 steps, degenerate single-step for ~99% of traffic):

```
Plan {
  plan_id, origin_turn_id
  steps: [ PlanStep {
      step_id
      intent + pre-bound slots (from the compound utterance)
      capability, workflow ref
      depends_on: [step_id]          # ordering edges
      condition?: Predicate           # closed vocabulary, evaluated at step start
      on_failure: abort | continue | ask   # from policy + step cost
  } ]
  confirmation_plan: per-step | plan-level | none   # computed from policy matrix (Part 8)
}
```

### Planner responsibilities (the Board's list, mapped)

| Responsibility | How |
|---|---|
| Intent decomposition | Consumes the Dialogue Manager's segmentation of compound utterances (conjunction splitting + per-segment classification is *linguistic* work — Dialogue Manager's; deciding what the segments become is the Planner's). Low-confidence segmentation → clarification request ("One at a time — volume first?"), preserving the platform-review §11 stance as the *fallback*, now upgraded with a happy path for clean compounds. |
| Multi-action requests | Compile segments to steps; bind cross-segment references ("increase volume and *then* start streaming" → dependency edge from "then"). |
| Dependency ordering | Explicit ordering words → edges; otherwise utterance order → sequential edges (conservative default; see Part 7 for why parallel is rejected). |
| Clarification planning | Emits *structured* clarification needs (ambiguous referent, unresolvable segment, over-cap plan) — one at a time, ranked by information gain (resolve the segment that gates the most steps first), bounded by the policy clarification budget. |
| Retry planning | None at plan time — retries are conversational (ADR-002 B7). The Planner only stamps each step's `on_failure` from policy. |
| Execution ordering | The DAG *is* the ordering; the Orchestrator walks it. The Planner never executes. |
| Conflict detection | Closed rule set over step pairs: contradictory actions on one target (`volume.increase` + `volume.mute`), duplicate steps, condition contradictions. Conflict → clarification, never silent resolution. |
| Capability selection | Trivial in v1 (intent → capability is a bundle fact) but the seam matters: when two capabilities could serve one goal (future: `help` article vs. `diagnostics` run for "my aids sound wrong"), selection policy lives here, driven by config ranking — not in the classifier and not in capabilities. |
| Workflow selection | Bundle fact lookup (intent → workflow); the Planner binds it, the Workflow Engine runs it. |
| Recovery planning | On step failure the Orchestrator asks the Planner to *re-plan the remainder* given the failure (e.g., step 1 `device_unreachable` → remaining device steps inherit the block → single consolidated failure response instead of three sequential apologies). Re-planning is the same pure function with updated context. |

### The Planner ↔ Workflow Engine boundary (review comment #1)

The reviewer's example resolves cleanly once one rule is stated:

> **The Planner reasons about *which* workflows run and in what order; the Workflow Engine owns *everything that happens inside one workflow*, including all of its user interaction.** Planning is utterance-scoped and one-shot; workflow execution is interactive and multi-turn. The seam is the PlanStep.

So the sequence `Reminder → Ask Date → Ask Time → Confirmation → Create Reminder` is **entirely Workflow Engine**. It is one step of a (usually single-step) plan: the intent's schema-defined micro-flow — slot prompts, validation, clarification, intra-step confirmation, completion — exactly as ADR-001.1 Part 6/7 already defines it. The Planner never sees "Ask Date"; it sees `PlanStep{intent: reminders.add}` and is finished before the first prompt is spoken.

**Exclusively Planner:** utterance decomposition into steps; step ordering (dependency edges); cross-step conflict detection; step-level condition binding; plan admission against policy caps; pre-binding slot values *already present in the utterance* ("remind me tomorrow at 3 to walk" arrives at the Workflow Engine with all slots pre-bound — the workflow then has nothing to ask); re-planning a failed plan's remainder.

**Exclusively Workflow Engine:** slot elicitation order and prompting; entity resolution for slot answers; per-slot validation and clarification; slot attempt budgets; intra-step schema confirmations; producing the fulfillment-ready action request. It never knows it is inside a plan — a step executes identically whether the plan has one step or three. This ignorance is deliberate: it keeps today's workflow semantics (and tests) unchanged, and keeps the Workflow Engine reusable by any future planner.

**Planner → Workflow Engine (via Orchestrator):** `PlanStep{intent, workflow ref, pre-bound slots, condition (already evaluated by the Orchestrator at step start), on_failure}`. The Workflow Engine receives an intent + partial slot map — the same inputs it receives today from a single-intent turn.

**Workflow Engine → Orchestrator (per turn of the step):** one of `NeedsUser(prompt)` (slot/clarification/confirmation prompt to render — the step is now interactive and subsequent user turns route back into it), `ReadyToExecute(action, params)` (all slots filled and validated → Orchestrator hands to Dispatcher), `Abandoned(reason)` (attempt budget exhausted), or `RepairApplied` (a correction act modified a slot). The Orchestrator alone decides what these mean for the *plan* (continue, recover, suspend) — the Workflow Engine never sees the plan.

**Why the separation matters:** (1) it keeps the Planner pure and stateless (Part 3 contract) — a planner that participated in slot prompting would hold conversation state and become a second dialogue manager; (2) it makes plans cheap to re-plan (no interactive state to unwind — only un-started steps are re-planned); (3) it preserves the single-intent fast path bit-for-bit (a one-step plan is today's behavior); (4) it gives the future LLM planner a hard safety floor — an LLM can propose steps but can never bypass the schema-defined validation and confirmation inside each workflow.

```
User: "Set a reminder and send a message"
  │
  ▼ [UNDERSTANDING] DialogueMgr: segments → [reminders.add], [messaging.send]
  ▼ [PLANNING] Planner: Plan{s1: reminders.add(slots:{}), s2: messaging.send, s1→s2}
  ▼ [EXECUTING s1] Orchestrator → WorkflowEngine.start(reminders.add, {})
  ◄ NeedsUser("What should I remind you about?")        ── prompt rendered
User: "charging my aids tomorrow at 8"
  ▼ Orchestrator routes turn → active step's workflow
  ◄ NeedsUser? no — slots complete → ReadyToExecute(reminders.add, {name, datetime})
  ▼ Orchestrator → Dispatcher → Success → notifyExecution
  ▼ [EXECUTING s2] Orchestrator → WorkflowEngine.start(messaging.send, {})
  ◄ NeedsUser("Who should I send it to?") … (step 2's own micro-flow proceeds)
  ▼ … ReadyToExecute → Success → [COMPLETED] — plan done, narrated once
```

### Planner interface and extensibility (review comment #4)

The Planner is consumed through one contract, owned by the Orchestrator's side of the seam:

> **PlannerContract:** `plan(interpretation, context_view, policy_view) → PlanResult`, where `PlanResult = Plan | NeedsClarification(structured need) | Rejection(reason)`. Required properties of *any* implementation: pure (no state, no I/O — context and policy arrive as read-only views); bounded (responds within a per-turn budget or the Orchestrator falls back to single-intent handling); and **untrusted** — every returned Plan passes the Orchestrator's *plan admission validator* regardless of who produced it (steps reference only bundle-registered intents; caps respected; conditions from the closed vocabulary; conflict rules clean). The validator, not the planner, is the safety boundary.

Only the Orchestrator depends on this contract (the Dialogue Manager produces its input; the Workflow Engine consumes its output *via the Orchestrator*; neither knows the planner exists). That single dependency edge is what makes the planner swappable. Evolution path, each behind the same contract with zero Orchestrator change:

| Generation | Implementation | Notes |
|---|---|---|
| v1 (this ADR) | Deterministic rule-based compiler | Ships with M4 |
| v2 | **Hybrid router**: rule-based first; unresolved interpretations optionally routed to a stronger planner by policy (connectivity, consent, complexity) | The router is itself a PlannerContract implementation composing two others |
| v2+ | **On-device LLM planner** (propose-verify): model proposes steps constrained to the bundle's intent registry; admission validator enforces everything else | The platform-review §20 seam, now formalized |
| v2+ | **Cloud reasoning / tool calling**: the capability action contracts (ADR-002 A4) double as a tool manifest; a cloud planner returns tool-call sequences that compile to PlanSteps | Requires the stage-4 consent gate; never available offline — hence the hybrid router, never a replacement |
| v2+ | **Knowledge retrieval**: not a planner variant but a *step kind* — the `help` domain's RAG conversion makes retrieval an intent workflow; planners simply plan it like any step | Keeps retrieval out of the planning abstraction |

The invariant across all generations: **planners propose, the platform disposes.** Admission validation, policy caps, per-workflow schema validation, and confirmation matrices apply identically to a regex-built plan and an LLM-built plan. Trust lives in the validator and the workflows — which is why they were kept out of the Planner in the first place.

### Conditional requests — the one hard scoping call

"Remind me to charge if the battery drops below 20%" contains a **standing trigger**, which is fundamentally different from a conversation plan: it outlives the conversation, requires background condition monitoring, and autonomously initiates action later. Platform-level background trigger infrastructure (Alexa Routines-class) is a large, privacy-sensitive system this product does not need generically. **Decision:** the Planner supports *immediate* conditions (evaluable now from DeviceContext: "if the battery is low, remind me tonight" → evaluate now, plan or politely decline) and compiles *standing* triggers **only as capability-parameterized actions** — i.e., it becomes `reminders.add{condition: battery_below_20}` *if and only if* the reminders capability declares conditional triggers in its manifest. No capability support → honest limitation response from schema content. The platform never grows a hidden background rules engine as a side effect of this ADR.

### What must NEVER be inside the Planner

Classification or any NLU (it consumes interpretations); execution or dispatch; user-facing text (requests acts; never renders); state (stateless by contract — all inputs are arguments); device I/O or availability *evaluation* (reads the snapshot, never probes); policy *values* (reads the Policy Engine's tables); learned/ML planning in v1 (a future LLM-planner slots behind the same `interpretation → Plan` contract — the seam is designed, the model is not); cross-session goals (excluded scope).

---

# Part 4 — Conversation State Machine

Two levels, honoring the ADR-001.1 boundary: the **app-level** view includes native states (Idle, Listening — mic/ASR are native); the **runtime-owned** machine begins at text receipt. The runtime emits lifecycle events so the app can mirror state in UI without owning any of it.

```
 APP LEVEL (native)                    RUNTIME LEVEL (Conversation Orchestrator)
 ┌──────┐  mic   ┌───────────┐  text  ┌═══════════════════════════════════════════┐
 │ IDLE │──────► │ LISTENING │──────► ║ UNDERSTANDING                             ║
 └──────┘        └───────────┘        ║   └─ cascade + dialogue-act typing        ║
    ▲                                 ╠═══════════════════════════════════════════╣
    │            events               ║ PLANNING ──────► CLARIFYING ─┐ (budget    ║
    └─────────────────────────────────║   │  plan ready      ▲       │  exhausted ║
                                      ║   ▼                  └───────┘  → FALLBACK)║
                                      ║ CONFIRMING (policy-required only)         ║
                                      ║   │ confirmed          declined → CANCELLED║
                                      ║   ▼                                        ║
                                      ║ EXECUTING ◄──────────────┐                ║
                                      ║   │ │ │                  │ next step      ║
                                      ║   │ │ └─ async initiated → WAITING ───────║──► late result:
                                      ║   │ │                                     ║    announce|silent
                                      ║   │ └─ step failed → RECOVERING           ║
                                      ║   │        │ re-plan ok ──────────────────║──► EXECUTING
                                      ║   │        │ unrecoverable → FAILED(resp) ║
                                      ║   │ new high-conf intent → INTERRUPTED    ║
                                      ║   │        │ suspend frame → UNDERSTANDING║ (new frame)
                                      ║   ▼                                        ║
                                      ║ COMPLETED          CANCELLED (verb, any   ║
                                      ║ (responses emitted)  non-terminal state)  ║
                                      ╚═══════════════════════════════════════════╝
```

| State | Entry | Exit | Failure path |
|---|---|---|---|
| UNDERSTANDING | text arrives (new turn, any active state — every turn passes here first for act-typing) | act typed: answer→ active frame; verb→ verb handling; new intent→ PLANNING | unintelligible → clarify (counts against budget) or GenAI fallback per cascade |
| PLANNING | interpretation ready | Plan admitted → CONFIRMING or EXECUTING; needs info → CLARIFYING; rejected → FALLBACK response | Planner internal error → FALLBACK (safe apology), telemetry `planner_failure` |
| CLARIFYING | Planner/Dialogue needs one fact | user answers → PLANNING (re-compile) | budget (default 2) exhausted → graceful abandon (the existing MAX_SLOT_ATTEMPTS pattern, generalized) |
| CONFIRMING | policy matrix demands it (Part 8) | yes → EXECUTING; no → CANCELLED; unclear → re-ask (shares clarification budget) | context TTL expiry → conversation reset (existing 90s rule) |
| EXECUTING | confirmed plan, next ready step; per-step slot-filling runs *inside* this state via the Workflow Engine (slot prompts are sub-states, unchanged from today) | all steps done → COMPLETED; async step → WAITING; failure → RECOVERING; interrupt → INTERRUPTED | dispatcher timeout → RECOVERING |
| WAITING | async action initiated | out-of-band outcome → EXECUTING (continue plan) or COMPLETED | context TTL → late result downgrades to silent (ADR-002 B6) |
| RECOVERING | step outcome ∈ Failed/unavailable | re-plan viable → EXECUTING; else → FAILED-with-response → IDLE | recovery itself fails → FALLBACK, frame dropped, telemetry |
| INTERRUPTED | interrupt-eligible new intent during EXECUTING/CLARIFYING (policy: priority classes, Part 8) | frame suspended (stack push) → UNDERSTANDING for the interrupter | stack full (depth 1) → oldest frame abandoned with notice |
| COMPLETED / CANCELLED / FAILED | terminal per frame | pop frame stack: suspended frame exists → offer resume; else → IDLE | — |

Rules: every state has a TTL (inherited from context TTLs — no state can strand a conversation); CANCELLED is reachable from *every* non-terminal state by the cancel verb (universal, non-negotiable); all transitions emit lifecycle events (app UI, telemetry) — the state machine is thereby the observability spine, not just control flow.

### The state machine is CLOSED (review comment #3 — normative)

**Capabilities can never introduce conversation states.** The reviewer's preference is adopted and made law. `StreamingConnecting`, `ReminderWaiting`, `DeviceSyncing` and their kin are **execution states, not conversation states** — they describe what a *capability* is doing, not what the *conversation* is doing. The conversation's view of all of them is already representable: a sync action in flight is EXECUTING; a long-running one is WAITING (with capability-supplied narration content: "Connecting to your hearing aids…" is a *prompt string in the bundle*, not a state); a capability session's internal lifecycle lives natively and surfaces only through the Capability Context snapshot (ADR-002 B6).

Why closed, beyond preference: (1) **testability** — the conversation corpus enumerates transitions of a ~12-state machine exhaustively; N capabilities × M private states makes the machine's behavior a function of installed features, and the corpus stops being a spec; (2) **the fixed-size-brain guarantee** (Part 11) is *this* guarantee — an extensible state machine is precisely how orchestrators become God objects one capability at a time; (3) **cross-platform identity** — a state only a streaming-capable build can enter reintroduces per-configuration behavioral drift, the disease this platform exists to cure; (4) **capabilities already have the right expressive tools**: WAITING-state narration content, Capability Context state flags (which policies and the Planner can read), and native UI (a connecting spinner is an app concern, not a conversation concern).

Enforcement: the state enum and transition table are part of the runtime's versioned contract (Part 12); the bundle compiler rejects content referencing unknown states; adding a state is a platform-team runtime release with a corpus extension — deliberately expensive, exactly as rare as it should be. If a capability's design appears to need a new state, the design review question is "which existing state plus what content/context expresses this?" — in every example examined so far, the answer exists.

---

# Part 5 — Context Architecture (runtime only)

Layered, narrow-to-wide; each layer readable by the layers that need it, writable by exactly one owner. No long-term memory, no personalization.

| Context | Purpose | Owner (single writer) | Lifetime | Expiration | Synchronization | Visibility |
|---|---|---|---|---|---|---|
| **Turn Context** | Scratch for one turn: raw text (never persisted), cascade trace, dialogue act, candidate interpretations | Dialogue Manager | one turn | end of turn (raw text dies here — the ADR-001.1 privacy invariant is enforced by this lifetime) | none needed (single-threaded turn) | Dialogue Manager, Planner (interpretation only, not raw text) |
| **Conversation Context** | The active frame: plan + step cursor, clarification budget spent, pending confirmation, **action history ring** (last N fulfilled actions + restorable prior values — powers "do that again", "turn it back", "cancel that") and the **frame stack** (suspended conversations, depth 1) | Conversation Orchestrator | conversation (frame) | terminal state + grace period; hard cap = session TTL | single writer; serial executor already guarantees ordering (ADR-002 B8) | Orchestrator, Dialogue Manager (for referent resolution), Planner (read) |
| **Session Context** | Cross-conversation short-term state: `last_fulfilled` per intent, prev/last memory preset, partial datetime parking — *exactly today's `Session`*, unchanged | Context Store (as today) | session | 10-min idle TTL (existing) | as today | Orchestrator, Workflow Engine |
| **Capability Context** | Per-capability sanctioned state: dynamic vocabulary (memory presets), session-mode status (streaming active), conditional-trigger support flags | Native capability → SDK snapshot push (writer); runtime holds read-only copy | until next snapshot push | replaced-on-push (no staleness window: push-on-change, ADR-002 A5) | snapshot swap on the serial executor — turns see a consistent frozen view | Planner (conditions, selection), Workflow Engine (dynamic entities), Policy Engine |
| **Device Context** | Availability snapshot: connection state, aid features, permissions, flags, battery — the *facts* conditions evaluate against | Native CapabilityRegistry (writer) via same push path | until next push | replaced-on-push | same as above | Planner, Policy Engine, Orchestrator (recovery) |
| **Dynamic Runtime State** | Active bundle id/version, loaded models, degradation flags (stage-3 dropped under memory pressure) | BundleManager/RuntimeHost | bundle activation → swap | swap (live conversations finish on old bundle, ADR-001.1 §9.2) | atomic swap | everything (read-only); telemetry stamps it on every event |

Design invariants: **strictly one writer per layer** (all races become impossible by construction, not by locking); **raw text never leaves Turn Context**; **snapshots are pushed, never pulled** (the runtime does no I/O — ADR-001.1 Part 8 holds); referent resolution ("that reminder", "the left one") reads *only* Conversation + Session contexts — never global state — so "cancel that" can never resolve to something the user wasn't just talking about.

### Storage & persistence model (review comment #5)

The runtime performs no I/O (ADR-001.1 Part 8), so *all* persistence follows one pattern: **the runtime serializes, the native host stores.** Serialization uses one versioned, schema-stable, implementation-agnostic format (not language-native serialization — the Rust migration must read blobs a Python engine wrote, and vice versa during M-phases). Thread safety is uniform: the single-writer rule plus the serial executor (ADR-002 B8) means no context is ever concurrently mutated; snapshot layers swap atomically between turns.

| Context | Stored where | Storage owner | Survives app restart? | Survives bundle upgrade? | Restored how | Why this design |
|---|---|---|---|---|---|---|
| Turn | Memory only — storage prohibited | — | No | — | Never | The privacy invariant *is* this row: raw text is unstorable by construction, not by policy |
| Conversation (frames, plans, step cursors) | Memory only — deliberately not persisted | — | **No** | n/a (dies with process) | Never — fresh start after process death | A half-executed plan must not resume autonomously on a medical device (Part 6); the cost (rare "start over") is the safety |
| Session (last-fulfilled, prev-memory, datetime parking, action-history ring) | Runtime-serialized blob → platform-encrypted storage (Keychain-wrapped key / Keystore-wrapped key; ADR-001.1 §17) | Native SDK (RuntimeHost triggers serialize on backgrounding) | **Yes**, within its 10-min TTL — TTL is re-checked *at restore* against the injected clock, so a stale blob restores to nothing | **Yes** — blob format version is independent of bundle version; unknown *newer* format → discard + telemetry (never crash, never partial-parse) | `restore(blob)` on SDK start; TTL sweep first | Continuity across backgrounding is the UX requirement; encryption because reminder topics are health-adjacent; discard-on-unknown because session loss is benign, corruption is not |
| Capability Context | Not stored by the runtime; each capability persists its own state natively however it likes | Owning capability (native) | Capability's choice | Capability's choice | Re-pushed via snapshot on SDK start — the runtime always begins empty and receives fresh truth | The runtime must never hold stale device claims; native re-push makes staleness structurally impossible |
| Device Context | Not stored; rebuilt from live native state | Native CapabilityRegistry | n/a (rebuilt) | n/a | Registry evaluates availability on start → first push | Same as above: connection state and permissions must be observed, never remembered |
| Dynamic Runtime State | Bundle files in the two-slot store (ADR-001.1 §7.5) | Native BundleManager | Yes (files) | This *is* the upgrade mechanism | Verify → warm → activate on start | Already specified; listed for completeness |

Bundle-upgrade interaction, stated precisely: an in-flight conversation completes on the old bundle (ADR-001.1 §9.2); the Session blob survives the swap because it references *stable identifiers only* (intent ids, slot names, canonical entity values — never bundle-internal indices); if a restored session references an intent the new bundle deprecated, the `superseded_by` mapping (ADR-002 A7) resolves it, and an unmappable reference degrades to an empty history entry, not an error. This "stable-identifiers-only" rule for anything serialized is a contract-level requirement in Part 12.

---

# Part 6 — Multi-Turn Conversations

**Clarifications.** One question per turn, most-load-bearing fact first (Planner ranks), budget of 2 per conversation (policy), budget exhaustion → graceful abandon with fallback offer. Clarification answers re-enter UNDERSTANDING typed as `answer(frame)` — they never risk being classified as new intents unless they *fail* to answer (then normal interrupt rules apply, reusing today's slot-fill interrupt logic).

**Corrections.** New dialogue act family, resolved against Conversation Context: *pre-execution* ("no, I meant the left one" during CONFIRMING) → repair the pending plan's slot (Dialogue Manager maps "the left one" → `target: left_aid` via referent tables in the bundle) and re-confirm only the changed fact; *post-execution* ("no, not that memory") → treated as implicit undo + re-slot: consult action history, if the prior action is `restorable` propose the correction ("Switch from Restaurant back to Outdoors?"). Corrections never silently re-execute high-cost actions — policy matrix applies as if new.

**Interruptions.** Generalizes today's slot-fill interrupt to plan scope: interrupt-eligible intent during EXECUTING/CLARIFYING suspends the frame (stack push) rather than destroying it (today's behavior destroys — strictly better). Eligibility = existing threshold + tier demotion rules, now plus a policy priority class (Part 8: e.g., `device.volume` may interrupt `help` reading; `help` may not interrupt an executing plan).

**Cancel.** Scope resolution ladder, most-recent-first against Conversation Context: pending confirmation → decline it; executing plan → halt after current step (never mid-action — actions are not preemptible), report partial completion honestly ("Volume's up; I didn't start streaming"); just-completed restorable action within the history window → offer undo ("cancel that reminder" → `reminders.complete`/delete via the capability's declared inverse); nothing cancellable → "nothing to cancel" from schema. Ambiguity between two candidates → one clarification, not a guess.

**Resume.** On frame pop: suspended frame *with pending user obligation* (mid-slot-fill) → explicit offer ("Back to your reminder — when should I remind you?"); suspended frame merely mid-plan → auto-continue with a one-line re-anchor. Suspended frames inherit context TTLs — a frame nobody returns to dies silently (no zombie prompts hours later; the Sprint-1 stale-context lesson, elevated to plans).

**Nested conversations.** Frame stack **depth 1** in v1 (one suspension). Deeper nesting is a UX failure mode, not a feature: interrupting the interruption abandons the oldest frame with notice. Revisit only on telemetry evidence.

**Context switching.** Covered by interrupt + resume; the explicit rule: switching is *frame*-granular, never *step*-granular — you cannot resume into the middle of a half-confirmed step; the step restarts its own micro-flow.

**Conversation recovery.** After app process death: Session Context restores (encrypted blob, ADR-001.1 §9.3); Conversation Context (frames/plans) deliberately does **not** — a plan half-executed before a crash must not resume autonomously on a medical device. Recovery = honest fresh start; the action history ring persists with the session so "did that go through?" remains answerable via the last-fulfilled record.

---

# Part 7 — Multi-Action Planning Semantics

| Mode | Decision | Reasoning |
|---|---|---|
| **Sequential** | Default and, in v1, the only execution mode | "and then" implies user-perceived ordering; per-step feedback gating (ADR-002 B5 order: decide → execute → feedback → next) means step N+1 sees step N's outcome — required for honest partial-failure narration; TTS response channel is serial anyway, so parallel completion cannot even be *reported* concurrently |
| **Parallel** | Rejected in v1, seam preserved (DAG already expresses independence) | Marginal latency win (device commands are ~100ms-scale) vs. real costs: BT command interleaving to the same aids, race-prone recovery, un-narratable outcomes. Revisit if telemetry shows multi-step plans with independent capabilities *and* user-perceived latency complaints — the plan model needs no change, only the Orchestrator's walker |
| **Conditional** | Immediate conditions: yes (closed predicate vocabulary over Device/Capability Context, evaluated at *step start*, not plan time — "if battery is low, remind me tonight" checks battery when the step runs). Standing triggers: capability-parameterized only (Part 3) | Step-start evaluation keeps conditions honest against live state; the closed vocabulary keeps the anti-DSL line (ADR-001.1 Part 6) |
| **Transactions** | No atomicity, ever. Compensation-based instead: capabilities may declare `restorable: true` + inverse action; the Orchestrator records prior values in the action history; failure mid-plan → halt + honest partial report + *offer* of undo where available — never automatic rollback | You cannot roll back sound that already reached someone's ears or a message already sent. Automatic rollback of device state on a hearing aid (un-asked-for volume reversal) is itself a wrong-action. Offered, user-approved compensation is the only safe semantics for this product class |

Plan admission caps (policy): ≤3 steps, ≤1 high-cost step per plan (a plan with two high-cost steps → per-step confirmation converts it into what is effectively two conversations — deliberate friction for the medical context).

---

# Part 8 — Conversation Policies

All policies are **bundle configuration** interpreted by a Policy Engine (a read-only rule-table service — not a component with behavior of its own). Platform owns the *vocabulary and defaults*; capabilities may **tighten but never loosen** within declared bounds — the single governance rule that keeps 25 future capabilities from 25 safety dialects.

| Policy | Platform-owned | Capability-tunable (tighten only) |
|---|---|---|
| Confirmation matrix | The matrix semantics: `cost × confidence-band × plan-size → none / plan-level / per-step confirm` | A capability may raise its actions' cost class; never lower |
| Confidence thresholds | Threshold *semantics* and defaults (existing calibrated values become the policy table — ADR-001 §7.3 fulfilled) | Per-intent floors may rise |
| Retries | Conversational-only rule (no silent re-execution) — non-negotiable platform law | Whether "try again" is offered proactively in failure responses |
| Timeouts | State TTLs, step timeout defaults (sync 3s / async 15s from ADR-002 B4) | Per-action timeout within platform max |
| Safety / medical commands | High-cost actions: always confirmable, never in standing triggers without explicit prior consent flow, never >1 per plan, never auto-rolled-back, never fired from `contains`-tier keyword evidence | Capability declares which of its actions are high-cost (may only add) |
| Execution failures | Outcome→dialogue mapping semantics; consolidated-failure re-planning rule | The localized response *text* per failure condition (already ADR-002 content) |
| Unavailable capabilities | Recognition-not-execution rule (ADR-002 A6) and its dialogue shape | The unavailable-response text and whether to offer alternatives |
| Clarification strategy | Budget (2), one-question-per-turn, ranking rule | Per-slot clarification text; per-intent budget reduction |
| Interruption priority | Priority classes and the eligibility algebra (threshold + tier + class) | A capability may declare its flows `non-interruptible-by` lower classes (e.g., an executing self-check resists a help query) — from the closed class list only |

Why this split: policies that shape *trust and safety* must be uniform (a user cannot be expected to learn per-feature confirmation habits on a medical device); policies that are *content* (what the apology says) belong to the feature owner. The Policy Engine is also where A/B experiments on conversation behavior plug in (bundle variants carry policy-table diffs — ADR-001 §14.2's A/B mechanism now has its conversation-layer payload).

### Policy lifecycle: authored → compiled → loaded → evaluated (review comment #2)

**Policies are compiled, not interpreted.** All merging, override resolution, and conflict checking happens in the **bundle compiler** at build time; the runtime loads flat, fully-resolved tables and evaluation is a pure lookup. This is the single most important implementation decision in this section: it moves every failure mode (bad override, incomplete matrix, conflicting rules) from *runtime on a user's device* to *compile time in CI*, and it means the runtime Policy Engine has no merge logic to drift across implementations.

**Authoring & override resolution (compile time).** Sources, in precedence order: platform defaults (`content/policies.yaml`) → capability tightenings (`capability.yaml` policy blocks) → experiment overlays (bundle-variant diffs). The compiler applies the **tighten-only algebra** per policy dimension (each dimension declares its tightening direction: thresholds may only rise, budgets only shrink, cost classes only escalate, timeout maxima only decrease, confirmation levels only strengthen). A capability override that loosens is a **compile error with the diff in the message** — conflicts are not "resolved," they are rejected; there is no runtime arbitration because no conflicting table can exist in a signed bundle. The compiler also proves **matrix completeness**: every `(cost × confidence-band × plan-size)` cell filled, every failure condition mapped, every interruption class-pair ordered — a partial policy table cannot compile.

**Versioning.** The resolved tables carry two versions: the *policy schema version* (the vocabulary of dimensions — owned by the runtime contract, Part 12; the runtime refuses bundles with a newer schema than it understands, same compat gate as everything else in `bundle.json`) and the *policy content version* (monotonic, logged in every telemetry event so fleet behavior changes are attributable to the exact policy release). Policy *values* changing is a content release; policy *vocabulary* changing is a runtime release — the same fact/behavior split as everywhere else (ADR-001.1 Part 6).

**Loading.** Tables load with the bundle (verify → warm → activate); a live conversation finishes on the tables it started with — mid-conversation policy shifts would make behavior unexplainable to the user and unattributable in telemetry.

**Evaluation.** Read-only lookups at fixed, named decision points — the Policy Engine is passive; it never initiates anything:

```
Decision point (Orchestrator)            Lookup
────────────────────────────             ──────────────────────────────────────────
Plan admission (PLANNING)                caps: max_steps, max_high_cost_steps
Confirmation decision (pre-EXECUTING)    matrix[cost][confidence_band][plan_size]
Interrupt arbitration (any active)       class_priority[incoming][active] + threshold
Clarification spend (CLARIFYING)         budget remaining
Step dispatch (EXECUTING)                timeout, on_failure default
Recovery selection (RECOVERING)          failure_condition → strategy
```

```
POLICY EVALUATION — confirmation decision for a plan step
Orchestrator                 PolicyEngine                  (tables: flat, pre-resolved)
     │  plan ready: step=MemoryChange, conf=0.81, plan_size=1
     │─ lookup(confirmation, {cost_class(MemoryChange)─┐
     │                        =high, band(0.81)=mid,   │ one indexed read,
     │                        plan_size=1})            │ no merging, no I/O
     │◄─ decision: per-step CONFIRM ───────────────────┘
     │  [CONFIRMING] "Switch to Restaurant?"
     │   … user: yes …
     │─ lookup(interrupt_priority, …) etc. — same pattern at each decision point
```

**Delivery.** Policy changes ride ordinary bundle releases (ADR-001 §14): a policy-only change is a small bundle diff → staged rollout → telemetry keyed by policy content version → promote or roll back. Because tables are compile-resolved and signed, a policy rollback is a bundle rollback — one mechanism, already built, no special path. Experiments are bundle variants whose *only* diff is a policy overlay, making conversation-behavior A/B tests exactly as cheap as they should be.

---

# Part 9 — Runtime Interaction (sequence)

**Compound request, one step failing:** *"Increase my volume and then start streaming"* with aids disconnecting mid-plan.

```
App        Orchestrator    DialogueMgr   Planner    WorkflowEng   Dispatcher   Capability
 │ text ────►│                                                                        
 │           │─ act? ──────►│ (segments: [volume.increase]→then→[streaming.start])    
 │           │◄─ interpretation ─┘                                                    
 │           │─ compile ────────────────►│ (2 steps, edge, no conflicts,              
 │           │◄─ Plan{s1,s2} ────────────┘  confirmation: none — low cost)            
 │           │  [EXECUTING s1]                                                        
 │           │─ run intent workflow (s1) ─────────►│ slots ok, FULFILL volume.increase│
 │◄─ dispatch(volume.increase) ────────────────────────────────►│─ handler ──►│ ok    
 │           │◄─ notifyExecution(Success) ──────────────────────┘                     
 │           │  [s1 done → EXECUTING s2]           (aids disconnect; snapshot push ►) │
 │           │─ step-start check s2: streaming requires aids_connected → unavailable  │
 │           │  [RECOVERING] ─ re-plan remainder ─►│ (nothing viable)                  
 │           │◄─ consolidated failure ─────────────┘                                  
 │◄─ events: Executed(volume) + response: "Volume's up — but I've lost                 
 │   connection to your hearing aids, so I couldn't start streaming."                  
 │           │  [FAILED→IDLE, action history records volume prior value]              
```

**Correction during confirmation:** *"Set my memory to Restaurant" → "Actually, the Outdoor one."*

```
 text#1 ► Orchestrator ► DialogueMgr(intent) ► Planner(1-step) ► policy: MemoryChange
          = state-changing → CONFIRMING: "Switch to Restaurant?"
 text#2 ► Orchestrator ► DialogueMgr: act = correction(referent: memory slot,
          value: Outdoors — via bundle referent tables + Capability Context vocab)
        ► Orchestrator repairs pending plan slot ► re-confirm changed fact only:
          "Outdoors instead — go ahead?"  ► yes ► EXECUTING ► dispatch ► Success
        ► history: {memory: prior=Everyday, new=Outdoors}   ("change back" now works)
```

The component chain the Board sketched (Orchestrator → Planner → Dialogue Manager → Registry → Workflow → Dispatcher → Native → Feedback → Context) holds, with two corrections of arrow-direction reality: the Dialogue Manager runs *before* the Planner within a turn (interpretation precedes planning), and the Capability Registry is consulted by *both* Planner (availability at plan time) and Orchestrator (re-check at step start) — availability is checked twice by design, because plans outlive snapshots.

---

# Part 10 — Failure Handling

Uniform pattern: **detect (owner) → classify (taxonomy) → select recovery (Orchestrator, per policy) → narrate (Dialogue Manager, from bundle content) → record (telemetry)**. Nobody recovers unilaterally.

| Failure | Detected by | Recovery decision owner | Strategy |
|---|---|---|---|
| Execution failure (action) | Dispatcher (outcome/timeout) | Orchestrator (policy: step `on_failure`) | RECOVERING → re-plan remainder → consolidated honest narration; conversational retry offer |
| Bluetooth disconnect | Native registry → snapshot push | Orchestrator | Mid-plan: step-start check catches it (§9); mid-flow: interruption-with-notice (ADR-002 A6); action in flight: dispatcher returns `device_unreachable`, same path |
| Permission denied | Dispatcher precondition / handler | Orchestrator + app | `NeedsUser(permission_flow)` → prompt + structured app event; no auto-resume in v1 (ADR-002 B10) |
| Capability unavailable | Runtime fulfillment gate | Orchestrator | Declared unavailable-response; plan-time known → Planner rejects step at compile (earlier, cheaper) |
| Workflow timeout (user silent) | State TTLs | Orchestrator | Existing context expiry; optional `on_timeout` re-prompt per schema (roadmap item, unchanged) |
| Planner failure (internal) | Planner (caught) | Orchestrator | Never strand: FALLBACK safe response + `planner_failure` telemetry; single-intent path (bypass planner) remains available as degraded mode — the assistant loses compounds, not commands |
| Dialogue failure (unintelligible act) | Dialogue Manager | Orchestrator | Clarify within budget → GenAI fallback (existing cascade behavior, unchanged) |
| Interrupted execution | Orchestrator (arbitration) | Orchestrator | Halt after current step (no preemption), suspend frame, honest partial narration on resume/abandon |
| Conversation cancellation | Dialogue Manager (verb) | Orchestrator | Scope ladder (Part 6); partial-completion honesty; undo offer where restorable |

Non-negotiable invariants across all rows: no silent retries of device actions; no automatic rollback; every recovery produces a user-comprehensible narration from bundle content; every recovery emits a telemetry pair (decision + outcome) so fleet-level recovery rates are measurable per capability.

---

# Part 11 — Extensibility: 59 → 500 Intents, 25 Capabilities, Multiple Teams

**What grows linearly (content, no orchestrator changes):** intents, workflows, prompts, entities, keyword rules, capability manifests, policy *values*, referent tables, unavailable/failure responses. A new capability integrates by: manifest + content package + native handlers + registration (ADR-002 A5). The Orchestrator learns of it entirely through data — registry entries, policy tables, plan-fact lookups. **Zero orchestrator code paths reference any capability id** (enforced: the A9 guardrail applied inward).

**What stays constant (the fixed-size brain):** the state machine (~12 states — states model *conversation shapes*, not features; 500 intents produce the same shapes); the plan model (steps/edges/conditions); the dialogue-act vocabulary (~8 acts); the outcome taxonomy; the closed policy/condition/predicate vocabularies. Each closed vocabulary is a governance point: growth there is a deliberate platform-team decision with a version bump — the mechanism ADR-001.1 established, now protecting the orchestration layer.

**Where scale pressure actually lands, and the pressure valves:** (1) *classifier label space* — 500 intents strain a flat TF-IDF/59-class design; the capability/domain hierarchy enables two-stage classification (domain → intent) without any orchestration change, and the `help` domain's planned RAG conversion (platform review §20) removes ~30 labels — both are ML-plane changes behind the same interpretation contract; (2) *policy matrix complexity* — 25 capabilities × policy dimensions stays reviewable only because capabilities tighten-not-loosen (one default table + small per-capability deltas); (3) *multi-team contention* — content is per-capability-owned (CODEOWNERS), the orchestrator is platform-owned, and the only shared artifacts are the closed vocabularies, which change rarely and loudly; (4) *multiple applications* — a second app (watch, TV remote app) reuses runtime + SDK + bundles, installing a different capability subset; ADR-002 A7's bidirectional tolerance already covers partial installation.

**The 5-year seam:** an LLM-based planner/interpreter (platform review §20) replaces the Dialogue Manager's segmentation and the Planner's compilation *behind their existing contracts* (`text → interpretation`, `interpretation → Plan`), while the Orchestrator, state machine, policies, and safety invariants — the parts that encode *trust* — remain exactly as specified here. That is the test of this design: the reasoning engine is swappable; the conversation's constitution is not.

---

# Part 12 — Component Interface Contracts (review comment #6)

These contracts are **stable interfaces**: any implementation — today's Python engine, the future Rust core, a test double — must preserve them exactly. They are versioned with the runtime contract; changing one is a platform release with a deprecation window. Universal rules applying to every contract: all calls occur on the serial executor (no contract is internally thread-safe because none needs to be); all errors are closed enums per component (no stringly errors, no exceptions crossing component seams undeclared); anything serialized uses stable identifiers only (§5); every component emits its events through the Orchestrator's lifecycle stream — components never talk to telemetry directly except via their declared events.

### 12.1 Conversation Orchestrator
**Responsibilities:** conversation lifecycle (state machine), frame stack, plan admission + walking, arbitration (interrupts, cancel scope), recovery selection. **Inputs:** turn interpretations (from Dialogue Manager), execution outcomes (`notifyExecution`), snapshot updates, lifecycle triggers (serialize/reset). **Outputs:** `NLUResult` per turn; lifecycle events (state transitions, frame push/pop, plan progress). **Events:** the platform's event spine — every transition, exactly once. **Errors:** consumes every other component's error enums; itself emits only `ConversationFailed(reason)` results, never throws outward. **Ownership:** sole writer of Conversation Context; owner of the state machine and the plan admission validator. **Lifecycle:** singleton per runtime instance; stateless between conversations except the frame stack. **Dependencies:** all five services below + Context Store — and *nothing else* (the Part 2 guardrail as a dependency rule: adding a dependency edge to the Orchestrator is an architecture review).

### 12.2 Dialogue Manager
**Responsibilities:** turn interpretation — dialogue-act typing (new-intent / answer / correction / verb / confirmation-response), compound segmentation, referent resolution, prompt selection/rendering from bundle content. **Inputs:** raw turn text + read views of Conversation/Session/Capability contexts (for referents) + active-frame expectation (what answer is awaited). **Outputs:** `Interpretation{acts, segments, referents, confidence}`; rendered prompt strings on request. **Events:** act-typing decisions (into the turn trace). **Errors:** `Unintelligible` (routes to clarify/fallback), `AmbiguousReferent(candidates)` (routes to clarification). **Ownership:** sole writer of Turn Context; sole author of user-facing text (nothing else composes sentences — ADR-002 B7 rule, restated as ownership). **Lifecycle:** stateless service; per-turn invocation. **Dependencies:** cascade classifier, bundle content tables, Context Store (read).

### 12.3 Planner
**Responsibilities & contract:** as §3 (`plan(interpretation, context_view, policy_view) → PlanResult`); pure, bounded, untrusted. **Inputs:** interpretation + read-only context/policy views. **Outputs:** `Plan | NeedsClarification | Rejection`. **Events:** plan-compiled trace (steps, edges, rejected alternatives) for the Studio. **Errors:** internal failure → `Rejection(planner_error)`; the Orchestrator's degraded single-intent path handles it (Part 10). **Ownership:** owns nothing — stateless by contract. **Lifecycle:** per-turn invocation during PLANNING/RECOVERING only. **Dependencies:** bundle fact tables (intent→workflow/capability), policy views. *Must not* depend on: Context Store write access, Dispatcher, Workflow Engine, any I/O.

### 12.4 Workflow Engine
**Responsibilities:** execute one intent's schema-defined micro-flow (§3 boundary): slot elicitation, entity resolution, validation, intra-step confirmation, repair application. **Inputs:** `start(intent, pre_bound_slots)`; subsequent routed turns for the active step; repair directives. **Outputs:** `NeedsUser(prompt_ref) | ReadyToExecute(action, params) | Abandoned(reason) | RepairApplied`. **Events:** slot-fill progress (trace). **Errors:** `SchemaInvalid` (compile-time-impossible in a signed bundle; present for defense), validation failures surface as `NeedsUser(clarification)`, never as errors. **Ownership:** owns the step's interactive state (awaited slot, attempts); writes slot values into the step, not into session (the Orchestrator commits fulfilled params to Session Context on FULFILL — single-writer preserved). **Lifecycle:** one instance per active step; discarded on step end. **Dependencies:** Entity Engine, bundle schema tables, Context Store (read: session for back-references, capability for dynamic vocab).

### 12.5 Capability Registry (native, SDK-side)
**Responsibilities:** static handler registration; dynamic availability evaluation; snapshot assembly + push. **Inputs:** capability module registrations; native state observations (permissions, connection, features, flags). **Outputs:** availability/capability/device snapshots (pushed); handler lookups for the Dispatcher. **Events:** snapshot-push events (telemetry: availability transitions). **Errors:** `HandlerMissing` (CI-prevented; runtime-fatal at bundle activation, not per-turn), `ContractVersionMismatch` (bundle activation gate, ADR-002 A7). **Ownership:** sole writer of Device + Capability contexts (via push). **Lifecycle:** app-scoped singleton; evaluates on start and on observed state change. **Dependencies:** capability modules, platform APIs — the one contract legitimately full of platform code.

### 12.6 Action Dispatcher (native, SDK-side)
**Responsibilities:** ADR-002 B4–B6 unchanged — precondition checks, typed dispatch, timeout, dedup, mode lifecycles, outcome feedback. **Inputs:** `(action, params, execution_descriptor)` from the Orchestrator via FFI. **Outputs:** outcome taxonomy via `notifyExecution(turn_id, outcome)`. **Events:** execution events (telemetry pair with turn events). **Errors:** the outcome taxonomy *is* its error surface — a dispatcher never throws into the runtime. **Ownership:** owns in-flight action tracking; never owns conversation meaning. **Lifecycle:** app-scoped singleton. **Dependencies:** Capability Registry (lookup), capability handlers.

### 12.7 Policy Engine
**Responsibilities:** serve pre-resolved policy tables at named decision points (§8). **Inputs:** resolved tables (bundle load); lookup keys. **Outputs:** decisions (pure values). **Events:** none of its own (decisions appear in the Orchestrator's traces with policy content version stamped). **Errors:** `TableIncomplete` — impossible in a signed bundle (compiler-proved); present as an activation-time assertion only. **Ownership:** owns nothing mutable; read-only by construction. **Lifecycle:** rebuilt per bundle activation. **Dependencies:** bundle tables only.

### 12.8 Context Store
**Responsibilities:** typed access to the six context layers with per-layer single-writer enforcement; TTL sweeps; serialization/restore of the Session layer (§5). **Inputs:** layer reads (typed views), layer writes (owner-authenticated — a write carries its writer's identity and the store rejects non-owners: the single-writer rule as a checked invariant, not a convention), snapshot swaps, clock ticks. **Outputs:** read views (immutable), serialized session blobs. **Events:** expiry events (context/session TTL fires — the Orchestrator turns these into state transitions). **Errors:** `RestoreRejected(version|ttl|corrupt)` — always non-fatal (empty context is the recovery). **Ownership:** owns storage *structure*; each layer's content is owned by its declared writer. **Lifecycle:** runtime-instance singleton; Session layer round-trips through the host. **Dependencies:** injected clock. Nothing else — the store is the bottom of the dependency graph.

# Part 13 — Package Boundaries (review comment #7)

Module structure mirroring the contracts one-to-one, so the architecture diagram *is* the build graph. Dependency arrows point strictly downward; a package may not import a peer above it (enforced by build tooling — import-linter now in Python, crate visibility later in Rust).

```
packages/runtime/                      (shared runtime — Python now, Rust at the ADR-001 trigger)
│
├── orchestrator/          state machine, frames, plan walker, admission validator,
│                          recovery selection            [depends: all below]
├── planner/               PlannerContract + v1 rule-based impl + (later) hybrid router
│                          [depends: policy-views, bundle-facts]
├── dialogue-manager/      act typing, segmentation, referents, prompt rendering
│                          [depends: cascade, bundle-content, context-store(read)]
├── workflow-engine/       step micro-flows, slot filling      [depends: entity-engine,
│                          bundle-schema, context-store(read)]
├── cascade/               keyword→tfidf→semantic orchestration, calibration, gates
│                          [depends: inference-port, bundle-models]
├── entity-engine/         enum/fuzzy/datetime resolvers       [depends: bundle-tables]
├── policy-engine/         resolved-table lookups              [depends: bundle-tables]
├── context-store/         six layers, TTLs, session serialization  [depends: clock-port]
├── bundle-runtime/        verification, table access, compat gates [depends: —]
└── api/                   the FFI/PyO3/UniFFI surface: Session, notifyExecution,
                           snapshot push, lifecycle events; the ports (InferenceBackend,
                           Clock) the host implements          [depends: orchestrator]

packages/sdk-ios/  packages/sdk-android/        (native, per ADR-002 B3)
├── facade/                NluFacade, conversation event stream
├── runtime-host/          serial executor, port implementations (CoreML/ORT, clock)
├── capability-registry/   registration, availability, snapshot push
├── action-dispatcher/     dispatch, timeout, dedup, outcome feedback
├── bundle-manager/        download, two-slot store, swap orchestration
└── telemetry-agent/       event stitching, consent, transport

packages/buildtime/                    (Python — never on device)
├── nlu-compiler/          content+policy compilation, override algebra, completeness
│                          proofs, codegen (action constants, contract tests)
├── nlu-training/          trainers, calibration, evaluation, gates
└── nlu-export/            model export, bundle assembly, signing
```

Three structural notes. (1) `orchestrator/` is the *only* runtime package that may depend on planner, dialogue-manager, and workflow-engine — peers among the services are invisible to each other, making the Part 2 ownership rule a build error rather than a review comment. (2) `bundle-runtime/` and `context-store/` sit at the bottom with zero internal dependencies — they are the first packages the Rust migration ports (ADR-001 Phase 1 leaf-extraction maps to this exact graph). (3) The `api/` package is the *entire* FFI surface — if a symbol isn't in `api/`, no host can reach it, which keeps the UniFFI interface definition honest and reviewable as one file.

## Trade-offs, Risks, Migration

**Trade-offs accepted:** sequential-only execution (latency ceiling on compound requests — accepted for narratability and BT safety); frame depth 1 (rare deep-nesting UX sacrificed for predictability); no plan persistence across process death (safety over convenience); deterministic planner (no emergent cleverness — by design); conditions limited to closed predicates (expressiveness ceiling — anti-DSL law).

**Risks:** R1 — Orchestrator scope creep despite the guardrails (mitigation: the Part 2 review rule + the fixed-size-brain inventory as a CI-checked component manifest); R2 — compound-request segmentation quality gates the whole planner's value; if the Dialogue Manager can't split reliably, the planner idles (mitigation: ship clarify-fallback first, measure compound frequency in telemetry *before* investing in segmentation quality); R3 — policy matrix misconfiguration is now a safety surface (mitigation: policy tables get golden tests + the compiler validates matrix completeness); R4 — frame suspension introduces the first cross-turn mutable structure beyond today's session — the highest-value target for the conversation corpus (mitigation: corpus stanzas for every Part 6 scenario are the acceptance gate).

**Migration (no big bang, per house rule):**
*M1* — Reframe: today's `engine.handle()` priority ladder becomes the degenerate Orchestrator (single-frame, single-step plans, no planner). Pure refactor, zero behavior change, existing tests prove it. Lands in the Python engine now (ADR-001.1 Part 9 discipline: adopt target shape pre-Rust).
*M2* — History + verbs: action history ring, cancel-scope ladder, "again"/undo generalization, correction acts. Each is corpus-tested content + Dialogue Manager work; ships behind bundle-versioned schema additions.
*M3* — Frames: suspension/resume replaces destroy-on-interrupt. Single feature flag; strictly-better fallback is today's behavior.
*M4* — Planner: compound segmentation behind a flag with clarify-as-fallback; plan DAG walker in the Orchestrator; conditional steps last, only alongside a capability that declares trigger support.
Each phase is independently shippable, independently revertible, and the conversation corpus grows before each phase as its acceptance spec.

## Action Items

1. [ ] Board ratifies conversation ownership (Part 2) and the planner scope boundaries (Part 3), especially the standing-trigger scoping call.
2. [ ] Add the dialogue-act vocabulary, plan model, referent tables, and policy matrix to the bundle format spec (versioned).
3. [ ] Extend the conversation-corpus format with frames, corrections, cancel-scopes, and plan stanzas (before M2 code).
4. [ ] Execute M1 in the current Python engine as part of the already-approved interface refactor.
5. [ ] Add compound-request frequency counters to the telemetry schema now — M4's investment case should be built on measured demand.
6. [ ] (rev. 2) Freeze the Part 12 contracts as the versioned runtime interface set; wire the import-linter/package rules of Part 13 into CI before M1 lands.
7. [ ] (rev. 2) Implement the policy compiler's override algebra + completeness proofs alongside the bundle compiler (they gate M2's policy tables).
