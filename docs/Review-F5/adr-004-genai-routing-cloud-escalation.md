# ADR-004: GenAI Routing & Cloud Escalation Architecture
## Governing All Interaction Between the On-Device Platform and Cloud AI

**Status:** Accepted — ratified 2026-07-14; was Proposed — rev. 2 (2026-07-14): capability-vs-provider-abstraction justification (§1.1), prompt ownership & multi-provider architecture (§5.1), streaming scoping (§5.2), determinism rationale for the egress boundary (§5.3), routing-reason telemetry (§2, §10 actions)
**Date:** 2026-07-14
**Depends on:** ADR-001 (shared runtime), ADR-001.1 (boundaries — esp. Part 8: no networking in the runtime), ADR-002 (capabilities + dispatch), ADR-003 (orchestration, policy engine, closed state machine)
**Formalizes:** the existing stage-4 fallback (`FALLBACK/GENAI` results; app-constructed requests; the deliberate rule that the raw utterance never enters an `NLUResult`) into a complete, policy-governed architecture.

---

## Executive Summary

The platform's identity is deterministic on-device execution; GenAI is the **fallback of last resort**, not a co-assistant. This ADR formalizes that stance into mechanism, and every design choice follows from one framing decision:

> **GenAI is a capability** — `assist.cloud`, registered like any other (ADR-002), with an action contract (`genai.query`), availability conditions (network, consent, auth), execution mode `async`, localized failure responses, and policy bounds. It is *not* a new layer, a new routing subsystem, or a special path through the Orchestrator.

Everything the Board asked for then falls out of already-approved machinery: routing is a **named policy decision point** in the Orchestrator (the Routing Decision Engine is a policy table + the existing cascade, not a new component); conversation continuity is ordinary frame semantics around an async step; offline behavior is ordinary capability unavailability; privacy is the existing egress rule (the runtime never sees the network; only the SDK-side handler — which already holds the utterance — builds the request); cost control is mostly *routing discipline* (clarify-before-escalate, never-send-context) rather than caching cleverness; and future native capabilities reduce cloud traffic automatically, because every intent added to the bundle is an utterance class the router never sees again. The state machine gains no states; the Orchestrator gains no code paths; the closed vocabularies gain one capability and a handful of policy rows.

Two hard rules anchor the design: **cloud output can never cause device action** (GenAI responses are narration-only; no path exists from a cloud response to the Action Dispatcher), and **escalation sends the current utterance only** — never conversation history, never device state, never context (which is simultaneously the privacy boundary and the largest cost saving).

---

# Part 1 — Where Routing Occurs

| Candidate location | Verdict |
|---|---|
| **Before the Planner** (i.e., inside the cascade) | Half right — the cascade already produces the primary routing *signal* (confident intent / learned-OOS / low-confidence), and that stays. But the cascade lacks conversation context: it cannot know a low-confidence turn is actually a slot answer, or that the clarification budget is spent. Routing purely here re-creates today's behavior, which this ADR exists to improve. |
| **Inside the Planner** | No. The Planner is pure, stateless, and untrusted (ADR-003 §3); escalation depends on live context (frames, budgets, availability) and is an *arbitration*, which is the Orchestrator's noun. The Planner's only involvement: it may return `Rejection(unplannable)`, which is one *input* to the routing decision. |
| **Inside the Conversation Orchestrator** | **Yes — the decision.** Escalation is a conversation-lifecycle arbitration ("this turn cannot be served locally") made at a named policy decision point (ADR-003 §8 gains a seventh row: *Escalation decision*). The Orchestrator consults policy + cascade output + context and, if escalating, admits a one-step plan: `PlanStep{intent: sys.cloud_assist → action: genai.query}`. |
| **Inside the Capability Registry** | Availability only. The registry answers "is `assist.cloud` usable right now?" (network, consent, auth, rate-budget) via the ordinary snapshot — it never decides *whether a turn should* escalate, only whether it *can*. |
| **Inside the SDK** | Execution only. The `assist.cloud` handler owns the HTTP call, timeout, auth, retry/backoff — plumbing per ADR-002 B. The SDK makes no routing decisions (ADR-002 B3 design rule holds). |

**Architecture:** decision in the Orchestrator (runtime — it changes what the user experiences, Q3), capability in the registry (native availability facts), execution in the SDK handler (native networking, ADR-001.1 Part 8). No new components; one new capability; one new policy decision point.

```
 cascade signal ─┐
 planner result ─┤                                    ┌─► local plan (99% path)
 context (frame, ├─► ORCHESTRATOR: escalation ────────┤
  budgets, acts) │    decision point (policy table)   └─► PlanStep(genai.query)
 availability ───┘                                         └─► Dispatcher ─► assist.cloud
                                                               handler (SDK, native)
```

### 1.1 Why a Capability, not a dedicated Fallback Provider abstraction (review comment #1)

A `FallbackProvider` interface — a special platform slot for "the thing that handles what we can't" — is the conventional design, and it was considered. It loses to the capability model on every axis that matters here:

1. **It would duplicate machinery that already exists.** A fallback provider needs registration, availability (network/consent/auth), dispatch with timeout, an outcome taxonomy, failure responses, policy bounds, and telemetry stitching. That list *is* the ADR-002 capability contract. A second abstraction with the same shape is the definition of accidental complexity.
2. **It would create a second kind of thing.** Every future component — Studio dashboards, policy tables, availability snapshots, the completeness gates — would need to special-case "capabilities, and also the fallback provider." The capability model keeps the platform's ontology at exactly one executable unit.
3. **The abstraction would encode the wrong variability.** A `FallbackProvider` interface implies the useful swap axis is *the fallback itself*. It isn't — there will be one escalation path for the product's lifetime. The real variability is *providers behind it* (OpenAI/Azure/Gemini/Claude), which lives entirely inside the handler (§5.1) where an interface is actually warranted.
4. **Safety bounds come free.** The no-cloud-action law is enforceable precisely *because* `assist.cloud` is an ordinary capability whose action contract simply declares no device actions — the Dispatcher's existing contract checking does the enforcement. A privileged fallback slot would need bespoke containment rules.
5. **Honest cost of the choice:** capability semantics carry concepts GenAI doesn't use (slots, workflows — its workflow is a degenerate zero-slot pass-through). This is harmless surplus, and cheaper than the alternative's parallel machinery.

The general principle, worth recording for future reviews: **when a new need fits an existing closed vocabulary at ~90%, extend the instance table, not the type system.**

# Part 2 — Routing Decision Engine

The "engine" is a policy-governed decision ladder evaluated once per fresh turn (mid-flow turns short-circuit at rule 1). Rules in order; first match wins:

| # | Condition | Route | Why |
|---|---|---|---|
| 1 | Active frame expects this turn (slot answer, confirmation response, clarification answer) | **Local — always.** Mid-workflow turns never escalate | A garbled slot answer is a re-prompt (attempt budget exists for exactly this); escalating mid-flow would strand a half-collected workflow and leak flow context into a cloud query |
| 2 | Dialogue act = universal verb / correction / back-reference | **Local — always** | Acts over history are meaningless to a stateless cloud query |
| 3 | Confident known intent (cascade ≥ policy threshold, incl. agreement gate) | **Local execute** | The product's whole point |
| 4 | Known intent, capability unavailable (no aids connected, feature absent, permission missing) | **Local unavailable-response — never cloud** | *Unavailability is not un-understanding.* "Start streaming" with no aids paired must yield "connect your hearing aids first," not a cloud essay about streaming. Escalating here would be actively wrong |
| 5 | Near-threshold with in-domain evidence (semantic top-N contains real intents at moderate score, or partial entity hits) | **Clarify first** (one question, within budget), escalate only if clarification fails | Clarification is free, local, private, and resolves most near-misses; also the primary cost lever (Part 7) |
| 6 | Learned OOS class predicted, or below all thresholds with no in-domain evidence, or Planner `Rejection(unplannable)` on a well-formed request | **Escalate** (subject to rule 7) | The trained rejection path — genuinely out-of-scope ("what's the weather") is GenAI's intended diet |
| 7 | Escalation chosen but `assist.cloud` unavailable (offline / no consent / auth / rate budget) | **Local honest limitation** response per condition (Part 6) | Ordinary ADR-002 A6 unavailability semantics |
| 8 | *Safety override, evaluated before 6:* utterance shows device-control evidence (device-domain keyword tier hit or device-domain in semantic top-N) but confidence insufficient | **Clarify, never escalate** | A possibly-misheard device command must resolve on-device; the failure mode "user asked for volume, cloud chatted about volume" is a trust-destroying outcome on a medical device |

Two standing prohibitions restated as routing law: **no cloud → action path** (the `assist.cloud` outcome taxonomy contains only `Success(narration)` and failures — the Dispatcher cannot receive an action from a cloud response, structurally); and **no speculative dual-path** (never query cloud in parallel "just in case" — cost, privacy, and determinism all forbid it).

**Every routing decision emits its reason (review comment #5).** The ladder is instrumented: each turn's telemetry event carries a `routing_reason` from a closed enum matching the rules one-to-one — `frame_turn (r1)`, `dialogue_act (r2)`, `confident_local (r3)`, `capability_unavailable (r4)`, `clarify_first (r5)`, `escalated_oos (r6a)`, `escalated_low_confidence (r6b)`, `escalated_planner_rejection (r6c)`, `cloud_unavailable_{offline|consent|auth|budget|breaker} (r7)`, `safety_override_clarify (r8)` — plus, for rule-5 turns, the follow-up outcome (`clarify_rescued_local` / `clarify_escalated` / `clarify_abandoned`). This makes the ladder's production behavior fully analyzable: escalation composition by reason and language, the clarify-rescue rate (the rule-5/6 boundary's tuning signal), rule-4 volumes (feature-demand on unsupported hardware), and rule-8 fires (near-miss device commands) each drive a different owner's roadmap. The enum is part of the telemetry schema version; reasons are decision metadata, never content.

# Part 3 — Confidence Strategy

Classifier confidence alone is insufficient — it is one signal in a context-aware gate (the ladder above operationalizes this). The full signal set:

| Signal | Contribution |
|---|---|
| Calibrated cascade confidence + stage + agreement gate | Primary signal (exists today — temperature-scaled, per-language) |
| Semantic top-N composition | Near-domain vs. far-domain discrimination: top-N dominated by real intents → rule 5 (clarify); dominated by OOS/flat → rule 6 (escalate) |
| Entity/slot extraction | Successful entity hits (a datetime, a memory name) are in-domain evidence → bias to clarify, not escalate |
| Conversation context | Active frame → rule 1; recent same-domain fulfillment → mild in-domain prior (a user mid-volume-session saying something garbled is probably still talking about volume) |
| Clarification budget | Already spent (2) → skip rule 5, escalate directly — never trap the user in a clarify loop when cloud can help |
| User corrections | A correction act routes to repair (rule 2) — *never* escalate a correction; "no, I meant the left one" sent to cloud is nonsense |
| Capability availability | Distinguishes rule 4 from rule 6 — the single most important disambiguation this ADR adds over today's behavior |

Decision process: rules 1–2 (context) → cascade (3) → availability check (4) → evidence assessment (5/8 vs 6) → cloud availability (7). All thresholds and the evidence heuristics' parameters are policy-table configuration (compiled, versioned, A/B-able per ADR-003 §8) — the routing engine ships tunable, because the correct clarify-vs-escalate balance is an empirical, per-language question.

# Part 4 — Conversation Continuity

A cloud turn is an ordinary conversation: PLANNING admits the one-step `genai.query` plan → EXECUTING dispatches → mode `async` → WAITING → response arrives via `notifyExecution(Success(narration))` → the Dialogue Manager renders it (TTS/UI) → COMPLETED → frame pops → IDLE. **No new states (ADR-003 §4 closed-machine law holds), no orchestrator changes.** The Board's example is therefore trivial: "What's the weather?" completes its frame; "Increase my volume" is a fresh turn, classified confidently, executed locally — the platform never left its own state machine, so nothing needs "restoring."

The subtler cases:

- **Cloud follow-ups** ("…what about tomorrow?" after a weather answer): the device holds no cloud conversation content (Part 5), so continuity is server-side. The `assist.cloud` handler receives an opaque `cloud_session_token` with each response and stores it in **Capability Context** (the sanctioned dynamic-state path, ADR-002 A6.3) with a short TTL (~60s). The routing ladder gains one refinement: an escalation-bound turn (rule 6) with a live token is sent *with* the token, letting the server thread its own context. The runtime knows only that a token exists — never what was said. Token TTL expiry or any local execution in between simply yields a fresh cloud conversation; graceful degradation is "the assistant is a bit forgetful about chit-chat," which is the correct priority for this product.
- **Interruption during WAITING:** "Increase my volume" while a cloud answer is pending is the existing interrupt path — volume executes immediately (it will beat the cloud round-trip); the late cloud result follows ADR-002 B6 async policy (`announce` while context lives, `silent` after).
- **History:** the action-history ring records `{genai_query, timestamp}` — the *event*, never the content — so "do that again" on a cloud turn re-escalates the fact of a query but cannot replay text the runtime never kept. ("Again" without the original utterance yields a clarification — accepted, correct behavior.)

# Part 5 — Privacy Architecture

The structural guarantee comes first: **the runtime cannot leak what it cannot reach.** The runtime has no network (ADR-001.1 Part 8), never embeds the utterance in results (existing invariant, preserved), and raw text dies with Turn Context (ADR-003 §5). The only egress point in the entire platform is the `assist.cloud` handler in the SDK — one component, natively owned, consent-gated, auditable in isolation.

| Data | May leave device? | Rule |
|---|---|---|
| Current utterance transcript | **Yes — the only payload.** Only the single escalated turn, only after rule-6 routing, only under consent | The handler already holds the text (app layer, per existing design); pre-egress redaction pass (digits-in-context, platform NER for names) before send |
| Language + locale | Yes | Required for a useful answer; not identifying |
| Opaque `cloud_session_token` | Yes (round-trip) | Server-issued, content-free on device, short-TTL |
| Conversation history / prior turns | **Never** | Continuity is server-side via token or not at all |
| Device state, battery, BT state, aid model | **Never** | No cloud answer needs it; anything device-related routed to cloud is a routing bug (rule 4/8), not a data need |
| User preferences, memory presets, reminders content | **Never** | Health-adjacent; capability contexts are not readable by `assist.cloud` (ADR-002 A9 isolation does the enforcement) |
| Medical information | **Never** (and the redaction pass is the backstop for utterances that volunteer it) | |
| Planner/orchestrator state, confidence values, traces | **Never** | Internal telemetry stays in the telemetry pipeline (aggregated, anonymous, separate consent) |
| Device/user identifiers | **Never** in query payloads; auth uses service credentials not user identity where the service design permits | Escalated queries must not be joinable to a user profile server-side |

Consent model: GenAI escalation is **off until explicitly enabled** (first-run choice, revocable in settings). No consent → `assist.cloud` permanently unavailable → rule 7 honest limitation ("I can help with your hearing aids and reminders; enable cloud assistant in settings for general questions"). Consent state is an availability condition — no special code path. The existing repo defect (placeholder `genai.yourcompany.com` reachable in production paths) is closed by this ADR: the handler refuses to activate without a signed, non-default endpoint configuration — availability again, not a hidden crash.

### 5.1 Prompt construction, template ownership, and multi-provider support (review comments #2, #3)

**The device never constructs prompts. It constructs a request envelope.** The `assist.cloud` handler sends a minimal, versioned envelope — `{utterance (redacted), language, cloud_session_token?, envelope_version}` — to a **company-owned GenAI Gateway**, and the gateway owns everything LLM-shaped: system prompts, persona, guardrail instructions, provider selection, provider-specific request formats, model tier, and context threading via the session token. Ownership by layer:

| Artifact | Owner | Why there |
|---|---|---|
| Request envelope schema (fields, redaction guarantees) | SDK handler contract (versioned with the SDK) | It is the egress boundary — auditable in one place (§5) |
| System prompts / persona / guardrail templates | **Gateway configuration** (server-side, independently versioned and deployed) | Prompt iteration cadence is days, not app-release cycles; prompt-injection hardening evolves server-side without fleet updates; prompts are provider-specific (see below) so they belong next to provider adapters |
| Provider adapters (OpenAI / Azure OpenAI / Gemini / Claude request formats, auth, streaming consumption) | Gateway | Provider churn — pricing, models, APIs — must never touch a shipped app, let alone the runtime |
| Routing/escalation policy | Bundle policy tables (ADR-003 §8) | Decision-plane, as established |
| **Not** the NLU bundle | — | The bundle is signed *runtime* content; the runtime has no network and never sees a prompt — putting prompts there would couple prompt iteration to content releases and violate the boundary spec for zero benefit |
| **Not** the Policy Engine | — | Policy governs *whether* to escalate, never cloud content |

**Multi-provider support therefore requires zero runtime changes by construction** — the runtime knows one action (`genai.query`), the SDK knows one endpoint (the gateway), and providers are a gateway deployment concern: swap, blend (tiering cheap/expensive models by query class), or fail over between OpenAI, Azure, Gemini, and Claude behind a stable envelope. This also centralizes provider credentials off devices entirely (the app authenticates to the gateway with service credentials; provider keys never ship in an app).

*Fallback stance if a gateway is not built* (accepted for dev/pilot only): the handler holds a provider-adapter interface internally and templates live in SDK remote-config — never in the NLU bundle. This preserves every boundary but forfeits centralized prompt iteration, key custody, and provider blending; the ADR recommends the gateway before GA.

### 5.2 Streaming responses: intentionally out of scope at the conversation layer (review comment #4)

**The runtime contract is non-streaming and stays so**: `genai.query` returns one complete `Success(narration)`. Reasons: the state machine's WAITING→outcome semantics (and the closed-machine law, ADR-003 §4) would otherwise need partial-result states; timeout, cancellation, and telemetry semantics are well-defined only for atomic outcomes; and the primary output channel is TTS, where sub-sentence token streaming has no UX value for the short factual answers this fallback serves. Two permitted uses of streaming that don't touch the contract: the **gateway/handler may consume** provider streaming APIs internally (often lower time-to-full-response) and deliver the assembled narration as one outcome; and a future **progressive text rendering** in the app UI would be an SDK/app presentation concern fed by the handler — the runtime would still receive exactly one outcome. If long-form cloud content ever becomes a product goal, streaming enters through a revision of the outcome taxonomy — a deliberate platform release, not a handler hack.

### 5.3 Beyond privacy: why context and runtime state never leave — and why cloud output never enters (review comment #6)

The egress rules of this Part are usually read as privacy. They are equally **determinism architecture**, and the boundary is one-way in *both* senses:

**Outbound (context never leaves):** local behavior must remain a pure function of `(bundle version, device state, user input)` — the property that makes the conversation corpus a real spec, cross-platform parity checkable, A/B results attributable, and any fleet incident reproducible from a telemetry record. The moment escalation payloads include conversation context or runtime state, the *cloud's* behavior becomes a function of local state — and pressure inevitably follows to let its richer answers flow back into that state, completing a feedback loop through an unversioned, nondeterministic external system.

**Inbound (cloud output never enters runtime state):** the stronger and less obvious rule. A cloud narration is **rendered and discarded** — it is never parsed, never entity-extracted, never written to any context layer, never consulted by the router, the Planner, or the Workflow Engine. The single sanctioned residue is the opaque session token (content-free by construction) and the content-free history event (`{genai_query, ts}`). Consequences: a hallucinating, drifting, or outright compromised provider **cannot steer the assistant** — it cannot influence a future routing decision, bias a classification, pre-fill a slot, or trigger an action, because no data path exists from its output to any component that decides anything. This extends the no-cloud-action law to its full form: **no cloud → action path, and no cloud → state path.** The assistant's deterministic core behaves identically whether the cloud answered brilliantly, badly, or not at all — which is precisely the guarantee a medical-adjacent device must be able to state in an audit.

# Part 6 — Offline & Failure Behaviour

All five scenarios are `assist.cloud` availability/outcome conditions with schema-configured, localized responses — no new mechanism:

| Condition | Detected by | User experience | Notes |
|---|---|---|---|
| No internet | Registry (connectivity observer) → unavailable *before* dispatch | Rule 7: "That needs an internet connection — I can still control your hearing aids and reminders." | **Do not ask to rephrase** — rephrasing cannot fix connectivity; do not clarify (rule 5 already ran; this turn is confirmed out-of-scope). Honest + capability reminder |
| Cloud timeout (async 15s per ADR-002 B4) | Dispatcher | "I couldn't get an answer just now — try again in a moment." Conversational retry only (no silent re-send) | Frame completes as FAILED-with-response; nothing pends |
| Auth failure | Handler → `Failed(internal)` | Same generic "having trouble reaching the assistant" — auth is never a user problem | Telemetry-distinguished (`auth_failure`) for ops; SDK refreshes credentials off the conversation path |
| Rate limiting (server or local budget) | Handler / registry budget check | Generic trouble response; if local budget: "I've answered a lot of general questions today — I can still help with your hearing aids." | Local budget is a policy value (Part 7); the honest variant respects the user more than fake errors |
| Server unavailable (5xx) | Handler | Same as timeout | SDK-side circuit breaker (cooldown after N failures) flips availability off temporarily — subsequent turns get the *fast* offline response instead of N users × 15s hangs |
| — Universal rule | — | **Never queue utterances for later transmission** | A stored transcript awaiting connectivity is a privacy liability no retry convenience justifies |

# Part 7 — Cost Optimisation

Ordered by leverage; the biggest levers are routing discipline, already designed:

1. **Clarify-before-escalate (rule 5)** — the primary filter. Every clarification that recovers a local intent saves one API call *and* converts a would-be cloud user into a served on-device user. Tuning the rule-5/rule-6 boundary is a cost knob exposed in policy, evaluated against escalation telemetry.
2. **No context transmission** — the privacy rule is the cost rule: single-utterance prompts are minimal tokens by construction. Context summarization is a *server-side* concern (threading via token); the device never pays tokens for history it refuses to hold.
3. **Local rate budget** (policy: per-device daily escalation cap, generous but real) — bounds worst-case spend from pathological inputs (a TV in the room feeding the mic), with the honest budget-exhausted response.
4. **Short-window duplicate suppression** — identical normalized utterance within ~30s (ASR echo, user repetition at a slow server) reuses the in-memory response instead of re-querying. Memory-only, TTL-bound, dies with the process — *not* a persisted response cache (persisted cloud content on device is a privacy surface and a staleness bug factory for negligible hit-rate: real duplicate queries across sessions are rare and weather answers go stale in hours). This is the deliberate trade-off: accept near-zero cache sophistication, keep zero stored content.
5. **The structural lever — capability growth (Part 8):** escalation telemetry (intent-less clusters: `{semantic neighborhood, count}` — no text) is the product roadmap for new capabilities/intents. Every escalation cluster converted to a bundle intent is permanently removed cloud spend. The router's cost curve is designed to *decay with content investment* — that, not caching, is the long-term cost architecture.
6. Prompt optimization (system-prompt tuning, model tier selection) is server-side and out of this ADR's scope, but the contract helps: single-turn, single-utterance, language-tagged queries are the cheapest possible shape to serve.

# Part 8 — Future Evolution

New native capabilities (Weather, Calendar, Contacts, Knowledge/RAG, Smart Home) integrate exactly as ADR-002 defines — manifest + content + handlers — and the routing architecture responds **without any change to the Orchestrator, the ladder, or the policy vocabulary**:

- A weather capability adds `weather.*` intents to the bundle → the cascade now recognizes weather utterances confidently → rule 3 routes them locally → cloud traffic drops by the weather cluster's volume. The router never learns "weather" exists; it only ever knew "confident intent → local."
- The `help` domain's RAG conversion (platform review §20) removes the informational-question escalations the same way — retrieval is an intent workflow (ADR-003 §3 "step kind"), not a router concept.
- Escalation telemetry prioritizes which capability to build next (lever 5) — the routing layer functions as a **demand-measurement instrument** for the capability roadmap.
- If a future hybrid Planner consults cloud reasoning (ADR-003 §3 v2+), it reuses this ADR wholesale: same consent gate, same availability conditions, same egress rules, same no-cloud-action law (a cloud-*planned* step still executes through local workflows and their validations). Cloud *planning* and cloud *answering* are two consumers of one escalation architecture.
- End state: GenAI's share shrinks toward true chit-chat and long-tail knowledge — the correct terminal role for a fallback capability. Nothing needs deprecating; the ladder's rule 6 simply fires less.

# Part 9 — Sequence Diagrams

```
[1] KNOWN INTENT (99% path — cloud never involved)
 text ► Orchestrator ► cascade: Cmd.VolumeIncrease @0.93 ► rule 3
      ► plan(1 step) ► Workflow ► Dispatcher ► volume handler ► Success ► "Volume increased."

[2] UNKNOWN INTENT (clean escalation)
 "what's the weather" ► cascade: OOS class ► rules 1–5 pass over ► rule 6
      ► availability(assist.cloud): OK ► plan{genai.query} ► [EXECUTING→WAITING]
      ► SDK handler: consent✓ redact ► POST(utterance, lang, token?) ► 200(narration, token')
      ► notifyExecution(Success) ► token'→CapabilityContext ► render ► [COMPLETED]

[3] LOW CONFIDENCE, IN-DOMAIN EVIDENCE (clarify saves a query)
 "make it better in here" ► cascade @0.48, top-N: {VolumeIncrease, MemoryChange, OOS}
      ► rule 5 ► [CLARIFYING] "Do you want the volume up, or a different sound setting?"
      ► "volume" ► rule 3 ► local execute        (cloud: 0 calls)

[4] CLOUD TIMEOUT
 …as [2] until WAITING ► 15s ► Dispatcher: Failed(timeout) ► [RECOVERING]
      ► policy: failure map ► "I couldn't get an answer just now — try again in a moment."
      ► [FAILED→IDLE]  (no retry, no queue; circuit breaker counts it)

[5] OFFLINE
 "what's the capital of France" ► rule 6 ► availability: unavailable(network) ► rule 7
      ► "That needs an internet connection — I can still help with your hearing aids."
      (fast: no dispatch, no timeout wait)

[6] CONTINUATION AFTER CLOUD RESPONSE  (Board's example)
 "what's the weather" ► [2] ► COMPLETED, frame popped, token in CapabilityContext
 "increase my volume" ► fresh turn ► rule 3 ► local execute      (state machine never
      left home; nothing to restore)
 "and what about tomorrow?" ► cascade: OOS ► rule 6 ► token alive ► escalate WITH token
      ► server threads context ► answer      (device still holds zero conversation content)
```

# Part 10 — Risks & Trade-offs

**Trade-offs accepted:** clarify-first adds one turn of friction for near-threshold utterances (chosen: friction over cost/privacy — and over wrong-action risk via rule 8); no persisted response cache (privacy over marginal cost savings); server-side-only continuity (a "forgetful" cloud persona over device-held transcripts); no cloud-initiated actions ever (capability ceiling for `assist.cloud` — deliberate and permanent for this product class); single-utterance context (occasionally worse cloud answers than history-aware assistants give — the identity trade of a privacy-first medical product).

**Risks:** R1 — *threshold miscalibration shifts the local/cloud balance silently* per language (mitigation: escalation-rate per language is a first-class fleet metric with alerting; thresholds are policy-versioned and A/B-able). R2 — *the redaction pass under-redacts volunteered medical content* (mitigation: redaction is testable in isolation with an adversarial corpus; consent language is explicit that general questions leave the device). R3 — *GenAI answer quality reflects on the assistant's trustworthiness* — users don't distinguish the deterministic assistant from its fallback (mitigation: distinct voice/visual treatment for cloud answers is an app UX decision this ADR recommends; telemetry separates satisfaction signals by route). R4 — *scope creep toward cloud-first* — each future feature will be tempted to "just use GenAI" instead of building a capability (mitigation: the no-cloud-action law makes GenAI structurally unable to serve device features; escalation telemetry makes the cost of not building capabilities visible). R5 — *the `assist.cloud` handler becomes the platform's most attack-relevant component* (mitigation: it is one auditable module; endpoint pinning + signed configuration; the runtime-side design means a compromised handler still cannot reach device actions or stored contexts).

## Action Items

1. [ ] Register `assist.cloud` as a capability: manifest, availability conditions (network, consent, auth, budget), `genai.query` contract, localized limitation/failure responses.
2. [ ] Add the escalation decision point + routing-ladder parameters to the policy schema (compiled tables, ADR-003 §8).
3. [ ] Implement rules 4 and 8 in the current Python engine now — both are corrections to today's behavior (unavailable ≠ unknown; device-evidence never escalates) and need no new infrastructure.
4. [ ] Specify the redaction pass + adversarial test corpus; wire the endpoint-configuration guard (closes the placeholder-URL defect).
5. [ ] Add escalation telemetry with the `routing_reason` enum (§2): rate per language × reason, clarify-rescue rate, rule-4/rule-8 volumes, duplicate-suppression hits, budget exhaustions, semantic-neighborhood clusters (no text) for the capability roadmap.
6. [ ] Define the consent flow + settings surface with product/legal (blocking for any production escalation).
7. [ ] (rev. 2) Decide gateway build vs. dev-only direct-provider fallback (§5.1) — gateway recommended before GA; specify the request-envelope schema + version either way.
8. [ ] (rev. 2) Add the no-cloud-state rule (§5.3) to the runtime contract tests: assert no context layer is writable from an `assist.cloud` outcome and no parser touches cloud narrations.
