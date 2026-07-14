# Runtime Interface Contract v1 (prose spec — no Rust)

Status: normative once merged; this is the **runtime contract version** that
`bundle.json:engine_compat` gates against (ADR-005 Part 3). Satisfies
ADR-001 AI#3 ("define `InferenceBackend` + session-state contracts in the
bundle/engine spec — pre-work, no Rust") and ADR-002 AI#5 ("specify
`notifyExecution` and the availability-snapshot push").

Purpose: the current Python engine and any future shared core implement the
**same seams**, so the ADR-001 Phase-2 migration is mechanical. Nothing here
requires new code today; the Python engine conforms incrementally.

## 1. Interface set (the five seams)

```
host ──► core   NluSession.handle(text, turn_id) -> NLUResult
host ──► core   NluSession.notify_execution(turn_id, outcome)
host ──► core   NluSession.push_availability(snapshot)
core ──► host   InferenceBackend.tfidf_logits(features) -> float[]
core ──► host   InferenceBackend.embed(token_ids) -> float[]
core ──► host   Clock.now_monotonic_ms() -> int
```

Calls **in** are synchronous from the host's single worker executor; calls
**out** are exactly the two inference callbacks plus the clock. The core is
otherwise side-effect-free: no I/O, no network, no threads, no wall-clock.

## 2. `InferenceBackend` (the inference inversion, ADR-001 Part 4)

The conversation logic **never links an ML runtime**; the host injects one.

- `tfidf_logits(features: sparse{index → weight}) -> float[n_labels]` —
  RAW decision-function logits, NOT softmaxed (the core applies
  `softmax(logits / T)` itself; a backend that bakes softmax in is
  non-conformant — the skl2onnx `raw_scores=True` lesson).
- `embed(token_ids: int[≤ max_len]) -> float[dim]` — the stage-3 encoder
  forward pass. Static shape, batch 1 (on-device constraint). `dim`,
  `max_len`, and dtypes come from `runtime/cascade.json`; the backend MUST
  assert them at model load (defensive half of compiler stage 8).
- Determinism: same inputs → same outputs within FP tolerance published by
  the parity suite (Tier-A/B: acc Δ ≈ 0, gate disagreements 0/30).
- Errors: backends throw/return a typed `InferenceError`; the core maps it
  to a fallback turn (never a crash). Backends never retry silently.
- Conformance today: Python = ORT; iOS = CoreML (ANE); Android = ORT Mobile.

## 3. Session-state blob

Owned and serialized by the core (schema below); **persisted and encrypted
by the host** (platform keystores own key material — ADR-001 Part 5).

```jsonc
{
  "v": 1,                        // blob schema version, independent axis
  "session_id": "…",
  "lang": "en",
  "updated_at_mono_ms": 123456,  // against the injected monotonic clock
  "frames": [                    // active conversation frames, newest last
    { "intent": "reminder.task.create",
      "slots_filled": { "when": "08:30" },
      "awaiting_slot": "period",  // or null
      "attempts": 1,
      "confirm_pending": false,
      "ttl_ms": 120000 }
  ],
  "last_result": { "intent": "…", "confidence": 0.91 },   // for referents
  "counters": { "cloud_escalations": 0 }                   // budgets
}
```

Rules: TTLs are evaluated against the monotonic clock at `handle()` entry
(rehydrate → sweep → route); the blob never contains raw utterance text
(privacy invariant, enforced in one audited place); unknown fields in a
newer blob minor version are preserved round-trip, never dropped.

## 4. `notify_execution(turn_id, outcome)` (ADR-002 §B7)

Feedback edge from the Action Dispatcher to the conversation, on the same
serial executor as turns (total order: decide → execute → feedback → next).

Closed outcome taxonomy v1 (extending it is a platform-team change,
versioned with the bundle format):

```
Success(result?)        | Success(initiated)        // async ack
Failure(reason_code)    | Unavailable(condition)    // condition from the
NeedsUser(permission_flow)                          //   availability vocab
```

Semantics: `sync` actions — the runtime holds the frame open until the
outcome arrives; `async` — completion arrives later with the originating
`turn_id`, and dialogue policy for late results is schema-declared
(`announce` | `silent`), auto-downgrading to `silent` after context TTL;
`session` — the conversation's job ends at successful start; stop/status
are new turns, never held handles. Reentrancy: a handler may NOT call the
facade (no self-injected turns).

## 5. Availability snapshot push (ADR-002 §A6/B4)

The native CapabilityRegistry evaluates each capability's `availability`
spec against live state and pushes:

```jsonc
{ "snapshot_id": 42,             // monotonic; stale pushes are dropped
  "capabilities": {
    "device.audio":       { "state": "available" },
    "assistant.reminder": { "state": "unavailable",
                             "condition": "app.notifications_enabled" }
  },
  "context": {                    // per-capability sanctioned state
    "device.audio": { "dynamic_entities": { "audio.memory.preset": ["outdoor", "tv"] },
                       "session_active": false }
  }
}
```

Rules: pushed on session start and on every state change (push-on-change,
no polling, no staleness window); swap is atomic on the serial executor —
a turn sees one consistent frozen snapshot; recognized-but-unavailable
intents route to the capability's `unavailable_response` (intents are
NEVER removed from the label space — ADR-002 A6 / STOP rule).

## 6. Ordering & threading contract

One serial executor owns all runtime calls — `handle`, `notify_execution`,
`push_availability` — totally ordered. Handlers run off-executor
(dispatcher hops), so a slow handler never blocks classification. The core
is single-threaded by contract; concurrency is the host's problem.

## 7. Versioning

This document defines **runtime contract version 1**. Bundles declare
`engine_compat: {min_runtime_contract, max_tested_runtime_contract}`
against this number. Changes: additive clarifications = editorial;
new interface members or vocab entries = contract version bump + bundle
`required_runtime_features` flag where feature-gated; semantic changes to
an existing member = major coordination per ADR-005 Part 10.

## 8. Conformance status (2026-07-14)

| Seam | Python engine today | Gap to v1 |
|---|---|---|
| `handle()` | `NLUEngine.handle` | turn_id threading; input bounding is present |
| `InferenceBackend` | ORT inline in classifier/semantic | extract behind an interface (mechanical; ND-2 M1 window) |
| Clock injection | wall-clock via `now` param in places | centralize monotonic provider |
| Session blob | `SessionStore` in-memory | serialize to the §3 schema; host encryption N/A server-side |
| `notify_execution` | not present (CLI executes nothing) | add no-op seam when SDK work starts |
| Availability push | not present | add with capability repartition |

These gaps are tracked as Phase-1/2 work items, not defects.
