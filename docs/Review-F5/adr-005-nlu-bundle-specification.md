# ADR-005: NLU Bundle Specification & Bundle Compiler Architecture
## The Official Contract Between the Training Pipeline, the Compiler, and Every Runtime

**Status:** Proposed
**Date:** 2026-07-14
**Depends on:** ADR-001 (bundle concept, D1), ADR-001.1 (Bundle Interface §9.2, format contract §9.6, stable-identifier rule), ADR-002 (capability manifests, action contracts), ADR-003 (compiled policies, plan facts, closed vocabularies, package boundaries), ADR-004 (routing tables, telemetry enums, nothing-LLM-shaped-in-bundle rule)
**Audience & force:** This is the engineering contract between the Python training pipeline, the Bundle Compiler, the iOS runtime, the Android runtime, and any future platform. Independent teams implement against this document. Changes to it follow the format-versioning rules it defines (Part 8).

---

## Executive Summary

The Bundle is the single deployable artifact of the conversational platform: one signed, versioned, self-describing archive containing everything a runtime needs to *be* the assistant — models, calibration, compiled intent workflows, entities, grammars, lexicons, policies, routing tables, prompts, and telemetry vocabulary. One artifact, one signature, one compatibility gate, one atomic swap. The Compiler is the only producer of bundles; runtimes are pure consumers that may assume every internal consistency guarantee the compiler proves at build time. The design principle throughout: **move every possible failure from a user's device at runtime to CI at compile time**, and make everything a runtime must trust *verifiable* and everything it must interpret *versioned*.

---

# Part 1 — Bundle Philosophy

**Why one artifact instead of individual models / JSON files / workflows / capabilities:**

1. **Atomic consistency is the product's correctness condition.** The platform's own history is the argument: the repo grew a label/schema parity assertion, a SHA-256 manifest, and startup cross-checks precisely because model, labels, schema, and calibration shipped as *separate files that could disagree*. A classifier from Tuesday with Monday's labels and Friday's thresholds is not a degraded assistant — it is an unpredictable one, on a medical device. A bundle makes the consistent set the *only* shippable unit; the inconsistent state becomes unrepresentable.
2. **One verification surface.** Security (signature), integrity (hashes), and compatibility (format gate) are each checked once, at one boundary, identically on every platform. N independent files means N×platforms verification paths and a combinatorial "which versions coexist" matrix no one will test.
3. **Atomic lifecycle.** Swap, rollback, A/B, and staged rollout are pointer operations on one artifact (ADR-001.1 §9.2 two-slot design). Per-file updates would require distributed-transaction semantics on a phone.
4. **The bundle is a testable snapshot of the assistant's mind.** Every evaluation report, conversation-corpus run, parity suite, and fleet telemetry record keys on one `bundle_id`. "What exactly was the assistant on 2026-09-03 for cohort B?" has a one-word answer.
5. **Team decoupling.** Content teams ship capabilities *into* the bundle pipeline; runtime teams consume *only* the bundle. Neither ever integrates against the other's working files — the compiler is the meeting point, and this spec is its law.

What the bundle deliberately is **not**: a code container (no executable logic — ADR-001.1's fact/behavior split; behaviors live in versioned runtimes), a prompt store (ADR-004 §5.1), or a user-data container (personalization overlays are device-side, Part 12).

# Part 2 — Complete Bundle Structure

Archive: **zip, stored with deterministic entry ordering and normalized timestamps** (reproducible byte-identical builds from identical inputs — Part 5). Extension `.nlu`. All structured files are JSON (compiled form; YAML exists only in source repos). All identifiers are the stable identifiers of ADR-001.1 §5 (intent ids, slot names, canonical entity values — never indices).

```
bundle.nlu
├── bundle.json                      # THE manifest (Part 3) — read first, trusted after signature
├── integrity/
│   ├── manifest.sha256              # path → SHA-256 for every other file in the archive
│   └── signature.sig                # Ed25519 over manifest.sha256 ⊕ bundle.json (Part 11)
├── models/
│   ├── intent/{lang}/model.{onnx|mlmodelc-ref}   # TF-IDF stage per language
│   ├── intent/{lang}/labels.json                 # ordered label list (the output contract)
│   ├── intent/{lang}/calibration.json            # temperature, thresholds derived, ECE record
│   ├── embedder/encoder.{onnx|ref} + vocab.txt   # stage-3 encoder + tokenizer vocab (paired forever)
│   └── semantic_head/{lang}/head.json            # weights, bias, labels incl. OOS class
├── runtime/
│   ├── cascade.json                 # stage wiring, per-stage tensor contracts (dims, dtypes)
│   ├── policies.json                # fully-resolved policy tables (ADR-003 §8: compiled, flat)
│   ├── routing.json                 # escalation-ladder parameters (ADR-004 §2)
│   └── plan_facts.json              # intent→capability→workflow map, admission caps, conflict rules
├── capabilities/{capability_id}/
│   ├── capability.json              # compiled manifest: version, actions+params+descriptors,
│   │                                #   availability conditions, platforms, languages
│   ├── workflows.json               # compiled intent micro-flows (slots, validation instances,
│   │                                #   confirmation/completion, back-references)
│   ├── entities.json                # capability-scoped entities
│   └── responses/{lang}.json        # prompts, clarifications, confirmations, fulfillment,
│                                    #   unavailable/failure responses
├── entities/
│   ├── shared/*.json                # cross-capability entity tables (synonyms, normalization,
│   │                                #   fuzzy flags, dynamic-source declarations)
│   └── system/datetime/{lang}.json  # datetime grammar tables per language
├── lexicons/{lang}.json             # yes/no, idioms, uncertainty, universal verbs, carriers,
│                                    #   referent tables, negation cues
├── keywords/{lang}.json             # compiled keyword rules: pattern (portable regex subset),
│                                    #   tier, guards, owning intent
├── telemetry/schema.json            # event schema version + closed enums (stages, routing_reason,
│                                    #   outcome taxonomy, lifecycle states) — so fleet analytics
│                                    #   can decode any bundle's events without code lookup
├── assets/{capability_id}/…         # typed opaque blobs a capability declares (e.g. future RAG
│                                    #   article index); type-tagged, size-capped, hash-covered
└── meta/
    ├── report_card.json             # evaluation snapshot: per-lang macro-F1, holdout, OOS recall,
    │                                #   wrong-action count, ECE, per-capability F1
    ├── lineage.json                 # dataset hashes (DVC), training env lock hash, compiler
    │                                #   version, git commits, experiment ids — full reproducibility
    └── release_notes.md             # human notes (optional)
```

Per-file contract summary (schema = JSON Schema in `spec/bundle/` — Part 13; "Val." = validated by compiler stage, Part 5):

| File | Purpose | Required fields (core) | Optional | Owner | Versioned by |
|---|---|---|---|---|---|
| `bundle.json` | Single source of truth | Part 3 | Part 3 | Compiler (generated only) | format_version |
| `integrity/*` | Trust chain | complete hash table; signature | — | Compiler/CI signer | signature scheme id |
| `models/intent/*` | Stage-2 classifier | model, ordered labels, calibration{temperature} | quantization tag | Training | model_version per lang |
| `models/embedder/*` | Stage-3 encoder | encoder ref, vocab, dim, max_len | quantization variants | Training | embedder_id (vocab+model paired — the ADR-001 P0-3 lesson, made structural) |
| `models/semantic_head/*` | Stage-3 head | weights, labels(+OOS), embedder_id it was trained against | — | Training | head tied to embedder_id — mismatch is a compile error |
| `runtime/cascade.json` | Stage wiring + tensor contracts | stages[], per-stage {input dtype/shape, output dim} | stage enable flags | Platform team | format_version |
| `runtime/policies.json` | All conversation policy | complete matrices (compiler-proved) | experiment overlay id | Platform defaults + capability tightenings | policy_schema_v + policy_content_v |
| `runtime/routing.json` | Escalation ladder params | thresholds, budgets, evidence params | — | Platform team | policy_schema_v |
| `runtime/plan_facts.json` | Planner lookup tables | intent→{capability, workflow}, caps | conflict pair rules | Compiler (derived) | format_version |
| `capabilities/*/capability.json` | Capability contract | id, version, actions[{key, params, descriptor}], availability | flags, superseded_by | Capability team | capability_version |
| `capabilities/*/workflows.json` | Intent micro-flows | per intent: slots, prompts refs, validation instances, completion | back_reference, followup | Capability team | capability_version |
| `capabilities/*/responses/*` | All user-facing text | every prompt key referenced by workflows/policies | per-string comments | Capability team + linguists | localization completeness table |
| `entities/*` | Entity tables | type, values/synonyms or grammar tables | fuzzy, dynamic source | Capability/linguist teams | entity table hash |
| `lexicons/*`, `keywords/*` | Language behavior facts | per-language completeness vs. declared languages | — | Platform + linguists | per-lang table hash |
| `telemetry/schema.json` | Event decoding contract | schema version, all closed enums | — | Platform team | telemetry_schema_v |
| `assets/*` | Typed capability blobs | type tag, format version per type | — | Owning capability | per-asset version |
| `meta/*` | Audit + reproducibility | report_card core metrics, lineage hashes | notes | Compiler (generated) | — |

Design rule for the whole tree: **a runtime interprets `runtime/`, `capabilities/`, `entities/`, `lexicons/`, `keywords/` through its closed vocabularies; it executes `models/` through the Inference Interface; it never interprets `meta/`** (device-ignored; exists for registry, Studio, and audit). Anything a future feature adds must land in one of those three roles.

# Part 3 — Bundle Manifest (`bundle.json`)

Read order: signature verifies `manifest.sha256 ⊕ bundle.json` first; nothing else in the archive is parsed before that check passes. Fields:

| Field | Req. | Meaning / rule |
|---|---|---|
| `bundle_id` | ✓ | Globally unique (content-hash-derived + monotonic counter). Keys every telemetry event, registry entry, and A/B cohort. |
| `format_version` | ✓ | **The compatibility contract** (Part 10). Integer major.minor: major = breaking (runtime must refuse newer major), minor = additive (unknown optional sections/fields are ignorable). |
| `content_version` | ✓ | Monotonic release number of the *content* (independent of format). Drives ordering, downgrade detection, telemetry attribution. |
| `compiler_version` | ✓ | Exact compiler build. Informational for runtimes (never gate on it — Part 10); essential for reproducibility and forensics. |
| `engine_compat` | ✓ | `{min_runtime_contract, max_tested_runtime_contract}` — expressed against the *runtime contract version* (the Part-12/ADR-003 interface set), not app versions. |
| `required_runtime_features` | ✓ | Closed feature-flag list this bundle needs (e.g. `frames`, `plans`, `conditional_steps`, `assist_cloud`). A runtime lacking any listed flag refuses the bundle — capability checks by declaration, not by crash (Part 10). |
| `languages` | ✓ | Declared languages with per-language completeness status (`full` \| `partial{missing: …}` — partial allowed only for `draft` channel bundles, never production). |
| `capabilities` | ✓ | `{capability_id: {version, status, superseded_by?}}` — the registry the SDK's contract-version gate (ADR-002 A7) reads. |
| `models` | ✓ | `{stage: {lang: {artifact, format, model_version, embedder_id?}}}` — what the Inference Interface must load per platform. |
| `policy_versions` | ✓ | `{schema, content}` (ADR-003 §8). |
| `telemetry_schema_version` | ✓ | Must match `telemetry/schema.json`. |
| `training` | ✓ | `{dataset_hashes, run_ids}` — summary of `meta/lineage.json` for registry display. |
| `checksums_root` | ✓ | Hash of `manifest.sha256` (binds the manifest to the file table under one signature). |
| `signature_info` | ✓ | Scheme (`ed25519-v1`), signing key id (for rotation, Part 11). |
| `created_at`, `channel` | ✓ | Build timestamp (normalized); `dev | beta | production` — runtimes on production builds refuse non-production channels. |
| `report_card_summary` | opt | Headline metrics duplicated for registry listings. |
| `experiment` | opt | `{experiment_id, variant}` for A/B bundles — telemetry stamps it. |
| `release_notes_ref` | opt | Pointer into `meta/`. |

# Part 4 — Capability Packaging

**Source form** (what teams edit — ADR-002 A4): `content/capabilities/<id>/` with `capability.yaml`, per-intent workflow YAMLs, entity YAMLs, per-language response YAMLs, and DVC dataset references. **Compiled form** (what ships): the `capabilities/<id>/` bundle subtree of Part 2 — flattened, validated, reference-resolved JSON.

Per-capability compilation contract:

- **Metadata:** id, version, owner, status, platforms, languages — copied through, cross-checked against the bundle's declared languages.
- **Intents/workflows:** compiled against the *workflow behavior vocabulary* of the target `format_version`; every prompt reference resolved to a `responses/` key; every entity reference resolved to a capability-scoped or shared entity; validation rules bound to the closed predicate set with parameters type-checked.
- **Entities:** capability-scoped tables emitted locally; references to `entities/shared/` recorded as dependencies (below).
- **Dialogue content:** all response keys for all declared languages, or compile error (completeness is per-capability × per-language).
- **Policies:** tightenings only (the algebra runs here); emitted *into* the resolved global `runtime/policies.json`, never as per-capability policy files a runtime would have to merge.
- **Training data:** *never inside the bundle* — datasets feed the training stage; the capability's contribution is recorded in `meta/lineage.json` (per-capability dataset hashes) so per-capability model provenance survives.
- **Assets:** typed blobs declared in the manifest with format-version tags.
- **Dependencies — deliberately minimal:** a capability may depend on (a) shared entities, (b) system entities, (c) platform vocabularies of the target format version. It may **not** depend on another capability (ADR-002 A9 — enforced here at compile time: any cross-capability reference is an error). Result: capabilities are order-independent compilation units; adding one can never break another at compile time, which is the property that keeps 25 future capability teams unblocked on each other.

# Part 5 — Bundle Compiler

One deterministic pipeline; every stage either transforms validated input or fails the build with a diagnostic naming the owning team. Stages in order:

| # | Stage | What it does / guarantees |
|---|---|---|
| 1 | **Load & schema-validate** | Parse all source YAML/JSON against `spec/` JSON Schemas. Catches: malformed files, unknown fields (error, not warning — silent typos like `requred:` must not compile), wrong types. |
| 2 | **Vocabulary check** | Every behavior/predicate/condition/act referenced exists in the target format version's closed vocabularies. Catches: content written for a newer runtime. |
| 3 | **Reference resolution** | Intents→capabilities, slots→entities, prompts→response keys, keyword rules→intents, referent tables→entities, `superseded_by`→existing intents. Dangling reference = error with the full path. |
| 4 | **Dependency resolution** | Capability→shared-entity edges resolved; cross-capability references rejected; shared entities orphaned by nothing are flagged (→ stage 6). |
| 5 | **Policy resolution** | Merge defaults + capability tightenings + experiment overlay under the tighten-only algebra; prove matrix completeness (ADR-003 §8). Output: flat `policies.json`/`routing.json`. |
| 6 | **Dead-content removal** | Strip: intents with `status: draft` (production channel), responses in undeclared languages, unreferenced shared entities, keyword rules for stripped intents. Emit a removal report (dead content is usually a mistake — surfaced, not silent). |
| 7 | **Training-artifact ingestion** | Pull models, labels, calibration from the training pipeline's output (by run id, hash-verified against `lineage`). The compiler *never trains*; it ingests. |
| 8 | **Cross-artifact parity** | The load-bearing stage: `labels.json` ≡ compiled intent set per language (the repo's runtime parity assert, moved to compile time); semantic-head labels ≡ intent set + OOS; head's `embedder_id` ≡ shipped embedder; calibration present for every model; tensor contracts in `cascade.json` match actual model I/O (probed via ORT). |
| 9 | **Localization completeness** | Every declared language: full lexicons, keyword tables, datetime grammars, all response keys, all system prompts. Partial → error (production) / manifest-flagged (dev). |
| 10 | **Portability checks** | Keyword/regex patterns conform to the portable regex subset (the ADR-001 §2.1 dialect problem, solved by construction); identifiers conform to the stable-id grammar; string encodings normalized (NFC). |
| 11 | **Optimization** | Table layout for load speed (sorted keys, interned strings), model quantization variant selection per declared platform targets, asset size-cap enforcement. Never semantic changes. |
| 12 | **Side-output codegen** | *Alongside* the bundle (not inside it): Swift/Kotlin action-key constants + typed param structs, contract-test skeletons, handler-completeness manifests (ADR-002 A5), golden fixtures regenerated. |
| 13 | **Manifest + lineage generation** | `bundle.json`, `meta/lineage.json` (dataset hashes, env lock, git SHAs), `meta/report_card.json` (ingested from the evaluation gate — a bundle cannot be built from an evaluation that didn't run). |
| 14 | **Deterministic packaging** | Normalized timestamps, fixed ordering, canonical JSON serialization → byte-identical rebuilds from identical inputs (verifiable supply chain). |
| 15 | **Signing** | `manifest.sha256` over all entries → Ed25519 signature (CI/KMS key, Part 11). Local dev builds sign with a dev key the production runtime refuses (`channel` + key id). |

Compiler property demanded by ADR-001.1 §9.6 and enforced here: **one validator.** The Studio, CI, and the compiler share stages 1–10 as a library — there is no second, slightly different validation path anywhere in the platform.

# Part 6 — Validation Semantics

Every reviewer-listed failure, mapped to its stage and blast radius had it escaped:

| Failure | Caught at | Runtime failure prevented |
|---|---|---|
| Missing intent (referenced, undefined) | 3 | Classifier predicts a label the orchestrator can't act on → silent dead command |
| Duplicate entity / conflicting synonym→different canonicals | 3/4 | Nondeterministic extraction; "restaurant" resolving differently per load order |
| Broken workflow (slot→missing entity, prompt→missing key) | 3 | Mid-conversation crash or blank prompt on a user's device |
| Invalid policy (loosening, incomplete matrix) | 5 | Un-confirmed high-cost action — a safety incident, not a bug |
| Unknown capability / cross-capability reference | 2/4 | Dispatch to nowhere; hidden coupling between teams |
| Missing localization | 9 | English prompt in a Danish session — trust damage |
| Invalid references (superseded_by, referent tables) | 3 | Broken deprecation routing; corrections resolving to nothing |
| Version mismatch (head vs embedder, labels vs model) | 8 | The silent-wrong-space bug class (embeddings in mismatched vector spaces) — the worst diagnosable-only-by-accuracy-drift failure the platform knows |

The consequence of the pipeline, stated as the runtime's entitlement: **a runtime holding a signature-valid production bundle may assume every guarantee above and must not re-validate content semantics.** Runtime checks are exactly three — signature, hashes, compatibility — plus defensive assertion of tensor contracts at model load. Everything else re-checked on device would be dead weight duplicating a proof already done.

# Part 7 — Runtime Loading

Consolidates ADR-001.1 §7.5/§9.2 and ADR-002; normative sequence:

```
STARTUP            locate slots {active, candidate, baked} → select active (fall back
                   baked if none) → verify signature → verify hashes → compat gate
                   (format major ≤ supported; engine_compat; required_runtime_features ⊆
                   runtime's flags; channel check) → parse runtime tables → register
                   models via Inference Interface (host loads; tensor-contract assert)
                   → warm-up inference per stage → mark healthy → serve
OTA UPDATE         BundleManager downloads to candidate slot (TLS; resumable; wifi/
                   charging policy native) → full verify+gate on candidate → warm in
                   background (tables + models) → smoke suite (fixed utterance set,
                   compiled into the runtime, must classify identically to fixtures
                   shipped as codegen side-outputs) → atomic activate (pointer swap;
                   in-flight conversations finish on old bundle) → old active → retired
                   (kept as last-known-good)
ROLLBACK           remote-config repoint or local health trigger → activate last-known-
                   good (already verified; instant) → report bundle_id change
PARTIAL FAILURE    download corrupt → discard candidate, active untouched, retry later;
                   verify fail → discard + telemetry (signature fail additionally raises
                   a security event); warm/smoke fail → discard candidate + telemetry
                   (the smoke suite is what catches "valid but broken on this device") —
                   in every case the user-visible assistant never changes until a fully
                   healthy candidate exists
VERSION MISMATCH   format major too new → refuse + report (registry sees fleet readiness
                   before promoting); required feature missing → refuse + report;
                   content_version ≤ active → refuse unless the instruction is an
                   authenticated rollback directive (downgrade protection, Part 11)
STARTUP DISASTER   active fails verify (storage corruption) → last-known-good → baked
                   bundle (the floor that always exists, shipped in the app binary) —
                   the assistant can degrade to launch-day intelligence but never to none
```

# Part 8 — Versioning Strategy

Independent axes, each with its own evolution law:

| Axis | Lives in | Changes when | Consumers gate on it? |
|---|---|---|---|
| `format_version` | bundle.json | The *shape* of the bundle changes. Minor = additive (old runtimes ignore unknowns); major = breaking (coordinated, Part 10) | **Yes — the only structural gate** |
| `content_version` | bundle.json | Every release. Monotonic | Ordering/downgrade only |
| `compiler_version` | bundle.json + lineage | Every compiler build | **Never** (a runtime gating on compiler version couples independent release trains — forbidden) |
| Capability versions | capability.json | Action/intent contract breaking change | SDK contract gate (ADR-002 A7) |
| Workflow vocabulary | implied by format_version | New behavior kinds (runtime release) | Via format + required_features |
| `policy_schema` / `policy_content` | policies.json | Vocabulary (runtime release) / values (any content release) | Schema: yes; content: telemetry only |
| Localization completeness | manifest languages table | Per language per release | Production channel requires `full` |
| Model/training versions | models section + lineage | Retraining | Never gated; audit + attribution |
| `telemetry_schema_version` | telemetry/schema.json | Enum/event changes (additive-preferred) | Analytics pipeline |

The discipline that makes independence real: **runtimes gate on exactly three things** — format version, engine compat/features, signature. Every other version is attribution metadata. The moment a runtime branches on any other axis, the axes recouple and independent evolution dies; this is a contract-test-enforced prohibition.

# Part 9 — OTA Lifecycle

```
REGISTRY SIDE                                DEVICE SIDE
built ──eval gates──► validated              discovered (remote config: cohort→bundle_url,
   │ CI sign                                   version, hash)
   ▼                                             │ policy-timed download (wifi/charging)
signed ──publish──► candidate                    ▼
   │ internal dogfood cohort                 downloaded ──verify sig+hash+gate──► verified
   ▼                                             │ background warm + smoke
staged (5% → 50%) ── tripwires watch             ▼
   │ promote            │ halt              warmed ──atomic swap──► ACTIVE
   ▼                    ▼                        │                     │ health trigger /
active (100%)      rolled-back ◄─────────────────┼─── remote directive │ config repoint
   │ superseded by next release                  ▼                     ▼
retired (registry keeps: artifact, report    retired (last-known-good slot; one kept)
   card, lineage — forever; audit trail)
```

Transition rules: publish requires green evaluation gates *and* a recorded human approval (ADR-001 §16); promotion requires tripwire health (fallback rate, wrong-action proxies, latency p95, crash-free — per `bundle_id`); halt at any stage repoints config, and devices that already activated roll back on next config fetch; recovery is always to a *previously fully-verified* artifact (last-known-good or baked) — devices never "repair" a bundle in place.

# Part 10 — Compatibility (the concrete scenario)

*Given: Python Compiler v5, iOS Runtime v3, Android Runtime v4.* These numbers are **deliberately not comparable** — the mistake to avoid is inventing a rule relating them. Each declares a relationship to the *format*:

- Compiler v5 emits format `3.2` (and can emit `2.x` on request for transition windows).
- iOS Runtime v3 supports formats `2.0–3.x`; Android Runtime v4 supports `3.0–3.x`.
- Fleet rule: the registry may only promote a bundle whose format is within the intersection of the *supported ranges of the runtime versions actually in the fleet* (the registry knows fleet composition from telemetry). Compiler and runtimes never negotiate directly; the registry is the meeting point.

**Forward compatibility (old runtime, newer bundle):** minor bumps are safe by the additive rule — unknown optional sections/fields are ignored (a `3.0` runtime uses a `3.2` bundle, minus features it doesn't know exist, which the `required_runtime_features` list makes explicit: if the new content *requires* the new feature, the bundle declares it and the old runtime refuses cleanly instead of degrading silently). Major bumps: refuse + report.
**Backward compatibility (new runtime, older bundle):** runtimes support ≥ current and previous major format (N and N−1) — the guarantee that lets a user restore a last-known-good bundle after an app update.
**Breaking changes:** new *required* section, changed semantics of an existing table, vocabulary removal → major bump → coordinated: runtime release supporting {N−1, N} ships first, fleet saturates, then the compiler flips default emission to N, then N−1 emission support is dropped after a deprecation window (≥ two release cycles).
**Migration rules:** content migrates via the compiler (recompile old source to new format — source YAML is the durable form); *bundles are never migrated in place*, on device or in registry.
**Deprecation:** vocabularies deprecate before removal (compiler warns one window, errors the next); intents deprecate via `superseded_by` (ADR-002 A7) independent of format versions.
**Runtime capability checks:** exclusively declarative — `required_runtime_features` vs. the runtime's compiled-in flag list at the compat gate. No probing, no reflection, no try-and-catch feature detection.

# Part 11 — Bundle Security & Trust Model

**Trust anchor:** Ed25519 public keys **pinned in the app binary** (two active pins for rotation). Trust derives from the signature alone — TLS protects transport but confers zero trust (a CDN, a proxy, or a compromised config service must not be able to produce an installable bundle).

- **Signing:** CI-only, key in KMS/HSM-backed secret storage, signing step gated on evaluation gates + human approval; the signature covers `manifest.sha256 ⊕ bundle.json`, and the hash table covers every file — one signature transitively covers every byte.
- **Integrity/tamper detection:** full hash verification at download *and* at every activation/boot (ADR-001.1 §9.2 — a bundle tampered at rest fails the next boot check and the device falls back). Signature failure is a security telemetry event, distinct from corruption.
- **Trusted publisher:** exactly one publishing identity per channel; `channel` + signing-key id in the manifest lets production runtimes refuse dev-signed artifacts categorically.
- **Certificate/key rotation:** two pinned keys (A active, B standby) → sign with B for one release cycle while A remains pinned → app release drops A, pins C as standby. Compromise response: config directs fleet to last-known-good signed by the surviving key; emergency app release rotates pins. Rotation procedure is documented and *rehearsed* (an unrehearsed rotation plan is a plan to fail during an incident).
- **Downgrade protection:** devices refuse `content_version` ≤ active unless the instruction arrives as an authenticated rollback directive (signed remote-config payload naming the exact target `bundle_id`) — prevents an attacker who controls only the CDN from replaying an old, buggier-but-validly-signed bundle.
- **Secure OTA, summarized:** TLS for privacy of *what* was downloaded; signature for authenticity; hashes for integrity; channel+key for provenance; monotonicity+directive for freshness. Each property has exactly one mechanism.

# Part 12 — Bundle Evolution (five-year test)

The evolution rule that everything below reduces to: **new facts are new sections/tables (minor bump); new behaviors are new runtime vocabularies unlocked by `required_runtime_features` (minor bump + feature flag); only re-shaping existing tables is a major bump.**

| Future addition | Bundle impact |
|---|---|
| New capabilities | Pure content — new `capabilities/<id>/` subtree. Format untouched. (Part 4's order-independence is why.) |
| New languages | New `{lang}` entries across models/lexicons/grammars/responses + completeness row. Format untouched. Downloadable per-language bundles are a *packaging profile* of this same spec (a bundle with one language), not a new format. |
| Knowledge/RAG capability | A capability whose `assets/` carry a typed article-index blob + a `retrieval` workflow step kind (new runtime vocabulary → feature flag `retrieval_steps`). Minor bump. |
| Cloud routing policies | Rows in `runtime/routing.json` — ADR-004 already reserved the table. Content release only. |
| Tool calling | The capability action contracts *are* the tool manifest (ADR-003 §3); an export view, not a bundle change. |
| Personalization | **Not in the bundle** — device-side overlay referencing stable ids; the bundle contributes only the overlay *schema* (what may be overridden, within policy bounds). Keeps user data out of a signed, distributed artifact by construction. |
| Memory | Same pattern: device-side, bundle ships vocabulary/limits only. |
| LLM integration | New entry in `models/` (typed model sections are open by design) + planner/interpreter config tables + feature flags (`llm_planner`). The propose-verify contract (ADR-003 §3) means even this lands as: new model asset + new tables + new flag — minor bump. |

The five-year claim, falsifiably stated: **no addition on this list requires a format major bump**, because the bundle's extension points (typed model sections, typed assets, feature flags, additive tables, closed-vocabulary versioning) were chosen against exactly this list.

# Part 13 — Repository Organization

Refines the platform-review §13 / ADR-003 Part 13 layout with the spec-and-compiler additions; the load-bearing separation is **platform code vs. conversational content vs. datasets vs. specification**:

```
intent-platform/
├── spec/                            # ← NEW: the normative contracts (this ADR's home)
│   ├── bundle/                      # JSON Schemas per format version: manifest, policies,
│   │   ├── 3.0/…  3.1/…  3.2/…     #   workflows, entities, lexicons, telemetry, capability
│   │   └── portable-regex.md        # the regex subset definition + conformance corpus
│   ├── vocabularies/                # closed vocabularies, versioned: behaviors, predicates,
│   │                                #   conditions, acts, outcomes, routing reasons
│   └── examples/                    # golden bundles (tiny, valid, per format version) —
│                                    #   runtime teams' conformance fixtures
├── content/                         # conversational content (capability-owned, CODEOWNERS)
│   ├── capabilities/<id>/…          # per ADR-002 A4
│   ├── entities/shared/  lexicons/  policies.yaml  routing.yaml
├── datasets/                        # DVC-tracked (train/holdout/oos/conversations/augment)
├── packages/
│   ├── buildtime/ nlu-compiler/     # stages 1–15; exposes stages 1–10 as the shared
│   │                                #   validator library (Studio + CI import it)
│   ├── buildtime/ nlu-training/     #   trainers, calibration, evaluate
│   ├── buildtime/ nlu-export/       #   model export, packaging, signing client
│   ├── runtime/ …                   # per ADR-003 Part 13 (bundle-runtime reads spec/ schemas)
│   └── sdk-ios/ sdk-android/        # per ADR-003 Part 13
├── bundles/                         # local build output (gitignored)
├── tests/                           # incl. spec-conformance: every example bundle must load
│                                    #   in every supported runtime (the cross-team contract test)
└── docs/                            # ADRs (this file), runbooks (key rotation, rollback)
```

Rules: `spec/` changes follow Part 8's format-versioning law and require platform-team review plus a changelog entry; `content/` never imports from `packages/` and vice versa (the compiler is the only bridge); `spec/examples/` golden bundles are the *inter-team integration test* — iOS, Android, and Python runtimes all load the same examples in CI, which is the mechanism that makes "independent teams implement without further clarification" true rather than hopeful.

# Part 14 — Developer Workflow

The reviewer's pipeline, made concrete (CLI names indicative; each step names its gate):

```
nlu new-capability weather            scaffold: capability.yaml, intent/entity/response
      │                               templates, dataset stubs, CODEOWNERS entry
add training data (Studio or CSV+DVC) ── leakage guard runs on PR
define intents / entities / workflows ── shared validator (compiler stages 1–4) in editor+CI
define policies (tightenings only)    ── algebra check (stage 5) in CI
      │
nlu train --lang all                  trainers → models + calibration + eval report
      │                               ── evaluation gates (holdout, OOS, wrong-action, ECE)
nlu build-bundle --channel dev        compiler stages 1–14 → bundle.nlu + codegen side-outputs
      │                               ── dev-signed; loadable by dev runtimes + Studio console
nlu validate-bundle                   independent re-verification + golden-bundle conformance
      │
[PR merge]                            CI reruns everything reproducibly (byte-identical check)
      │
nlu publish --channel beta            CI signs (production key), uploads, registry entry
      │                               ── requires green gates + recorded human approval
[registry] stage 5% → 50% → 100%      tripwires per bundle_id; halt → auto-rollback
      │
devices: discover → download → verify → warm → smoke → swap        (Part 7/9)
```

Runtime-change requirement for this entire flow: **zero.** A new capability with new intents, entities, workflows, prompts, and policy tightenings reaches production without any runtime, SDK, or app code change beyond registering the capability's native handlers (ADR-002 A5) — and the codegen side-outputs plus the CI completeness gate make even that step compiler-assisted and compile-checked. That property — content velocity fully decoupled from code velocity, behind a signed, versioned, atomically-swapped artifact — is the entire point of the Bundle.

---

## Risks & Trade-offs

**Trade-offs accepted:** a compile step between editing and running (mitigated: the Studio embeds the validator and a dev-channel fast path); one artifact means one download even for a one-string fix (accepted: bundles are <30MB; delta encoding is a future packaging profile, not a format change); strict validation will occasionally block a release on a technically-harmless inconsistency (accepted deliberately — the alternative is re-learning each check's necessity via production incidents, which is how the current repo's assertions were earned).

**Risks:** R1 — *spec drift between this document and the schemas* (mitigation: `spec/` JSON Schemas are the normative machine truth; this ADR defers to them once they exist; CI proves examples ↔ schemas ↔ runtimes agree). R2 — *the compiler becomes a bottleneck team* (mitigation: stages are library-modular; content diagnostics name owning teams; the Studio surfaces validation locally so the compiler team isn't the help desk). R3 — *format-version discipline erodes under deadline pressure* ("just gate on compiler version this once") — the recoupling failure; mitigation: the Part 8 prohibition is a contract test, and this ADR is the citation. R4 — *signing-key operational failure* (mitigation: rehearsed rotation runbook, two-pin scheme, break-glass procedure documented before GA).

## Action Items

1. [ ] Ratify format `3.0` as the initial versioned format (the current ad-hoc artifacts are retroactively "format 2").
2. [ ] Author `spec/bundle/3.0/` JSON Schemas + the portable-regex subset definition + two golden example bundles (one minimal, one full-featured).
3. [ ] Implement compiler stages 1–10 as the shared validator library first (Studio and CI need it before packaging exists).
4. [ ] Migrate current artifacts (`models/*`, `nlu_schema.json`, `nlu_entities.json`, `config/calibration.json`, localization overlays) into the source layout of Part 13 — the compiler's first real input.
5. [ ] Stand up the signing pipeline (KMS key, CI integration, dev/production channels) and write + rehearse the key-rotation runbook.
6. [ ] Add the spec-conformance CI job: every golden bundle loads in the Python runtime today; iOS/Android join as their bundle loaders land.
