# Memory: Architectural Decisions (ADR log)

> Single responsibility: the durable "why" behind non-obvious choices on **this
> branch**. Append new decisions as short ADRs; keep entries terse and link to
> the detailed doc. Numbering here is local to this branch and does **not**
> match the ADR numbers on `feature/production-work`.

## ADR-001 — TF-IDF + LogisticRegression -> ONNX for the core classifier
**Status:** Accepted. **Why:** ~16KB, offline, deterministic, tiny latency on
device; no neural runtime needed. Trade-off: less semantic generalization than
embeddings — mitigated by the semantic-rescue layer (ADR-007).

## ADR-002 — No torch/transformers on the inference path
**Status:** Accepted. **Why:** on-device size/latency. Embeddings run through
ONNX Runtime; `torch`/`coremltools` are **export-only** (macOS/CI).

## ADR-003 — Temperature scaling over isotonic calibration
**Status:** Accepted. **Why:** isotonic calibration was **not rank-preserving**,
causing server↔device argmax disagreements and worse ECE on holdout. Temperature
scaling is rank-preserving by design and empirically better-calibrated. Detail:
`multilingual/TEMPERATURE_SCALING_DECISION.md`,
`multilingual/MODEL_CALIBRATION_DECISION.md`.

## ADR-004 — Per-language single-model calibration
**Status:** Accepted. **Why:** best server↔device parity and smallest size vs.
an isotonic ensemble. Residual parity issues traced to tokenizer divergence and
argmax-before-vs-after-calibration. Values live in `config/calibration.json`.

## ADR-005 — Fixed `(1, V)` CoreML input shape
**Status:** Accepted (hard constraint). **Why:** ANE eligibility + deterministic
CoreML conversion. Note the scope: this binds the **CoreML** model. The
classifier ONNX is exported with a dynamic batch (`StringTensorType([None, 1])`);
the MiniLM embedder is called one sentence at a time.

## ADR-006 — CoreML FP16 `mlprogram`, temperature in metadata, logits output
**Status:** Accepted. **Why:** FP16 halves size with acc Δ ≈ 0; shipping logits +
temperature lets the device own `softmax(logits/T)` and keeps parity.
`ComputeUnit.ALL` + `mlprogram` is what makes it ANE-resident — a
`neuralNetworkClassifier` package is a **defect**, not a variant. Detail:
`multilingual/COREML_RESULTS.md`, `docs/coreml-conversion-guide.md`.

## ADR-007 — Semantic rescue is a separate layer, and it ships OFF
**Status:** Accepted. **Why:** keeps the fast TF-IDF path primary; only pays the
embedding cost when the confidence/margin gates fail. On this branch the stage is
**pack-declared and disabled by default** (`packs/en/config.json`
`semantic_enabled: false`) — an explicit opt-in, not an ambient behavior.
Enabling it on a pack that declares no semantic stage is a hard error via arg,
a warning via env/config (a broad switch must not crash packs lacking the stage).

## ADR-008 — Language Pack: the engine is language-neutral, data lives in packs
**Status:** Accepted — **the defining decision of this branch.** **What:** every
language-specific input (intents, model, lexicons, keyword rules, entities, the
full datetime grammar, policy thresholds) is evicted out of `scripts/nlu/` into
`packs/<lang>/`, behind the locked contract in `packages/nlu_langpack/`
(`RUNTIME_CONTRACT_VERSION = "1.0"`). **Why:** adding a language becomes
authoring data, not editing engine code — the previous model required touching
`engine.py` for every language. **Enforcement:** zero `if language` branches and
zero English words in regex literals, checked in CI by
`scripts/ci/check_language_neutral.py` plus a hostile `zz` pack that must run
end-to-end. Detail: `langpack.md`, `docs/Review-F5/IMPLEMENTATION-PLAN.md` §9.

## ADR-009 — Big-bang migration, flat `packages/<name>` layout
**Status:** Accepted. **Why:** when the engine moves it moves wholesale
(`scripts/nlu` → `packages/nlu_engine`), not as a dual-run — a compatibility
shim would double the surface that has to stay neutral. Layout is flat
(`packages/nlu_langpack`), per plan §7. **Consequence:** the move is a discrete,
approval-gated checkpoint rather than a gradual drift; it has **not happened
yet** (see `roadmap.md`).

## ADR-010 — No Rust core
**Status:** Accepted. **Why:** owner directive. The platform stays Python for
training/tooling/reference engine, with native iOS and Android on device. The
pack boundary is shaped so adopting a shared native core later is a
swap-behind-interfaces, not a rewrite. Cross-platform parity is guaranteed by
the contract + fixture approach, not by a shared binary. No Rust exists or is
planned.

## ADR-011 — One featurizer of record: CoreML weights derive from the ONNX pipeline
**Status:** Accepted (fixes a real shipped bug). **Why:** `scripts/train.py` and
`scripts/export_weights.py` were producing **two different fits** — different
data, `min_df`, even upper-cased labels — so iOS and Android silently disagreed.
`export_weights.py` now derives labels/vocab/idf/coef/ngram/sublinear from the
same `models/intent_pipeline.pkl` the ONNX is exported from, and
`scripts/ci/verify_coreml_parity.py` + `scripts/test_ios_conformance.py` gate the
three featurizers (sklearn / ONNX / Swift) against each other. Detail:
`mobile.md`.

## ADR-012 — Ship only what clears a gate, as a versioned `.nlu` bundle
**Status:** Accepted. **Why:** a model that trains is not a model that ships.
`scripts/ci/evaluate_gate.py` + `config/gate_thresholds.json` block a release
below `min_accuracy` 0.80 / `min_macro_f1` 0.60 / `max_wrong_action` 15;
`scripts/ci/assemble_pack.py` emits a deterministic `pack-<lang>-v<ver>.nlu` with
a SHA-256 manifest, git-SHA lineage, and the embedded report card, carrying both
the ONNX (Android) and the ANE `.mlpackage` (iOS). Tighten the thresholds over
time; **never loosen them silently.**

## ADR-014 — Negation is pack data and guards regex rules too
**Status:** Accepted (2026-07-25, fixes a shipped wrong-action bug). **What:**
keyword negation cues move from a hardcoded `_NEGATIONS` tuple in
`classifier.py` into `packs/<lang>/lexicons.json:negations`, and the guard now
suppresses **`regex`** hits as well as `contains` hits. **Why:** two failures in
one. Regex rules were expected to spell negation into each rule's `not_regex`,
so the guard was silently lost when the translate rule migrated `contains` →
`regex` — "i don't want to translate anything" started translation on a
medical-context device. And because the cue list was English and lived in the
engine, negation was a **no-op for every non-English pack**, which breaks the
rule that adding a language touches only the pack. `exact` rules are unaffected
(a full-string match cannot have a cue before the term). Measured: no accuracy
change on the holdout (322/341, 6 wrong-action, unchanged).

## ADR-015 — Informational actions may never interrupt a slot flow
**Status:** Accepted (2026-07-25). **What:** `config.json`
`policy.non_interrupting_actions` lists action-ID prefixes (`["help."]` for
English) that cannot abandon an in-progress slot-filling flow. **Why:** "ask
about the translate feature" mid-reminder classifies as `Help_Translate` at
**0.991**, so no interrupt threshold can distinguish it from a real topic
switch — losing a half-built reminder to a question is a user-facing failure.
Matching on the action ID rather than the intent name or display text keeps it
language-neutral, and an empty list restores the old behaviour. Trade-off: the
help turn is consumed by the flow rather than answered; answering it while
holding the flow is a larger dialogue change, deliberately not attempted here.

## ADR-016 — Open free-text slots store the user's words, never a canonical match
**Status:** Accepted (pre-existing behaviour, ratified here). **What:** an open
entity slot (e.g. `@remind`) stores what the user actually said; the synonym
table is used for **recognition only, not storage**. **Why:** canonicalising
turns "prendre des médicaments" into "Take Medication" in a French session. The
engine returns content, not presentation — title-casing or truncating a reminder
title is the app layer's job. **Consequence:** two long-standing tests asserting
`"Take Medication"` were wrong and were corrected, not the code.

## ADR-013 — MCP transport: the code graph stays local; web uses Context7 only
**Status:** Accepted. **Why:** the code-graph server (CodeGraphContext, `cgc`)
runs as a **local stdio** process and indexes this **proprietary** codebase.
claude.ai web sessions can only reach **remote HTTPS** MCP servers, so putting
the code graph on the web would mean exposing a private code index over the
internet — an unacceptable default. Use the code graph **only on desktop**
clients; in web sessions use **Context7** (`https://mcp.context7.com/mcp`) for
library docs only. Revisit only if a private, authenticated hosting path is
explicitly approved.

## Related memory

Language Pack -> `langpack.md` · Training/calibration -> `training.md` ·
Mobile -> `mobile.md` · Roadmap -> `roadmap.md`.
