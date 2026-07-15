# Memory: Architectural Decisions (ADR log)

> Single responsibility: durable "why" behind non-obvious choices. Append new
> decisions as short ADRs; use the `engineering:architecture` skill for full
> ADRs. Keep entries terse — link to the detailed doc.

## ADR-001 — TF-IDF + LogisticRegression -> ONNX for the core classifier
**Status:** Accepted. **Why:** ~16KB, offline, deterministic, tiny latency on
device; no neural runtime needed. Trade-off: less semantic generalization than
embeddings — mitigated by the semantic-rescue layer.

## ADR-002 — No torch/transformers on the inference path
**Status:** Accepted. **Why:** on-device size/latency. Embeddings run through
ONNX; `torch`/`coremltools` are export-only (macOS/CI).

## ADR-003 — Temperature scaling over isotonic calibration
**Status:** Accepted. **Why:** isotonic calibration was **not rank-preserving**,
causing server↔device argmax disagreements and worse ECE on holdout. Temperature
scaling is rank-preserving by design and empirically better-calibrated. Detail:
`multilingual/TEMPERATURE_SCALING_DECISION.md`,
`multilingual/MODEL_CALIBRATION_DECISION.md`.

## ADR-004 — Per-language single-model calibration (Option B)
**Status:** Accepted. **Why:** best server↔device parity and smallest size vs.
ensemble isotonic. Residual parity issues traced to tokenizer divergence and
argmax-before-vs-after-calibration. Detail: `MODEL_CALIBRATION_DECISION.md` §4-5.

## ADR-005 — Static ONNX shapes, batch size 1
**Status:** Accepted (hard constraint). **Why:** ANE eligibility + deterministic
CoreML conversion. Consequence: embed one sentence at a time everywhere.

## ADR-006 — CoreML FP16 mlprogram, temperature in metadata, logits output
**Status:** Accepted. **Why:** FP16 halves size with acc Δ ≈ 0; shipping logits +
temperature lets the device own `softmax(logits/T)` and keeps parity. Detail:
`multilingual/COREML_RESULTS.md`, `docs/coreml-conversion-guide.md`.

## ADR-007 — Semantic rescue as a separate low-confidence layer
**Status:** Accepted. **Why:** keeps the fast TF-IDF path primary; only pays
embedding cost when confidence/margin gates fail. Threshold via
`nlu_schema.json:semantic_threshold`.

## ADR-008 — Tooling lenient-by-design, tighten incrementally
**Status:** Accepted. **Why:** never-linted code must stay green from day one;
enforce a lean Ruff rule set now; formatting is **format-on-touch** (darker:
Black on changed lines only) so legacy history isn't reformatted (avoids conflicts
across the many in-flight branches); MyPy stays non-blocking with per-module
tightening. Detail: `pyproject.toml`, `CONTRIBUTING.md`, `.pre-commit-config.yaml`.

## ADR-009 — MCP transport: code graph stays local; web uses Context7 only
**Status:** Accepted. **Why:** the code-graph server (CodeGraphContext, `cgc`)
runs as a **local stdio** process and indexes this **proprietary** codebase.
claude.ai web sessions can only reach **remote HTTPS** MCP servers, so putting the
code graph on the web would mean exposing a private code index over the internet
(tunnel or hosted box) — an unacceptable default. Decision: use the code graph
**only on desktop** clients (Claude Code / Desktop in PyCharm); in web sessions
use **Context7** (hosted at `https://mcp.context7.com/mcp`) for library docs only.
Revisit only if a private, authenticated hosting path is explicitly approved.


## ADR-010 — Review-F5 platform ADRs 001–005 ratified
**Status:** Accepted (owner ratification, 2026-07-14). **What:** the five
Review-F5 platform ADRs move Proposed → Accepted as written:
ADR-001 shared-runtime strategy (Option B now; Rust core only at the
Android-multi-turn trigger), ADR-002 capability & action-execution SDK,
ADR-003 conversation orchestration (Orchestrator/Planner/Dialogue/Policy),
ADR-004 GenAI routing & cloud escalation (narration-only, utterance-only
egress), ADR-005 NLU Bundle spec & compiler. **Why:** unblocks gated Phase-1
work (bundle schemas/compiler, capability repartition, contract freezing).
Phase 2 (Rust core / Android) remains **trigger-gated** despite ratification;
signing keys, consent/legal, restructure, and label-space changes remain
separately approval-gated (EXECUTION_STATUS ND-2/3/8/9). Detail:
`docs/Review-F5/adr-00{1..5}-*.md`.


## ADR-011 — No Rust core; Python + native iOS/Android is the strategy
**Status:** Accepted (owner directive, 2026-07-14). **What:** the shared Rust
runtime sketched in ADR-001 Phase 2 is removed from the active roadmap. The
platform remains: Python for training, compiler, tooling, and the reference
NLU engine; native iOS and native Android implementations on device. **Why:**
owner decision — explore a shared core later only if required, on explicit
request; the ADR-001 trigger (Android multi-turn dialogue) is no longer
sufficient on its own. **Consequences:** cross-platform parity continues to
be guaranteed by the contract + fixture approach (runtime-contract-v1.md,
golden bundles, parity CSVs, conformance tests) rather than by a shared
binary. No Rust code exists in the repo; none is planned.


## Related memory

Training/calibration -> `training.md` · Mobile -> `mobile.md` · Roadmap ->
`roadmap.md`.
