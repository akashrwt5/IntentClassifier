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

## Related memory

Training/calibration -> `training.md` · Mobile -> `mobile.md` · Roadmap ->
`roadmap.md`.
