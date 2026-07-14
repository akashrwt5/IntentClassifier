# Memory: Roadmap

> Single responsibility: what is done, in progress, and planned. Living doc —
> update as milestones move. Detailed status: `STATUS.md`.

## Done / stable

- English TF-IDF+LogReg classifier -> ONNX (~16KB), offline on iOS/Android.
- Full NLU engine (confirmation / slot-filling / classify, interruption
  detection, entity + datetime extraction, session context).
- Multilingual en/fr/de/da with per-language temperature-scaling calibration.
- Semantic rescue (MiniLM ONNX) for low-confidence turns, incl. multilingual.
- CoreML FP16 export with in-metadata temperature; Tier-A (Linux) + Tier-B
  (Apple-Silicon CI) parity: acc Δ ≈ 0, 0/30 gate disagreements.
- iOS switched to `softmax(logits/T)`; golden fixtures + parity XCTest in the
  STT repo (`akashrwt5/STT`).
- Dependency reproducibility: bounded direct deps + resolved `requirements.lock`
  (`make lock`); CI installs from the lock.
- French decimal-hour idiom bug fixed (parity suite 25/25) — see `known-issues.md`.

## Review-F5 production execution (charter-driven)

Phased plan in `docs/Review-F5/production-architecture-review-and-roadmap.md`;
live status + Needs-decision queue in `docs/Review-F5/EXECUTION_STATUS.md`.

- **Phase 0 (stop the bleeding):** deps lock [x] · pyproject [x] · PR CI [x] ·
  French datetime fix [x] · auto_label.py quarantined [x] · root README
  rewritten [x] · CODEOWNERS added [x] · GenAI placeholder-URL startup guard
  [x] · unknown_data privacy stance documented [x] · junk deletion (Engage.zip,
  checkpoints/) [x — already absent] · iOS parity CI secret [gated].
- **Phase 1 (platform foundations):** next up — spec/bundle/3.0 schemas,
  bundle compiler validator, capability repartition, nlu_training package,
  DVC/MLflow, real tests/ tree. packages/ restructure + label-space cleanup
  are approval-gated.
- **Phases 2–5:** ratification-gated (Rust core/Android) or blocked on Phase 1.

## In progress / near-term

- Repo productionization: tooling (Ruff/Black/MyPy/Pytest), pre-commit, CI,
  modular `.claude/` memory + agents.
- Danish quality lift (holdout F1 ≈ 0.74; lowest of the four languages).

## Planned (larger bets)

- On-device memory optimization & distillation — teacher selection (E5-small vs
  MiniLM-L6-v2), student architectures, distillation data/recipe, then reuse the
  existing export toolchain. Plan: `docs/on-device-memory-optimization-plan.md`.
- Static embeddings (Model2Vec) as a "dark horse" size/latency win.
- TFLite artifact path for Android parity with CoreML.
- Optional `src/` package reorg (make code importable) — pending approval.
- Tighten lint/type gates family-by-family; one-time Black format pass.

## Open decisions

Tracked in `decisions.md`. New architectural choices should be recorded as ADRs.

## Related memory

Decisions -> `decisions.md` · Known issues -> `known-issues.md` · Mobile plan ->
`mobile.md`.
