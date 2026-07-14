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
