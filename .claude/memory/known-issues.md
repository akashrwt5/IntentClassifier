# Memory: Known Issues

> Single responsibility: active bugs, gaps, and gotchas. Update when an issue is
> fixed (move a note to `decisions.md`/`roadmap.md`) or discovered.

## Bugs

_None open. Recently fixed:_

- **FIXED — French decimal-hour idioms dropped their minutes.** `et demie` /
  `et quart` / `moins le quart` were lost when a "N heures" clock hour was
  present (e.g. `"huit heures moins le quart"` -> 08:00 instead of 07:45).
  Cause: in `entities.py` (then `scripts/nlu/`, now `packages/runtime/nlu_engine/`), the digit+clock-marker step (D1) ran
  before the decimal-hour idioms (D3) and consumed the hour. Fix: run D1 after
  D3. Full datetime parity suite green (25/25) across fr/de/da. Note: the shared
  fixtures also drive the iOS Swift parity test — keep both in sync.

## Privacy / data-collection

- **FIXED (ND-5, 2026-07-14):** `predict.py` unknown-data logging now goes
  through `apps/cli/unknown_log.py` — aggregate counters by default, raw text
  only behind `NLU_COLLECT_RAW_UNKNOWN` (see `docs/privacy-unknown-data.md`).
  Open sub-item: retention window for consented raw text (legal decision).

## Artifacts

- **Danish model artifacts are tracked in git as a deliberate exception**
  (`multilingual/models/da/`): Danish fails the trainer's 0.80 accuracy gate
  (0.760), so `train_multilingual.py --all` exports nothing for it — the
  tracked copies are the only ones. All other model dirs were untracked +
  gitignored after a live regen check (ND-10, 2026-07-14). Untrack `da/` once
  the native-data program lifts it past the gate.

## Quality gaps

- **RESOLVED (2026-07-14): de macro-F1 dip.** Root cause was the TF-IDF
  recipe, not the migration: min_df=2 discarded once-occurring German
  compounds, starving small help.* classes. min_df=1 lifts EVERY language
  (en .907/.896, fr .866/.853, de .845/.821, da .800/.787 — de fully
  recovered above pre-migration; Danish now a hair from its 0.80 floor).
  ONNX grows ~2x (en 2.3MB, de 3.4MB) — inside the on-device budget.
- **(historical) de macro-F1 dipped −0.030 in the ND-3 label migration** (0.821 → 0.791;
  accuracy held at −0.010). Deterministic effect of re-optimizing the
  57-class softmax after removing the two dialogue-act classes; weakest
  classes are small-support `help.*` topics (worst: help.battery.show 0.25).
  Follow-up: per-class C/class-weight re-tune for de, or targeted data.
- **Semantic-rescue artifacts must be regenerated locally** after ND-3
  (heads/indexes are gitignored and still carry old labels wherever they
  exist on a dev machine): run the semantic training targets before using
  the engine's semantic stage. Same for the CoreML fixtures (macOS export).
- **iOS coordination pending:** ship datasets/label_migration_map.json to
  the STT repo and regenerate golden fixtures in the same release window.

- **Wrong-action budget still not met — improved 99 → 73 (ND-11 a+b, 2026-07-14).**
  Polarity guards + uncertainty-confirmation gate (<0.80 on device.volume/
  streaming) cut shipped-lang wrong actions 26% (en 39→25, fr 32→26,
  de 28→22) and intercepted 30 wrong guesses behind CONFIRM turns (friction:
  2–5% of turns ask first). Residual 73 is dominated by NON-gated domains
  (activity queries, translation/transcription session starts) and
  high-confidence (≥0.80) confusions. Next levers (owner decision):
  extend the gate to activity/translation/transcription/find; raise the
  confirm bar; re-measure with regenerated semantic rescue.
- **(original finding, superseded)** **Wrong-action budget NOT met at holdout scale (system level).** The
  engine-in-the-loop harness (wrong_action_harness.py, semantic disabled)
  measures 99 wrong actions across shipped langs (en 39 / fr 32 / de 28;
  da 33 waived) ≈ 2.0–2.7% of turns vs the ≤5 budget. Findings: (1) polarity
  confusions inside device.volume ("turn mute on"→unmute, "more quiet"→
  increase) dominate; (2) help-questions misfiring as device commands;
  (3) the CONFIRM defense layer never fires on the replay — no intents are
  confirmation-required in the current schema, so that whole gate is
  unused. Candidate mitigations (POLICY changes → owner approval, §7):
  confirmation-require volume/streaming actions at low margin; polarity
  keyword guards; higher actionable-intent threshold; semantic rescue after
  artifact regen. Baseline artifact:
  tests/parity/oracle_post_migration/wrong_action_system_report.json.
- **(superseded by the above) raw-classifier wrong-action upper bound.** The unified
  evaluate now reports the OFFICIAL definition (confident prediction of an
  actionable intent ≠ truth): 244 across 4 langs at the raw-classifier
  level (device 127). The ≤5 budget is a SYSTEM property — keyword tiers,
  thresholds, confirmation gates, and semantic agreement sit between the
  raw classifier and any action — so the budget gate requires replaying the
  holdout through NLUEngine.handle, not the bare pipeline. Build that
  harness (Phase 1 follow-up); until then wrong_action_count is a trend
  upper bound, not a CI gate.
- **Danish OOS recall is weak (0.51)** — fallback-class recall from the
  unified evaluate; compounds the known Danish macro-F1 gap.

- **Danish accuracy** is the weakest language (holdout macro-F1 ≈ 0.74 vs
  ~0.83-0.90 for en/de/fr). Consider more data / augmentation / threshold review.
- **Residual server↔device intent parity** on a few multilingual cases —
  tokenizer divergence + argmax-before-vs-after-calibration. Documented and
  bounded in `multilingual/MODEL_CALIBRATION_DECISION.md` §4.

## Tooling / CI

- Formatting is **format-on-touch** (darker = Black on changed lines only), so
  the ~58 unformatted legacy files are left alone; CI enforces formatting only on
  changed lines vs the base ref. MyPy stays non-blocking (gradual typing, ADR-008).
  `make format-all` is the deliberate one-time full-repo Black pass if ever wanted.
- Tier-B CoreML runtime + ANE checks are **macOS-only**; the Linux CI auto-skips
  them. iOS XCTest parity needs an `INTENTCLASSIFIER_PAT` secret in the STT repo.

## Gotchas

- ONNX string-normalizer needs a **UTF-8 locale** (`LC_ALL`/`LANG`); CI sets it.
- Trained artifacts (`*.onnx`, `*.pkl`, `*.mlpackage`) are **gitignored** —
  regenerate with `make train` / `make export-coreml`; do not commit them.
- The `nlu` package `__init__` pulls in numpy etc.; leaf modules like
  `entities.py` are intentionally importable standalone for light tests.

## Related memory

Datasets -> `datasets.md` · Mobile -> `mobile.md` · Roadmap -> `roadmap.md`.
