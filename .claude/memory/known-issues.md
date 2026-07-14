# Memory: Known Issues

> Single responsibility: active bugs, gaps, and gotchas. Update when an issue is
> fixed (move a note to `decisions.md`/`roadmap.md`) or discovered.

## Bugs

_None open. Recently fixed:_

- **FIXED — French decimal-hour idioms dropped their minutes.** `et demie` /
  `et quart` / `moins le quart` were lost when a "N heures" clock hour was
  present (e.g. `"huit heures moins le quart"` -> 08:00 instead of 07:45).
  Cause: in `scripts/nlu/entities.py`, the digit+clock-marker step (D1) ran
  before the decimal-hour idioms (D3) and consumed the hour. Fix: run D1 after
  D3. Full datetime parity suite green (25/25) across fr/de/da. Note: the shared
  fixtures also drive the iOS Swift parity test — keep both in sync.

## Privacy / data-collection

- **FIXED (ND-5, 2026-07-14):** `predict.py` unknown-data logging now goes
  through `scripts/unknown_log.py` — aggregate counters by default, raw text
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
