# Memory: Known Issues

> Single responsibility: real, currently-open problems on **this branch**. Read
> before reporting a bug — don't re-flag what's already tracked here. Move an
> entry out when it's fixed. Re-derived from this branch; the
> `feature/production-work` issue tracker does not apply.

## The 8 tracked pre-existing test failures

`pytest` fails on 8 cases that **predate** the Language Pack work (verified
failing on the pristine engine before any of it). CI runs the suite with these
`--deselect`ed — the authoritative list is in `.github/workflows/pr.yml`;
rationale in `docs/Review-F5/TO-REMOVE.md`.

| Test | Area |
|---|---|
| `test_sprint1_hardening.py::test_keyword_negation_suppresses_contains_hit` | keyword negation |
| `test_sprint1_hardening.py::test_keyword_positive_contains_still_fires` | keyword negation |
| `test_sprint3_hardening.py::test_bare_contains_does_not_interrupt_slot_flow` | slot-flow interruption |
| `test_sprint3_hardening.py::test_holdout_gate_fails_when_floor_impossible` | holdout gate |
| `test_nlu.py::test_reminder_step_by_step` | reminder slot flow |
| `test_nlu.py::test_reminder_one_shot` | reminder slot flow |
| `test_datetime_parity.py::…[fr-dix heures et demie-today-10:30]` | French relative time |
| `test_datetime_parity.py::…[fr-huit heures moins le quart-today-07:45]` | French relative time |

Rules:
- **Any new failure is a real regression** — the deselect list is exhaustive.
- **Never weaken an assertion to make one pass.** Fix the bug, then delete its
  `--deselect` line so the case starts gating again.
- Three underlying bugs: French relative-time parsing (`et demie`,
  `moins le quart`), keyword negation, and the holdout gate.

## Deferred deprecations — do NOT remove yet

Superseded by the pack, but still used by the backward-compatible **no-pack**
path. Remove only once every caller constructs the engine with a Language Pack:

- `scripts/nlu/engine.py`: class-level EN fallback constants `_CARRIER`,
  `_UNCERTAIN`, `_NO_IDIOMS`, `_LEADING_CONNECTOR`, and the schema-sourced
  `affirmative`/`negative` fallback. Superseded by `packs/en/lexicons.json`.
- `scripts/nlu/entities.py`: built-in English `_WEEKDAYS` / `_WORD_NUMS` defaults
  (now injectable from `packs/en/datetime/grammar.json`) and the consolidated
  `_DEFAULT_DT_GRAMMAR` fallback table.

Deleting these early breaks every pre-pack caller.

## Stale CoreML artifacts are a live trap

A checked-in `models/IntentClassifier.mlpackage` was once found to be
`neuralNetworkClassifier` (**non-ANE**) with **1340 features vs the production
5433** — a different, stale model that looked fine by filename. Before trusting
any local `.mlpackage`:

```bash
python scripts/ci/verify_coreml_parity.py --inspect   # works on Linux
```

Related, already fixed but worth knowing (ADR-011): the ONNX and the CoreML
weights used to be two independent fits, silently drifting iOS from Android.
Keep `export_weights.py` deriving from `models/intent_pipeline.pkl`.

## Danish is the weakest language

`da` macro-F1 **0.7448**, ECE **0.0352** (vs `en` 0.9018 / 0.0184) —
`config/calibration.json`. Below the 0.80 accuracy posture in
`config/gate_thresholds.json`. Treat Danish results with more suspicion than the
other languages; it is a data-quality problem, not a calibration one.

## Cannot be verified on Linux

These auto-skip rather than fail — a green Linux run does **not** mean they pass:

- Tier-B Core ML runtime parity (`verify_coreml_parity.py --runtime`).
- Live ANE compute-plan placement.
- The iOS XCTest parity suite (lives in `akashrwt5/STT`; its workflow still needs
  a one-time `INTENTCLASSIFIER_PAT` secret before it can run).
- The CoreML export step itself (needs coremltools + torch on macOS).

## Repo hygiene

`bash scripts/cleanup.sh --dry-run` previews; `--all` also removes `Engage.zip`,
`checkpoints/`, and the empty `packages/{runtime,buildtime}` scaffold dirs.
Note the sandbox this was authored in **could not delete files**, so some cruft
is still tracked. A dry-run retrain overwrites `models/` — restore it afterwards.

## Related memory

Pack rules + parity baselines -> `langpack.md` · Export/parity -> `mobile.md` ·
Gate thresholds -> `training.md` · What's next -> `roadmap.md`.
