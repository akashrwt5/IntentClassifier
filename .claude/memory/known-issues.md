# Memory: Known Issues

> Single responsibility: real, currently-open problems on **this branch**. Read
> before reporting a bug — don't re-flag what's already tracked here. Move an
> entry out when it's fixed. Re-derived from this branch; the
> `feature/production-work` issue tracker does not apply.

## Tracked test failures — 2 remaining (was 8)

Six were fixed (see below); their `--deselect` lines are gone and they gate
normally. **Only the two French clock idioms remain deselected** in
`.github/workflows/pr.yml`:

| Test | Why it still fails |
|---|---|
| `test_datetime_parity.py::…[fr-dix heures et demie-today-10:30]` | `et demie` (half past) → 10:00 |
| `test_datetime_parity.py::…[fr-huit heures moins le quart-today-07:45]` | `moins le quart` (quarter to) → 08:00 |

**These are not engine bugs.** English gets clock idioms from
`packs/en/datetime/grammar.json`; French has no pack, so the vocabulary was
never authored. They start passing the moment `packs/fr/` exists — no engine
change. Treat them as the concrete cost of the second-pack gap in `roadmap.md`,
not as a parser defect.

Rules:
- **Any new failure is a real regression** — the deselect list is exhaustive.
- **Never weaken an assertion to make one pass.** Fix the cause, then delete the
  `--deselect` line so the case starts gating again.

### What the six fixes were (2026-07-25)

Three real bugs and three stale expectations. Worth knowing because the stale
ones encoded behaviour that is now deliberately different:

- **Negation was a no-op for regex rules** (real, wrong-action). "i don't want
  to translate anything" fired `Cmd.TranslationStart`. The guard only covered
  `contains` rules, so it was silently lost when a rule migrated to `regex`.
  Now applied to regex hits too, with cues from the pack — ADR-014.
- **A help question abandoned an active slot flow** (real). "ask about the
  translate feature" scores ~0.99 on TF-IDF, so no threshold could fix it;
  actions listed in `policy.non_interrupting_actions` can no longer interrupt.
- **The holdout gate's "impossible" floor became reachable** (real, and it made
  the test vacuous). It used `MIN_HOLDOUT_TOTAL=99` while the corpus grew to 341
  cases; now 999999.
- **Keyword tier expectation** (stale): the translate rule is `regex_guarded`
  (0.90), not `contains` (0.85).
- **Reminder title-casing ×2** (stale, and re-adding it would be a regression):
  open free-text slots deliberately store what the user said, not a canonical
  English synonym — otherwise "prendre des médicaments" comes back as "Take
  Medication" in a French session. Casing for display is the app layer's job.

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

## The holdout score is non-deterministic when semantic is ON

Repeated identical runs of `scripts/test_holdout.py` return **320–323 / 341**
(93.8–94.7%). With `--no-semantic` it is rock stable at **280/341** across every
run. Wrong-action is **6 in both configurations, every time** — the variance is
only ever a case flipping between "correct" and "safe GenAI fallback", never
into a wrong action. So it is an accuracy-reporting problem, not a safety one.

Ruled out by direct measurement:
- **Not the model.** `SemanticFallback._embed()` on the same text 5× gives
  max pairwise difference **0.0**, and `classify()` returns an identical
  confidence to 16 digits.
- **Not ONNX threading.** Forcing `OMP_NUM_THREADS=1` still varies.
- **Not engine construction.** A single engine instance varies between two
  passes over the same corpus.
- **Not the preceding turn.** The utterance in isolation, and after four
  different predecessors, gives an identical result every time.

Still suspect: **wall-clock state in `SessionStore`**. Contexts expire at
`context_ttl_seconds` 90 and sessions at 600 (`context.py` uses `time.time()`),
while a full pass takes tens of seconds — so how far the clock has advanced can
change routing for turns that open a context. Consistent with the evidence:
giving each utterance its own session id instead of reusing `"holdout"` with a
reset reduced the variance (4 → 2 differing cases) but did **not** remove it.

Practical impact: the gate floor is `min_total` 258, and the worst observed run
is 320, so it will not flake — but do not report a holdout number to 0.1% or
treat a ±3 movement as a real change. Re-run before concluding anything.
A fix would be injecting a fixed clock into `SessionStore` for benchmark runs
(the constructor already takes one — `clock: Callable[[], float] = time.time`).

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
