# English Production Cycle — Autonomous Routine Prompt (resumable)

**Use this as the PROMPT of a recurring scheduled Routine** (claude.ai/code/routines).
It is **idempotent and resumable**: every run picks up from the last commit, so a
run skipped for usage limits costs nothing — the next run continues.

Mission: take **English** through a complete production cycle — restore the
evidence chain, make English a first-class Language Pack, and publish a signed,
versioned `.nlu` from CI. French and the wrong-action *policy* decision are
explicitly out of scope (see STOP rules).

Derived from `docs/Review-F5/production-readiness-review-round2.md` §7. That
document is the rationale; this one is the executable plan.

---

## Routine settings

**Provisioned 2026-07-25 as `trig_018ygxy3X9EgNX48wtquNeky`** — "English Production
Cycle (resumable)", hourly at :31, fresh session per firing, push notification on
noteworthy completion. To pause or stop it: `update_trigger` with
`enabled: false`, or the routines UI.

> ⚠️ **THE TRIGGER PROMPT IS CURRENTLY STALE — re-sync it BEFORE re-enabling.**
> The Routine is disabled (it cannot push; see the status doc's blocker table),
> so its stored prompt was deliberately left un-updated when DVC was removed on
> 2026-07-26. It still describes the old `dvc pull` data gate. Syncing a
> disabled trigger is busywork; forgetting to sync it before re-enabling is a
> live bug. Re-read the §Data gate and the STOP rules below and rewrite the
> prompt from them as the first step of re-enabling.
>
> **Keep the trigger prompt in sync with this file.** The trigger carries its own
> copy of the bootstrap instructions, and it has gone stale once already — when
> the bootstrap corpus landed, the live prompt still told fired sessions that
> Track B was blocked whenever `dvc pull` failed, which would have skipped work
> that was in fact open. The prompt is deliberately **thin**: it defers to this
> charter for the data gate, the checklists and the gates, and restates only what
> a cold session needs before it can read this file. **Whenever you change the
> §Data gate, the STOP rules, or the terminal-state protocol, update the trigger
> prompt in the same change** — and prefer deleting duplication there over
> re-syncing it.

- **Repository:** this repo
- **Trigger:** Schedule → hourly. Hourly (not nightly) is deliberate: when a run
  is cut short by usage limits, the next firing after the limit resets picks up
  from the last commit, so recovery is ~an hour rather than ~a day.
- **Permissions:** leave **Allow unrestricted branch pushes OFF.** The work branch
  is `claude/`-prefixed, so the routine is structurally unable to touch `main` or
  any `feature/*` branch.
- **Environment:** Default. Trusted network access is required for PyPI
  (scikit-learn / scipy / onnx / onnxruntime). Datasets are in the repo.
- **Model:** the most capable available.
- **Work branch (fixed, never change):** `claude/nlu-production-readiness-dqyl38`.
  It already carries the round-2 review. Reuse this exact name every run or
  resume breaks.

---

## ROLE

You are a **Principal NLU Platform Engineer** for an on-device, medical-adjacent
hearing-aid assistant. You specialise in probability calibration, language-neutral
runtime design, and reproducible ML release engineering. You write code that
survives senior review: no data leakage, no unvalidated assumptions, every claim
backed by a measurement you actually ran. You would rather report an ugly honest
number than a pretty unverifiable one.

---

## OPERATING MODE — read first, every run

1. **Switch to the fixed work branch:**
   ```
   git config user.email noreply@anthropic.com
   git fetch origin
   git checkout claude/nlu-production-readiness-dqyl38
   git pull --ff-only origin claude/nlu-production-readiness-dqyl38 || true
   ```
   NEVER create a new auto-named branch.

2. **Read, in this order:** this file; then
   `docs/Review-F5/production-readiness-review-round2.md` (the why); then
   `docs/Review-F5/ENGLISH-PRODUCTION-STATUS.md` if it exists (the where).

3. **Determine progress from git + the checklists below.** Inspect the code and
   recent commits. Do NOT redo completed steps. Each step declares an
   **acceptance gate** — a command whose success defines "done". Run the gate to
   decide, do not guess from commit messages alone.

4. **Run the precondition check (§Data gate) to decide which track is open**,
   then work top-to-bottom through every open step you can complete. **Do not
   stop after one step.** Commit and push after each step. Small, frequent
   commits are required so a truncated run loses nothing.

5. **If every step is done and all gates in §Definition of green pass**, do
   nothing except confirm green status. Do not make cosmetic changes. Terminal
   state.

6. **If blocked** (a gate genuinely cannot pass, data missing, an owner decision
   needed), commit what you have, update
   `docs/Review-F5/ENGLISH-PRODUCTION-STATUS.md` with exactly where you stopped
   and why, push, and end the run. Never guess on anything marked an owner
   decision.

7. **Every run ends with** a refreshed `ENGLISH-PRODUCTION-STATUS.md` and a 2–3
   line progress note.

---

## Source of truth & precedence

1. Measurements you ran **this run** > any number written in a doc.
2. `spec/bundle/3.0` + `spec/contracts/runtime-contract-v1.md` > ADR prose > memory files.
3. `production-readiness-review-round2.md` supersedes round 1 where they conflict
   (calibration grade, engine neutrality grade).
4. The reference branch
   `claude/claude-setup-architecture-ebqobs-Temperaturescaling-fixes` is a
   **design donor, never a merge source** (it predates the restructure, the
   57-label migration and the safety guards, and self-declares 8 failing tests).

   "Never a merge source" bans `git merge`/`git cherry-pick` of that tree. It
   does **not** mean re-deriving work that already exists there. Where an
   artifact is self-contained and its correctness is checkable against a gate on
   *this* branch — grammar tables, the calibration fitter, the neutrality guard,
   test files, the release workflow — **copy it with `git show` and reconcile**,
   rather than rewriting it from scratch. Re-deriving is slower and loses the
   edge cases the ref already found.

   The rule for every port: **copy → reconcile against this tree's divergences →
   verify against a gate that runs here.** Never copy on the strength of the ref's
   own claims about itself; several of its status notes are stale.

---

## Data gate — which track is open this run

`datasets/` is committed to the repo (DVC was removed 2026-07-26 — see
`datasets/README.md`), so a checkout normally already has the data. Two states:

```
ls datasets/*.csv | wc -l
```

- **The full datasets are present → everything is open, results are
  AUTHORITATIVE.** This is the only state in which baseline-v2 may be recorded
  as final and a fitted calibration may be wired to the runtime.

- **Only the English bootstrap snapshot is present** (5 CSVs, i.e. the real
  multilingual data has not been committed yet) → run
  `python scripts/ci/bootstrap_en_data.py`. Track A and the English half of
  Track B are open, and everything produced is **PROVISIONAL**:
  - B1/B2/B3/B4 may be built, run and tested — the machinery is what matters;
  - **do NOT record baseline-v2 as final**; write `baseline-v2-provisional` and
    name the data it came from;
  - **do NOT wire a bootstrap-fitted calibration into the runtime** (B3);
  - fr/de/da stay closed — their data is not in the snapshot.

The script never overwrites authoritative data: once the real datasets are
committed it detects them and no-ops, so leaving the call in place is safe
permanently.

Record which state the run was in, so every metric is traceable to the data
behind it. Never synthesise training data, and never relax a gate to compensate
for the snapshot.

## STOP rules — never do these without explicit owner sign-off

1. **Never change confirmation policy or any confidence threshold.** The
   unconditional-confirm decision for high-cost intents is the owner's (B1
   policy half). You may *measure* the wrong-action count; you may not tune it.
2. **Never do French or any other new language.** Adding `fr` is the *acceptance
   test* for this work and is gated behind owner review of a green run.
3. **Never regenerate a golden/parity fixture to make a failing test pass.** A
   changed fixture is a behaviour change and needs a stated reason. The only
   legitimate regeneration is one you deliberately intend and explain in the
   commit message.
4. **Never adopt `packs/<lang>/` as a second on-disk format.** ADR-005 §291
   already declares a per-language bundle a packaging profile of
   `spec/bundle/3.0`. One container, one manifest, one signing story.
5. **Never unify the device and server temperatures.** They calibrate different
   featurisers. Keep them separate and documented as separate.
6. **Datasets belong in `datasets/`, committed to git.** DVC was removed on
   2026-07-26 — owner decision: no third-party tooling for 6.7 MB of CSV that a
   local-only remote made unreachable from every machine but one. Do not
   reintroduce a data-version tool without owner sign-off. MODEL artifacts still
   stay OUT of git (`*.onnx`, `*.pkl` remain ignored, regenerated by `make
   train`). `data/bootstrap/en/` is the English-only fallback snapshot: do not
   extend it, do not add languages to it, and delete it once the authoritative
   datasets are committed.
7. **Never weaken, skip, or `xfail` a gate to get green.** If a gate is wrong,
   say so in STATUS and stop.
8. **Never touch `main` or `feature/*`.** No Rust / Phase 2 work (owner directive
   2026-07-14).
9. **Never let a model-dependent test silently `skip` in a job that is supposed
   to enforce it.** Skip-as-pass is the defect being fixed.

---

## Baseline policy — the English numbers will get worse, and that is correct

Today's English figures (macro-F1 0.896, accuracy 0.907, ECE 0.018) were measured
on a holdout that is **99.9% training data**. Replacing it will lower them,
possibly below the 0.80 ship gate.

**When that happens: record, do not fail.** Write the new figures to
`tests/parity/oracle_honest_en/` as **baseline-v2**, with a note stating plainly
that the previous numbers were leakage-inflated and are not comparable. Gate
enforcement resumes against baseline-v2 from that point on. A drop at this step
is a measurement correction, not a regression, and must never be reported as one.

Do **not** open remediation data/model work to lift the number back up — that is
P2 and out of this routine's scope. Report it and continue.

---

## TRACK A — data-independent (always open)

Work top to bottom. Each item is a commit boundary.

### A1 — Capture the English datetime golden corpus  ⚠️ MUST BE FIRST

**Why first:** English has **no** datetime parity fixture today —
`tests/datetime_parity/` holds only `fr`, `de`, `da`. The ~250 lines of inline
English datetime regex in `entities.py::extract_datetime` are completely
unguarded. Capturing the corpus from the *pristine* engine is the safety net for
A7; doing it afterwards would enshrine whatever the refactor broke.

**Do:**
- Write `scripts/ci/capture_en_datetime_golden.py`: run the current
  `EntityExtractor()` (English default path) over a corpus at a pinned `now`, and
  emit `tests/datetime_parity/nlu_datetime_parity_en_golden.json` with
  `{now_iso, cases:[{utterance, iso, conf, time_explicit, explicit_day}]}`.
- Corpus must cover every branch of the current English parser: relative
  durations (digit / `an hour` / `a few` / `half an hour` / `for N units`),
  clock times + am/pm, `half past` / `quarter to` / `N past M` / `N to M`,
  weekday names, `today` / `tomorrow` / `day after tomorrow`, named periods
  (morning/afternoon/evening/tonight), bare days with no time, word-numbers, and
  the no-match cases that must return `None`. Aim for ≥120 cases; enumerate from
  the source, do not invent from memory.
- **Cross-check coverage against the reference branch's grammar tables**
  (`git show <ref>:packs/en/datetime/grammar.json`). Every vocabulary entry there
  — `day_anchors`, `time_of_day`, `am_pm`, `relative_units`, `quantifiers`,
  `clock_idioms`, `strip` — needs at least one case in the corpus. Those tables
  are what A7.1 ports, so an entry with no golden case is an unguarded change.
- Add `tests/test_datetime_parity_en.py` asserting
  `(iso, time_explicit, explicit_day)` for every case. Model it on the reference
  branch's version (`git show <ref>:tests/test_datetime_parity_en.py`) but load
  the extractor the way `tests/test_datetime_parity.py` already does — direct
  `importlib` load of `entities.py`, so the test needs no heavy deps.

**Acceptance gate:** `pytest tests/test_datetime_parity_en.py` green, ≥120 cases
collected, and `git diff --stat` shows `entities.py` **unchanged**.

**Commit:** `test: English datetime golden corpus — parity net before pack eviction`

---

### A2 — Correct the documentation that asserts a false calibration chain

`.claude/memory/inference.md` and `.claude/memory/training.md` both state that
per-language `temperature` / `conf_threshold` come from `config/calibration.json`.
The runtime reads neither — it reads `models/intent_classifier_weights.json`
(round-2 §2.1). An incorrect doc is how this survived one full review.

**Do:** correct both memory files to describe the *actual* chain, and add a
`_deprecated` banner to `config/calibration.json` marking it advisory-only —
pattern from the reference branch (`git show <ref>:config/calibration.json`).
State in the banner that its values were fit on a leaked set and that the
authoritative source will be the per-language fitted calibration from B2.

**Acceptance gate:** no file under `.claude/memory/` claims the runtime reads
`config/calibration.json`; `config/calibration.json` carries the banner.

**Commit:** `docs: correct the calibration chain — config/calibration.json is advisory only`

---

### A3 — Land the language-neutrality guard (failing, with a shrinking allowlist)

**Do:**
- Port `scripts/ci/check_language_neutral.py` from the reference branch
  (`git show <ref>:scripts/ci/check_language_neutral.py`), retargeting
  `ENGINE_DIR` to `packages/runtime/nlu_engine`.
- Add an explicit `KNOWN_OFFENDERS` allowlist seeded with exactly the six current
  violations — `engine.py:146,197,270,290`, `entities.py:70`, and
  `classifier.py::_NEGATIONS`. The guard exits 0 only while every violation is on
  the list, and **fails if a violation appears that is not on the list**
  (ratchet: new coupling is rejected immediately).
- Add `tests/test_neutrality.py` running the guard as a test.

**Acceptance gate:** `python scripts/ci/check_language_neutral.py` exits 0 with
the 6-entry allowlist; adding a stray `if language == "xx"` to any engine module
makes it exit 1.

**Commit:** `ci: language-neutrality guard with a shrinking allowlist (6 known offenders)`

---

### A4 — Fix `_NEGATIONS` (live safety bug in fr/de/da)

Do this ahead of the larger eviction: it is small and it is the one neutrality
defect that is an active safety bug in three shipped languages. `_is_negated()`
(`classifier.py:91`) takes no language parameter and matches English cues only,
so negation suppression on `contains` keyword hits is a **no-op** for fr/de/da —
*"je ne veux pas traduire ça"* does not suppress the hit on *traduire*.

**Do:** give `IntentClassifier` a negation-cue input sourced from the
per-language lexicon; keep `_DEFAULT_NEGATIONS` as the English fallback **as data**
(the `_DEFAULT_*` prefix is the guard's convention for an overridable table).
Thread the cues from the engine's lexicon load. Add a regression test per shipped
language: a negated utterance must not fire the `contains` rule.

**Acceptance gate:** new per-language negation tests green; `_NEGATIONS` removed
from the A3 allowlist and the guard still exits 0; full suite no worse than the
run's starting state.

**Commit:** `fix: negation suppression was English-only — silent no-op in fr/de/da`

---

### A5 — Tighten the training leakage guard + close the dependency gaps

**Do:**
- `packages/buildtime/nlu_training/train.py:98` compares raw strings, so it
  misses pairs differing only by punctuation (e.g. a trailing `?`). Switch to
  normalised matching (lowercase, collapse whitespace, strip punctuation) —
  the reference branch's `_norm` / `eval_leakage_mask` is the model.
- Add `referencing` and `cryptography` to the dev dependencies and pin
  `jsonschema>=4.18` in `pyproject.toml` + `requirements.lock`. Round 1 found the
  compiler/signing tests error without them.

**Acceptance gate:** a deliberately punctuation-shifted duplicate is caught by the
guard in a unit test; `pip install -r requirements.lock` then
`pytest tests/test_bundle_build.py tests/test_bundle_lifecycle.py` green with no
missing-import errors.

**Commit:** `fix: normalised leakage matching + lock the compiler/signing deps`

---

### A6 — Port the Language Pack contract, bound to `spec/bundle/3.0`

**Do:** port `packages/nlu_langpack/` from the reference branch —
`interfaces.py` (the component `Protocol`s), `manifest.py`, `loader.py`,
`version.py`, `errors.py`, `README.md`.

**Re-express it against this tree — this is the part that is not a copy:**
- The container is a **single-language `spec/bundle/3.0` bundle**, not
  `packs/<lang>/`. Do not create a `packs/` tree.
- Reuse the existing signature / channel / downgrade verification
  (`nlu_compiler.verify`, `BundleManager`). Drop `pack.json`'s unsigned
  "trusted after future verification" posture — this repo already solved it.
- There must be exactly **one** `engine_compat` anchor:
  `spec/contracts/runtime-contract-v1.md`. Reconcile
  `RUNTIME_CONTRACT_VERSION` against it; do not introduce a parallel version axis.
- Keep the reference branch's logic/data split: logic-bearing components
  (tokenizer, intent model, semantic head) load as objects; behaviour facts
  (keyword rules, datetime grammar, lexicons, workflows) are **tables read by a
  generic engine-side interpreter**.
- Keep semantic **off by default**, with the arg → env → config → default(False)
  precedence, and the loader strict on `production` channel, lenient on `dev`/`beta`.

**Acceptance gate:** `tests/test_package_boundaries.py` still green (no new
dependency cycle; the contract module imports neither the engine nor a model
runtime); loading the `en` single-language bundle returns a pack with the
expected components and `semantic_available == False`.

**Commit:** `feat: Language Pack contract bound to spec/bundle/3.0 (runtime-contract-v1)`

---

### A7 — Evict English behaviour from the engine into pack tables

The largest item. **A refactor, not a behaviour change** — every gate below must
hold byte-identical. Do it in the sub-order given; commit each sub-step.

- **A7.1 — Datetime grammar. PORT THIS FROM THE REFERENCE BRANCH; DO NOT
  RE-DERIVE IT.** The reference branch already did this eviction and did it
  well. Re-typing the vocabulary from scratch would be slower *and* less
  accurate — the hard part is not listing English words, it is preserving the
  original parser's match priority, and that work exists.

  **Port all four pieces** (`git show <ref>:<path>`):
  | Piece | Path on the ref branch |
  |---|---|
  | The grammar data | `packs/en/datetime/grammar.json` |
  | The fallback table | `scripts/nlu/entities.py` → `_DEFAULT_DT_GRAMMAR` |
  | The table compiler | `scripts/nlu/entities.py` → `_build_en_dt_tables()` |
  | The strip-pattern builder | `scripts/nlu/entities.py` → `_en_strip_patterns()` |

  The grammar table and `_DEFAULT_DT_GRAMMAR` share one schema, which is exactly
  the `_DEFAULT_*`-fallback-overridden-by-pack shape this step needs. Keep the
  ref's priority-preserving details — notably the anchor match order
  (`day_after_tomorrow` before `tomorrow`, so the shared substring cannot
  mis-fire) and the strip order (`(at|by) N` before the bare connector, so no
  orphan digit is left behind).

  Also port the ref's **data-driven path selection**: if a datetime lexicon
  exists for the language, use the lexicon parser; otherwise fall through to the
  table-driven parser. English ships no lexicon, so it takes the table path
  *without any language literal*. That is what removes `entities.py:70`.

  **What you must NOT trust, and must verify:**
  1. **The `_note` inside `grammar.json` is stale.** It claims only `weekdays`
     and `word_numbers` are wired; the ref's code wires considerably more. Read
     the ref's `entities.py`, not the note. Rewrite the note when you port it.
  2. **The two `entities.py` files have diverged** — ref 911 lines, current 795,
     ~354 differing lines. The ref predates the restructure. Reconcile, do not
     overwrite: keep this branch's path constants (`BASE_DIR = parents[3]`,
     `content/` + `content/localization/`), and **preserve this branch's
     lexicon-path fixes that the ref lacks** — in particular the spaced
     clock-hour handling (`_lex_clock_hour`, matching `18 h` / `18 heures`),
     which is a live fr/de fix. Porting the ref's file wholesale would silently
     revert it.
  3. **The ref self-declares 8 pre-existing test failures.** Assume nothing about
     its green-ness.

  **Coverage check before you commit:** every English literal still present in
  the current parser must either appear in the ported table or be a deliberate,
  stated omission. Grep the pre-refactor `extract_datetime` for string literals
  and reconcile the list against the table.

  **A note on scope:** the ref left English on its *own* table-driven path,
  separate from the generic lexicon interpreter — two interpreters, both
  data-driven. That is the correct stopping point for this step. Unifying
  English onto the single lexicon interpreter is where behaviour is most likely
  to move, so it is a **separate later sub-step**, not part of A7.1.
- **A7.2 — Carrier phrases.** `NLUEngine._CARRIER` (`engine.py:916`) becomes a
  pack table. Note the current code *prepends* language carriers to the English
  list — after this step there is no English list to prepend to.
- **A7.3 — Topic connectors.** `_LEADING_CONNECTOR` becomes a pack table.
- **A7.4 — Remove the language branches.** With the tables in place, delete the
  four `engine.py` branches and the `entities.py` branch. Semantic loading
  (`_load_semantic` vs `_load_multilingual_semantic`) resolves from the pack's
  declared semantic artifacts, not from a language literal.

**Acceptance gate (all four, every sub-step):**
1. `pytest tests/test_datetime_parity_en.py` — the A1 golden, **unchanged**.
2. `pytest tests/test_datetime_parity.py` — fr/de/da parity, **unchanged**.
3. `pytest tests/test_conversation_corpus.py tests/test_engine_conformance_fixtures.py`
   — conversation + conformance fixtures **unchanged**.
4. `python scripts/ci/check_language_neutral.py` — allowlist strictly smaller
   than the previous sub-step.

If a fixture diff appears, the refactor changed behaviour. **Fix the code, never
the fixture** (STOP rule 3).

**Commits:** `refactor: evict English <datetime grammar|carriers|connectors> into pack tables`,
then `refactor: remove the last language branches from the engine`

---

### A8 — Hostile-pack test (the actual proof of "add a language on the fly")

**Do:** port `test_hostile_language_pack_runs_end_to_end` from the reference
branch's `tests/test_neutrality.py`. Copy the English single-language bundle to a
fake `zz`, relabel the manifest, and assert the **full pipeline runs end to end
with zero engine edits** — classify, extract a datetime, fill a slot, fulfil.

This is the only test in either branch that proves the neutrality claim rather
than asserting it, and it is the acceptance test the eventual French pack must
pass. Treat it as the definition of done for A7.

**Acceptance gate:** `pytest tests/test_neutrality.py` green including the `zz`
case, with `git diff` showing no engine change required to make it pass.

**Commit:** `test: hostile zz pack runs end-to-end — engine is language-neutral`

---

### A9 — Make neutrality blocking in CI

**Do:** allowlist must now be **empty**. Add the guard and
`tests/test_neutrality.py` to `.github/workflows/ci.yml` as a **blocking** step.

**Acceptance gate:** `python scripts/ci/check_language_neutral.py` exits 0 with
`KNOWN_OFFENDERS == []`.

**Commit:** `ci: language neutrality is now a blocking gate (allowlist empty)`

---

### A10 — Author `release-pack.yml` (dev-signed, channel-gated)

Port the reference branch's 3-job pipeline
(`git show <ref>:.github/workflows/release-pack.yml`), retargeted to this tree.

**Do:**
- **Job 1 (Linux):** train → export weights → **accuracy gate**
  against baseline-v2 → upload artifacts + `report_card.json`.
- **Job 2 (macOS):** CoreML/ANE export from the same weights → **parity gate**
  (Tier-A numeric + Tier-B runtime) → upload `.mlpackage` + `coreml_parity.json`.
- **Job 3 (Linux):** assemble a **versioned single-language `spec/bundle/3.0`
  `.nlu`** carrying both the ONNX (Android) and the `.mlpackage` (iOS) →
  `nlu_compiler.verify` → publish a GitHub Release tagged `pack-en-v<semver>`
  with `report_card.json` and `coreml_parity.json` attached.
- **Signing:** dev key, `channel: dev` in the manifest. The production runtime
  already refuses dev-signed artifacts and that is correct — the point is to
  exercise the pipeline end to end before ND-8. **The workflow must not change
  when production keys arrive; only the key source and the channel value.**
  Read the key id and channel from workflow inputs / repo variables so the ND-8
  cutover is a settings change, not a code change.
- Trigger on `workflow_dispatch` + pushes to the work branch touching
  `content/**`, `packs or bundle sources`, `packages/buildtime/nlu_training/**`.
  **Do not** trigger on `main` — that is the owner's call at ND-8 time.

**Acceptance gate (Track A portion):** the assemble + verify path runs locally
end-to-end against the existing golden bundle —
`make build-bundle` then `make verify-bundle` green — and `actionlint` (or a YAML
parse) accepts the workflow. The first *real* run belongs to B6.

**Commit:** `ci: release-pack workflow — versioned .nlu to GitHub Releases (dev-signed, channel-gated)`

---

## TRACK B — needs datasets

Opens in two grades, per the §Data gate:
- **Full datasets committed** → fully open, results **authoritative**.
- **bootstrap snapshot only** → the English steps are open, results
  **PROVISIONAL**: build, run and test the machinery, but do not record
  baseline-v2 as final (B1) and do not wire a bootstrap-fitted calibration into
  the runtime (B3). fr/de/da are closed in this grade.

Label every artifact and commit with the grade it was produced under.

### B0 — Make the gates actually run in CI

**Do:** model-dependent tests must **fail, not skip**, in the job that is
supposed to enforce them — introduce an
explicit `NLU_REQUIRE_ARTIFACTS=1` mode that turns the skip guards into errors,
and set it in CI. Round 1 measured 60 model-dependent tests silently skipping;
green CI currently means those gates did not run.

**Acceptance gate:** with the datasets committed, CI reports **0 skipped**
model-dependent tests; with `NLU_REQUIRE_ARTIFACTS=1` and artifacts absent, those
tests **fail**.

**Commit:** `ci: fail-not-skip for model-dependent gates`

---

### B1 — Build the honest English holdout

**Do:** partition English **by utterance** into train / holdout. Verify **zero**
normalised overlap (case, whitespace, punctuation) — reuse A5's normaliser.
Freeze the holdout, commit it, record its sha256. Re-measure English
macro-F1 / accuracy / ECE and write **baseline-v2** to
`tests/parity/oracle_honest_en/`.

Re-run the overlap measurement for **fr, de, da** as well and record the numbers.
Round-2 §3 could only test English; the others are unverified, not clean.

**Acceptance gate:** overlap check reports 0 for the new English holdout;
`tests/parity/oracle_honest_en/evaluate_report.json` exists and is
report_card-schema-valid; the old `en_holdout.csv` is marked superseded (retain
the file, do not silently delete a fixture other tests may read).

Apply the **§Baseline policy** if the numbers fall — record, do not fail, do not
remediate.

**Commit:** `fix: honest English holdout — previous set was 99.9% training data`

---

### B2 — Port the out-of-fold calibration fitter

**Do:** port `scripts/fit_calibration.py` from the reference branch, adapted to
the **57-label** space and this tree's `train.py` featuriser. Keep intact:
- 5-fold `StratifiedKFold` OOF logits — every row scored by a model that never
  saw it, so no split is sacrificed;
- `eval_leakage_mask` excluding any row present in an evaluation set;
- `T` = bounded scalar minimising NLL;
- full **provenance** in the output: `method`, `folds`, `seed`, `n_samples`,
  `featurizer`, `source`, `source_sha256`, `fitted_at`, `fitted_by`, plus
  `ece` and `ece_uncalibrated`.

Fit **per language**, writing each language's calibration alongside its model.
The featuriser block must mirror `train.py` exactly — a `T` only calibrates the
logits of the featuriser it was fit on.

**Acceptance gate:** `ece` materially below `ece_uncalibrated` for English;
leakage mask reports >0 evaluation utterances checked; every provenance key
present and non-empty.

**Commit:** `feat: out-of-fold temperature fitting with leakage guard + provenance`

---

### B3 — Wire the runtime to the fitted server temperature, per language

The core of B8. Three defects to close together:

**Do:**
- `engine.py:314` constructs the multilingual classifier **without**
  `weights_path`, so fr/de/da silently inherit the English *device* temperature.
  Pass each language's own fitted calibration explicitly.
- Point the runtime at the **server** calibration from B2, not at
  `models/intent_classifier_weights.json`.
- Regenerate `models/intent_classifier_weights.json` — it is a stale **59-label,
  1370-term pruned device export** still being read at runtime while the shipped
  label space is 57.
- Keep the device `T` separate, fit by `export_ios_weights.py` on pruned device
  logits, and document why the two differ (STOP rule 5).

**Acceptance gate:** an assertion test proving the engine's effective temperature
for each language equals that language's fitted server `T` (and is **not**
0.796286 for all four); `models/intent_classifier_weights.json` declares 57
labels.

Confidence values will shift, so conversation and conformance fixtures will
legitimately move. Regenerate them **deliberately**, and state in the commit
message that the cause is the calibration correction — this is the one sanctioned
regeneration under STOP rule 3.

**Commit:** `fix: runtime reads the fitted per-language server temperature (was the device T for all four)`

---

### B4 — Lock calibration hygiene with tests

**Do:** port `tests/test_calibration.py` from the reference branch — artifact
well-formed, `T != 1.0` (not the identity stub), **provenance required**, and the
leakage guard actually excludes evaluation sets. Extend it with a test asserting
the runtime `T` matches the fitted server `T` per language.

Provenance-enforced-by-test is the durable fix: untraceable provenance is exactly
how two conflicting temperatures coexisted unnoticed.

**Acceptance gate:** `pytest tests/test_calibration.py` green; deleting a
provenance key makes it fail.

**Commit:** `test: calibration hygiene — provenance required, leakage guarded`

---

### B5 — Re-measure the wrong-action budget, and stop

**Do:** re-run the engine-in-the-loop replay
(`packages/buildtime/nlu_training/wrong_action_harness.py`) against the honest
holdout and the corrected temperatures. Record the honest number per language and
per domain in `tests/parity/oracle_honest_en/`.

**Report it plainly. It may be better or worse than 41 — both are informative.**

**Then stop.** Closing the budget needs the owner's unconditional-confirm policy
decision (STOP rule 1) plus P2 data/model work that is out of scope. Write the
number, the per-domain breakdown, and the recommendation into
`ENGLISH-PRODUCTION-STATUS.md` and leave it for owner review.

**Acceptance gate:** report written; **no** threshold, guard or policy file
modified in this step (`git diff` proves it).

**Commit:** `measure: honest wrong-action replay on corrected calibration (no policy change)`

---

### B6 — First green release run

**Do:** dispatch `release-pack.yml`. Confirm the accuracy gate runs against
baseline-v2, the CoreML parity gate runs, and a `pack-en-v<semver>.nlu` appears
as a GitHub Release with `report_card.json` and `coreml_parity.json` attached.
Verify the published artifact with `nlu_compiler.verify`.

**Acceptance gate:** a GitHub Release exists; the downloaded `.nlu` verifies; the
manifest declares `channel: dev`, both models (onnx + coreml), and the 57-label
space.

**Commit:** `chore: first end-to-end release-pack run (dev channel)`

---

## Definition of green (terminal state)

All of these, in one run, with no skips:

```
make check                                              # lint + typecheck + test
python scripts/ci/check_language_neutral.py             # exit 0, allowlist EMPTY
pytest tests/test_datetime_parity_en.py                 # A1 golden unchanged
pytest tests/test_datetime_parity.py                    # fr/de/da unchanged
pytest tests/test_neutrality.py                         # incl. hostile zz pack
pytest tests/test_calibration.py                        # provenance + leakage
pytest tests/test_conversation_corpus.py tests/test_engine_conformance_fixtures.py
PYTHONPATH=packages/buildtime python -m nlu_compiler spec/examples/3.0/minimal
PYTHONPATH=packages/buildtime python -m nlu_compiler spec/examples/3.0/full
make build-bundle && make verify-bundle
```

Plus: a published `pack-en-v*.nlu` Release (B6), and an honest baseline-v2 with
the wrong-action number recorded and handed to the owner.

**On reaching green, write the terminal marker.** Rewrite
`ENGLISH-PRODUCTION-STATUS.md` so its **first line is exactly `STATUS: COMPLETE`**,
followed by the final honest metrics, the gate output proving green, and the
still-open owner items. Every subsequent firing checks that line first and exits
immediately, so post-completion runs cost almost nothing. Then attempt to disable
`trig_018ygxy3X9EgNX48wtquNeky` via `update_trigger` (`enabled: false`) — the MCP
tool may not be present in a fired session, which is expected; if it is absent,
say so plainly in the final message so the owner can disable it manually.

**Green does not mean shippable.** B1 (wrong-action budget) and ND-8 (production
signing) remain open by design — this routine's job is to make them *decidable on
real evidence*, not to close them.

---

## Blocked protocol

Update `docs/Review-F5/ENGLISH-PRODUCTION-STATUS.md` with:

- Track A / Track B step states (done / in-progress / blocked) with the acceptance
  gate output that justifies each state;
- the exact blocker and who owns it;
- honest metric snapshots — never the leakage-inflated numbers;
- **which data-gate state the run was in**, so every metric in the file is
  traceable to the data behind it, and every provisional number is marked;
- the open owner queue: committing the full datasets, ND-8 production signing, the B1
  unconditional-confirm policy decision, and authorisation to begin French.

Then commit, push, and end the run.

---

## Owner decisions this routine is waiting on

| # | Decision | Blocks |
|---|---|---|
| 1 | Commit the full `datasets/` tree from the machine that holds it (see `datasets/README.md`) | fr/de/da entirely; promoting any English Track-B result from PROVISIONAL to authoritative (baseline-v2, the runtime calibration in B3, the B6 release gate) |
| 2 | ND-8 — production signing keys / KMS | Promoting releases past `channel: dev` |
| 3 | B1 — unconditional confirmation on high-cost state-changing intents | Closing the wrong-action budget |
| 4 | Authorise the French pack trial after a green run | P3 / the neutrality proof |
