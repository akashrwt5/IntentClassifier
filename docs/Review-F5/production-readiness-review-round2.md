# Production Readiness Review — Round 2 (independent second pass)

**Role:** Principal Software Architect / NLU Architect / Production Readiness Reviewer
**Date:** 2026-07-25
**Branch reviewed:** `claude/nlu-production-readiness-dqyl38` @ `93b1888` (descendant of `feature/production-work`)
**Reference branch evaluated:** `claude/claude-setup-architecture-ebqobs-Temperaturescaling-fixes` @ `0089b89`
**Prior art:** `docs/Review-F5/production-readiness-review.md` (round 1, 2026-07-24)

**Method:** I did not accept round 1 or `EXECUTION_STATUS.md` as given. I re-read
the runtime code, traced the calibration chain end-to-end, ran the reference
branch's language-neutrality guard against the current engine, and measured
train/holdout overlap myself. Every claim below cites what I ran or read.

---

## 1. Verdict

**NO-GO, and further from GO than round 1 concluded.**

Round 1 named two hard blockers (wrong-action budget, CI/data reproducibility)
and graded the runtime engine 🟢 and calibration "genuinely good." I agree on the
two blockers. I do **not** agree on those two grades. Two additional blockers sit
underneath them, and both are load-bearing for the safety argument:

| # | Blocker | Status in round 1 | My finding |
|---|---|---|---|
| **B1** | Wrong-action budget violated (41 vs ≤5) | Identified ✅ | Confirmed — and **not currently measurable**, see B8 |
| **B2** | DVC local remote; safety/quality tests skip in CI | Identified ✅ | Confirmed |
| **B8** | **The shipped confidence scale is not the calibrated one.** Runtime uses `T=0.796` (device featurizer, stale 59-label export); the evaluation report uses `T=0.6055`. Every confidence gate is tuned against a scale the engine does not run. | **Missed** — graded 🟢 "calibration is genuinely good" | **New hard blocker** |
| **B9** | **The English holdout is 99.9% training data.** en macro-F1 / accuracy / ECE are memorisation scores, not generalisation. | **Missed** | **New hard blocker** |
| **B10** | **The engine is not language-neutral.** English behaviour is hardcoded *in code*; other languages are data-driven. English is the one language that cannot be shipped as a pack. | **Missed** — graded 🟢 "well-structured" | **New architectural blocker** |
| B3–B7 | Signing (ND-8), consent (ND-9), infra (ND-12), iOS PAT (ND-7), language data (ND-13) | Identified ✅ | Confirmed, owner-gated |

B8 and B9 matter more than their line items suggest, because **the entire safety
architecture is confidence-gated**. The polarity guards, the help-marker guard
and the uncertainty-confirmation gate all trigger off a probability. If that
probability comes from the wrong temperature, and the accuracy behind it was
measured on memorised data, then the "99 → 73 → 65 → 55 → 46 → 41" wrong-action
trajectory is a measurement of an artifact, not of the product. B1 cannot be
honestly closed until B8 and B9 are closed first.

---

## 2. New finding: the calibration chain is broken (B8)

Confidence is `softmax(logits / T)`. `T` is rank-preserving — it never changes
*which* intent wins — so a wrong `T` produces no test failure anywhere. It
silently mis-tunes every gate at once. That is exactly what has happened.

### 2.1 Three different temperatures, and the runtime uses the worst one

| Where | Value (en) | Fit against | Read by |
|---|---|---|---|
| `models/intent_classifier_weights.json` | **0.796286** | **Device** logits — 1370-term pruned vocab, 59-label pre-migration export (`export_ios_weights.py::_fit_temperature`) | **The runtime engine** |
| `config/calibration.json` | 0.6055 | Full-vocab server logits, but on `en_holdout.csv` | `evaluate.py` only — **nothing at runtime** |
| Documented in `.claude/memory/inference.md` | "comes from `config/calibration.json`" | — | **False** |

Verified:

- `packages/runtime/nlu_engine/classifier.py:126` → `self.temperature = _load_temperature(weights_path)`, default `WEIGHTS_PATH = models/intent_classifier_weights.json`.
- That file: `vocab` length **1370** (pruned device vocabulary), `labels` length **59** (pre-ND-3 taxonomy; the shipped schema and `intent_labels.json` are **57**), `temperature: 0.796286`.
- `packages/buildtime/nlu_training/evaluate.py:37,91` → report card divides by `config/calibration.json[lang]["temperature"]`.
- `grep` for `calibration.json` across the runtime package: **zero hits**.

So the engine calibrates full-vocab ONNX logits with a temperature fit on a
*different featurizer* and a *different label space*. The report card that says
`ECE = 0.018` measures a configuration that does not ship.

### 2.2 Per-language calibration is computed, reported — and never applied

`packages/runtime/nlu_engine/engine.py:314`:

```python
return IntentClassifier(model_path=onnx_file, labels_path=labels_file, schema_path=SCHEMA_PATH)
```

No `weights_path`. The multilingual branch therefore falls back to the **same**
`models/intent_classifier_weights.json`. Consequence: **fr, de and da all run at
the English device temperature 0.796**, not their fitted 0.6609 / 0.6433 / 0.6664.
The per-language temperature-scaling work exists in `config/calibration.json`, is
quoted in the docs, and has no runtime effect whatsoever.

This is a one-line-shaped bug with a four-language blast radius, and it is
invisible to the test suite by construction.

---

## 3. New finding: the English holdout is not a holdout (B9)

I compared `multilingual/test/*_holdout.csv` against the English training master
(`data/04_GENERATED_MASTER_training_data.csv`, 9,986 unique texts), normalising
case and whitespace:

```
en_holdout  n=1461   overlap = 1460   (99.9%)
fr_holdout  n=1358   overlap = 1      ( 0.1%)
de_holdout  n= 942   overlap = 102    (10.8%)
da_holdout  n=1258   overlap = 0      ( 0.0%)
```

**1460 of 1461 English "holdout" utterances appear verbatim in training data.**
Every English number derived from that file — macro-F1 0.896, accuracy 0.907,
ECE 0.018, the tuned `conf_threshold`, and the temperature in
`config/calibration.json` — measures recall of memorised strings. English's
true generalisation performance is currently **unknown**, not 0.90.

*Scope caveat, stated precisely:* the current branch's datasets are DVC-local and
absent from a fresh clone (this is B2), so I used the training master committed on
the reference branch. That file predates the ND-3 label migration, which renamed
labels but did not rewrite utterance text, so the text-level overlap conclusion
holds. The fr/de/da rows above are **not** evidence those languages are clean —
that master is English-only, so their overlap is untested. **Re-run this check per
language against the live datasets before trusting any per-language number.**

The reference branch reached the same conclusion independently and wrote it into
`config/calibration.json` as a `_deprecated` banner. That finding is correct and
should be adopted.

---

## 4. New finding: the engine is not language-neutral (B10)

This is the direct answer to *"is the engine completely independent of language, so
a language can be added on the fly?"* — **No.** Not today.

I ran the reference branch's CI guard against the current engine package:

```
FAIL: language-specific branch(es) found in the engine:
  engine.py:146: elif self.language in ("en", "", "multilingual"):
  engine.py:197: if language in ("en", ""):
  engine.py:270: if language in ("en", "", "multilingual"):
  engine.py:290: if language in ("en", "", "multilingual"):
  entities.py:70: if language and language != "en":

FAIL: hardcoded natural-language vocabulary found in the engine:
  classifier.py:61: _NEGATIONS = ["don't", 'do not', 'no need to']…
```

### 4.1 The architecture is inverted

The current design is **two tracks**: English is the hardcoded path, everything
else is the data-driven path.

| Concern | English | fr / de / da |
|---|---|---|
| Datetime parsing | ~250 lines of English regex inline in `entities.py::extract_datetime` | `_extract_datetime_lex()`, driven by `nlu_entities.<lang>.json` |
| Carrier phrases | `NLUEngine._CARRIER`, hardcoded class constant (`engine.py:916`) | lexicon `carrier_phrases`, *prepended to* the English list |
| Topic connectors | `_LEADING_CONNECTOR`, hardcoded English | same hardcoded English |
| Negation suppression | `_NEGATIONS`, hardcoded English | **same hardcoded English** |
| Semantic stage | `_load_semantic()` | `_load_multilingual_semantic()` |
| Schema / entities | canonical files | overlay merge |

Two consequences, both bad, and the second is a live defect:

1. **English is the one language that cannot be packaged, versioned, or
   OTA-updated**, because its behaviour is code, not data. The whole bundle/OTA
   investment (ADR-005, `BundleManager`, two-slot lifecycle) cannot actually
   deliver a change to English datetime handling or carrier stripping — that needs
   an app release.
2. **Negation suppression is a silent no-op for fr/de/da.** `_is_negated()`
   (`classifier.py:91`) takes no language parameter and matches only English cues
   (`not`, `don't`, `never`, `without`, `stop`, `cancel`). For a French user,
   *"je ne veux pas traduire ça"* does not suppress the `contains` keyword hit on
   *traduire*. Given that this is a hearing-aid product and the wrong-action budget
   is the governing safety constraint, a negation guard that only works in one of
   four shipped languages is a safety defect, not a tidiness issue.

### 4.2 "On the fly" — what is and isn't true today

| Claim | Reality |
|---|---|
| Add a language with no engine code change | **No.** Adding `zz` needs `_load_semantic` routing, a `language in (...)` audit, and it inherits English negation/connectors. |
| Add a language with no app release | **No.** Language assets are repo files (`content/localization/*`, `multilingual/models/*`) resolved from `BASE_DIR`, not from a downloadable pack. |
| Language behaviour is fully declared in data | **Partially** — schema/entities/lexicon yes; datetime grammar partly; negation, connectors, carriers no. |
| The bundle format supports a per-language pack | **Yes, on paper.** ADR-005 §291 already says: *"Downloadable per-language bundles are a packaging profile of this same spec (a bundle with one language), not a new format."* The spec is right; the runtime does not implement it. |

That last row is the good news, and it drives my central recommendation in §7.

---

## 5. Direct answers to the four questions asked

### Q1 — How ready is it for production?

**Platform: strong. Product: not shippable, and the gap is larger than the tracker
shows.** Round 1's §8.1/§8.2/§8.3 grading is fair on structure — the bundle spec,
the single-path validator, the `packages/apps/spec` restructure with byte-identical
oracle parity, the compiler, the two-slot `BundleManager`, the telemetry redaction
model and the config-driven guard stack are all genuine, well-built platform work.

What is not ready is the **evidence chain**. Right now the project cannot answer
"how accurate is English?" (B9), "what confidence does the shipped engine emit?"
(B8), or "did anything regress?" (B2 — the gates skip in CI). B1 is downstream of
all three. A hearing-aid assistant whose safety argument rests on confidence
thresholds needs those three answers before the wrong-action number means anything.

### Q2 — Does it create the NLU package with all the details on GitHub?

**No. Nothing is published to GitHub today.**

- `.github/workflows/` contains exactly two workflows: `ci.yml` (lint / type /
  test / bundle-spec conformance) and `coreml-macos.yml`, which triggers **only on
  branch `claude/coreml-export`**.
- Neither builds, signs, versions, or publishes a `.nlu`.
- `grep` for release/publish machinery across workflows and the Makefile: **zero
  hits** for GitHub Releases, `GITHUB_TOKEN`, or artifact upload.
- The capability *does* exist locally: `make build-bundle` → compiler stages 11–15
  produce a deterministic `.nlu` with an Ed25519 dev signature and a 3-gate
  verifier. It fails closed correctly. But it is a developer command, not a
  pipeline, and the production runtime correctly refuses dev-signed artifacts — so
  no releasable package can be produced at all until ND-8 lands.

The reference branch **does** have this, and it is the single most complete thing
it has: `.github/workflows/release-pack.yml` is a 3-job pipeline —
train + accuracy gate (Linux) → CoreML/ANE export + parity gate (macOS) → assemble
a versioned `.nlu` carrying **both** the ONNX (Android) and the `.mlpackage` (iOS)
→ publish a GitHub Release tagged `pack-en-v<semver>` with `report_card.json` and
`coreml_parity.json` attached. GitHub Releases as the model registry; clients fetch
from Releases only. That is the right shape and should be ported (§7, P1).

### Q3 — Is the engine language-independent, can a language be added on the fly?

**No** — see §4. English is hardcoded, other languages are data-driven, and the
negation guard silently fails outside English. The bundle *spec* anticipates
per-language packs; the *runtime* does not implement pack loading.

### Q4 — What should we take from the reference branch?

See §6 and §7. Short version: **take the diagnoses, the contract, the guards and
the release pipeline. Do not take the tree.**

---

## 6. Evaluation of the reference branch

### 6.1 What it gets right

1. **The temperature diagnosis (`scripts/fit_calibration.py`).** The docstring
   states the principle exactly: *"Temperature is a property of a (model,
   featurizer) pair, not of a language."* It correctly separates the device `T`
   (fit in `export_ios_weights.py` on pruned logits) from the server `T`, and
   explicitly warns against unifying them. Correct.
2. **Out-of-fold calibration fitting.** 5-fold `StratifiedKFold`, every row scored
   by a model that never saw it, `T` = bounded scalar minimising NLL. Standard,
   correct, and it avoids sacrificing a split — which matters when the honest data
   is scarce. Result for en: `ECE 0.116 → 0.0084` with recorded provenance.
3. **An explicit eval-leakage guard** (`eval_leakage_mask`) that excludes any
   calibration row appearing in an evaluation set — normalised on case *and*
   punctuation. Its docstring notes that `train.py`'s own guard compares raw
   strings and therefore misses pairs differing only by a trailing `?`. That is a
   real hole in the current branch's guard too.
4. **Provenance as a hard requirement.** `packs/en/intent_model/calibration.json`
   carries `method`, `folds`, `seed`, `n_samples`, `featurizer`, `source`,
   `source_sha256`, `fitted_at`, `fitted_by` — and `tests/test_calibration.py`
   *fails* if any is missing. Untraceable provenance is precisely how two
   conflicting temperatures coexisted. This is the fix that prevents recurrence.
5. **The Language Pack contract (`packages/nlu_langpack/`).** `Protocol`-based
   structural typing, a dependency-free contract module both sides can import
   without a cycle, a manifest that fails closed on unknown/missing components, an
   `engine_compat` version gate, and a loader that is strict on `production` packs
   and lenient on `dev`/`beta`. The design is sound and it is the right boundary.
6. **The deliberate logic/data split.** Logic-bearing components (tokenizer, intent
   model, semantic head) ship as objects; behaviour facts (keyword rules, datetime
   grammar, lexicons, workflows) ship as **tables interpreted by a generic
   engine-side interpreter**. This is the distinction the current engine is missing,
   and it is what makes a new language authorable rather than programmable.
7. **Neutrality enforced in CI, not by convention** (`check_language_neutral.py`).
   Two checks: no `if language ==` branches, and no module-level match vocabulary
   outside the `_DEFAULT_*` fallback convention. The second check exists
   *specifically because* `_NEGATIONS` slipped past the first — the guard was
   written from a real escape. It works: I ran it unmodified against the current
   engine and it caught all six defects (§4).
8. **The hostile-pack test.** `test_hostile_language_pack_runs_end_to_end` copies
   `packs/en` to `packs/zz`, relabels it, and asserts the full pipeline runs with
   zero engine edits. This is the only test I have seen in either branch that
   actually *proves* "a language can be added on the fly" rather than asserting it.
9. **`release-pack.yml`** — §5, Q2.

### 6.2 What is wrong with it, and why it must not be merged

The reference branch is **architecturally ahead and structurally behind**. It
branched before the current line's major work and does not contain it:

| Dimension | Current branch | Reference branch |
|---|---|---|
| Engine location | `packages/runtime/nlu_engine/` | `scripts/nlu/` (pre-restructure) |
| Label space | **57** (`domain.object.action`, ND-3) | **59** (`Help_FallAlert`, `Cmd.VolumeIncrease`) |
| Bundle format | `spec/bundle/3.0`, 16 JSON Schemas, signed `.nlu`, `BundleManager` | none — `packs/<lang>/` directories |
| Safety guards | polarity, help-marker (ND-14), confirm gate | absent |
| Content model | `content/capabilities/` (12 × 57) + drift guard | flat `data/` |
| Datasets | DVC | committed CSVs |
| Test suite | 150 nominal / 90 pass + 60 skip clean | **8 pre-existing failures, self-declared** |

Merging it would revert the ND-3 label migration, the restructure, the capability
content tree, the bundle spec and the entire safety guard stack. **A diff-based
merge is not on the table.** Its value is as a *design donor*: port the ideas,
the contract, the guards, the scripts and the workflow — re-expressed against the
current tree.

### 6.3 Where I disagree with the reference branch

- **`packs/<lang>/` is a second package format, and the project should not have
  two.** ADR-005 already declares that a per-language bundle is a *packaging
  profile* of `spec/bundle/3.0`, not a new format. Adopting `packs/` verbatim
  would create a competing container with its own manifest (`pack.json`), its own
  versioning (`format_version`), its own integrity story (none — no signing), and
  no `BundleManager` integration. Take `nlu_langpack` as the **runtime-side
  contract and loader for a single-language `.nlu` bundle**; do not take `packs/`
  as a parallel on-disk format.
- **`pack.json` has no signature verification** — the loader's own docstring says
  "trusted after (future) signature verification." The current branch already
  solved this properly (3-gate verify, channel + key-id enforcement, tamper
  detection, downgrade refusal). Keep the current branch's answer.
- **`content_version: int` is weaker than the current lineage model.** Prefer the
  bundle manifest's existing versioning and lineage fields.
- **`RUNTIME_CONTRACT_VERSION` must be reconciled with
  `spec/contracts/runtime-contract-v1.md`**, not introduced as a second, parallel
  version axis. There should be exactly one `engine_compat` anchor.

---

## 7. What to implement, English-first

The user's stated sequencing is right and I want to endorse it explicitly:
**complete the production cycle for English, then prove the structure by adding a
French pack and confirming it needs no engine change.** That is the correct order,
and §4.1 is the reason it is *also* the shortest path — making English a pack is
what forces the last hardcoded language logic out of the engine. Doing French
first would leave English as a permanent special case.

One correction to how that goal is usually framed: "productionise English" is not
mostly a modelling task. Sequence P0 below is entirely about **being able to
measure English at all**. Until B8/B9 close, any accuracy or wrong-action number
produced for English is not evidence.

### P0 — Restore the evidence chain (nothing else counts until this is done)

| # | Task | Port from ref | Why |
|---|---|---|---|
| 1 | **Build an honest English holdout.** Partition by utterance, verify zero normalised overlap with training, freeze it, DVC-track it. Re-measure en macro-F1 / accuracy / ECE. **Expect the numbers to drop** — plan for it and do not treat the drop as a regression. | — | B9. Without this there is no English baseline. |
| 2 | **Port `fit_calibration.py`** — OOF temperature fitting + `eval_leakage_mask` — adapted to the 57-label space and the current `train.py` featurizer. Emit `temperature`, `ece`, `ece_uncalibrated` and full `provenance`. | ✅ direct | B8/B9. |
| 3 | **Make the runtime read the fitted server temperature**, per language. Pass `weights_path`/calibration explicitly at `engine.py:314`; stop defaulting fr/de/da to the English device `T`. Keep the device `T` separate and documented as separate. | design | B8 §2.2. |
| 4 | **Regenerate `models/intent_classifier_weights.json`** — it is a stale 59-label pre-migration export still being read at runtime. | — | B8 §2.1. |
| 5 | **Port `tests/test_calibration.py`** — provenance-required, no-identity-`T`, leakage-guard tests. | ✅ direct | Prevents recurrence. |
| 6 | **Tighten `train.py`'s leakage guard** to normalised matching (case + punctuation), per the ref's finding that raw-string comparison misses trailing-`?` pairs. | design | Closes the hole that produced B9. |
| 7 | **Fix the docs that assert the false chain** — `.claude/memory/inference.md` and `training.md` both state the runtime reads `config/calibration.json`. Either delete the file or mark it advisory-only, as the ref branch did. | ✅ pattern | An incorrect doc is how this survived review once already. |
| 8 | **Close B2**: real shared DVC remote + `dvc pull` in CI; add `referencing`, `cryptography`, pin `jsonschema>=4.18`; make model-dependent tests **fail rather than skip** in the gating job. | — | Round 1 P0. Without it, P0.1–P0.7 cannot be enforced. |

### P1 — Make English a pack (the neutrality work)

| # | Task | Port from ref | Why |
|---|---|---|---|
| 9 | **Port `check_language_neutral.py`** into `scripts/ci/`, pointed at `packages/runtime/nlu_engine`. Land it **failing**, with the six known offenders as an explicit allowlist that shrinks to zero. | ✅ direct (retarget `ENGINE_DIR`) | Makes B10 visible and monotonic. |
| 10 | **Fix `_NEGATIONS` first** — it is the one neutrality defect that is a live safety bug in three shipped languages, and it is small: give `IntentClassifier` a language/lexicon input and move the cues into the per-language lexicon, keeping `_DEFAULT_NEGATIONS` as the English fallback. | ✅ pattern | §4.1(2). Do this ahead of the rest. |
| 11 | **Port `packages/nlu_langpack/` as the runtime contract** — `interfaces.py` Protocols, `manifest.py`, `loader.py`, `version.py`, `errors.py — but bind it to `spec/bundle/3.0` and reuse the existing signature/channel verification instead of `pack.json`'s unsigned manifest. One `engine_compat` anchor: `runtime-contract-v1.md`. | ✅ contract, ❌ `packs/` format | §6.3. Realises ADR-005 §291. |
| 12 | **Evict English behaviour into pack tables**: datetime grammar (the ~250-line inline path), `_CARRIER`, `_LEADING_CONNECTOR`, negation cues. Keep `_DEFAULT_*` fallbacks as data. Follow the ref's logic/data split — generic interpreter in the engine, tables in the pack. **Guard with byte-identical parity** on `multilingual/test/` fixtures and the datetime corpus; this is a refactor, not a behaviour change. | ✅ approach | The largest item. Removes the last `if language` branches. |
| 13 | **Port the hostile-pack test** — copy the `en` pack to `zz`, relabel, assert the full pipeline runs unmodified. | ✅ direct | The only real proof of "add a language on the fly". Also the acceptance test for the French trial. |
| 14 | **Port `release-pack.yml`** — train + gate → CoreML/ANE export + parity gate → assemble a versioned single-language `.nlu` (ONNX + `.mlpackage`) → publish a GitHub Release with `report_card.json` and `coreml_parity.json`. Adapt `assemble_pack.py` to emit a `spec/bundle/3.0` bundle, not a `packs/` archive. **Ship it dev-signed and channel-gated until ND-8**, so the pipeline is exercised before the keys exist. | ✅ direct (retarget format) | Q2. This is what makes "the NLU package on GitHub" true. |

### P2 — Then, and only then, the wrong-action budget (B1)

Re-run the engine-in-the-loop replay **after** P0, against the honest English
holdout and the correct runtime temperature. The current 41 is not a trustworthy
number in either direction — it could be better or worse. Round 1's two levers
remain correct and I endorse both:

- **Policy:** unconditional confirmation on the highest-cost state-changing intents
  regardless of confidence — the only lever that catches the 0.87–1.00 residue.
  Owner decision; put the friction numbers in front of them.
- **Data/model:** `device.volume` polarity, OOS rejection (0.68 recall), help-vs-action
  separation. Add **per-domain budgets** so `device` carries its own ceiling.

Then wire the replay as a blocking CI gate, which P0.8 makes possible.

### P3 — Prove it with French

Only after P1.13 is green: author `fr` as a pack + training data, change **no
engine code**, and confirm the neutrality guard and hostile-pack test still pass.
If that requires an engine edit, P1 is not finished.

### Deliberately not recommended

- Merging or rebasing the reference branch. §6.2.
- Adopting `packs/<lang>/` as a second on-disk format. §6.3.
- Re-opening Phase 2 (Rust). Owner directive 2026-07-14 stands; the contract work
  delivers the value without it.
- Tuning thresholds to improve the wrong-action count before P0 lands. That would
  be fitting to a broken measurement.

---

## 8. Bottom line

Round 1 concluded "the platform is production-grade, the product is not." I would
sharpen that: **the platform is production-grade, and the measurement system
underneath the product is not.** Three of the numbers the project steers by —
English accuracy, English ECE, and the shipped confidence scale — do not describe
the artifact that runs. Everything downstream, including the wrong-action budget
that is the stated reason for the No-Go, inherits that error.

The reference branch found two of these three independently and fixed them
properly, with the right method (out-of-fold), the right principle (temperature
belongs to a *(model, featurizer)* pair) and the right durable defence (provenance
required, enforced by test). It also built the language-neutrality boundary this
codebase needs and the release pipeline it does not have. It is on the wrong tree,
but it is right about the architecture. Port it forward; do not merge it back.

Closing P0 is roughly a fortnight of unglamorous measurement work with no visible
product improvement, and it will probably make the English metrics look *worse*.
It is still the highest-value work available, because it is what converts every
subsequent number from an assertion into evidence.

---

### Appendix — evidence index

| Claim | How verified |
|---|---|
| 5 language branches + `_NEGATIONS` in the engine | Ran ref `check_language_neutral.py` against `packages/runtime/nlu_engine` (`ENGINE_DIR` retargeted) |
| Runtime `T = 0.796286` from the device export | `classifier.py:126`; `models/intent_classifier_weights.json` → `vocab` 1370, `labels` 59, `temperature` 0.796286 |
| Shipped label space is 57 | `models/intent_labels.json` (57) and `content/nlu_schema.json` (57) vs weights `labels` (59) |
| `config/calibration.json` unread at runtime | `grep -rn calibration.json packages/runtime/` → no hits; readers are `evaluate.py:37`, `calibrate_languages.py`, `test_smoke.py` |
| fr/de/da inherit the English device `T` | `engine.py:314` omits `weights_path`; `classifier.py` falls back to `WEIGHTS_PATH` |
| Report card uses a different `T` than the runtime | `evaluate.py:91` → `calibration[lang]["temperature"]` |
| en holdout 99.9% train overlap | Normalised set intersection, `multilingual/test/en_holdout.csv` × `data/04_GENERATED_MASTER_training_data.csv` (ref-branch copy — see §3 caveat) |
| Negation guard is language-blind | `classifier.py:91` `_is_negated(text, term)` — no language parameter; `_NEGATIONS` module-level English |
| No GitHub release path | `.github/workflows/` = `ci.yml` + `coreml-macos.yml` (branch `claude/coreml-export` only); no release/upload steps in workflows or Makefile |
| DVC remote is local, no `dvc pull` in CI | `.dvc/config` → `url = ../../dvc-store`; `grep -c dvc .github/workflows/ci.yml` → 0 |
| Ref branch is pre-restructure / pre-migration | `git ls-tree` → `scripts/nlu`, `data/*.csv` committed; training master labels `Help_FallAlert` (59-label space) |
| Ref branch has 8 pre-existing test failures | `docs/Review-F5/IMPLEMENTATION-PROGRESS.md`, self-declared |
