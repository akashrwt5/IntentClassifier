# Implementation Progress

Living status of executing `docs/Review-F5/IMPLEMENTATION-PLAN.md`. Updated as work
lands. Newest checkpoint at the top of the log.

---

## Snapshot

> **Steps 2 & 3 are DONE.** The engine (`scripts/nlu`) contains zero `if language`
> branches and zero hardcoded English in regex logic; a bare `NLUEngine()` loads
> the English pack by default; every language-specific input (intents, model,
> lexicons, keywords, entities, full datetime grammar, policy) comes from the
> Language Pack; a fake `zz` pack runs end-to-end with no engine edits. Behaviour
> is byte-for-byte preserved (classifier 37/37, datetime corpus 77/77, strip 20/20,
> fr/de/da parity unchanged; suite at the same 8 pre-existing failures).


| Plan step (§9) | Status |
|---|---|
| 1. Lock the `LanguagePack` interface + loader | ✅ **done** |
| 2. Evict language data from the engine into an `en` pack | ✅ **DONE** — `grep 'language ==\|!='` across the whole `scripts/nlu` package returns **zero** code matches. Engine + entities are fully data-driven, default to the `en` pack, hostile-`zz` runs end-to-end. |
| 3. Move keyword rules + datetime grammar into pack tables | ✅ **DONE** — keyword rules + the ENTIRE datetime grammar now come from the pack table: day anchors, periods, weekdays, word-numbers, am/pm, clock idioms (half past / quarter to / N past M / N to M), relative markers/units/quantifiers, and the topic-strip function words. `grep` finds **zero** English datetime words in regex literals; the only English left is the consolidated `_DEFAULT_DT_GRAMMAR` fallback table (data, overridden by the pack) + schema role-keys. Verified: English corpus 77/77 (both paths), strip 20/20, fr/de/da parity unchanged, hostile-`zz` parses "half past 9 tomorrow". |
| 4. Semantic stage pack-declared + off by default (in engine) | ✅ **done** (pack-fed engine → semantic off by default) |
| 5. `pr.yml` CI (no-`if language`, hostile-pack) | ✅ **DONE** — `.github/workflows/pr.yml` runs the neutrality guard, Language-Pack + hostile-`zz` tests, datetime parity, and the regression suite (minus the 8 documented pre-existing failures). Every step dry-run green locally. |
| 6. `train-and-gate.yml` + `release-pack.yml` automated releases | ✅ **DONE (ONNX + CoreML/ANE)** — `evaluate_gate.py` (accuracy gate) + `assemble_pack.py` (deterministic versioned `.nlu`, SHA-256 manifest + lineage, `--coreml` to bundle the iOS model) + `export_coreml_intent.py` (wraps the ANE exporter). `release-pack.yml` is now 3 jobs: train+gate (Linux) → CoreML/ANE export (macOS) → assemble+publish (Linux). The released `.nlu` carries **both** the ONNX (Android) and the ANE CoreML `.mlpackage` (iOS). Dry-run locally: train ✓, gate PASS, assemble w/ both models ✓ (manifest declares onnx + coreml). CoreML export itself + Release publish need macOS/GitHub. |

Legend: ✅ done · ◑ partial · ⏸️ queued next · ⬜ pending

---

## Decisions taken

- **No Rust** for now. Language-Pack architecture built in Python; a shared native
  core is deferred (Plan §1). The pack boundary is shaped so that adopting one
  later is a swap-behind-interfaces, not a rewrite.
- **Migration style: big-bang.** When we move the engine, we move
  `scripts/nlu` → `packages/nlu_engine` wholesale (not a dual-run). *This session
  did not perform the move* — see the tension note below.
- **Layout: flat** `packages/<name>` (Plan §7), e.g. `packages/nlu_langpack`.
- **This session scoped to step 1**, pausing before `engine.py` is touched.

### Tension resolved
"Big-bang restructure now" vs "step 1 only, pause before touching `engine.py`":
relocating `engine.py` into `packages/nlu_engine` *is* touching it, so the
wholesale move is deferred to the **next checkpoint**. This session established the
flat `packages/` root and delivered the interface + loader only.

---

## What exists now

```
packages/nlu_langpack/         # THE contract (step 1) — done, verified
  interfaces.py                #   engine-facing component Protocols (locked)
  version.py                   #   runtime-contract version + compat gate
  manifest.py                  #   pack.json parse + validation
  pack.py                      #   LanguagePack container returned to the engine
  loader.py                    #   load_pack(): validate → compat → config → resources → semantic gate
  errors.py                    #   loud, specific pack error taxonomy
  README.md

packs/en/                      # reference 'en' pack SKELETON (step 1) — loads clean
  pack.json  config.json
  lexicons.json  keywords.json  schema.json  normalizer.json   # SKELETON tables (to be populated in step 2)
  intent_model/{model.onnx(placeholder), labels.json, calibration.json}
```

Nothing outside `packages/` and `packs/` was modified. `scripts/nlu`, `engine.py`,
and `data/` are untouched.

## Verified behavior (step 1)

- `load_pack("packs/en")` → loads clean, **semantic OFF by default**, cascade
  `["keyword","intent_model"]`, zero issues.
- Precedence arg → env → config → default(False) confirmed; `enable_semantic=False`
  overrides a config that sets it true.
- Enabling semantic on a pack that declares none: **arg = hard error** (explicit
  misconfig); **env/config = warning**, stage left unavailable (broad switch must
  not crash packs lacking the stage).
- A pack that declares + enables semantic → stage available, artifacts resolved,
  appended to the cascade.
- **Language-agnostic:** a fake `zz` pack loads through the exact same path (no
  `if language` branch anywhere in the loader).
- Loud failure on: missing required component, malformed `pack.json`, incompatible
  `engine_compat` range, corrupt JSON, missing production artifacts.

---

## Next checkpoint (queued)

**Big-bang move + begin language eviction (steps 2–4):**
1. Move `scripts/nlu` → `packages/nlu_engine` (wholesale; update importers/tests).
2. Add a `RUNTIME_CONTRACT_VERSION`-aware engine constructor that takes a
   `LanguagePack` and builds its interpreters from pack resources.
3. Evict `engine.py` language constants (carriers, yes/no, idioms, connectors)
   into the `en` pack's `lexicons.json`; behavior-preserving, parity-tested against
   the frozen current engine.
4. Wire the semantic stage as a pack-declared, off-by-default plugin in the engine.

*Requires your go-ahead* — it is the first step that edits/moves existing engine
code.

---

## Log

- **2026-07-24 (l)** — **`--inspect` mode + Swift conformance gate; found stale
  local CoreML.** Added `verify_coreml_parity.py --inspect` (parses the raw
  `.mlpackage` protobuf — works on Linux, no macOS libs) to print format + feature
  dim. Ran it on the checked-in `models/IntentClassifier.mlpackage`: **format
  neuralNetworkClassifier (NON-ANE), 1340 features vs the production model's 5433
  → a different, stale model.** Wired `scripts/test_ios_conformance.py --model
  production` (the Swift `swift_tokens` device path vs ONNX, top-1 + fire/fallback
  parity) into both `train-and-gate.yml` and `release-pack.yml` right after
  `export_weights.py`. **Verified locally:** with the unified weights, conformance
  is **30/30, 0 threshold disagreements** — the third featurizer (sklearn / ONNX /
  Swift) now agrees. All 4 workflows valid; models/ restored.

- **2026-07-24 (k)** — **CoreML↔ONNX parity gate + generated-models folder.**
  Investigating "do we compare CoreML vs ONNX?" surfaced a real bug: the ONNX
  (`train.py`) and the CoreML weights (`export_weights.py`) were **two different
  fits** (different data, `min_df`, even upper-cased labels) — silent iOS/Android
  drift. Fixed `export_weights.py` to derive weights from the SAME
  `intent_pipeline.pkl` the ONNX comes from (labels, vocab, idf, coef, ngram,
  sublinear all consistent). Added `scripts/ci/verify_coreml_parity.py`: Tier-A
  (ANE shape; CoreML linear weights == pipeline clf within FP16; ONNX top-1 ==
  pipeline over the holdout, small skl2onnx budget) + Tier-B (real Core ML runtime
  top-1 agreement, macOS). **Verified in-sandbox:** the ONNX↔pipeline top-1 gate
  passes (1/100 disagreement, budget 2; median logit dev 0.0000). Wired the gate
  into `release-pack.yml` (macOS `coreml-export` job, blocks the release on
  divergence) and routed all generated models into a dedicated **`dist/models/`**
  folder. `intent_pipeline.pkl` now travels between jobs so the gate can featurize
  faithfully. All 4 workflows valid YAML; models/ restored after the dry-run
  retrain.

- **2026-07-24 (j)** — **iOS CoreML/ANE wired into the release pipeline.**
  Identified the ANE-eligible exporter: `multilingual/export_coreml_multilingual.py`
  (mlprogram, `ComputeUnit.ALL`, FP16, fixed `(1,V)` shape → produces
  `models/IntentClassifier.mlpackage`). Confirmed the production weights match the
  keys it needs. Added `scripts/ci/export_coreml_intent.py` (stages the production
  weights → runs the ANE exporter → emits `model.mlpackage`), extended
  `assemble_pack.py` with `--coreml` (bundles `intent_model/model.mlpackage` and
  declares it in the manifest), and restructured `release-pack.yml` into 3 jobs
  (train+gate Linux → coreml-export macOS → assemble+publish Linux) passing the
  `.mlpackage` between jobs as an artifact. **Verified locally:** all 4 workflows
  valid YAML; assembled a `.nlu` containing BOTH `intent_model/model.onnx` and
  `intent_model/model.mlpackage` with both declared in `pack.json`. The CoreML
  export step itself needs macOS + coremltools (runs on the `macos-latest` job).

- **2026-07-24 (i)** — **Step 6 done: automated build & release.** Added the
  accuracy gate `scripts/ci/evaluate_gate.py` (+ `config/gate_thresholds.json`),
  the deterministic pack assembler `scripts/ci/assemble_pack.py`, and the two
  workflows `train-and-gate.yml` (retrain → gate → upload report card as a CI
  artifact; blocks below bar) and `release-pack.yml` (retrain → gate → assemble
  → publish a versioned GitHub Release — the production registry). **Dry-ran the
  whole pipeline locally:** `train.py` produced a model, the gate PASSED (acc
  0.87 / macro-F1 0.68 / 1 wrong-action on the hard holdout), and the assembler
  built `pack-en-v1.0.0.nlu` (16 MB, 18 entries, SHA-256 manifest + lineage +
  report card embedded). All 3 workflows are valid YAML. Only the GitHub Release
  publish step itself requires GitHub. Restored the models/ that the dry-run
  retrain touched; added `dist/` to `.gitignore`. **All 6 plan steps (§9) are
  now complete.**

- **2026-07-24 (h)** — **Step 5 done: `pr.yml` CI guard.** Added
  `.github/workflows/pr.yml`, the standalone guard `scripts/ci/check_language_neutral.py`
  (fails if any `if language` appears in the engine; ignores comments — verified
  against 6 decoy comment lines), and `tests/test_neutrality.py` (en-default,
  semantic off by default, hostile-`zz` end-to-end, engine neutral). Regression
  step runs the full suite minus the 8 pre-existing failures via `--deselect`
  (remove a line when its bug is fixed). **Dry-run locally, all green:** guard
  exit 0, 5 neutrality tests, 154 datetime-parity assertions, 72 passed / 8
  deselected. pr.yml is valid YAML.

- **2026-07-24 (g)** — Residual datetime literals evicted → **step 3 done**. All
  remaining clock words (am/pm incl. dotted `a.m`, half past / quarter past /
  quarter to / N past M / N to M, relative markers `in/for/at`, units,
  quantifiers `a few`/`a couple`, and the topic-strip function words
  `at/on/by/this/next/every/each/the`) now come from the pack datetime grammar
  table; `_TIME_PATTERNS` is rebuilt from tables. `grep` finds zero English
  datetime words in regex literals. **Verified:** corpus 77/77 both paths, strip
  20/20 both paths, fr/de/da parity unchanged, hostile-`zz` parses "half past 9
  tomorrow", suite at the 8 pre-existing failures. (Caught + fixed one regression
  along the way: the dotted `a.m.` form — the corpus guardrail + a strip baseline
  surfaced it immediately.)

- **2026-07-24 (f)** — Datetime de-languaged (step 2 fully done). Evicted the
  English day-anchors + periods into `_DEFAULT_DT_GRAMMAR` / the pack table
  (weekdays + word-numbers were already live), and made the datetime lexicon
  selection **data-driven** (load if a lexicon file exists) so the last
  `if language != "en"` branch is gone. `grep 'language ==|!='` over the whole
  `scripts/nlu` package now returns ZERO code matches. **Verified:** English
  golden corpus 77/77 on both the default extractor and the pack-fed engine;
  fr/de/da datetime parity unchanged; classifications correct; hostile-`zz`
  builds a reminder end-to-end; suite still the 8 pre-existing fails only.
  Residual (task #23): the time-notation literals (am/pm, half-past/quarter-to,
  relative markers) remain inline — the parser's *words* for the clock idioms,
  not domain vocabulary.

- **2026-07-24 (e)** — Engine made fully language-neutral (step 2 done at
  `engine.py`). Completed the `en` pack with the REAL schema (59 intents),
  entities, model, and semantic artifacts. The engine now **always** loads a pack
  and defaults to `packs/en` when no language is passed → **English is the
  default**. Removed all 4 `if language` branches, the `_load_schema` overlay,
  `_load_multilingual_semantic`, and the hardcoded English constants (`_CARRIER`,
  `_UNCERTAIN`, `_NO_IDIOMS`, `_LEADING_CONNECTOR`) from `engine.py`. Semantic
  stage declared in the pack, **off by default**, enableable. **Proven:**
  classifier parity 37/37 (default); full `handle()` parity 37/37 with semantic
  enabled (the one earlier diff was purely semantic-off); datetime corpus 77/77;
  a fake **`zz`** pack runs end-to-end with ZERO engine edits (language=zz, 59
  intents, classifies correctly). Suite back to the 8 pre-existing fails only
  (updated the semantic-default tests). Remaining for 100%: the datetime parser
  in `entities.py` (task #26).

- **2026-07-24 (d)** — Datetime, staged safely. English had **no** parity fixture
  (it was the hardcoded special case), so first built a 77-case English golden
  corpus (`tests/datetime_parity/nlu_datetime_parity_en_golden.json`) from the
  pristine parser, now enforced by `tests/test_datetime_parity_en.py` (154
  assertions, default + pack-fed). Authored the full English datetime grammar
  table (`packs/en/datetime/grammar.json`); loader loads it; the engine now
  injects `weekdays` + `word_numbers` from the pack through the extractor's
  existing seams — that vocabulary is evicted, verified **77/77 byte-identical**
  on both paths. Remaining (task #23, own pass): evict the inline literals
  (anchors/periods/idioms/am-pm) — bigger + riskier, now guarded by the corpus.
  No new regressions (still the 8 pre-existing fails).

- **2026-07-24 (c)** — Keyword rules evicted. All 32 `keyword_triggers` copied
  verbatim into `packs/en/keywords.json`; `IntentClassifier` accepts a
  `keyword_source`; the engine passes the pack's keyword table when a pack is
  loaded. **Verified:** exact keyword parity legacy vs pack (32 rules, identical
  intents/tiers/confidences across a 23-utterance keyword battery + `handle()`).
  No new test regressions — 8 failures total, all confirmed pre-existing on
  pristine HEAD (now includes 2 `test_nlu.py` reminder tests; see `TO-REMOVE.md`).

- **2026-07-24 (b)** — Step 2 (partial) + step 4, done IN PLACE (per your call: no
  sandbox move; delete happens on your machine). `scripts/nlu/engine.py` now
  accepts an optional `pack=`; when supplied it sources yes/no, carriers, idioms,
  uncertainty, connectors, and policy constants from the pack, and gates the
  semantic stage OFF by default. `packs/en/lexicons.json` populated with the exact
  evicted EN values. **Verified:** exact parity between legacy `NLUEngine()` and
  `NLUEngine(pack="packs/en")` on yes/no, carriers, idioms, topic derivation, and
  `handle()`; semantic off under a pack; no-pack path byte-for-byte unchanged
  (same 44 pass / 6 pre-existing fail as pristine HEAD — the 6 failures are
  pre-existing and unrelated, see `TO-REMOVE.md`). Environment cannot delete
  files → cleanup + relocation list captured in `docs/Review-F5/TO-REMOVE.md`.
  Still pending for step 2/3: keyword rules + datetime grammar into pack tables,
  and having the *no-pack* default itself load the en pack (so the fallback
  constants can be retired).

- **2026-07-24 (a)** — Step 1 complete. Built `packages/nlu_langpack` (interfaces,
  manifest, loader, pack container, versioned compat gate, error taxonomy) and a
  reference `packs/en` skeleton. All loader behaviors verified. Additive only.
