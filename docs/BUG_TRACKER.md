# Bug Tracker

Defects found during the VoiceIntentKit pack-contract work (2026-08-03). Feature
and contract *requests* live in the iOS team's `PROMPT_FOR_NLU_COMPILER_TEAM.md`;
this file is only for things that are **wrong**.

**Summary:** 14 fixed, 13 open.

| ID | Area | Sev | Summary | Status |
|---|---|---|---|---|
| BUG-001 | runtime/datetime | **High** | English never parsed month names or ordinals | **Fixed** |
| BUG-002 | content | **High** | `datetime.json` missing the tables the engine reads | **Fixed** |
| BUG-003 | runtime/datetime | **High** | Bare ordinal fabricated a datetime (fr/de/da path) | **Fixed** |
| BUG-004 | runtime/datetime | Med | No bare day-of-month or separator support (fr/de/da path) | **Fixed** |
| BUG-005 | runtime/datetime | Med | Impossible date silently re-read against the wrong month | **Fixed** |
| BUG-006 | compiler | **High** | `help_marker_guard` unreachable from the v3 surface | **Fixed** |
| BUG-007 | compiler | Med | `uncertain_confirm` band + cancel message lost in v3 | **Fixed** |
| BUG-008 | typing | Low | 3 new MyPy errors introduced by the C3 rewrite | **Fixed** |
| BUG-009 | build | **High** | 4 `make` targets point at a deleted tree; `make check` broken | Open |
| BUG-010 | build | Med | `make typecheck` and CI MyPy check different trees | Open |
| BUG-011 | build | Low | `make format-check` stricter than CI | Open |
| BUG-012 | eval data | **High** | Two eval fixtures on the dead `Cmd.*` taxonomy score 0% silently | Open |
| BUG-013 | pack | **High** | `head.json` declared in `bundle.json`, absent from the pack | Open |
| BUG-014 | pack | **High** | No MiniLM embedder or vocab despite `embedder_id` | Open |
| BUG-015 | pack | Med | Server temperature shipped device-side with its warning stripped | Open |
| BUG-016 | pack | Med | `temperature_int8` not shipped, but `model_int8.tflite` is | Open |
| BUG-017 | pack | Low | Entity id separator differs between v3 and the root shim | Open |
| BUG-018 | pack | Low | `.DS_Store` shipped inside the pack | Open |
| BUG-019 | pack | Med | 56% of pack bytes are never read by any mobile client | Open |
| BUG-020 | pack | Low | `labels.pkl` (Python pickle) shipped to mobile clients | Open |
| BUG-021 | runtime | Med | Startup integrity check verifies files the engine does not load | Open |
| BUG-022 | compiler | **High** | Validator rejects `.mlmodelc` internals; release pipeline cannot package | **Fixed** |
| BUG-023 | compiler | Med | Schema exemption misses the `_full` device-weights variant | **Fixed** |
| BUG-024 | compiler | Med | Pack shipped a temperature read from a file it did not ship | **Fixed** |
| BUG-025 | compiler | **High** | `open` entity flag dropped by the v3 projection | **Fixed** |
| BUG-026 | compiler | Med | `dynamic_source` does not say which builtin an entity is | **Fixed** |
| BUG-027 | compiler | **High** | Non-portable carrier silently dropped from every bundle | **Fixed** |

---

## Fixed

### BUG-001 — English never parsed month names or ordinals
`entities.py` has two datetime paths. `_build_lex_tables` (fr/de/da) reads
`weekdays`, `months`, `numbers_0_to_31`, `ordinals_1_to_31`; `_build_en_dt_tables`
(English) read none of them, and English ships no `lexicon.json` so the lexicon
path never runs. `june 5` and `the 25th` returned `None`.

Two tests asserted this and were failing before the fix
(`test_slot_value_validation.py::test_legitimate_datetimes_still_resolve`).

**Fix:** `_build_en_dt_tables` now reads `months` and `weekdays` from the same
grammar dict; new section 4a in `extract_datetime` resolves `june 5` / `5 june` /
`the 25th`; `_normalise_word_ordinals` rewrites spelled-out ordinals to digits
*before* the cardinal pass, which used to strand them ("twenty" → "20", "fifth"
orphaned). Weekday matching moved from the hardcoded `_WEEKDAYS` list to a
synonym map, so `mon`/`tue` now work.

**Verified:** the two failing tests pass; 130 golden parity cases unchanged;
`the twenty fifth` → 2026-08-25, `june twenty fifth` → 2027-06-25,
`the fifth of june` → 2027-06-05. `wait a second` still → `None`.

`_WORD_NUMS` deliberately NOT moved to data: it carries 40/50 for clock minutes
("four fifty" → 16:50) which a 0..31 date table cannot express.

### BUG-002 — `datetime.json` missing the tables the engine reads
`language_packs/en/datetime.json` had none of `weekdays`, `months`,
`numbers_0_to_31`, `ordinals_1_to_31`, `clock_hour_markers`, `grammar` — the
exact keys `_build_lex_tables` looks up. Root cause of BUG-001, and it also meant
the compiled pack could not describe English dates to any client.

**Fix:** all six keys added, plus `ordinal_context` (see BUG-003).
`compile_lexicon` copies the file verbatim, so
`lexicons/en.json → datetime_grammar` now carries them.

### BUG-003 — Bare ordinal fabricated a datetime
`_build_lex_tables` merged ordinals into `_lex_number`, and
`_lex_normalise_numbers` rewrote every one to a bare digit unconditionally. A
standalone ordinal then became a number the clock parser claimed.

Measured before/after on a synthetic fr pack:

```
BEFORE   deuxième  ->  2026-08-04T02:00     fabricated
AFTER    deuxième  ->  None
```

This is the invariant `test_non_temporal_word_never_resolves_to_a_datetime`
enforces for English; the lexicon path had no equivalent.

**Fix:** ordinals split out of the cardinal normalisation pass into
`_lex_normalise_ordinals`, gated on an `ordinal_context` marker or an adjacent
month — the mirror of the English gate. A pack with no `ordinal_context` gets no
ordinal rewriting rather than unconditional rewriting: failing toward "do not
invent a date".

Also corrected the English gate, which initially hardcoded `the|of`. Both paths
now read `ordinal_context` from the grammar, so no vocabulary sits in the engine.

### BUG-004 — No bare day-of-month or separator support (fr/de/da)
C3 only matched `<day> <month>` adjacently. `le vingt-cinquième` and
`le cinquième de juin` returned `None`; English resolved both equivalents.

**Fix:** C3 accepts an optional separator drawn from `ordinal_context` and
month-first order; new C3b handles a bare day. `_lex_normalise_ordinals` emits
`25.` rather than `25` — the European ordinal marker, which C3 already tolerated
and which lets C3b distinguish an ordinal day from a cardinal clock hour after
normalisation has erased the word form. `à 14 heures` stays 14:00.

### BUG-005 — Impossible date silently re-read against the wrong month
Found by the BUG-004 test run. `le trente-et-unième de février` returned
**2026-08-31**: C3 correctly rejected Feb 31, then C3b picked up the orphaned
`31.` and resolved it against the current month.

**Fix:** `named_month` flag; C3b is skipped when C3 matched a month. If the user
names a month we honour it or fail — never substitute a different one. Now
returns `None`. English was already correct here by structure.

### BUG-006 — `help_marker_guard` unreachable from the v3 surface
Lived only in the root `nlu_schema.json` shim. A client binding the normalized
surface lost it silently — and without it 11 command intents fire on the question
*about* them ("how do i turn up the volume" changes the volume).

**Fix:** new `runtime/guards.json` (+ `spec/bundle/3.0/guards.schema.json`,
+ validator mapping — unmapped files hard-fail stage 1). Carries `help_marker`
and `polarity`. Kept out of `runtime/routing.json` deliberately: routing decides
what to do when confidence is low, a guard fires regardless of confidence.

Build fails if a redirect names an intent absent from the pack.

### BUG-007 — `uncertain_confirm` band + cancel message lost in v3
`policies.confirmation` carried *which* intents are gated but not *when* the gate
fires, so a runtime could only confirm always or never. `cancel_message` sat in a
policy table where it could never be localised.

**Fix:** `uncertain_confirm_below` / `uncertain_confirm_floor` added to
`policies.thresholds`; `cancel_message` moved to
`capabilities/sys/responses/<lang>.json` as `sys.confirm.cancelled`. The intent
list is not duplicated — `confirmation` already has it.

`semantic_rescue_enabled` needed no action: `compile_cascade` already reads it.

### BUG-008 — 3 new MyPy errors from the C3 rewrite
`day`/`mon` are `int | None` at the `1 <= day <= 31` guard. Non-blocking in CI
but self-inflicted. **Fix:** explicit narrowing. Zero MyPy errors remain in the
rewritten ranges.

### BUG-022 — Validator rejects `.mlmodelc` internals, blocking the release pipeline
`release-pack.yml` fails at `nlu_compiler.build`:

```
[stage 1] ERROR UNMAPPED_FILE models/intent/en/iOS/IntentClassifier.mlmodelc/metadata.json
[stage 1] ERROR UNMAPPED_FILE models/intent/en/iOS/IntentClassifier_full.mlmodelc/metadata.json
FAIL: nlu_compiler.build failed
```

`_bundle_files()` walked every `*.json` in the tree except those under a
`.mlpackage` directory. A `.mlmodelc` contains its own `metadata.json`, which is
CoreML's format and has no bundle-spec schema — so stage 1 flagged it as an
unknown file and `build_bundle` refused to package.

Latent since the exclusion was written; surfaced when the pipeline began
shipping pre-compiled `.mlmodelc` artifacts (ADR-017, "Compile the CoreML models
to .mlmodelc" in `release-pack.yml`). Before that only `.mlpackage` ever reached
a pack, so one suffix was enough.

**Fix:** `_OPAQUE_MODEL_DIR_SUFFIXES = (".mlpackage", ".mlmodelc")` — packaged
and compiled model artifacts are both opaque to the spec walker.

**Verified:** reproduced the exact CI diagnostics by seeding two `.mlmodelc`
dirs into a built bundle and running the pre-fix exclusion, then confirmed 0
errors after. `build_bundle` packages 65 files and `integrity/manifest.sha256`
still covers all four `.mlmodelc` entries, so a compiled model remains signed.
436 passed, ruff clean, both golden example bundles 0 errors.

**Note for later:** `build.py` canonicalises *every* `.json` it packages,
including model-internal ones — so CoreML's `metadata.json` and a `.mlpackage`'s
`Manifest.json` are re-serialised with sorted keys and NFC strings. Semantically
equivalent, and `.mlpackage` has shipped this way already, but it means the
bytes in the pack differ from what the compiler emitted. Worth confirming CoreML
never byte-compares these.

### BUG-023 — Schema exemption misses the `_full` device-weights variant
`stage_1_schemas` exempted device weight blobs with:

```python
if rel in ("nlu_schema.json", "nlu_entities.json") or "intent_classifier_weights.json" in rel:
```

That substring never matches `intent_classifier_weights_full.json`, because the
`_full` suffix sits *before* the extension. The moment the full-vocabulary
weights were added to a pack, stage 1 flagged them `UNMAPPED_FILE` and
`nlu_compiler.build` refused to package — the same failure shape as BUG-022, and
the same root cause: a membership test written against the artifacts that
existed at the time.

Latent until the file first shipped, so it had never run.

**Fix:** `_is_device_weights()` matches on the filename STEM
(`intent_classifier_weights*` + `.json`), which covers both variants and any
future one.

**Verified:** validator 0 errors on a pack containing both weight files; 436
passed; both golden example bundles clean.

### BUG-024 — Pack shipped a temperature read from a file it did not ship
`assemble_pack.py` opened both `ios_weights` and `ios_weights_full` to lift
`temperature_coreml` / `temperature_coreml_full` into `calibration.json`, then
copied neither file into the pack. The vocabulary shipped separately, via
`compile_models`.

So a pack's device temperature and its device vocabulary came from two
independent sources with nothing tying them together. They agreed in
`pack-en-v1.0.28` (both 0.791486), which is why it went unnoticed — but a
regenerated weights file with a refitted temperature would have produced a pack
whose T was fitted for a vocabulary it was not shipping. iOS applies
`softmax(logits / T)` and gates at 0.70, so the failure is a quiet confidence
shift, not an error.

Underlying principle, now stated in both repos: **a device head, its vocabulary
and its temperature are one triple** and must be selected together.

**Fix:** each weights file is copied into the pack in the same loop that reads
its temperature, so the number and the vocabulary it was fitted against cannot
be separated. `ClassifierVariant` on the Swift side binds the same triple.

### BUG-025 — `open` entity flag dropped by the v3 projection
`language_packs/*/nlu_entities.json` marks `remind` as `"open": true` — its value
list is a hint, not a closed set, so a free-text answer is acceptable.
`compile_entities` read `type` and `fuzzy` and nothing else, and
`entities.schema.json` declares `additionalProperties: false` with no `open`
property, so the field was unexpressible even if emitted.

`EntityExtractor.is_open` has always read this flag, so any consumer bound to
the v3 surface saw every entity as closed. The symptom is a slot that will not
fill from free text — "remind me to call the plumber" cannot supply its own
name — and it presents as a re-prompt, never an error.

**Fix:** `open` added to the schema; `compile_entities` emits
`bool(spec.get("open"))` alongside `fuzzy`. Verified: emitted entities validate
against the registry, `remind.open == true`, unknown keys still rejected.

### BUG-026 — `dynamic_source` does not identify the builtin
`sys.date_time` and `sys.number_integer` both emitted
`"dynamic_source": "runtime.builtin"`, which states that the runtime resolves
the entity but not what to resolve it as. A date parser and an integer parser
are not interchangeable, so the only discriminator left was the entity id —
meaning ids carried semantics the format says they do not have, and renaming one
would break date slots on device with no error anywhere.

**Fix:** `runtime.builtin.datetime` / `runtime.builtin.integer`, from a table
keyed by content-tree entity name, with a warning (not an error) for an unmapped
builtin. No schema change — `stableId` already permits dotted segments.

**Still open:** the vocabulary of builtin sources is not published, so a consumer
cannot decide whether an unknown one should be a refusal or a degraded load.
Same question as `required_runtime_features` (iOS ask B5).

### BUG-027 — Non-portable carrier silently dropped from every bundle
`compile_lexicon` runs each carrier through `portable_regex.check_pattern` and
appends failures to `gaps`, which surfaces as one line in the build summary.

`^\s*set(?:\s+up)?\s+(?:an?\s+)?(?:reminder|alarm)\b\s*(?:to|about|for\s+(?!\d))?\s*`
contains a negative lookahead, which the subset forbids. So it has been dropped
from **every bundle ever built**, while `engine.py::_DEFAULT_CARRIERS` kept it.
The engine and the bundles have been resolving the same utterance differently
since format 3.0 shipped: the engine derives the topic "go to the airport" from
"set a reminder to go to the airport", and a bundle-driven runtime stores the
entire sentence as the reminder's name.

This is a divergence between our two release trains. It was reported by the iOS
client, but it is not an iOS defect.

**Fix, in three parts:**

1. The `for` branch is removed rather than the guard rewritten — a leading "for"
   is stripped by `leading_connectors` one step later, so it was redundant.
   Verified identical on 10/11 reminder utterances and better on the other two:
   "set a reminder for 5pm" previously produced a reminder literally named
   "for".
2. That last case was `_build_leading_connector` anchoring on `^(?:...)\s+`,
   which cannot match a connector with nothing after it. Now `(?:\s+|$)`.
3. Changed in `platform.yaml`, `nlu_schema.json` and `_DEFAULT_CARRIERS`
   together, so content and reference cannot drift again — asserted by
   `_DEFAULT_CARRIERS == lexicon.carriers`.

**And the guard that hid it:** a non-portable carrier is now a build failure.
A carrier changes what the runtime extracts; it is not metadata, and demoting it
to a log line is why this survived. Push back if a warning is preferred, but
then it has to reach whoever consumes the bundle, not just whoever runs the
build.

---

## Open

### BUG-009 — `make` targets point at a deleted tree (**High**)
`multilingual/` does not exist. Still referenced by:

```
Makefile:51   typecheck            mypy ... multilingual/*.py
Makefile:77   train-multilingual   multilingual/train_multilingual.py
Makefile:104  export-coreml        multilingual/export_coreml_multilingual.py
Makefile:108  export-coreml-test   multilingual/test/test_coreml_multilingual.py
```

`typecheck` is inside `check`, so **`make check` fails on a clean checkout**. It
does not surface in CI because `ci.yml` calls `mypy` directly. The real exporter
is `packages/buildtime/nlu_export/export_coreml.py`.

### BUG-010 — `make typecheck` and CI check different trees (Med)
Makefile: `multilingual/*.py`. CI: `legacy_research/*.py`. They should agree.

### BUG-011 — `make format-check` stricter than CI (Low)
CI wraps darker in `|| true`; the Makefile does not. Local fails where CI passes.

### BUG-012 — Eval fixtures on the dead taxonomy score 0% silently (**High**)
`language_packs/en/extras/holdout_paraphrase.csv` and `extras/benchmark_250.csv`
still use `Cmd.BatteryLevel`-style labels — **zero overlap** with the model's 57
dotted labels, so both score 0.0000 and look like catastrophic regressions.
`calibration.json` lists `holdout_paraphrase.csv` under `eval_sets_excluded`, i.e.
treats it as a live eval set. Either migrate the labels or retire the files.

### BUG-013 — `head.json` declared but absent (**High**)
`bundle.json` declares `models/semantic_head/shared/head.json`; only the
`.mlpackage` ships. A client validating declared artifacts fails closed.

### BUG-014 — No MiniLM embedder or vocab in the pack (**High**)
`embedder_id: minilm-l6-v2` is declared, no artifact and no vocab file exists.
Masked today only because `cascade.json` disables the semantic stage — meaning
that flag can never be flipped back on.

### BUG-015 — Server temperature shipped device-side, warning stripped (Med)
The repo's `calibration.json` carries an explicit note that the server/ONNX and
device temperatures calibrate different featurizers and **must not be unified**
(Review-F5 B8). The pack's copy keeps `temperature` but drops `_note`,
`provenance` and `temperature_int8` — so a device-facing path ships the server
value with nothing marking it as such. Carry the note, or name the fields
`temperature_server` / `temperature_device`.

### BUG-016 — `temperature_int8` not shipped (Med)
`model_int8.tflite` ships; its temperature (0.649641) does not. Anyone using the
int8 model applies the wrong one.

### BUG-017 — Entity id separator differs between surfaces (Low)
v3 `entities/shared/content.json` uses `sys.date_time`; root
`nlu_entities.json` uses `sys.date-time`.

### BUG-018 — `.DS_Store` shipped in the pack (Low)
Four present. Correctly excluded from `manifest.sha256`, but should not be in the
archive. Add a packaging exclusion.

### BUG-019 — 56% of pack bytes are never read by mobile (Med)
`model.onnx` (1,478,675) + `model.tflite` (1,076,952) + `model_int8.tflite`
(270,920) + `labels.pkl` (1,403) = 2,827,950 of 5,009,150 bytes, downloaded on
every OTA. Needs platform-scoped packs or a slicing mechanism — compiler-side,
since the signature and `checksums_root` must cover the sliced form.

### BUG-020 — `labels.pkl` shipped to mobile clients (Low)
A Python pickle duplicating `labels.json`. No mobile client can read it.

### BUG-021 — Startup integrity check verifies files the engine does not load (Med)
`verify_manifest()` runs at engine startup (`classifier.py:162`) over
`manifest.py`'s `TRACKED_FILES`, which lists only the **legacy flat** paths:

```
models/intent_model.onnx      1,676,051 bytes   2026-07-25
models/intent/en/model.onnx   1,478,675 bytes   2026-08-03   <- what actually loads
```

`model_paths.py` resolves pack → `models/intent/<lang>/` → legacy flat, and
documents the flat tree as a pre-per-language fallback. So the engine loads the
per-language model while the startup guard hashes a different, ~10-day-older
one. The check currently *passes* — `train.py` rewrites `models/manifest.json`
on every run, re-hashing whatever sits at the legacy paths — which is worse than
failing: it certifies a stale artifact and would stay green if the model the
engine really loads were corrupted or swapped. That is the exact failure
`manifest.py`'s own docstring says it exists to catch.

Compounding it, all eight legacy files match `.gitignore` rules
(`models/*.onnx`, `models/*.pkl`, `models/**/*.npz`) but are tracked from before
those rules existed, so "regenerated, not committed" is untrue for them.

**Do not fix with `git rm --cached`** — the files would vanish from a fresh
clone and `verify_manifest()` would raise at startup for everyone. The fix is to
point `TRACKED_FILES` at the artifacts actually resolved (or drive it from
`model_paths.py`), then retire the flat tree deliberately.

---

## Not a bug

- **Two different temperatures in one pack.** Deliberate — server/ONNX vs
  device/pruned featurizers, per Review-F5 B8. An earlier draft of the iOS
  request asked to unify them; withdrawn. Only the labelling is at issue
  (BUG-015).
- **The `en` device-weights retrain.** Reproducible: `export_ios_weights` refits
  `T=0.7915` matching the working tree, RFE prune to 1317 features is
  deterministic, and the accuracy gate passes (test-split 0.91, holdout 0.86).
  Measured +1.22pp on `holdout_honest` (n=1470) and +0.91pp on
  `holdout_leakage_guard` (n=331) vs HEAD.
