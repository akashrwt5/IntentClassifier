# Session handover — NLU quality work (Python + iOS)

**Last updated:** 2026-09-17 — 17 commits here, 2 in `STT`. Everything committed,
**nothing pushed** (see §3).

> **START AT §10.** Everything from 2026-09-17 is there: the memory-entity
> correction (the pack carried 38 memories; the product has 11), the resolved /
> unresolved slot design the owner settled, the iOS featurizer defect, and the
> corpus triage that is the next job. §§1–9 are the earlier session and remain
> accurate except where §10 says otherwise.
**Branch:** `feature/removeing_confirmation_code_APIVersion-Bugtracker-fixes-ReminderFixes-Layer2-QAAuditDataFix`
**iOS repo:** `STT` (VoiceAIKit) — same engagement, separate working tree
**Android:** explicitly out of scope this session.

Read this before touching anything. It records what we are trying to do, what
is already done and proven, what is broken, and what is next — with the
measurement that backs each claim.

---

## 0. Rules of engagement (owner-stated, still in force)

1. Do not switch branches or start work on a new branch without the owner's
   confirmation.
2. Ask before every decision — commits, new branches, anything irreversible.
3. **No claim from assumption or speculation.** Believe the code, not the
   docs. Docs in this repo are frequently stale or incomplete; if a doc says
   something, verify it against source before acting on it.
4. Work on Python and iOS only. Do not modify Android.
5. When context runs low (10–15% left), ask whether to write a fresh handover.

A corollary that cost real time this session: **a plausible-sounding mechanism
is not evidence.** Twice a conclusion had to be retracted because it was
reasoned rather than measured (see §7.3 and §8.2). Measure first.

---

## 1. What we are trying to do

Close the wrong-action and misrouting gap in the on-device NLU, using the QA
phrase reports as the evidence base. Four analysis documents were written in
the iOS repo under `docs/` and one consolidated plan:

- `QADataBasedDecision_SingleTokenCollapse.md`
- `QADataBasedDecision_Help.md`
- `QADataBasedDecision_Cmd_Reminders_Fallback.md`
- `NLU-Quality-Plan.md` (priorities P0–P8 + architecture track)

Everything below is execution against that plan.

---

## 2. The build chain — read this first

This was the single biggest operational discovery of the session, and it
invalidated a commit that looked complete.

```
content/capabilities/**/*.yaml  +  language_packs/en/platform.yaml
        |
        |  python -m nlu_compiler.content_source assemble      <-- EASY TO FORGET
        v
language_packs/en/nlu_schema.json          (COMPILED, git-tracked)
        |
        |  python -m nlu_compiler.content_bundle --lang en --out dist/bundle-en \
        |        --report dist/report_card.json
        v
dist/bundle-en/   (nlu_schema.json, lexicons/en.json, models/, runtime/, ...)
```

`content_bundle.py:1030` reads `language_packs/en/nlu_schema.json`. It does
**not** read `platform.yaml` or `content/capabilities/`. So editing a source
file and building a bundle changes nothing until `assemble` runs.

**Rule: touch `platform.yaml` or `content/capabilities/**` → run `assemble` →
commit BOTH the source and `nlu_schema.json`.**

Drift check (read-only, safe, and also the CI guard via
`tests/test_content_source.py`):

```bash
PYTHONPATH=packages/buildtime python3 -m nlu_compiler.content_source check
# "in sync"  or  "DRIFT: source tree and compiled schema disagree"
```

### How this bit us

`nlu_schema.json` had not been regenerated since **4 Sep**. Commit `f21df3e3`
(14 Sep, 49 capability yamls — "empty command fulfillments, restandardise help
copy") never ran `assemble`; its message even says *"No schema … is touched"*,
which is exactly the bug. Commit `cce284e7` (help-marker widening) was dead for
the same reason. Both are live now — the owner ran `assemble` on 16 Sep and
`check` reports **in sync**.

---

## 3. Current repo state

### Commits on this branch (all local — NOTHING IS PUSHED)

Neither branch exists on its remote. The Cowork VM has no git credentials
(no helper, no `gh`, no `GITHUB_TOKEN`) — reads work, pushes cannot. Push from
the owner's own terminal, where the auth lives. First push creates the remote
branch, and note the size: this branch is **344 commits ahead of
`origin/main`** and `STT` is **200** — most of that predates this work and
arrived with the branch point, so a PR diff will be large.

**IntentClassifier — 14 commits**

| sha | what |
|---|---|
| `cce284e7` | help-marker `how` clause drops the pronoun restriction |
| `bb0838d0` | +30 rows: `Cmd.MemoryChange` plurals, audiologist asks |
| `2b0c0430` | compiler carries `cancel_cues` + the lemma table into the bundle |
| `fe56d953` | pack-own the cancellation cues (VIK-068) |
| `0aed80cd` | regenerate the compiled schema — the `assemble` step (see §2) |
| `af35ed46` | three export sites normalise text the way the vectorizer was fitted |
| `05049adc` | plural folding, fitted by ablation (`fit_lemmas.py`) |
| `e51bf33b` | refit en artifacts on the folded featurizer |
| `84fa18a0` | `Cmd.MemoryChange` learns the hearing-aid carriers |
| `64c59b91` | the corpus learns what a timer is (Bug 2, data half) |
| `ef5bea4c` | strip the timer carrier — a timer is not named "set a timer" |
| `6ac0e904` | a bare duration answers "when?", but only there |
| `f9778942` | seconds are a relative unit; one `_UNIT_DELTA` |
| `834c9548` | **bare-value guard** — a bare entity value is not a request |
| `33213eeb` | unpoison "put it on", teach "go to" |
| `328b732e` | this handover |

**STT (VoiceAIKit) — 2 commits**

| sha | what |
|---|---|
| `c445488` | cancel cues + bare-value guard, both from the pack |
| `86d705d` | docs: P4(b) reversal recorded |

`c445488` carries both engine changes because they touch the same regions of
`NLUEngine` and `PackEngineFactory`; splitting would have meant rewriting a
diff whose Xcode build was already verified. The message covers both under
separate headings.

### Do not commit

`STT`: `VoiceAIKit/docs/VoiceAIKit_Architecture_Review.md` (owner's deletion).
`IntentClassifier`: `scratch_*.py`, `models.bak-*/`, `docs/help-intent-misrouting.md`,
`scripts/analysis/*`, `tests/fixtures/help_phrases_en.json`.


## 4. Work completed and verified

### 4.1 Help-marker widening (`cce284e7`)

Dropped the pronoun restriction: `how do i` → `how do`, `how can i` → `how can`.

Measured on the **full corpus** (train.csv + holdout_honest.csv, 9,930 distinct
phrases), old compiled schema vs new:

```
OLD: 9548/9930 = 96.15%
NEW: 9550/9930 = 96.17%    delta +2

rows whose prediction changed: 2  (both fixes, zero regressions)
  'how do turn the volume down'   fallback -> Help_Volume
  'how do turn the volume up'     fallback -> Help_Volume
```

Regex level: 30 phrases newly match, **0 matches lost**. Honest holdout
unchanged at 90.68%.

### 4.2 VIK-068 — cancellation cues are now pack-owned

`cancel_cues` moved from a Python code default (`engine.py:242`) into
`platform.yaml`, through `assemble`, into `nlu_schema.json` and
`dist/bundle-en/lexicons/en.json`.

**Python behaviour did not change** — the same 8 words were already the code
default. Verified: identical pass on 12 cancel/non-cancel cases before and
after. What changed is ownership, proven by substituting the pack's list:

```
pack = [cancel, stop, never mind, ... , abort]  -> "abort" cancels, "scrap that" does not
pack = ["scrap that"]                           -> "abort" does NOT cancel, "scrap that" does
```

If the engine were silently using its hardcoded list, `abort` would still
cancel in the second case. It does not. The pack drives it.

The value is on **iOS**, which had no cancellation at all.

#### Why a separate `cancel_cues` list instead of reusing `negative`

`engine.py:862` `_is_cancel` has two branches:

```python
if any(cue in t for cue in self._cancel_cues):          # explicit cue, any length
    return True
return self._yes_no(t) is False and len(t.split()) <= 2  # bare refusal, <= 2 tokens
```

Branch 2 already uses `negative`. Measured with `cancel_cues=[]`: `cancel`,
`stop`, `never mind`, `nevermind`, `forget it` still cancel; **`quit`, `abort`,
`forget about it`, `cancel the reminder` do not**. So the separate list buys
exactly those.

Measured the reverse — using `negative` directly as the cue list breaks 3 of 11
realistic slot replies:

```
'remind me to not forget the keys'  -> CANCELS (the reminder body kills the flow)
"don't make it 5, make it 6"        -> CANCELS (a correction, not a cancel)
'hold on let me check the time'     -> CANCELS
```

`negative` contains `skip`, `leave message`, `leave it`, `not now`, `hold on` —
refusals that must not cancel. Two lists is correct.

Note: `engine.py:1021` guards with `_is_cancel(text) and not answers_prompt`, so
a valid slot answer overrides a cancel ("no, tomorrow at 5" survives).
**iOS does not have `answers_prompt` (VIK-067).** The Swift `isCancel` covers
the common case with the ≤2-token purity guard; exact parity is still open.

### 4.3 Training data (`bb0838d0`) + retrain

30 rows added (22 `Cmd.MemoryChange` plural/target-less forms, 8
`Help_RemoteProgramming`). Verified 0 duplicates and 0 holdout leakage before
committing.

Owner retrained (16 Sep 14:49). Measured against the previous model:

```
OLD model (7 Sep)  : 1333/1470 = 90.68%
NEW model (16 Sep) : 1336/1470 = 90.88%    +3 gained, 0 lost
  'music memory'                             fallback -> Cmd.MemoryChange
  'switch over to my gym preset'             fallback -> Cmd.MemoryChange
  'help connecting to my provider remotely'  fallback -> Help_HearingCareAnywhereConnect
```

### 4.4 Calibration refit

`make calibrate` is the **wrong target** — it runs `calibrate_languages.py`
which writes `config/calibration.json`, a file whose own header says
*"ADVISORY REPORT ONLY — NOT runtime config. Nothing in
packages/runtime/nlu_engine/ reads this file."* It also deletes the fr/de/da
blocks that the file says exist nowhere else. Owner restored it with
`git checkout`.

The runtime artifact is `models/intent/en/calibration.json`, read by
`classifier.py::_load_temperature` (precedence documented there), written by:

```bash
PYTHONPATH=packages/buildtime python3 -m nlu_training.fit_calibration --lang en --write
```

Done. Provenance now matches the current train.csv sha exactly:

```
temperature   : 0.671457 -> 0.68127
ECE           : 0.1043 uncalibrated -> 0.0122
n_samples     : 8430   source_sha256 matches language_packs/en/train.csv
```

**The refit costs −3 on the holdout (90.88% → 90.68%). That is correct, not a
regression.** Temperature is rank-preserving; it cannot change which intent
wins. A higher T lowers confidence, so three rows that were firing on
*overconfident* scores fell below the 0.70 gate. Do not "fix" this by reverting
T.

What it does imply: `confidence_threshold: 0.7` (`platform.yaml:67`) was set
against the old T and should be re-examined.
`fit_oov_guard`, `fit_decision_ladder`, `fit_confirm_gate` are **report-only**
(no `write_text`) — safe to run, they print recommendations for a human.
`fit_slot_thresholds` is the only one that writes.

---

## 5. Current measurement baselines

**Install `onnx` before measuring anything.** Without it
`OrtIntentBackend.unigram_vocabulary()` returns an empty set and the
out-of-vocabulary guard disables itself SILENTLY — `inference.py` catches every
exception there. Numbers taken without it read about a point high on the
holdout and eleven points high on the leakage guard. Every figure below is with
the guard live. An earlier version of this file quoted 1333/251; those were
measured with it off and were wrong.

| set | rows | current | notes |
|---|---|---|---|
| `holdout_honest.csv` | 1,470 | **1337** | full engine, FIRE 0.70 |
| `holdout_leakage_guard.csv` | 331 | **245** | |
| out-of-scope fires | 195 | **6** | the metric this work protects; never moved |
| raw model argmax | 1,470 | **1355** | classifier only, no gates — this is the number a `pipeline.predict` script reports, ~2 points above the engine |
| P4 command phrases | 11 | **10** | |
| explicit verb x memory name | 342 | **340** | |
| bare memory names firing `Cmd.MemoryChange` | 47 | **0** | guaranteed by the guard, not by the model |

Scoring note: the engine returns intent `GENAI` for the fallback route
(`engine.py:1060`, `:1334`). Map it to `Default Fallback Intent` before
comparing against gold, or accuracy reads ~12 points low.

**Back up `models/` before any retrain** — it is gitignored, so a retrain
destroys the ability to compare against the previous model:

```bash
cp -a models models.bak-$(date +%Y%m%d-%H%M)
```

### Progress across this branch

```
                  start   folding   hearing-aid   timer   carriers
raw model          1351     1354       1354       1355     1355
engine holdout     1332     1332       1333       1334     1337
leakage guard       243      243        243        243      245
out-of-scope          6        6          6          6        6
P4 phrases         0/11     7/11       9/11       9/11    10/11
```

## 6. P4 — root cause (kept for the analysis; see §8 for what is left)

11 of the 30 rows we added are still wrong:

```
MISS 'change memories'                -> fallback
MISS 'change programs'                -> fallback
MISS 'change my programs'             -> fallback
MISS 'switch memories'                -> fallback
MISS 'put it on custom'               -> fallback
MISS 'change my hearing aid memories' -> Help_ChangingMemories
... (11 total)
```

This is **not** a capacity problem. The model usually ranks the right intent
first — it just never clears 0.70:

```
change memories      Help_ChangingMemories=0.553, Cmd.MemoryChange=0.397   rule: none
change programs      Help_ChangingMemories=0.621, Cmd.MemoryChange=0.344   rule: none
change my programs   Cmd.MemoryChange=0.614,      Help_ChangingMemories=0.351
change the memories  Cmd.MemoryChange=0.662,      Help_ChangingMemories=0.198
                                                  confidence_threshold = 0.70
```

### Root cause, measured

The featurizer has **no stemmer and no lemmatizer** (`train.py:183`:
`TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)`, stock
defaults). `memory` and `memories` are two unrelated vocabulary slots, and the
counts are lopsided:

| token | `Cmd.MemoryChange` | `Help_ChangingMemories` | model learns |
|---|---|---|---|
| `memory` | 617 | 27 | "memory = command" |
| `program` | 623 | 23 | "program = command" |
| `memories` | 7 | **33** | "memories = help" |
| `programs` | 6 | **44** | "programs = help" |

Adding 22 rows moved the plural side from ~0 to 6–7. Help still has 33–44.
**Plurality itself became the intent predictor.**

Useful related fact: of 134 `Help_ChangingMemories` rows, **0** start without an
interrogative (`how/what/where/can/is/why`). "Bare imperative = command" is a
clean, data-backed discriminator if a rule is ever wanted.

Separate problem, do not conflate: `Help_ChangingMemories` (134 rows) and
`Help_MemoryOptions` (83 rows) overlap heavily, and
`help_marker_guard.pairs` maps `Cmd.MemoryChange -> Help_MemoryOptions` while
these phrases' gold is `Help_ChangingMemories`.

---

## 7. Plural normalisation ("Option B") — SHIPPED. Kept for the method,
which is the reusable part: two cheap proxies were tried and measured
wrong before ablation settled it.

### 7.1 The hook already exists — do not invent a new one

`packages/runtime/nlu_engine/text_norm.py` (99 lines). Its own docstring states
the contract:

> `normalize_text()` MUST be applied at every entry point that feeds the
> English model: training (`nlu_training/train.py`), Python inference
> (`nlu_engine.classifier`), **and on-device / Swift (port this exact logic for
> iOS/Android parity)**.

It is already pack-data driven: `_DEFAULT_CONTRACTIONS` loads
`language_packs/en/contractions.json`, `normalize_text(text, contractions=...)`
accepts an override, `content_bundle.py:580-582` ships it into
`lexicons/en.json`. `_DEFAULT_` is a CI convention for an overridable data
table (`scripts/ci/check_language_neutral.py`, check 2).

**Option B = add a `lemmas` table on exactly this path.** Order inside
`normalize_text`: lowercase → apostrophes → contractions → drop residual
apostrophes → **lemmas** → collapse whitespace. Must stay idempotent.

Callers that already apply it (so they need no change): `train.py:100,119,225`;
`classifier.py:255,339`; `fit_calibration`, `fit_oov_guard`,
`fit_confirm_gate`, `fit_slot_thresholds`, `fit_decision_ladder`;
`evaluate.py:70-77` (which already passes `contractions=` and will need
`lemmas=`).

### 7.2 Export surface — audited, and the earlier summary of it was wrong

There are **four** Stage-2 artifacts, in two tokenizer situations:

| artifact | built by | tokenizer |
|---|---|---|
| `model.onnx` | `train.py:257` `convert_sklearn(pipeline)` | **baked into the graph** (skl2onnx TF-IDF subgraph, string input) |
| `intent_classifier_weights.json` | `export_ios_weights.py` | `vocab`+`idf` only; tokenizer **re-implemented in Swift** (`_swift_tokenize:126` is the reference, `PackTFIDFVectorizer.tokenize():105` ships) |
| `IntentClassifier.mlpackage` FP16/FP32 | `export_coreml.py` | **none** — input is `tfidf_vector`, a dense array. Built from weights.json, not ONNX: `export_coreml.py:24-30` — *"coremltools cannot convert ONNX string tensors"* |
| TFLite fp32 + int8 | `export_tflite.py` | **none** — input is the dense vector from `tfidf.transform()` |

**All four descend from the one `TfidfVectorizer` fitted on `normalize_text`-ed
text.** So changing the normalisation requires **zero changes to any export or
conversion code** — the vocabulary changes automatically. What must change is
every place that feeds *text* to a fitted artifact.

Incidental findings: `make export-coreml` points at
`multilingual/export_coreml_multilingual.py`, **which does not exist** — the
target is stale. `export_onnx_to_coreml.py` exists but is not wired anywhere.

### 7.3 Three defects found by that audit — fix these regardless of Option B

1. **`export_ios_weights.py:57` and `:207`** — `str.lower().str.strip()` only;
   the file never imports `text_norm`. But the vocabulary was built from
   normalised text. So the shipped device temperature (`T=0.822109`) is fit on
   text that mis-tokenises against its own vocab — every apostrophe row.
   **This is a live defect today.**
2. **`export_tflite.py:135`** — raw text straight into `tfidf.transform()`.
   Same class of bug for the int8 temperature.
3. **iOS has no `normalize_text` port.** `PackLexicon.swift:41,66` decodes
   `contractions` from the pack and **nothing in `VoiceAIKit/Sources/` ever
   uses it.** `PackTFIDFVectorizer.tokenize()` takes raw text.

`export_weights.py:21` has the same bug but is legacy and unwired (only
referenced from a comment and `legacy_research/`).

Adding plural folding without fixing 1–3 makes them worse: the exports would
see `programs` while the new vocabulary contains only `program`.

### 7.4 Also: the keyword layer has the same singular/plural gap, and
normalisation will NOT fix it

`classifier.py:286-287`:

```python
kw_intent = self._keyword_match(text)      # RAW text
p = self._model_distribution(text)         # normalize_text(text)
```

`nlu_schema.json:147` keyword regex:
`\b(party|outdoor|restaurant|meeting|tv|music|quiet|calm|noise)\b.{0,10}\bmode\b`
— all singular, so "meetings mode" misses today. Separate fix, separate track.

### 7.5 What the map should be — and the measurement that is blocking it

The owner's objection to a hand-written table is correct: a coverage table only
covers what someone typed into it, so `meeting`/`meetings` would not work
unless someone remembered to add it. Resolution: **the map is generated by
`train.py`, not maintained by hand** — same status as `vocab` and `idf`.

Generation recipe, all four steps measured:

1. **Suffix rule** — `-ies→y`, `-es→""`, `-s→""` with length guards.
   Alone it is destructive: of 2,064 distinct corpus tokens it rewrites 250,
   and **149 of those invent a non-word** — `this→thi` (244 occurrences),
   `tinnitus→tinnitu` (193), `does→doe` (185), `minutes→minut` (55).
2. **Vocabulary guard** — fold only when the stem actually occurs in the
   corpus. Kills every case in (1). 250 → 101 candidates.
   `meetings→meeting` appears here automatically, with no table entry.
3. **Ratio guard** — fold only when the plural is clearly the minority form of
   the same concept (`plural share < 30%`). This alone excludes
   `aids` (52%), `settings` (38%), `adjustments` (56%), `aerobics` (88%),
   `outdoors` (93%). 101 → ~58.
4. **Exception list** — small, hand-written, for English words that end in `-s`
   but are already singular: `news, aerobics, outdoors, basics, diagnostics,
   metrics, athletics, gymnastics, headphones, glasses, series, species, means,
   lens`. This is an **exception** list, not a coverage list — a word missing
   from it is still handled correctly by the rule.

A blocklist derived from intent ids / actions / keyword regexes was tried and
**rejected**: it blocked `memories`, the headline case. The premise was wrong —
the model fits on text, never on labels (`train.py:196`), and the keyword layer
matches raw text, so neither is affected by normalisation.

The generated map is at `~/verify/lemmas_B.json` on the session VM (58 entries);
regenerate it rather than trusting that path.

#### THE BLOCKER — read before implementing

Every measurement available today is **inference-only**: the model was trained
without folding, so folding at inference takes away signal it has already
learned. Under that test:

| map | holdout (base 1333) | leakage guard (base 251) | P4 rows |
|---|---|---|---|
| 3-word table (`memories`,`programs`,`environments`) | 1334 (**+1**) | 251 (0) | 8/11 |
| B map, 58 folds | 1330 (**−3**) | 249 (−2) | 8/11 |
| full rule, 101 folds | 1318 (**−15**) | — | — |

Two things this does and does not establish:

- It does **not** settle the question. A symmetric retrain (folding applied at
  train time too) could flip the sign, and cannot be measured in the Cowork VM
  because **sklearn is not installed there** — training must run on the owner's
  Mac.
- It does establish that **the extra 55 folds buy nothing on P4** (8/11 either
  way) while costing measurably elsewhere, and that the damage scales with the
  number of folds. All current evidence points away from the wide map.

Typical B losses (`reminders` is a word the model learned in its own right):

```
'what are reminders'             Help_Reminder  -> fallback
'where do i see my reminders'    Help_Reminder  -> fallback
'can i set recurring reminders'  Help_Reminder  -> fallback
```

`environment` is **not** a plural problem at all: 4 rows spread over 4
different intents, `environments` 0 rows, not in the device vocab. It needs a
labelling decision and training data, not normalisation.

### 7.6 Agreed approach (owner asked for B; scope was then split)

**Part 1 — DONE and committed (`05049adc`). Landed behaviour-neutral: with
`lemmas.json` empty every metric was bit-identical; the table was filled by
ablation afterwards.**

Verified with the table empty: `holdout_honest` **1333/1470 = 90.68%** and
`leakage_guard` **251/331 = 75.83%** — both exactly the §5 baseline.
`content_source check` still reports **in sync**. The bundle now carries
`lexicons/en.json -> "lemmas": {}`.

| file | change |
|---|---|
| `language_packs/en/lemmas.json` | NEW, generated artifact, currently `{}` |
| `language_packs/en/lemma_exceptions.json` | NEW, 14 hand-written `-s`-final singulars |
| `nlu_engine/text_norm.py` | `_DEFAULT_LEMMAS`, `_lemma_re`, `lemmas=` param, fold step after apostrophe removal |
| `nlu_compiler/content_bundle.py` | ships `lemmas.json` into `lexicons/<lang>.json` beside `contractions` |
| `nlu_training/train.py` | `--lemmas {keep,generate,off}`, `--lemma-max-plural-share`, `derive_lemmas()`, and a run-local `_norm()` used by all three data paths |
| `nlu_training/evaluate.py` | **bug fix** + `lemmas=` (see below) |
| `nlu_export/export_ios_weights.py` | `_featurize_text()` replaces `.str.lower().str.strip()` at both sites; `LANG` global added |
| `nlu_export/export_tflite.py` | normalises before `tfidf.transform()` |
| `tests/test_text_norm.py` | NEW, 17 tests pinning the transform and the shipped tables |

**Fourth defect found while implementing:** `evaluate.py:_normalize` read
`.get("contractions", {})` on `contractions.json`, which is a FLAT map with no
such key — so it always got `{}` and evaluation ran with contraction expansion
**disabled** while training ran with it on. Fixed; `"what's up"` now normalises
to `"what is up"` there as it does in the trainer.

`--lemmas` defaults to **keep** (use the file as it stands) so every run is
reproducible and the table is reviewable in git. `generate` derives and
overwrites; `off` ignores it. The generator reads **train.csv only** — never the
holdout — so the eval set cannot leak into the featurizer.

Generator output today, run against the shipped code (`--lemmas generate` would
write this): **54 entries.**
`memories->memory`, `programs->program`, `meetings->meeting`,
`batteries->battery`, `restaurants->restaurant`, `reminders->reminder`.
Correctly NOT folded: `aids`, `settings` (ratio guard), `aerobics`, `news`,
`outdoors` (exception list), `this`, `does`, `tinnitus` (vocabulary guard).

Remaining for Part 1: nothing. Part 2 is the three retrains.

~~**Part 1 — plumbing. Safe, reversible, behaviour-neutral while `lemmas.json` is empty.**~~

- `language_packs/en/lemmas.json` (generated artifact)
- `text_norm.py`: `_DEFAULT_LEMMAS` + `lemmas=` parameter + the fold step,
  mirroring `_contraction_re` (lru_cache, longest-first alternation)
- `content_bundle.py`: ship `lex["lemmas"]` next to `contractions` (~line 583)
- `evaluate.py:77`: pass `lemmas=` alongside `contractions=`
- `train.py`: generate the map (rule + vocab guard + ratio guard + exception
  list) and write `lemmas.json`
- fix the three defects in §7.3 (Python side: items 1 and 2)

**Part 2 — the map's contents, decided by measurement, not opinion.** Three
retrains on the owner's Mac, ~7 minutes each:

| run | `lemmas.json` | proves |
|---|---|---|
| 1 | empty | plumbing changed nothing — holdout must read exactly 1333 |
| 2 | 3 words | the small, safe option |
| 3 | B's 58-word map | whether the wide map pays off after a symmetric retrain |

Ship whichever wins. **Hard gate: nothing ships that scores below the 1333
holdout baseline.** If run 3 does not beat run 2, only the 3 words ship and the
generation machinery stays for later.

**Part 3 — iOS, only after Python is green.** Port `normalize_text` to Swift
reading `PackLexicon.contractions` and the new `PackLexicon.lemmas` (no
hardcoded tables); call it *before* `PackTFIDFVectorizer`, **not inside
`tokenize()`** — that function mirrors sklearn's token pattern and must stay
as it is. Then run the conformance fixtures in Swift CI.

---

## 8. Pending, in priority order

Everything through the bare-value guard, the carrier fix and the iOS port is
COMMITTED. Nothing is pushed — see §3. What is left, in the order it should be
done:

1. **Push both branches, then build a pack and put it on a device.** Both
   guards are code-complete on Python and iOS but INERT on any pack built
   before `fe56d953` / `834c9548` — they decode with `decodeIfPresent ?? []`.
   Nothing is really verified until a device runs a pack from this branch.
   `dist/bundle-en` is current.
2. **Android** — both guards are absent there. Until ported, a bare memory name
   still switches programs on Android while Python and iOS refuse it. Out of
   scope by owner directive; recorded so it is not forgotten.
3. **Conformance fixtures** cannot be regenerated: the generator needs de/fr/da
   models and this repo has only `en` (`models/intent/` — the backups too).
   Native CI is validating against 3 Sep expectations. Needs either an en-only
   mode or the other languages trained.
4. **Bug 2, structural half.** The OOV guard's ratio is `unknown / total`, so
   utterance LENGTH decides whether an unknown word is refused. The `timer` data
   fix closed the symptom, not the mechanism. Two candidates, both need
   measuring: count unknown CONTENT tokens instead of a ratio, or take the ratio
   over content words so padding cannot dilute it.
5. **Carrier duplication.** `_DEFAULT_CARRIERS` in `engine.py` and the pack's
   `lexicon.carriers` are two copies of one list, currently identical.
   `_build_carrier_patterns` looks for `carrier_phrases` in
   `language_packs/<lang>/nlu_lexicon.json`, a key and a file no build stage
   produces, so Python has always used its constant while the device reads the
   pack. Making the engine read the pack breaks
   `tests/test_hostile_language.py`, which pins the fallback contract — that
   reconciliation is its own change. **Until then the two lists must be edited
   together.**
6. **`tomorrow at 3` resolves to 03:00, not 15:00.**
7. **`month` is not a relative unit** — `timedelta` has no months, so it needs
   calendar arithmetic. `in 2 months` returns nothing.
8. **Three timer phrasings have no intent to carry them**: `cancel the timer`,
   `stop the timer`, `how long is left on my timer`. `reminders.complete` means
   completed, not cancelled, and there is no query intent. Product call.
9. **`mute the tv`** falls back; every other `mute X` form works.
10. **Two tests fail, both product decisions, neither a regression:**
    `test_confirmation_branches[Cmd.SendMessage]` expects a spoken fulfillment
    that `f21df3e3` deliberately emptied; `test_legacy_label_compat` expects
    `how do i change my memories` to fall back, and the engine now answers
    `Help_MemoryOptions` (the corpus labels every near variant
    `Help_ChangingMemories`, so a help card is right and the sibling is wrong).
11. **P4 remainder** — `change my hearing aid memories` at 0.624 (right label,
    under the gate); `Help_ChangingMemories` ↔ `Help_MemoryOptions` overlap;
    `help_marker_guard.pairs` maps `Cmd.MemoryChange` → `Help_MemoryOptions`
    while that family's gold is `Help_ChangingMemories`.
12. **P4(c)** — 18 rows reading as other commands (`mute`, `decrease volume`).
    The engine is right; these are QA labels. Needs a decision, not a fix.

### Settled decisions — do not reopen without the owner

- **S1:** a fallback is better than a wrong help card. Losses that land on
  fallback are the cheap kind; losses that fire a wrong action are not.
- **S2:** leave the confidence re-read behaviour as it is.
- **Q1:** `change program(s)` / `change memor(y|ies)` are commands unless the
  user says "help".
- **Q2:** when the memory is ambiguous, ask which memory, then pass the intent
  **and** the memory name to the app.
- **Q4:** whatever memory name is matched, hand it to the app.
- **Q5:** audiologist / hearing-care-professional asks route to
  `Help_RemoteProgramming`.
- **opt3a** (a `features>=2` variant) was measured and **rejected**: 4 good vs
  8 bad under the owner's cost rule.
- **P1b** (removing `\w+\s+help\b` from the marker) was measured and
  **rejected**: it looked free on the iOS QA corpus but costs 15 rows on
  holdout+train, and would make `transcribe help` start transcribing.

---

## 9. Environment notes

- **sklearn / scipy are not installed in the Cowork VM.** Training, calibration
  fitting and anything that unpickles `pipeline.pkl` must run on the owner's
  Mac. The ONNX runtime path works in the VM, so engine evaluation, holdout
  scoring and all the measurements in this document can be reproduced there.
- **`dist/` and `models/**/*.pkl` are gitignored.**
- Pre-commit cannot run in the VM (see §3).
- `.git/index.lock` has been left stale twice; clearing it needs
  `device_request_delete_permission` for the repo folder.

### Commands used throughout

```bash
# drift check (read-only)
PYTHONPATH=packages/buildtime python3 -m nlu_compiler.content_source check

# source -> compiled schema
PYTHONPATH=packages/buildtime python3 -m nlu_compiler.content_source assemble

# compiled schema + model -> bundle
PYTHONPATH=packages/buildtime python3 -m nlu_compiler.content_bundle \
    --lang en --out dist/bundle-en --report dist/report_card.json

# train / calibrate  (Mac only)
make train
PYTHONPATH=packages/buildtime python3 -m nlu_training.fit_calibration --lang en --write

# exports
PYTHONPATH=packages/buildtime python3 -m nlu_export.export_ios_weights --lang en
PYTHONPATH=packages/buildtime python3 -m nlu_export.export_coreml --lang en
make export-tflite LANG=en

# cross-platform golden fixtures — regenerate after ANY dialogue/policy/model change
PYTHONPATH=packages/buildtime:packages/runtime \
    python3 -m nlu_training.generate_conformance_fixtures
```

The conformance fixtures under `tests/parity/engine_conformance/` are the
drift-prevention mechanism for native iOS/Android (ADR-011, the Rust core was
removed by owner directive). **A diff there in review IS the cross-platform
behaviour change** — read it, do not rubber-stamp it.

---

# 10. 2026-09-17 — the memory entity was wrong, and what follows from it

Read this section before §§4–8; where they disagree, this is current.

## 10.1 The finding that reframes P4

**The pack declared 38 memories. The product has 11.** The owner supplied the
real list:

```
Personal  Restaurant  Outdoors  Meeting  Crowd  Car
Auditorium  Music  Television  Mute  Custom
```

Everything else in `language_packs/en/nlu_entities.json` was invented:

```
Telephone Automatic Everyday Golf Gym Home Kitchen Master Noise Office
Pub Quiet School Speech Theater Tinnitus Train Work Worship Mask
one two three four   Custom One..Four
```

Provenance, checked rather than assumed: the 38-value list arrives in the
FIRST commit that touches the file (`b6c2e830 "Refactored the language_pack"`).
Nobody in this repo added it — it came with the Dialogflow-era import and was
never validated against the device.

**Consequence, measured:** of 1,984 `Cmd.MemoryChange` rows in the corpus,
**989 name a memory that does not exist**, and the phantom names are the
everyday words an always-on microphone hears all day — gym, home, kitchen,
office, work, quiet, train, school. That is the root cause of the precision
complaint that opened this thread, not the guards and not the enum mechanism.

**This also indicts `33213eeb` (mine).** That commit added 20 `go to <memory>`
and 30 `put it on <memory>` rows to strengthen two carriers, with zero
counter-examples, and reported "out-of-scope fires 6/195 unchanged" as
evidence. The claim was true *of the labelled corpus* — which contains no
conversational `go to <memory>` rows at all, so it could not have caught this.
Measured against the pre-change pack:

```
                            v1.0.54 (before)          v1.0.56 (after)
shall we go to the gym   Default Fallback 0.602 -> Cmd.MemoryChange 0.850
go to the theater        Cmd.MemoryChange 0.621 -> Cmd.MemoryChange 0.967
                         (under the gate)          (fires)
```

`shall we go to the gym` used to fall back correctly. Strengthening a carrier
without measuring what it costs on the precision side is what broke it. **The
durable fix is not another guard — it is a measurement for this failure class**
(see §10.6), because the reason it shipped is that nothing measured it.

## 10.2 Owner decisions settled on 2026-09-17

1. **The microphone stays open for the whole session**, until an action is
   needed. So the engine hears conversation, not just commands, and a weak
   carrier is not enough to authorise a state change.
2. **Numbered memories are invalid.** `memory one`, `program four`,
   `switch to number two` — the product has names, not numbers.
3. **`Custom` is a generic label; the USER names their custom memory.** A
   build-time list can therefore never hold the real name, which is what the
   passthrough path below exists for.
4. **The 11 names are identical across all hearing-aid models**, so no
   host-supplied runtime entity list is needed. (This was on the table; it is
   now out of scope.)
5. **Option 2 — carrier strength decides first-turn trust** (see §10.3).

## 10.3 The slot contract the owner settled

The rule, in full. The discriminator is *did the engine ask?*, refined by how
explicit the carrier is:

```
turn 1, STRONG carrier + unknown name   -> FULFILL, passthrough to host
        (switch to / change to / set it to — no conversational reading)
turn 1, WEAK carrier + unknown name     -> FALLBACK
        (go to / put it on — everyday speech; "put it on the shelf")
turn 1, any carrier + KNOWN name        -> FULFILL, resolved
turn 1, no name at all                  -> PROMPT   ("change my memory")
after the PROMPT, any name              -> FULFILL, passthrough to host
```

Why not PROMPT on an unknown name: the user *did* name a target. If it is not
a memory, the premise was wrong — this was never a memory request. Asking
"which memory?" hijacks the conversation, which in an always-open session
happens constantly.

The result must tell the host which case it got, or the host cannot distinguish
`"Restaurant"` from `"shelf"`:

```
matched   -> MemoryName = "Restaurant"   memory_resolved = true
unmatched -> MemoryName = "temp"         memory_resolved = false
```

A host that ignores the new field behaves exactly as today for resolved cases.

**The carrier-strength list must be DERIVED from the corpus, not typed.** That
is the whole lesson of §10.1. Do not hand-author it.

### The one rule that must not be broken

Passthrough lives on the slot-filling path ONLY. It must never reach
`entities.extract_enum` / Swift `isWholeValue`, because the **bare-value guard**
(`834c9548`, `c445488`) asks "is this whole utterance one value of the closed
entity?" — and if every string is a value, the guard suppresses everything.
Keep it in a separate method (`extract_or_passthrough`), leave `extract` and
`extract_enum` untouched.

## 10.4 Step 1 — DONE, measured, UNCOMMITTED

Four files, working tree only:

| file | change |
|---|---|
| `language_packs/en/nlu_entities.json` | `memory` 38 -> 11. `tv` added as a Television synonym (its absence is why "mute the tv" fell back). `Custom` added. `automated_expansion: true` dropped — no code reads it, and it is false of a list we now declare authoritative. |
| `language_packs/en/platform.yaml` | the `"<X> mode"` keyword rule listed phantom names (`party\|quiet\|calm\|noise`); aligned to the real 11. **SOURCE file — `content_source assemble` was run; `check` says "in sync".** |
| `language_packs/en/nlu_schema.json` | the assemble output |
| `tests/test_slot_thresholds.py` | `test_memory_name_that_is_also_a_command_fills_the_slot` had 4 of its 9 cases naming phantom memories. The case list is now **read from the entity file**, so it cannot go stale again. |

Measured (no retrain — model unchanged, so only extraction and the keyword rule move):

```
engine holdout               1337/1470    unchanged
out-of-scope ACTION fires    20 -> 14     all six Cmd.MemoryChange ones gone
suite                        677 passed / 2 failed   (the same two known)

shall we go to the gym       FULFILL -> PROMPT     <- the device action is gone
go to gym / go to the theater  FULFILL -> PROMPT
```

Do NOT compare "explicit verb x memory" against the old 340/342: that was over
38 names, half of them phantom. Same probe, same model, before vs after:

```
                BEFORE(38)   AFTER(11)
switch to         10/11        10/11
change to         10/11        10/11
put it on         10/11        10/11
go to             10/11        10/11
set it to          1/11         1/11
use the / activate / select    0/11 each
TOTAL             41/88        45/88     (+4, from tv and custom)
```

`set it to`, `use the`, `activate`, `select` have **never** worked. Corpus gap,
not a regression — step 2 material.

Still broken after step 1, because `Restaurant` and `Music` are real so the slot
fills: `lets go to a restaurant` and `shall we put it on music` still FULFILL.
Corpus, not entity.

## 10.5 Step 2 — the corpus. NEXT JOB. Triage complete, nothing edited.

All `Cmd.MemoryChange` rows across `train.csv`, `holdout_honest.csv`,
`holdout_leakage_guard.csv`, classified by rule (name in the real list / phantom
/ numbered), never by looking at model output:

```
keep       687   (580 name a real memory; 107 name-less -> PROMPT is correct)
phantom    989   (851 train + 138 holdout)
numbered   308   (263 train +  45 holdout)
------------------------------------------
to fix   1,297 / 1,984 = 65% of the class
```

The triage is written to `/tmp/memchange_triage.json` by the script inlined in
the transcript; regenerate it rather than trusting a stale copy.

Plan, still requiring owner sign-off on the row lists before anything is edited:

1. Command-shaped generated rows (`go to gym`, `switch to program two`) —
   **delete**. They are synthetic, not captures.
2. Genuine ASR captures that happen to name a phantom — **relabel to
   `Default Fallback Intent`**. They are real speech and belong in the negative
   class. Do not launder the corpus (the `33213eeb` reasoning still holds).
3. The holdout rows (138 + 45) get the same treatment. This is a **label
   correctness** fix, done by rule, so blocker B9 is not violated — but it must
   never be driven by model output.
4. Add the counter-example families that were designed and validated but not
   inserted: conversational lead-in + `go to <memory>`, `go to <non-memory>`,
   `put it on <non-memory>`, lead-in + `put it on <memory>`. 61 rows, checked
   for domain-noun poisoning (0), train duplicates (0), holdout leakage (0).
   Generator: the `gen.py` in the transcript. **Note:** with the phantom purge
   done first, the `gym`/`kitchen`/`office` rows in that set become less
   load-bearing — re-validate before inserting.
5. Retrain -> recalibrate -> re-export -> measure at every step.

Expect the headline accuracy to MOVE, possibly down, and expect that to be
correct: today's number is partly the model reciting memories that do not exist.

## 10.6 The measurement that has to exist

`wrong_action_harness` replays `holdout_honest.csv` only, and that corpus
contains none of this failure family — which is exactly why `33213eeb` looked
green. Before or alongside step 2, add an **always-on precision probe set**:
conversational sentences that must never fire an action, committed to the repo
and wired into the report card as a gate.

It must use **different phrasings from the training counter-examples**, or the
gate tests its own training data.

Baseline from the 28 probes already run: 24 of 28 correctly fall back. The four
that fire are `shall we go to the gym`, `lets go to a restaurant`,
`the music is too loud in here` (-> VolumeDecrease) and `quiet please`
(-> VolumeMute). **The last two are arguably correct for a hearing aid and are
an open product question, not a defect.**

## 10.7 iOS — a shipped defect, fixed in the working tree, UNCOMMITTED

`pack-en-v1.0.56-ios` was verified: it carries `cancel_cues`, `bare_value`,
the widened `help_marker`, the timer carriers, `lemmas`, `sys.confirm.cancelled`,
and a lemma-folded vocabulary. All present and correct.

**But VoiceAIKit never applied the lemma table, and never applied contractions
either.** `PackLexicon` decoded `contractions` and nothing consumed it; `lemmas`
was not decoded at all; `PackTFIDFVectorizer.tokenize` only lowercased. The head
is fitted on normalised text, so the device fed it a surface form it was never
trained on. Measured on the pack's own shipped weights (the **full** head —
`BundleDataLoader` binds `variant: .full`; an earlier measurement of mine used
the pruned head and was wrong):

```
185 rows contain a folded plural:  v1.0.54 162 correct -> v1.0.56 130
                                   the pack made iOS WORSE by 32
"i'm outdoors now"     device: Cmd.VolumeUnmute 0.946   reference: Cmd.MemoryChange
"don't mute it"        OOV ratio 0.333 -> 0.000 once contractions expand
```

Fix in the working tree, five files, none committed:

- `Pack/Loader/PackTextNormalizer.swift` **(new)** — port of
  `text_norm.normalize_text`. No regex: ICU's `\b` is not Python's, and the type
  stays a value type so `PackTFIDFVectorizer` remains `Sendable`.
- `PackLexicon.swift` — decodes `lemmas`
- `PackTFIDFVectorizer.swift` — holds a normalizer, applies it at the head of
  `tokenize`, so `oovRatio` and `vectorize` both inherit it
- `PackIntentClassifier.swift` / `PackEngineFactory.swift` — thread it through

Verified without a Swift compiler by porting the algorithm back to Python and
diffing against `normalize_text` over all 10,045 corpus rows plus 18 edge cases
(`reprograms` must not fold, `o'clock`, curly apostrophes, empty string):
**10,063 compared, 0 mismatches.** Impact on the full head: **better 72, worse
11, zero new wrong actions.**

**Known gap, deliberately left:** the new init parameters are defaulted to
`.identity`, so the two test call sites in `ReferenceParityTests` and
`PackEntityAndClassifierTests` still construct the classifier WITHOUT the
normalizer. They compile and pass, but the parity suite therefore does not cover
the production path. Two lines to fix; left undone because this VM cannot
compile Swift.

**Also open on iOS, owner deferred it:** slot cancellation returns
`message: ""` at `NLUEngine.swift:508`. The reference returns
`uncertain_confirm.cancel_message`, which the bundler ships as
`sys.confirm.cancelled` and which `PackEngineFactory` already reads for the
confirm gate. The string is on the device; the engine does not use it.

## 10.8 Commits added on 2026-09-17

| sha | what |
|---|---|
| `941e687a` | `nlu_export` never got the runtime bootstrap `nlu_training` has — the release died in the TFLite step with `ModuleNotFoundError: nlu_engine`. Fixed at the package, not just in the two workflow steps, because `export_ios_weights` imports at module level and would have died again in the macOS job. |
| `1184a08a` | the conformance-fixture generator deleted what it could not rebuild; it now merges and reports. `slot_interruption.yaml` rewritten as the VIK-038 contract test it should always have been. |

## 10.9 Where things stand

**Uncommitted, IntentClassifier** — step 1 (4 files, §10.4).
**Uncommitted, STT** — the iOS featurizer fix (5 files, §10.7), plus the owner's
own deletion of `VoiceAIKit_Architecture_Review.md`.
**Nothing pushed, either repo.** The VM has no git credentials.

Suite: `677 passed, 2 failed, 34 skipped`. The two failures are the same open
product decisions recorded in §8 — `test_confirmation_branches[Cmd.SendMessage]`
and `test_legacy_label_compat`.

The release test gate in `release-pack.yml` is still **commented out** (4 lines).
Re-enable it once those two are settled.

Order from here:

1. Owner reviews the step-2 row lists -> edit corpus -> retrain -> measure
2. Always-on precision probe set + report-card gate (§10.6)
3. Passthrough: Python, then iOS (§10.3)
4. Pack v1.0.57, and only then the iOS test call sites (§10.7)
5. Android — still untouched, now divergent on three counts: both guards and
   the featurizer normalisation.

---

## 10.10 Step 1 + Step 2 — EXECUTED, measured, UNCOMMITTED

### What the owner settled while this ran

- **`Mask` IS a real memory.** It was missing from the list the entity was
  rebuilt against; caught on review of the deletion CSV. Final entity: **12**
  values — `Personal Restaurant Outdoors Meeting Crowd Car Auditorium Music
  Television Mute Mask Custom`.
- **Option A on the singular/plural gap** (see below), not turn-1 fuzzy.
- **The holdout gets the same purge** as training. Explicitly approved.

### Entity, final

Synonyms added are the OTHER grammatical number and nothing invented:
`restaurants, outdoor, meetings, crowds, cars, auditoriums, televisions, masks`,
plus `tv`. 22 surfaces over 12 values. Personal / Music / Mute / Custom take no
variant.

Why this was needed at all: `engine.py:1412 _extract_all_slots` runs the
first-turn scan with `fuzzy=False` on purpose ("so a common word (e.g. 'care',
'cup') doesn't get mis-read as a memory name"). So `please change to outdoor`
missed on turn 1 while `outdoor` answered the PROMPT fine — fuzzy is on there.
Exact synonyms close it without touching that decision.

### Corpus purge — applied

The list the owner reviewed is `step2_memorychange_triage_REVIEW.csv` at the
repo root (**do not commit it**). 1,250 rows, all of them generated templates —
there were no genuine ASR captures in the set, so nothing was relabelled.

```
train.csv                          8575 -> 7501   (-1074)
holdout_honest.csv                 1470 -> 1294   (-176)
holdout_leakage_guard.csv           331 ->  331   (0)
sources/01_source_base            8205 -> 6985   (-1220)
sources/02_manual_corrections       527 ->  503   (-24)
sources/03_generated_augmented     1337 -> 1333   (-4)

Cmd.MemoryChange class            1984 ->  734
   names a real memory                     627
   name-less (the PROMPT path)             107
   phantom / numbered remaining              0
```

**TRAP, hit and fixed — read this before rewriting any corpus CSV.**
`csv.writer` defaults to `lineterminator="\r\n"`. The first pass rewrote all six
files with CRLF where the repo uses LF, which turned a pure-deletion diff into
2,536 changed lines and then broke
`test_calibration.py::test_provenance_matches_the_current_training_data`,
because calibration records the sha256 of the train.csv it was fit on and the
bytes had changed under it. Normalise to LF and RETRAIN — same rows, different
sha. Always `git diff --stat` a corpus rewrite before trusting it.

### Measured, after retrain + recalibration

Holdout accuracy compared on the SAME 1,294 rows, old model vs new — the
pre-purge 1337/1470 is NOT comparable, the holdout shrank:

```
OLD model (pre-purge corpus)   1159/1294   89.6%
NEW model (purged corpus)      1157/1294   89.4%     -2 rows
temperature 0.668   ECE 0.1240 -> 0.0112
```

Removing 63% of the class cost **two holdout rows**.

```
                            before step 1     after step 2
shall we go to the gym      FULFILL       ->  FALLBACK 0.423
go to gym                   FULFILL       ->  FALLBACK 0.879
go to the theater           FULFILL       ->  FALLBACK 0.768
please change to outdoor    PROMPT        ->  FULFILL  (Outdoors)
switch to mask              PROMPT        ->  FULFILL  (Mask)
outdoors (bare)             FALLBACK      ->  FALLBACK 0.028  (guard holds)
```

Legitimate commands, 12 memories x carrier — `set it to` went from 1/12 to
11/12, which was a pre-existing corpus gap, not a regression this work caused:

```
switch to 12/12   change to 12/12   put it on 12/12   go to 12/12
set it to 11/12   use the    2/12
```

### The one regression, measured rather than argued

`test_oov_guard.py::test_out_of_scope_action_budget_is_met` now fails: **6 -> 8**
out-of-scope holdout rows FULFILL. The old model was rebuilt from the stashed
corpus to get an exact list rather than guessing. The eight are the old six plus:

```
  'please help me find a paper'  -> Help_FindMyHearingAids
  'can you help me find a paper' -> Help_FindMyHearingAids
```

So it is **one utterance with two polite prefixes**, it is a **help card and not
a device action**, and the `Cmd.*` count is unchanged at three
(`Cmd.StreamingStart`, `Cmd.FindMyPhone`, `Cmd.ListenMessage`). The cause is
class rebalancing: 1,250 rows left `Cmd.MemoryChange`, so every other class
carries slightly more mass. That phrase was already a known misroute — the owner
raised it earlier in the engagement.

Needs an owner decision: fix the phrase with data, or re-baseline
`OOS_ACTION_BUDGET`. **Do not raise the budget silently** — the constant's own
comment says to lower it with data, not by loosening the gate.

### Still open after step 2 (these need the slot contract, §10.3)

```
FULFILL  lets go to a restaurant        Restaurant is real, so the slot fills
FULFILL  shall we put it on music       Music is real
FULFILL  i'm outdoors now               Outdoors is real  <- a device action
PROMPT   put it on the shelf            conversation hijack
PROMPT   go to church
FALLBACK switch to tv  0.679            model has no "tv" rows; entity is fine
```

The first three are what the counter-example families in §10.5 were designed
for; they were NOT inserted, because the phantom purge changed what they need to
counterbalance. Re-validate the generator's output before using it.

### Suite

`685 passed, 3 failed, 34 skipped` — the two known product decisions plus the
budget regression above.

### Uncommitted after all of this

```
IntentClassifier:  .claude/SESSION-HANDOVER.md
                   language_packs/en/{nlu_entities,nlu_schema}.json
                   language_packs/en/platform.yaml
                   language_packs/en/{train,holdout_honest}.csv
                   language_packs/en/extras/sources/0{1,2,3}_*.csv
                   models/intent/en/calibration.json
                   tests/test_slot_thresholds.py
                   step2_memorychange_triage_REVIEW.csv   <- DO NOT COMMIT
STT:               the iOS featurizer fix, 5 files (§10.7)
```

`models/intent/en/model.onnx`, `pipeline.pkl` and `labels.pkl` are untracked, so
the retrained model does not appear in `git status`. The device weights have NOT
been re-exported yet — run `nlu_export.export_ios_weights` before building a
pack, or the CoreML head will not match this model.

## 10.11 Step 2b — weak carriers dropped, "find an object" taught. Budget recovered.

Two further owner decisions, both executed and measured.

### The evidence that settled the weak carriers

`Cmd.MemoryChange` rows per carrier in the corpus BEFORE `33213eeb` (mine):

```
                rows   with a domain noun
switch to        243        166
change to        254        173
go to              8          0
put it on          4          3   <- and those three are "put me in ... mode"
```

So the corpus's real memory carriers are `switch to` and `change to`. `go to` was
marginal and literal "put it on <memory>" did not exist — `33213eeb` created it
from a phrase matrix I built myself, not from QA evidence. The owner's rule:
**`go to` / `put it on` are not memory carriers without a domain noun**
(`memory / program / mode / setting / profile / environment`).

27 bare rows removed (14 mine, 13 inherited). Kept: `put me in restaurant mode`,
`go to the outdoors memory`, `switch to personal memory`.

Cost, stated plainly: `put it on outdoors` now falls back. `go to outdoors` still
fires (the classifier keeps it from the surrounding data).

### "help me find a paper"

All three 'paper' rows live in `holdout_honest.csv` and NONE in `train.csv`, so
the model had never seen the pattern — a generalisation failure, not a stubborn
model. Meanwhile `Help_FindMyHearingAids` carries 264 rows, **9 of which name no
device at all** ("help me track them", "find my equipment", "locate my missing
earpiece"), which is what lets "help me find a X" reach it.

48 out-of-scope rows added: `find / locate / look for / help me find` x 14
everyday objects x {bare, please, can you}, plus 6 non-template shapes. The
holdout phrasings are deliberately NOT reused (blocker B9). **"charger" was
excluded on purpose** — a hearing aid has one, so "find my charger" is a
legitimate device question.

### Measured

```
OOS budget test        8/195 -> 6/195     PASSES again (budget 6)
engine holdout         1155/1290  (89.5%)
suite                  686 passed, 2 failed, 34 skipped
                       (only the two known product decisions remain)
temperature 0.6673   ECE 0.1233 -> 0.0133
```

Every case the owner reported is now correct:

```
FALLBACK  lets go to a restaurant        0.726
FALLBACK  shall we put it on music       0.863
FALLBACK  shall we go to the gym         0.774
FALLBACK  put it on the shelf            0.750
FALLBACK  go to church                   0.686
FALLBACK  please/can you help me find a paper
FULFILL   switch to <memory>             12/12
FULFILL   change to <memory>             12/12
FULFILL   put me in restaurant mode / switch to personal memory
```

### THE ONE THING LEFT, and it is a product question, not a defect

```
FULFILL  Cmd.MemoryChange 1.000   "i'm outdoors now"     <- a device action
```

This is **deliberate corpus design**, not a bug. There are 18 "situational"
`Cmd.MemoryChange` rows that teach the engine to switch program when the user
DESCRIBES their surroundings:

```
i'm outdoors now            i'm in the car now        i'm heading into a meeting
i'm at a noisy restaurant now   i'm somewhere noisy   i'm walking into a loud bar
i'm settling in for tv time     i'm in a loud cafeteria now
```

It directly contradicts the direction taken above — an always-open microphone
hears exactly these sentences in ordinary conversation, and a program change is
the most visible thing this engine can do. But auto-switching on context may be
an intended product feature. **Do not remove these without the owner saying so.**

### Corpus, final state

```
train.csv           8575 -> 7526      holdout_honest.csv  1470 -> 1290
Cmd.MemoryChange    1984 ->  707      leakage guard        331 ->  331
```

Line endings: the second and third passes wrote with
`csv.DictWriter(..., lineterminator="\n")`. Verify with
`grep -qU $'\r' <file>` after ANY corpus rewrite — see the trap in §10.10.
