# Session handover — NLU quality work (Python + iOS)

**Last updated:** 2026-09-16 — 14 commits here, 2 in `STT`. Everything committed,
**nothing pushed** (see §3).
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
