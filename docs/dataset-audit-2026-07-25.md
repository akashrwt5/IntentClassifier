# NLU Dataset Quality Audit

**Dataset:** `data/04_GENERATED_MASTER_training_data.csv`
**Audit date:** 2026-07-25
**Branch:** `claude/claude-setup-architecture-ebqobs`

| Snapshot | Value |
|---|---|
| SHA-256 | `8d41839d67e0424cf92d53281cd78524e40c199878a592feae96a08425a1e867` |
| Size | 499,397 bytes |
| Rows | 9,986 |
| Intents | 59 |
| Last commit touching the file | `ac353b8f` |

> Re-run the audit after any data change: the SHA above identifies exactly which
> snapshot these findings describe. Findings are **not** valid for a different hash.

**Scope:** intent-classification training data for a medical-grade hearing-aid
voice assistant. Audited for labelling correctness, taxonomy design, ambiguity,
duplication, coverage, entity completeness, and out-of-scope handling.

**Status:** no dataset changes have been made. This document records findings only.

---

## Executive Summary

**Quality score: 62 / 100**
**Verdict: ⚠️ Needs Improvements Before Training**

The dataset is structurally clean — no exact duplicates, no contradictory labels
at the surface, no nulls, and it deliberately includes realistic ASR noise. The
annotation is more disciplined than is typical at this stage.

However it carries three defects that are the **measured root cause of four of
the six confident wrong-actions** observed in end-to-end holdout testing, plus
one gap that means the product cannot execute a common hearing-aid request.

Critical issues: **ISS-01, ISS-02, ISS-03**.

### Headline findings

1. A **19:1 class prior** between `Cmd.MemoryChange` and `Help_MemoryOptions`
   mathematically forces help questions into the command class.
2. **43% of `Help_*` data is imperative-framed**, destroying the one signal that
   separates a question from a command.
3. **No laterality entity** — 253 utterances specify a left/right ear and the
   system has no way to capture which one.
4. `Default Fallback Intent` is 13% of the dataset and conflates true
   out-of-scope, ASR noise, and genuinely in-scope queries.

---

## Dataset Statistics

| Metric | Value |
|---|---|
| Intents | 59 |
| Utterances | 9,986 |
| Mean / median per intent | 169 / 80 |
| Largest class | 1,884 — `Cmd.MemoryChange` (18.9% of dataset) |
| Smallest class | 53 — `Help_DemoMode` |
| Imbalance ratio | **36:1** |
| Intents with < 60 examples | 13 |
| Intents with > 500 examples | 3 |
| Exact duplicates (text + intent) | **0** |
| Normalised duplicates (within intent) | 84 |
| Contradictory labels (identical text, different intent) | **0** |
| Entities defined / referenced by slots | 5 / 4 |
| Median utterance length | 6 words (p90 = 10, max = 31) |

### Cross-intent semantic conflicts

Cosine similarity over MiniLM embeddings of all 9,986 utterances; a "conflict"
is a pair above threshold carrying **different** labels.

| Cosine ≥ | Pairs | Distinct utterances | % of dataset |
|---|---|---|---|
| 0.95 | 69 | 106 | 1.1% |
| 0.92 | 317 | 281 | 2.8% |
| 0.90 | 595 | 391 | 3.9% |
| 0.85 | 2,179 | 969 | 9.7% |

**Interpretation:** high similarity is *not* proof of a labelling error. The
largest group (volume polarity) is correctly labelled and simply hard. See
ISS-05 and the Cross-Intent Ambiguity Report.

### Top conflicting intent pairs (cosine ≥ 0.92)

| Pairs | Intent A | Intent B | Nature |
|---|---|---|---|
| 33 | `Cmd.VolumeDecrease` | `Cmd.VolumeMute` | polarity — correct |
| 26 | `Cmd.VolumeDecrease` | `Cmd.VolumeUnmute` | polarity — correct |
| 24 | `Cmd.VolumeMute` | `Cmd.VolumeUnmute` | polarity — correct |
| 15 | `Help_Reminder` | `reminders.add` | **help ↔ action** |
| 13 | `Cmd.VolumeIncrease` | `Cmd.VolumeMute` | polarity — correct |
| 11 | `Cmd.MemoryChange` | `Help_Tinnitus` | **domain overlap** |
| 9 | `Cmd.MemoryChange` | `Help_ChangingMemories` | **help ↔ action** |
| 7 | `Cmd.VolumeIncrease` | `Help_Volume` | **help ↔ action** |
| 6 | `Cmd.VolumeDecrease` | `Help_Volume` | **help ↔ action** |
| 5 | `Help_Home` | `Help_WhatsNew` | topic overlap |
| 3 | `Default Fallback Intent` | `Help_DeviceSettings` | **mislabel** |
| 2 | `Cmd.BatteryLevel` | `Help_Battery` | **help ↔ action** |

---

## Methodology

Every finding is measured, not inspected by sampling.

| Check | Method |
|---|---|
| Duplicates / contradictions | Exact + normalised (`lower`, strip punctuation, collapse whitespace) grouping |
| Semantic conflicts | MiniLM (`packs/en/semantic/minilm-l6-v2.onnx`) embeddings, 384-dim L2-normalised, full pairwise cosine in 1,000-row chunks |
| Question framing | Regex `^(how|what|where|why|can i|is there|does|do i|tell me|explain)` or contains `?` |
| Class balance | `value_counts()` over the intent column |
| Template diversity | Distinct first-two-word openers ÷ class size |
| Entity coverage | `data/nlu_entities.json` cross-referenced against slot declarations in `data/nlu_schema.json` |
| Laterality / degree gaps | Regex over utterances, checked against defined entities |

---

## Detailed Findings

### ISS-01 — Class prior overwhelms the help/command distinction

| | |
|---|---|
| **Category** | Dataset balance → wrong-action safety |
| **Severity** | **CRITICAL** |
| **Affected intents** | `Cmd.MemoryChange` (1,884), `Help_MemoryOptions` (98), `Help_ChangingMemories` (158) |
| **Affected utterances** | Class-level; manifests on any borderline memory-related phrasing |

**Description.** `Cmd.MemoryChange` holds 18.9% of the entire dataset and
outweighs its help counterpart **19:1**.

**Explanation.** A logistic classifier trained on this prior resolves any
ambiguous memory-related utterance to the command. This is confirmed by
production holdout testing — two confident wrong-actions trace directly to it:

```
"can i set a favourite program"  → Cmd.MemoryChange @ 1.00   (expected Help_MemoryOptions)
"change memory explained"        → Cmd.MemoryChange @ 1.00   (expected Help_ChangingMemories)
```

**Impact.** The device changes a hearing program when the user asked a question.
On a medical device, for a hearing-impaired user, that is a harm event. At
confidence 1.00 no confidence threshold can intercept it — this cannot be fixed
downstream in the engine.

**Recommendation.** Cap `Cmd.MemoryChange` to ~300 examples (the current
`MAX_PER_INTENT = 500` in `scripts/train.py` is too loose to help here). Raise
`Help_MemoryOptions` and `Help_ChangingMemories` to 150+ each. Adopt a standing
rule: **no Cmd/Help pair on the same topic may exceed 3:1**.

---

### ISS-02 — 43% of `Help_*` data is imperative-framed

| | |
|---|---|
| **Category** | Annotation consistency / conversational design |
| **Severity** | **CRITICAL** |
| **Affected intents** | All 33 `Help_*` intents |
| **Affected utterances** | 1,501 of 3,470 `Help_*` rows |

**Description.** Question framing is inconsistent between the two class families:

| Class family | Question-framed | n |
|---|---|---|
| `Help_*` | **56.7%** | 3,470 |
| `Cmd.*` | 6.0% | 4,298 |

**Explanation.** Interrogative framing is the *only* reliable separator between
"how do I change memories" and "change memories". Placing 1,501 imperative-looking
utterances in `Help_*` teaches the model that framing carries no information.

Some are outright mislabels — genuine commands sitting in help classes:

```
'add mask as a customized memory'      [Help_MemoryOptions]  ← imperative CREATE command
'personalize my hearing aid settings'  [Help_MemoryOptions]  ← imperative
'i want to create a new program?'      [Help_MemoryOptions]  ← statement with spurious '?'
'help with changing memories on my aid'[Help_ChangingMemories] ← borderline: "help with X"
```

**Impact.** The model cannot learn question → help. This is why a rules-based
help-marker guard tested against the six known wrong-actions catches only **1 of
6** — the signal is not consistently present in the data to begin with.

**Recommendation.** Re-annotate all `Help_*` intents against a written rule:

> **If executing the utterance literally would change device state, it is a `Cmd.`
> intent, regardless of politeness or phrasing.**

Move true imperatives to the appropriate command intent, creating
`Cmd.MemoryCreate` where no target exists. Target ≥ 85% question-framing for
`Help_*`.

---

### ISS-03 — No laterality (ear/side) entity

| | |
|---|---|
| **Category** | Entity coverage — functional gap |
| **Severity** | **CRITICAL** |
| **Affected intents** | `Cmd.Volume*` (103 rows), `Help_FindMyHearingAids` (71), `Help_SelfCheck` (32), `Help_Pairing` (31) |
| **Affected utterances** | 253 total |

**Description.** Entities defined in `data/nlu_entities.json`: `memory`,
`recurrence`, `remind`, `sys.date-time`, `sys.number-integer`. **There is no
ear/side entity.**

Yet the training data is full of laterality:

```
'turn my left hearing aid up'        'increase the right hearing aid'
'adjust right ear for higher sound'  'volume up left aid'
'turn down the volume in my left hearing aid'
```

**Impact.** "Turn up my **left** hearing aid" and "turn up my **right** hearing
aid" are indistinguishable after classification — both yield
`Cmd.VolumeIncrease` with no slot carrying the side. For binaural fitting, where
asymmetric hearing loss is the norm, this is a core product capability that
silently does the wrong thing or applies the change bilaterally.

This is the one finding that is a **missing capability**, not a quality issue.

**Recommendation.** Add an `ear` entity with values `left` / `right` / `both`
and synonyms (*left side, left ear, that ear, this side*). Attach it as an
optional slot to all `Cmd.Volume*` and `Cmd.MemoryChange` intents. Confirm with
product whether an absent side means "both" or should prompt.

---

### ISS-04 — `Default Fallback Intent` conflates three different things

| | |
|---|---|
| **Category** | Out-of-scope design |
| **Severity** | **HIGH** |
| **Affected intents** | `Default Fallback Intent` (1,308 rows = 13.1% of dataset) |
| **Affected utterances** | 87 contain hearing-domain terms; ~15–20 judged genuinely in-scope |

**Description.** The class currently mixes four distinct categories:

| Category | Example | Correct? |
|---|---|---|
| True out-of-scope | `'what caused the fall of rome'`, `'what are stock options'` | ✅ |
| Useful hard negatives | `'electric car battery'`, `'car battery'` | ✅ valuable — adjacent to `Cmd.BatteryLevel` |
| ASR noise | `'bustling hearing aid'`, `'mavis by stupid hearing aids'` | ✅ realistic |
| **Genuinely in-scope** | see below | ❌ **mislabelled** |

Mislabelled in-scope examples:

```
'where can i get my hearing checked'                            ← hearing-health query
'how do i know if i need hearing aids'                          ← hearing-health query
'how do i stop pandora from playing on my hearing aids'         ← streaming stop
'where can i find videos for the hearing aid for instructions'  ← help / tutorial
'i said other program'                                          ← repair / correction turn
'how do i activate the default settings on the hearing aids'    ← Help_DeviceSettings
```

**Impact.** The model is explicitly taught to reject real user needs.
`'where can i get my hearing checked'` is precisely the query a hearing-health
assistant exists to answer. The `'i said other program'` case is worse — it is a
**conversational repair turn**, and labelling repairs as fallback teaches the
system to abandon users who are correcting themselves.

**Content note:** `'how do i dispose of a body'` appears in this class. Whatever
its label, it is inappropriate in a medical-device training corpus and should be
removed on review grounds.

**Recommendation.** Split into `oos.general` (trivia / unrelated) and
`oos.unintelligible` (ASR noise) so the two can be measured separately. Relabel
the in-scope rows. Add the missing intents `Help_HearingTest` and
`Help_FindProvider` — currently there is nowhere for hearing-health queries to go.

---

### ISS-05 — Volume polarity minimal pairs

| | |
|---|---|
| **Category** | Cross-intent ambiguity |
| **Severity** | **HIGH** |
| **Affected intents** | `Cmd.VolumeIncrease` (253), `Cmd.VolumeDecrease` (301), `Cmd.VolumeMute` (118), `Cmd.VolumeUnmute` (130) |
| **Affected utterances** | 96 conflicting pairs at cosine ≥ 0.92 |

**Description.** The four volume intents dominate the conflict list:

```
0.974  'turn the volume up on the hearing aid'    [Cmd.VolumeIncrease]
       'turn down the volume on the hearing aid'  [Cmd.VolumeDecrease]

0.970  'turn my left hearing aid up'              [Cmd.VolumeIncrease]
       'turn my left hearing aid down'            [Cmd.VolumeDecrease]
```

**Explanation.** **These are correctly labelled.** They are near-identical in
embedding space because they differ by a single antonym. TF-IDF weights
*up*/*down* low (high document frequency); embeddings place antonyms adjacently
because they share context. This is the hardest legitimate discrimination in the
dataset and is why the engine carries explicit polarity guards.

**Impact.** Not a data error, but the highest-risk confusion surface: confusing
increase with decrease is directly harmful to a hearing-impaired user.

**Recommendation.** **Do not relabel.** Instead:
- Guarantee matched-pair coverage — for every "turn up X" phrasing, ensure a
  "turn down X" exists with identical structure.
- Raise `Cmd.VolumeMute` (118) and `Cmd.VolumeUnmute` (130) toward parity with
  Increase (253) / Decrease (301).
- Retain the engine-side polarity guard as defence in depth.

---

### ISS-06 — `Cmd.MemoryChange` conflates three user intentions

| | |
|---|---|
| **Category** | Intent taxonomy — should be split |
| **Severity** | **HIGH** |
| **Affected intents** | `Cmd.MemoryChange` (1,884) |
| **Affected utterances** | 18 implicit-context rows; create/manage rows split across `Help_*` |

**Description.** One intent covers three behaviours:

```
explicit switch  : 'switch memory to speech', 'please set program to master'
create / manage  : 'add mask as a customized memory'   (currently in Help_MemoryOptions)
implicit context : "i'm in the car now", "i'm about to join a phone call"   (18 rows)
```

**Explanation.** The implicit-context group encodes a **significant product
decision** — that merely stating your environment should silently change a
hearing program. Examples:

```
"i'm somewhere noisy"        "i'm walking into a loud bar"
"i'm getting in the car now" "i'm heading into a meeting"
"i'm about to join a phone call"
```

**Impact.** Three problems. (a) The product decision is undocumented and
implicit in training data. (b) At 18 examples it is badly under-sampled for a
feature that changes device behaviour without an explicit request. (c) Merging
create/manage with switch means "add a new memory" and "switch to a memory"
compete for the same label while their help counterparts sit elsewhere,
compounding ISS-02.

**Recommendation.** Split into `Cmd.MemoryChange` (explicit switch) and
`Cmd.MemoryCreate` (create / customise / delete). Make an explicit product
decision on implicit context-switching: if it is a v1 feature it needs its own
intent and ~100 examples; if not, remove those 18 rows.

---

### ISS-07 — Template-generated data lacks linguistic variety

| | |
|---|---|
| **Category** | Coverage |
| **Severity** | **MEDIUM** |
| **Affected intents** | `Cmd.MemoryChange`, `reminders.add`, `Help_Tinnitus`, `Help_Volume`, and others |
| **Affected utterances** | ~3,000 template-derived rows |

**Description.** Lexical diversity, measured as distinct first-two-word openers
divided by class size:

| Intent | n | Distinct openers | Ratio |
|---|---|---|---|
| `Cmd.MemoryChange` | 1,884 | 194 | **0.10** |
| `reminders.add` | 832 | 97 | 0.12 |
| `Help_Tinnitus` | 225 | 32 | 0.14 |
| `Help_Volume` | 170 | 28 | 0.16 |
| `Help_ChangingMemories` | 158 | 27 | 0.17 |

Whole-corpus phenomenon coverage:

| Phenomenon | Present |
|---|---|
| Polite forms (*please / could you / would you*) | 36.5% |
| Short commands (≤ 3 words) | 10.8% |
| Long utterances (≥ 12 words) | 4.0% |
| Contractions | 6.9% |
| Question marks | 14.6% |
| **Fillers (*um, uh, hmm, erm*)** | **0.0%** |
| **Self-correction (*no wait, I mean, sorry*)** | **0.1%** |

**Impact.** 1,884 rows of `Cmd.MemoryChange` carry far less information than the
count implies — they inflate the class prior (ISS-01) without adding
generalisation. Zero disfluency coverage is a notable gap for a **speech-driven**
assistant: real ASR output routinely contains fillers and restarts, and the
dataset contains no examples of either.

**Recommendation.** Cap template families per intent. Add disfluent variants
(*"um, can you turn it up"*), self-corrections (*"turn it down — no wait, up"*),
and long-form natural phrasing. Prioritise this for the classes named above.

---

### ISS-08 — Four naming conventions across 59 intents

| | |
|---|---|
| **Category** | Taxonomy hygiene |
| **Severity** | **MEDIUM** |
| **Affected intents** | All 59 |

**Description.**

| Convention | Count | Example |
|---|---|---|
| `Cmd.<PascalCase>` | 23 | `Cmd.VolumeIncrease` |
| `Help_<PascalCase>` | 33 | `Help_MemoryOptions` |
| `lower.dotted` | 2 | `reminders.add`, `reminders.complete` |
| Space-separated | 1 | `Default Fallback Intent` |

**Impact.** Blocks programmatic reasoning about intents. The engine's
`non_interrupting_actions` safety guard has to match on the `action` field
precisely *because* intent names are unreliable as a source of structure. Any
future rule of the form "all command intents must X" cannot be expressed.

**Recommendation.** Migrate to one convention. Note that
`feature/production-work` already performed this migration
(`domain.object.action`, 59 → 57 intents, with a documented migration map).
**Align with that scheme rather than inventing a third** — this is
cross-branch technical debt, not a fresh decision.

---

### ISS-09 — No degree / level entity

| | |
|---|---|
| **Category** | Entity coverage |
| **Severity** | **MEDIUM** |
| **Affected intents** | `Cmd.VolumeIncrease`, `Cmd.VolumeDecrease` |
| **Affected utterances** | 48 |

**Description.** Utterances express magnitude with no entity to capture it:

```
'max loudness'   'maximum volume'   'minimum sound'   'a bit louder'
```

**Impact.** `'max volume'` and `'a little louder'` both resolve to an
undifferentiated `Cmd.VolumeIncrease`. The device cannot honour the requested
magnitude, and "max volume" on a hearing aid has safety implications that a
generic increment does not.

**Recommendation.** Add a `level` entity (`max`, `min`, `slight`, `moderate`) as
an optional slot on the volume intents. Define clinically safe behaviour for
`max` with audiology input.

---

### ISS-10 — Malformed rows from the augmentation pipeline

| | |
|---|---|
| **Category** | Data hygiene |
| **Severity** | **LOW** |
| **Affected utterances** | 7 |

**Description.**

```
'can you train memory'    [Cmd.MemoryChange]   ← "train" is almost certainly "change"
'can you train program'   [Cmd.MemoryChange]
'can you train setting'   [Cmd.MemoryChange]
'how do i go the the main screen?'  [Help_Home]        ← "the the"
'how do i go the the home screen?'  [Help_Home]        ← "the the"
'rain. the the app me'    [Default Fallback Intent]
'what are some river cruise options in the united states (1)'  ← spreadsheet artifact
```

**Impact.** Individually trivial. Collectively they indicate
`scripts/build_augmented_data.py` has no output validation, so future
augmentation runs may introduce similar corruption at scale.

**Recommendation.** Fix the 7 rows. Add a validation pass to
`build_augmented_data.py` rejecting: repeated adjacent tokens, trailing `(n)`
artifacts, and rows whose token set is not covered by the source template bank.

---

## Cross-Intent Ambiguity Report

### Conflict 1 — Volume polarity

| | |
|---|---|
| Phrase 1 | `'turn the volume up on the hearing aid'` → `Cmd.VolumeIncrease` |
| Phrase 2 | `'turn down the volume on the hearing aid'` → `Cmd.VolumeDecrease` |
| Similarity | **0.974** |
| Also affects | Decrease↔Mute (33), Decrease↔Unmute (26), Mute↔Unmute (24), Increase↔Mute (13) |

**Why they confuse the classifier.** The sole discriminator is one function word.
TF-IDF assigns *up*/*down* low weight due to high document frequency; embedding
models place antonyms adjacently because they occur in identical contexts.

**Which intent is more appropriate.** Both are correct — these are opposite
actions and must stay separate.

**Should the taxonomy change?** No.

**Additional examples required?** Yes — matched contrastive pairs, and raise
Mute/Unmute toward parity with Increase/Decrease.

**Confidence: HIGH** (correctly labelled, genuinely hard).

---

### Conflict 2 — On/off versus up/down

| | |
|---|---|
| Phrase 1 | `'turn my left hearing aid on'` → `Cmd.VolumeUnmute` |
| Phrase 2 | `'turn my left hearing aid up'` → `Cmd.VolumeIncrease` |
| Similarity | **0.966** |
| Related | `'turn on the volume in my right hearing aid'` → `Cmd.VolumeUnmute` vs `'turn up the volume in my right hearing aid'` → `Cmd.VolumeIncrease` (0.967) |

**Why they confuse the classifier.** *"Turn on my hearing aid"* is genuinely
ambiguous **to a human**: it could mean unmute, power on, or increase volume.
The dataset silently resolves it to Unmute without documenting the rule.

**Which intent is more appropriate.** Undecidable from the utterance alone —
this requires a product decision, not an annotation fix.

**Should the taxonomy change?** Possibly — if "power on" is a distinct device
action, it needs its own intent, and it is currently absent.

**Additional examples required?** After the product decision, yes.

**Confidence: MEDIUM** — flagged as a design gap rather than a defect.

---

### Conflict 3 — Help versus Command (the safety-critical conflict)

| | |
|---|---|
| Phrase 1 | `'can i set a favourite program'` → `Help_MemoryOptions` (98 examples) |
| Phrase 2 | `'switch to office'` → `Cmd.MemoryChange` (1,884 examples) |
| Similarity | ≥ 0.92 |
| Also affects | `Cmd.MemoryChange`↔`Help_ChangingMemories` (9), `Cmd.Volume*`↔`Help_Volume` (13), `Help_Reminder`↔`reminders.add` (15), `Cmd.BatteryLevel`↔`Help_Battery` (2) |

**Why they confuse the classifier.** Two compounding causes: the 19:1 class
prior (ISS-01) and the 43% imperative-framed help data (ISS-02). The model has
neither a prior nor a linguistic signal favouring the help reading.

**Which intent is more appropriate.** The help intent — "can I…" is a
capability question, not a request to act.

**Should the taxonomy change?** No, but the balance and annotation must.

**Additional examples required?** Yes — this is the primary remediation target.

**Confidence: HIGH — measured end-to-end, not inferred.** This conflict directly
produces confident wrong-actions at 1.00 confidence.

---

### Conflict 4 — Fallback versus in-scope

| | |
|---|---|
| Phrase 1 | `'how do i activate default setting on the hearing aids'` → `Help_DeviceSettings` |
| Phrase 2 | `'how do i activate the default settings on the hearing aids'` → **`Default Fallback Intent`** |
| Similarity | **0.980** |

**Why they confuse the classifier.** They are the same sentence differing by one
article and a plural. Opposite labels teach the model that this region of the
space is arbitrary.

**Which intent is more appropriate.** `Help_DeviceSettings`, unambiguously.

**Should the taxonomy change?** No — this is a straightforward mislabel.

**Confidence: VERY HIGH.**

---

### Conflict 5 — `Cmd.MemoryChange` versus `Help_Tinnitus`

| | |
|---|---|
| Conflicting pairs | 11 at cosine ≥ 0.92 |

**Why they confuse the classifier.** Tinnitus relief is delivered *through* a
hearing program, so "switch to the tinnitus program" is legitimately both a
memory change and a tinnitus topic. The domain overlap is real, not an
annotation slip.

**Which intent is more appropriate.** Requires clinical input — depends on
whether tinnitus-program activation is modelled as a command or a guided flow.

**Should the taxonomy change?** Possibly. Consider `Cmd.TinnitusProgramStart`
as an explicit intent.

**Confidence: MEDIUM — needs audiologist review, not data-analyst judgement.**

---

## Prioritised Recommendations

### Critical — block retraining until resolved

| # | Action | Issue |
|---|---|---|
| 1 | Rebalance every Cmd/Help topic pair to ≤ 3:1 | ISS-01 |
| 2 | Re-annotate `Help_*` against a written state-change rule; target ≥ 85% question-framing | ISS-02 |
| 3 | Add the `ear` laterality entity and attach it to volume + memory intents | ISS-03 |

### High

| # | Action | Issue |
|---|---|---|
| 4 | Split `Default Fallback Intent`; relabel in-scope rows; add `Help_HearingTest` / `Help_FindProvider` | ISS-04 |
| 5 | Matched-pair polarity coverage; lift Mute/Unmute toward Increase/Decrease | ISS-05 |
| 6 | Split `Cmd.MemoryChange`; decide on implicit context-switching | ISS-06 |

### Medium

| # | Action | Issue |
|---|---|---|
| 7 | Cap template families; add disfluency and self-correction coverage | ISS-07 |
| 8 | Unify intent naming; align with the `domain.object.action` scheme | ISS-08 |
| 9 | Add a `level` (degree) entity | ISS-09 |

### Low

| # | Action | Issue |
|---|---|---|
| 10 | Fix 7 malformed rows; add validation to `build_augmented_data.py` | ISS-10 |

---

## Remediation Tracker

| Issue | Severity | Status | Owner | Notes |
|---|---|---|---|---|
| ISS-01 | Critical | ☐ Open | | Blocks retraining |
| ISS-02 | Critical | ☐ Open | | Blocks retraining |
| ISS-03 | Critical | ☐ Open | | Missing capability |
| ISS-04 | High | ☐ Open | | Needs new intents |
| ISS-05 | High | ☐ Open | | Coverage, not relabelling |
| ISS-06 | High | ☐ Open | | Needs product decision |
| ISS-07 | Medium | ☐ Open | | |
| ISS-08 | Medium | ☐ Open | | Align with production-work |
| ISS-09 | Medium | ☐ Open | | Needs clinical input on `max` |
| ISS-10 | Low | ☐ Open | | |

---

## Limitations of This Audit

Stated so findings are weighted correctly:

1. **Semantic similarity was computed with the same MiniLM model the system
   uses.** The analysis therefore shares that model's blind spots — conflicts it
   cannot perceive are invisible here.
2. **Hearing-health terminology was not validated against clinical sources.**
   ISS-06 and Conflict 5 need an audiologist's judgement. Terminology observed
   (memory/program, telecoil, tinnitus, streaming) is internally consistent, but
   consistency is not correctness.
3. **In-scope/out-of-scope judgements in ISS-04 are the auditor's reading**, not
   a product definition. The count "~15–20 genuinely in-scope" should be
   confirmed against the product's intended capability list.
4. **No entity-level annotation audit was performed** on slot *values* within
   utterances — only entity definitions and coverage gaps. Reminder names and
   memory values inside utterance text were not verified.
5. **Only `04_GENERATED_MASTER_training_data.csv` was audited.** Upstream sources
   (`01_source_base`, `02_source_manual_corrections`,
   `03_generated_augmented_phrases`) were not separately reviewed, so it is not
   established which stage introduced each defect.

---

## Conclusion

## ⚠️ Needs Improvements Before Training

The annotation reflects careful work: clean structure, no contradictions, and
realistic ASR noise deliberately retained. The defects are **distributional and
taxonomic rather than careless** — the data teaches the model that framing does
not matter and that memory-changing is 19× more likely than memory-help.

Recommendations 1–3 are the gate. They are the measured root cause of four of
the six confident wrong-actions seen in production testing, and all three are
fixable **in the data alone** — no model or architecture change required.
Recommendation 3 addresses a capability the product currently cannot deliver.

Once 1–3 are complete, re-run this audit against the new SHA-256 and re-measure
the wrong-action count end-to-end before declaring readiness.

---

## Reproducing This Audit

```bash
# fingerprint
python3 -c "import hashlib;print(hashlib.sha256(open('data/04_GENERATED_MASTER_training_data.csv','rb').read()).hexdigest())"

# end-to-end wrong-action count (the metric these findings target)
python scripts/test_holdout.py --no-semantic --verbose | grep "❌" | grep -v "Default Fallback Intent"
```

Semantic-conflict detection embeds every utterance with
`packs/en/semantic/minilm-l6-v2.onnx` via `scripts/nlu/semantic.py`
(`SemanticFallback._embed`) and computes full pairwise cosine similarity in
chunks, reporting pairs above threshold that carry different labels.
