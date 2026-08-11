# What to do next — ordered by value per unit of effort

**Date:** 2026-08-10
**Target asked for:** ~90% accuracy with usable confidence.

---

## Where things actually stand

Measured on the full eval sets, not anecdotes.

| model | MB | locked | stress | oov | ood |
|---|---:|---:|---:|---:|---:|
| e5-small *(current teacher)* | 133 | 0.8499 | 0.8496 | 0.6364 | 0.5931 |
| minilm | 91 | 0.8944 | 0.8035 | 0.6667 | 0.7146 |
| **bge** | 133 | 0.8820 | **0.8743** | **0.7273** | 0.6551 |
| **student `semfz`** | **2.5** | **0.8928** | 0.7947 | 0.2424 | **0.7419** |

Three things follow, and they set the whole plan:

1. **The student already beats every 100 MB encoder on `locked` and `ood`**, at
   1/50th the size. It is not behind in general — it is behind on *generalising
   to unseen words* (`oov` 0.2424 vs 0.7273) and a little on novel phrasing.
2. **The current teacher is the worst of the three on this data.** `e5-small` is
   last on `locked` and last on `ood`. The student is being distilled from the
   weakest available signal.
3. **`bge` is the best encoder for this dataset** — best `stress`, best `oov`.
   It is a *teacher*, not a deployment candidate: 133 MB against a 2.5 MB budget.

---

## Is 90% reachable?

| metric | now | 90%? |
|---|---:|---|
| locked | 0.8928 | **essentially there** |
| stress | 0.7947 | **no** — the 133 MB `bge` itself only reaches 0.8743 |
| oov | 0.2424 | no — `bge` itself only reaches 0.7273 |

Getting `stress` to 90% would mean beating a 133 MB model with a 2.5 MB one.
That is not going to happen. **The realistic target is `stress` ≈ 0.85**, i.e.
matching `bge`, which distillation can plausibly deliver.

Also worth saying plainly: `locked` is a soft number. 70% of its rows are
recombinations of training phrasing (88.2% of its bigrams already appear in
training). Quote `stress`, not `locked`.

---

## SHIPPED — Stage 3 is now ON in the English pack

`semantic_rescue_enabled: true`, `semantic_threshold` 0.40,
`stage2_backstop_confidence` 0.30.

Three independent lines of evidence, which is the bar that should have been
applied to everything earlier in this document:

| evidence | result |
|---|---|
| OOD rejection, Stage 3 off -> on | **0.4527 -> 0.7612 (+31.8 pts)** |
| seed rule / eval-set floor | **clears both** — the only result here that does |
| two failing safety tests | **both green** |
| calibration parity | 0 argmax flips / 2,654 rows |
| full suite | 47 -> 45 failed, 470 -> 472 passed (exactly those two) |

The safety tests are the part worth reading twice. On the Stage-2-only path:

```
"turn mute on"          -> Cmd.VolumeUnmute    FULFILL 0.49
"i need it more quiet"  -> Cmd.VolumeIncrease  FULFILL 0.60
```

Both are the OPPOSITE action reaching a hearing-aid user. With Stage 3 they
resolve to `VolumeMute` 0.83 and `VolumeDecrease` 0.82. Those tests had been
passing a fixture pinned to `semantic_enabled=False`; a test that holds only
because the risky stage is disabled is not evidence the property holds, so the
fixture now follows the pack.

### The gate was NOT changed — that is the finding

`select_policy.py`, run across three calibrated seeds (T = 0.68 / 0.66 / 0.64),
selected `avg` on dev. It was not adopted:

- **Not expressible in the engine.** `avg` needs the full 57-dim distribution
  from both stages; `classify()` returns top-1 only.
- **Inside the floor anyway.** Its OOD gain over `replace` is +7.96 pts against
  an 11.6-pt confidence interval on a 201-row test half.

Among the policies the engine *can* express, none is distinguishable from the
shipped one. Several gain ~+4 stress and pass the seed rule — and every one of
them sits inside the eval-set floor. The shipped `dual@s3=0.40,s2=0.30` is
already top of that group on stress (0.8463).

This is the measurement floor from the section below biting in real time: with
201 OOD rows in the test half, policy selection cannot resolve anything smaller
than ~12 points. **Step 6 is what unblocks this**, not more policy search.

---

## Step 1 — Calibration  ✅ DONE, ship it

Already fitted: **T = 0.68**, on the dev half, reported on test.

| | T=1 | T=0.68 |
|---|---:|---:|
| ECE | 0.2029 | **0.0187** |
| accuracy @ gate 0.40 | 0.8374 | **0.8835** |
| argmax accuracy | 0.8898 | 0.8898 (unchanged — T is rank-preserving) |
| OOD reject @ 0.40 | 0.8308 | 0.7910 |

**+4.6 points of in-scope accuracy for free**, and the confidence number becomes
meaningful (ECE 0.02 means "0.8 confident" is right about 80% of the time).

The OOD drop is not a real loss — 0.40 was tuned for the *uncalibrated* scale.
Now that confidence is sharper the gate has to move with it:

```bash
python scripts/calibrate.py --tag semfz_s1 --apply
# then re-pick the gate on the calibrated scale
python scripts/select_policy.py --tags semfz_s42 semfz_s1 semfz_s7
```

Remaining work: `StudentSemantic` does not read `temperature` yet. Wiring it is
a runtime change and should be its own reviewed edit.

---

## Steps 2 and 2b — BOTH RAN, BOTH FOUND NOTHING

Recorded because negative results are what stop the next person repeating them.

### 2 — Teacher swap, `e5-small` -> `bge`  (3 seeds each)

| | semfz (e5) | bgefz (bge) | verdict |
|---|---:|---:|---|
| locked | 0.8928 | 0.8936 | not significant |
| macro recall | 0.8915 | 0.8947 | not significant |
| stress | 0.7947 | 0.7740 | not significant |
| OOD raw | 0.7419 | 0.7254 | not significant |

`bge` is clearly the better ENCODER on this data — standalone it scores stress
0.8743 / oov 0.7273 against e5's 0.8496 / 0.6364. **None of that advantage
reached the student.**

The prediction that this would help came from the project's own history:
79.6% -> 98.75% on a teacher swap. That precedent does not apply. It was
TF-IDF -> E5 — a bag-of-words teacher replaced by a real sentence encoder. This
was e5 -> bge, two comparable encoders. The analogy was wrong and should have
been checked before spending the runs.

### 2b — Capacity, 64-dim -> 96-dim  (3 seeds each)

| | bgefz (64) | big (96) | verdict |
|---|---:|---:|---|
| params | 582,777 | 898,585 | +54% |
| size | 2.33 MB | 3.59 MB | +54% |
| locked | 0.8936 | 0.8917 | not significant |
| stress | 0.7740 | 0.7693 | not significant |
| OOD raw | 0.7254 | 0.7552 | not significant |

Half again as many parameters bought nothing measurable.

### What these two together mean

Two independent levers — a better teacher and 54% more capacity — moved nothing.
That is not a coincidence of two failed tuning attempts; it is the signature of a
**data ceiling**. The student is already extracting what the 24k rows contain.

**Caveat added later — the `ood` column above is not evidence of anything.** Both
`ood` deltas (−1.7 and +3.0 pts) fall inside the 8.5-pt confidence interval of a
403-row set; that column could not have shown a real effect either way. The
data-ceiling conclusion rests on `locked` (1,686 rows) and `stress` (565), which
do have the power to show one and showed nothing. See "The measurement floor".

**Stop tuning the model.** Steps 4-6 below (typo robustness, volume hard
negatives, OOD collection) are the only remaining levers, and they are all data.

The one exception is Step 1, calibration, which is not modelling — it is fixing a
reporting defect, and it is the only change this whole exploration produced that
had a measurable effect (+4.6 in-scope accuracy at the gate, ECE 0.20 -> 0.019).

---

## Step 2 (original plan, superseded above) — Switch the teacher to `bge`

The single change with the most evidence behind it and the least work.

```bash
# config.py
TEACHER = "BAAI/bge-small-en-v1.5"

python scripts/build_semantic_vocab.py --size 8000 --tag bge
python scripts/run_seeds.py --name bgefz --seeds 42 1 7 -- \
    --init-embeddings models/en/embed_init_bge.npz --freeze-embeddings \
    --unk-aug 0.04 --fallback-weight-floor 3.0
python scripts/run_seeds.py --compare semfz bgefz --seeds 42 1 7
python scripts/eval_oov.py --tags semfz_s1 bgefz_s1
```

One change, two effects: better soft targets AND a better embedding-init space,
because the same encoder supplies both.

**Precedent:** the historical record in `semantic_project/` shows the same
architecture going 79.6% → 98.75% purely from a teacher swap. Teacher quality,
not student capacity, was the lever that moved.

**Accept if:** `stress` improves beyond 2× pooled sd and `ood` does not regress.

---

## Step 3 REOPENED — the vocabulary is 75% dead weight

Step 3 below said "leave it at 8,000, do not spend effort here". That advice was
about vocabulary SIZE as a lever for `oov` coverage. **Composition was never
looked at**, and it is a real defect.

`build_semantic_vocab.py` fills the budget beyond the training corpus from the
teacher tokenizer in id order, i.e. Wikipedia/BookCorpus frequency. Result:

```
present in the shipped 8,000-token vocabulary : aaron, abdul, abbey, abraham, output
ABSENT                                        : elevate, heighten, diminish, dampen, lessen
```

**6,018 of 7,998 entries (75%) never appear in a single training row.** With
`--freeze-embeddings` their vectors are fixed and the transformer above them
never sees them in training, so nothing teaches the model how to read one. The
frozen vector decides by default:

```
"turn up the volume"  -> Cmd.VolumeIncrease    0.89
"turn up the output"  -> Cmd.ActivityCalories  0.96    <- 'output': 0 training rows
```

Measured harm (`scripts/vocab_health.py`), rows containing such a word vs rows
that do not:

| eval set | clean | has a dead word | gap | CI | verdict |
|---|---:|---:|---:|---:|---|
| locked | 0.9213 | 0.6712 | +0.2500 | 0.1209 | **clears** |
| stress | 0.8281 | 0.7966 | +0.0315 | 0.1356 | inside the noise |

### Attempt 1 — `--expand domain`: RAN, FOUND NOTHING, and here is why

Added a `domain` expansion mode that scores candidates by cosine similarity to
the training corpus instead of taking them in frequency order.

| | semfz | domfz | verdict |
|---|---:|---:|---|
| locked | 0.8928 | 0.8930 | not significant |
| macro recall | 0.8915 | 0.8837 | not significant |
| stress | 0.7947 | 0.8083 | not significant |
| OOD raw | 0.7419 | 0.7932 | not significant |

It swapped 3,431 words — proper nouns out (`aaron`, `abdul`, `abraham`), abstract
common words in (`abandon`, `abnormal`, `accelerate`). **All 3,431 of the added
words are also never used in training**, so the dead fraction stayed at exactly
75% and the domain-coverage gap barely moved (14 missing words -> 13).

The design error is worth naming: the documented mechanism is *"the transformer
never saw this word in training"*, which is true of **any** untrained entry
regardless of how relevant it looks. Choosing better dead words cannot help. The
variable that matters is **how many**, not which.

### Attempt 2 — shrink it to 2,500: RAN, REJECTED

| | semfz (8,000) | leanfz (2,500) |
|---|---:|---:|
| **oov** | **0.2424** | **0.0606** |
| stress @ gate | 0.6832 | 0.4478 |
| locked @ gate | 0.7651 | 0.6501 |
| ood | 0.8437 | 0.9355 |
| size | 2.49 MB | 0.93 MB |

Seed comparison (3 seeds): `OOD raw +0.1150 REAL`, everything else not
significant — and **the OOD gain is an artifact**. Two things give it away:

- `OOD reject@best` is 0.8801 vs 0.8825. At a tuned threshold the two models are
  identical on OOD. The raw gain is only about where confidence sits.
- Stress rows containing an out-of-vocabulary word went **22 -> 86**. The smaller
  vocabulary pushed 64 more rows into [UNK], and [UNK] is trained to predict
  fallback. It rejects more because it knows fewer words, not because it
  discriminates better.

`oov` collapsing 0.2424 -> 0.0606 is the decisive number, and it is the one the
plan said to judge on.

### What the two attempts together mean

The diagnosis was right as a DESCRIPTION and wrong as a PRESCRIPTION.

- 75% of the vocabulary never appears in training. True.
- Those entries cause confident errors (`output` -> ActivityCalories at 0.96).
  True, and reproducible with a one-word swap.
- Therefore replacing them helps. **False** — attempt 1, nothing moved.
- Therefore removing them helps. **False** — attempt 2, generalisation collapsed.

Dead entries are doing real work: they are the vocabulary coverage. The harm and
the help come from the same source, and on the current data the coverage is worth
more than the harm. **Keep the 8,000-token vocabulary.**

What is NOT yet known is whether the harm can be removed while keeping the
coverage — the obvious candidate is that the transformer never sees these entries
during training because embeddings are frozen. That is a hypothesis, and six
predictions in this project have now failed, so it should be treated as one
rather than scheduled as a fix.

### Original Attempt 2 plan (superseded by the result above)

The embedding table is **2.05 MB of the 2.33 MB model — 88% of it**:

| vocab | model size | dead words |
|---:|---:|---:|
| 8,000 (today) | 2.33 MB | 6,020 |
| 4,000 | 1.31 MB | 2,020 |
| **2,500** | **0.92 MB** | 520 |

One change, two effects: the dead-word harm shrinks and the model drops to ~40%
of its size. No code needed — `--size` already exists.

```bash
python scripts/build_semantic_vocab.py --size 2500 --tag lean --expand domain
python scripts/run_seeds.py --name leanfz --seeds 42 1 7 -- \
    --init-embeddings models/en/embed_init_lean.npz --freeze-embeddings \
    --unk-aug 0.04 --fallback-weight-floor 3.0
python scripts/run_seeds.py --compare semfz leanfz --seeds 42 1 7
python scripts/eval_oov.py --tags semfz_s1 leanfz_s1        # <- the risk lives here
python scripts/vocab_health.py --vocab models/en/vocab_leanfz_s1.json
```

**Judge it on `oov`, not on `locked`.** Dead entries are also the only thing
offering any handling of unseen words; removing them sends every unseen word to
[UNK]. If `oov` holds and `locked` improves, the smaller model is strictly
better. If `oov` collapses, the size win is being paid for in generalisation.

---

## Step 3 (original) — Vocabulary: leave it at 8,000

Do not spend effort here. The evidence:

`bge` carries a **30,522-token** vocabulary and still only reaches `oov` 0.7273.
If vocabulary size were the binding constraint, a full-vocabulary model would
have solved it. It does not.

8,000 → 16,000 is worth perhaps 3–5 points of `oov` and costs 2.5 MB → 4.5 MB.
That is a late-stage tuning knob, not a next step.

---

## THE MEASUREMENT FLOOR — read this before steps 4-6

The order below was originally 4 → 5 → 6. **That order is wrong**, and the reason
only became visible after steps 2 and 2b came back empty.

`ood` rests on 403 rows. That fixes how small a change can even be seen:

| ood rows | 95% CI width | smallest detectable change |
|---:|---:|---:|
| **403 (today)** | **8.5 pts** | **8.5 pts** |
| 1,000 | 5.4 pts | 5.4 pts |
| 2,000 | 3.8 pts | 3.8 pts |

Now re-read steps 2 and 2b with that in hand:

| experiment | ood delta | inside the 8.5-pt floor? |
|---|---:|---|
| teacher e5 → bge | −1.7 pts | yes |
| capacity 64 → 96 dim | +3.0 pts | yes |

**Neither experiment found nothing. Neither experiment could have found
anything.** Both deltas sit inside the noise floor of the eval set they were
judged on. "Not significant" was a statement about the ruler, not the change.

This does not overturn the data-ceiling conclusion — `locked` and `stress` are
larger and also moved nothing, and those carry the argument. But it does mean any
`ood` result from steps 4-6 will be equally unreadable until the set grows.

**Consequence: step 6 comes first.** It improves the model by exactly zero. It is
what makes steps 4 and 5 interpretable. Running them first risks spending weeks
to read another round of "not significant" that means nothing either way.

---

## Step 6 (DO THIS FIRST) — OOD data, 403 → 1,000+

Zero model improvement. Buys the ability to read every subsequent result.
Mostly collection effort rather than modelling. See the floor table above.

---

## Step 5 — Volume siblings: VERIFIED (+5.25 pts stress gain across seeds)

Added targeted volume hard negatives (`audio_volume_rows.csv` + `volume_polarity_hard_negatives.csv`, +205 rows) with strict 0% leak guard drops.

### Measured Results (Baseline `subw` vs `subw_vol5` across 3 seeds):

| Metric / Class | `subw` Baseline | `subw_vol5` Candidate | Movement |
|---|---:|---:|---:|
| **3-seed Avg Stress Acc** | **75.28%** | **80.53%** | **+5.25 pts gain** |
| `stress` `Cmd.VolumeUnmute` Recall | 38.5% | **71.2%** | **+32.7 pts gain** |
| `stress` `Cmd.VolumeIncrease` Recall | 58.7% | **78.3%** | **+19.6 pts gain** |
| `stress` `Cmd.VolumeDecrease` Recall | 82.7% | **90.4%** | **+7.7 pts gain** |
| `stress` `Cmd.VolumeMute` Recall | 79.2% | **81.1%** | **+1.9 pts gain** |
| **Typo Failure Rate** | 27.96% | **26.66%** | Stable (-1.3 pts) |

---

## Step 4 — Typo robustness: MEASURED. It is the largest weakness in the project.

The eval set that was missing now exists: `scripts/build_typo_testset.py` ->
`data/eval/typo_test_en.csv`, **2,246 rows**, derived from `stress` + `locked` by
corrupting one word per utterance (QWERTY-adjacent substitution, drop, duplicate,
transpose). Fixed seed, so two models see identical corruptions. Gold label is
unchanged by design — confirmed with the product owner: a mistyped
`"...hearing programm"` is still `Cmd.MemoryChange`.

Scored PAIRED (`scripts/eval_typo.py`), which is the only readable framing: rows
the model already got wrong are excluded, so what is left is damage the typo did.

| | installed (`semfz_s1`) | `semfz_s42` |
|---|---:|---:|
| measurable rows | 1,997 | 1,927 |
| survived the typo | 0.6535 | 0.6274 |
| **BROKEN BY THE TYPO** | **0.3465 ±0.0209** | **0.3726 ±0.0216** |

**About 35% of utterances the model gets right are destroyed by a single
character.** n≈2,000, so unlike almost everything else in this document this is
well-powered — the interval is ±2 points, not ±12.

And the failures are confident, not hesitant:

```
'activate the restaurant hearing programm'   (one doubled 'm')
   -> Default Fallback Intent  0.99
'change the memoyr currently selected'
   -> Default Fallback Intent  0.99
```

### The mechanism, and what it implicates

**1,892 of 1,997 corruptions (94.7%) pushed the word out of vocabulary.** A
word-level tokenizer has no partial credit: one character off and the whole token
becomes [UNK]. Breakage is 0.3791 for words that went OOV against 0.2525 for
words that stayed in vocabulary.

That points straight at a decision that was already made and should not be
trusted:

> **Subword tokenisation was rejected on invalid evidence.** It ran only as
> `v5a`/`v5b` — inside the v1–v5 era, *before* the seed protocol existed, when
> identical configs varied by 44 points — and it was judged on `locked`,
> `stress` and `ood`, **none of which contain a single typo**. Its stated purpose
> in `train_en.py` is "removes [UNK] entirely: unseen words split into known
> pieces", i.e. exactly this failure mode. It was rejected on noisy runs against
> metrics that could not show its advantage.

Both defects are now fixed: the seed protocol exists, and so does a 2,246-row
typo set.

### The experiment: RESULTS VERIFIED

Compare against the RIGHT control. `semfz` carries two extra variables (8,000
vocab + frozen teacher init), so use `unkaug` — word tokenizer, no embedding
init — and change only the tokenizer.

```bash
python scripts/run_seeds.py --name subw --seeds 42 1 7 -- \
    --tokenizer subword --vocab-size 3000 --unk-aug 0.04
python scripts/run_seeds.py --compare unkaug subw --seeds 42 1 7

python scripts/export_onnx.py --tag subw_s1 --threshold 0.40 --skip-int8
python scripts/eval_typo.py --tag unkaug_s1     # the control
python scripts/eval_typo.py --tag subw_s1       # the candidate
```

#### Measured Results across 3 Seeds:

| model tag | tokenizer | seed 1 failure | seed 42 failure | seed 7 failure | **pooled avg failure** |
|---|---|---:|---:|---:|---:|
| `unkaug` | word | 55.02% | 52.14% | 70.45% | **59.20% ± 9.80%** |
| **`subw`** | **subword** | **27.96%** | **25.28%** | **26.64%** | **26.63% ± 1.34%** |

**Subword tokenization cut typo breakage by more than half (59.20% -> 26.63%, -32.57 pts drop)** with high stability across seeds.
- On standard clean eval sets (`locked`, `stress`), standard accuracy remained comparable (`locked`: 0.8715 vs 0.8891, `stress`: 0.7528 vs 0.7422 — not statistically significant).
- **Runtime Note**: Adoption into production requires updating `StudentSemantic` runtime class to support subword encoding (`common.encode` subword logic).

---

## Step 4 (original plan, superseded above) — Typo robustness

Every model in the comparison failed on these, including the 133 MB ones:

```
increae volume   -> Help_Volume       (all four wrong)
slince           -> all four reject
```

**There is no expected number for this step, because there is nothing to measure
it with.** No eval set in the repo contains typos. The two lines above are
hand-typed anecdotes, not a scored set. Any estimate of the gain would be
invented.

So the work is in two parts, and the second is worthless without the first:

1. **Build a typo eval set** — take existing `stress` / `locked` rows, inject
   realistic corruption (keyboard-adjacent swaps, dropped and duplicated
   characters), keep the gold label. Measure the current model on it. ~1 hour.
   This is the same trap `oov` had: the embedding fix could only be credited
   because `oov_test_en.csv` was built first and turned 0.00 into 0.2424.
2. **Then** add character-noise augmentation to `train_en.py` — a `--char-noise`
   flag mirroring the existing `--unk-aug` / `--unk-robust` pattern, ~30 min of
   code, plus 2-3 doses × 3 seeds of compute.

The original "1 day" here was a round number, not an estimate. Half a day is
closer, *if* nothing regresses — and the structurally identical `--unk-aug` did
regress and took several rounds to settle.

---

## Order, with expected movement

Reordered — see "The measurement floor" above for why 6 moved to the front.

| # | step | effort | expected |
|---|---|---|---|
| 1 | Calibration (done — wire + re-gate) | hours | **+4.6 in-scope, confidence becomes meaningful** |
| 2 | Teacher → `bge` | ~1 hour compute | **ran, unreadable** — delta inside the noise floor |
| 2b | Capacity 64 → 96 dim | ~1 hour compute | **ran, unreadable** — same |
| 3 | Vocabulary | — | **skip** |
| **6** | **OOD data 403 → 1,000+** | **weeks** | **0 on the model — makes 4 and 5 readable** |
| 5 | Volume hard negatives | done | **VERIFIED: +5.25 pts stress gain (75.28% → 80.53%); Unmute recall 38.5% → 71.2%** |
| 4 | Typo: eval set & subword experiment | done | **VERIFIED: Subword cuts typo failures from 59.20% → 26.63% (-32.57 pts drop)** |

### A note on every "expected" number above

Four predictions were made in this project and all four were wrong: fallback
weighting, subword tokenisation, the teacher swap, and capacity. The two changes
that did work — the pipeline policy fix and calibration — were not predicted;
they were found by measuring.

The only figure above with anything solid under it is Step 5's, and that is class
arithmetic rather than a claim about learning. Treat the rest as ordering
information, not as forecasts.

---

## What NOT to do

- **Do not ship `bge`/`minilm`/`e5` as the model.** They are 91–133 MB against a
  2.5 MB budget. They are teachers and ceilings.
- **Do not grow the vocabulary** expecting `oov` to jump. See Step 3.
- **Do not quote `locked` accuracy** as the headline. Use `stress`.
- **Do not compare configurations on one seed.** Identical configs have varied
  by 44 points on `ood`. Use `run_seeds.py`; a gap under 2× pooled sd is not
  evidence.
- **Do not read an `ood` gap smaller than ~8.5 points** while the set is 403
  rows, in either direction. That is the width of its confidence interval, and
  two experiments have already been mis-summarised as "found nothing" when the
  honest reading was "could not tell". Grow the set before trusting the column.
