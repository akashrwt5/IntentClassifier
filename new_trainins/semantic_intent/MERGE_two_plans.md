# Merging the two plans — what each has that the other needs

Comparing `semantic_compression_plan.md` (rev 5, branch
`feature/Akash/semantic_with_new_csv-DataGeneration-Automation`) with the work
on `feature/new_train`.

---

## 0. The finding that reframes the question

These are not two competing plans. **They are the same product, the same
taxonomy, and two lineages that have never exchanged instruments.**

| | compression plan | this branch |
|---|---|---|
| intents | 57, moving to 60 | 57 |
| intent names | `Cmd.VolumeIncrease`, `Cmd.StreamingStop`, `Default Fallback Intent` … | identical — all 7 named in their document exist here |
| high-risk polarity pairs | Increase↔Decrease, Mute↔Unmute, Start↔Stop | the same three axes, already built as minimal pairs |
| training corpus | `language_packs/en/train.csv`, 8,430 rows | `data/raw/en.csv`, 9,826 rows |
| student | bge-L3, 384 hidden, 9.1 MB | bge→h256/l4, 4.75 MB |
| budget | ≤12 MB | 2–5 MB |
| status | **plan complete, nothing applied** | **executed; findings in hand** |

Two corpora for one taxonomy is itself worth resolving, but the larger waste is
that each side is missing exactly what the other built.

**One sentence version:** their plan is the better *method*; this branch has the
better *instruments* and the only *executed results*. Neither is complete alone.

---

## 1. Take from them → this branch

### 1.1 McNemar + power analysis (their §10) — the most important transfer

They tabulate the smallest difference each instrument size can resolve. Applied
to this branch's suites, the consequence is immediate and uncomfortable:

| suite here | rows | resolves (their table, 15% discordant) | how it has been used |
|---|---:|---:|---|
| structure probe, per arm | 18 | ≈**0.256** | the A−B gap is read as a result |
| accessories | 85 | ≈0.145 | quoted to 3 decimals, treated as decisive |
| minimal_pairs | 88 | ≈0.144 | same |
| contextual | 145 | ≈0.122 | same |
| negation | 156 | ≈0.118 | same |
| ood_test | 286 | ≈0.067 | same |
| hard_negatives | 916 | ≈0.036 | same |
| stt | 1,496 | ≈0.028 | same |

The 5-seed 2σ used here is a **noise floor** — it says how much a number moves
between runs. It does not say how large a difference must be before two models
can be told apart. Those are different questions and this branch has only been
answering the first.

**Consequence to accept:** several suites here are too small to decide anything
on their own, and the structure-probe arms at 18 rows each are the smallest of
all. The +0.611 → 0.000 change survives that easily; a 0.278 vs 0.222 comparison
does not. Adopt McNemar with a pre-registered minimum meaningful effect matched
to each suite's actual power.

### 1.2 Sealed Tier-2, re-sealed per candidate (their §12)

This is the direct answer to the failure recorded in
`ROADMAP_both_short_and_long.md` §7: validation here has been used for six
nested selections and stopped predicting held-out behaviour — one run reported
validation coverage 0.736 against test coverage 0.143.

Their design forbids exactly that, and their refinement matters: a failed
candidate gets a **newly sealed** set rather than one attempt at ship-or-die.

### 1.3 Per-run provenance record (their §2)

They spent a section proving which file a historical run consumed. This branch
had the same confusion today — a training run used a `train_augmented.csv` that
predated two corrections, and it was only caught by noticing a row count.

Adopt their JSON block, extended with what matters here: the augmentation
batches enabled (F1…F16), the dual-role shortcut table, and the encoder's
fine-tune status.

### 1.4 The artifact-contract CI (their P5, ten checks)

A broken artefact shipped into `models/final_student_256/` here and was found by
a person typing sentences into it — high-risk threshold pinned at 0.995, INT8
coverage 0.000. Their checks 2, 5, 6, 7 and 10 would have caught it in CI.

Add two checks their list does not have, both earned here:

11. no per-risk threshold equals the fitter's give-up value (0.995)
12. gated coverage on a fixture set exceeds a floor — an artefact that accepts
    nothing is broken, not cautious

### 1.5 The confidence table (their §11)

Claim, confidence, basis. A better device than prose caveats, and it makes the
difference between "measured" and "expected" impossible to blur.

---

## 2. Take from here → their plan

### 2.1 `structure_probe.py` — an instrument their Tier-2 does not have

Their report card measures paraphrase, negation, multi-clause and short-utterance
**accuracy**. None of those would surface this:

```
"it's a bit quieter here, can you make it louder"  ->  Cmd.VolumeDecrease
```

In a paraphrase set that is one wrong row. The score moves from 0.717 to 0.695
and nobody asks why.

The probe isolates the mechanism instead of counting errors — same request, same
frame, same gold label, only the describing word changes:

```
"it is rather faint in here can you make it louder"   1.000
"it is a bit quieter in here can you make it louder"  0.389
```

That is not an accuracy measurement. It is a controlled experiment, and it names
the cause without interpretation.

### 2.2 Shortcut strength in the training data

Their B5 (resampling) and B7 (44% near-duplicate holdout) are data-quality
defects. Neither measures whether the data **teaches a shortcut**:

```
P(intent | word present):   faint 1.000   harsh 1.000   soft 1.000
```

A model answering from those words is fitting, not failing. This is one pandas
groupby and it changes what "more data" means.

### 2.3 The measured capacity result — and why it threatens their P4

This is the finding they most need, because it lands directly on their Pareto
ladder.

```
12 layers x 384   lexical gap 0.000   contextual 0.855   corrective 0.869
 4 layers x 256   lexical gap 0.278   contextual 0.524   corrective 0.557
```

Two independent attempts failed to move the small model: distillation from the
fine-tuned teacher, and direct fine-tuning of the student. Same data, same
objective, same recipe.

Their P4 searches 3 → 4 → 5 layers. **Their whole ladder sits below the only
configuration measured to work**, and their instruments would not tell them —
their gates are `dev_hard`, paraphrase, retention and FAR, and a model can hold
all four while still answering on word association.

Their §6 layer probe ran on **MiniLM-L6 as a proxy** because HuggingFace was
unreachable. They say so honestly. But it means the architectural claim carrying
their layer decision is unconfirmed, and this branch has the real measurement.

### 2.4 The OOD warning they will hit in P6

Fine-tuning the encoder here cost the Mahalanobis OOD score **0.939 → 0.699
AUROC**. Training for classification compacts the classes and flattens the
distance geometry the score depends on.

Their P6 builds a four-signal ladder including a Fallback-score signal, after a
distillation and adaptation programme. They should expect the same degradation
and design for it rather than discovering it at the release gate.

### 2.5 The gate itself, and its failure modes

Six signals running and measured here, versus four designed there. More useful
than the design are the two ways it broke:

- **The per-risk step applies a second precision target on top of the operating
  point's.** In one run it cost 0.294 of coverage, and the reported validation
  coverage was measured at the *first* threshold while the gate shipped the
  *second* — 0.6827 printed, 0.431 delivered.
- **The operating-point search fell through to "maximise precision"** when its
  constraints were infeasible. Asking for `--min-coverage 0.80` returned
  coverage 0.299 — the refuse-everything corner its own docstring warns about.

Both are fixed here. Both would otherwise have been rediscovered there.

### 2.6 `acceptance_test.py` — OK / ASKS / WRONG

Every instrument in their plan speaks to an ML reader. A hearing-aid product
also needs a number a tester can act on, and the distinction that matters is not
accuracy: a refusal is recoverable, a wrong action is not.

---

## 3. Decide once, not twice

Same taxonomy means these should not be settled independently:

| decision | why joint |
|---|---|
| **Which corpus** — 8,430 vs 9,826 rows for 57 intents | two datasets for one taxonomy is duplicated authoring and divergent labels |
| **The high-risk intent list** | their SC-3 and this branch's `risk_tiers` must not disagree about what is dangerous |
| **The polarity pairs** | their §12 list is this branch's minimal-pair axes |
| **help→action** (their ND-14) | this branch's P1 result-vs-how-to policy, already written in `configs/intents.yaml` |
| **The size budget** | see §4 — the evidence says this is now one question, not two |

Their R1 — *"task adaptation eats generalisation,"* the adapted student 8.7
points behind a generic reference on paraphrases — and this branch's lexical
shortcut are **the same disease found from two directions**. That agreement is
the strongest evidence either document contains, and neither knows it.

---

## 4. The one experiment that decides both budgets

Only two configurations have been measured against the structure probe:

```
12 x 384   works
 4 x 256   does not
```

Everything between is unknown, and both projects have their budget sitting in
that gap. Their ceiling at 384 hidden is 5–6 layers (their §7: 5 layers ≈ 11.96
MB, 6 layers with FFN 1024 ≈ 11.37 MB). This branch's is 4 layers at 256.

**Neither team knows whether their budget can hold semantic structure.**

The experiment: fine-tune bge students at 4, 6 and 8 layers, 384 hidden, and
read the lexical gap per epoch. About 20 minutes each on an M-series Mac,
`scripts/finetune_encoder.py` already prints the number.

The result sets both budgets from evidence instead of from argument. If the
answer is 8 layers, this branch's 5 MB target cannot carry it and their 12 MB
one can. If it is 5, both are fine. If it is 10, both budgets have to move and
that is a conversation with the product owner, not with a model.

**Run this before either P4 or this branch's Phase 1b.** It is the cheapest
experiment in either document and the only one whose answer neither team can
proceed without.

---

## 5. What "perfect" is, honestly

It is not available. 57 intents on ~5,000 genuinely distinct examples, fully
offline, in single-digit megabytes, with a tail class at 53 examples — no
architecture fixes a thin tail.

What the two plans together can produce, which neither can alone:

| | from |
|---|---|
| a model that reads sentences, not words | measured here; capacity question open |
| decisions backed by tests that can actually resolve them | their §10 |
| numbers that survive being selected on | their sealed Tier-2 |
| artefacts that cannot ship broken | their P5 + the two checks added in §1.4 |
| a failure mode named before it is discovered | this branch's probes |
| a tester who can report something actionable | `acceptance_test.py` |

That is a materially better model than either branch reaches alone, and it is a
week or two of work rather than a research programme.

**The first move is not technical.** It is the two branches exchanging
`structure_probe.py` in one direction and §10 in the other, and agreeing which
corpus is the corpus. Everything else follows from that.
