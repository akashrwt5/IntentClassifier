# Plan: move the model from word matching to sentence reading

Written for a team review. Every claim below is either measured or marked as an
estimate. Nothing here blocks the current release — the shipped model is safe,
it is just narrower than it should be.

---

## 1. The problem, in one example

On the shipped model:

```
"it's a bit quieter here can you make it louder"
  -> Cmd.VolumeDecrease   0.6937      (runner-up: Cmd.VolumeIncrease 0.2894)
```

The room is described as quiet. The request is to make it louder. The model
followed the description and returned the opposite of what was asked. The gate
refused it — nothing wrong reached the hardware — but a reasonable request was
lost.

The tempting fix is to add "quieter" to a symptom word list. **That is the wrong
fix and it should be rejected in review.** It repairs one word and leaves the
next unseen comparative exactly as broken. This document is about the reason
behind the word.

## 2. Why it happens — three causes, all measurable

**(a) The architecture discards structure right after computing it.**

The pipeline is `4-layer transformer → mean pooling → L2 norm → MLP head`. The
transformer attends across the whole sentence. Mean pooling then averages the 64
token vectors into one:

```python
emb = (h * mask).sum(1) / mask.sum(1)        # scripts/encoders.py:116
```

After that average, "which word was the request and which was the description"
is largely gone. A strongly-polarised token contributes its vector whatever its
grammatical role.

**(b) The encoder was never trained to know what a request looks like here.**

The student was distilled to imitate `bge-small-en-v1.5`, a general-purpose
sentence embedder. Neither the teacher nor the student was ever trained on the
task of deciding which clause carries the instruction. The encoder is frozen;
only the head on top of it learns.

**(c) The training data never posed the question.**

```
corpus rows containing both a quiet-word and a loud-word:   0
```

A distinction that never varies in the training set cannot be learned from it.
The only sentences where both directions co-occur are generated *corrective*
frames ("i wanted it louder, not quieter"), which mean something else — so the
model has reasonably learned that two direction words together signal a
correction.

Measured shortcut strength in the current training set — P(top intent | word
present):

| word | rows | share | top intent |
|---|---|---|---|
| soft | 32 | **1.000** | Cmd.VolumeIncrease |
| faint | 35 | **1.000** | Cmd.VolumeIncrease |
| harsh | 31 | **1.000** | Cmd.VolumeDecrease |
| muffled | 26 | 0.923 | Cmd.VolumeIncrease |
| quieter | 143 | 0.636 | Cmd.VolumeDecrease |

A model that answers from these words is not malfunctioning. It is fitting the
data it was given, and the data offers a shortcut that works almost every time.

**This is the same root cause already documented for corrective negation**
(0.74 on taught pairs, 0.48 on a held-out control). Two independent symptoms,
one disease. That is the argument for fixing the cause rather than either
symptom.

## 3. Measure it first — run this before the meeting

`scripts/structure_probe.py` produces the numbers this plan is judged against.
It runs against the shipped INT8 file, not the training pipeline.

```bash
python scripts/structure_probe.py
```

**Probe 1 — word order.** Real test sentences, then the same sentences with
their tokens shuffled and reversed. A model that reads word order loses accuracy
when order is destroyed; a bag-of-words model does not. The headline is
*accuracy retained when scrambled*.

**Probe 2 — lexical association, controlled.** Four arms. Same request, same
frame, same gold label; only the describing word changes:

```
A  "it is rather faint in here can you make it louder"    -> VolumeIncrease
B  "it is a bit quieter in here can you make it louder"   -> VolumeIncrease
C  "it is rather harsh in here can you make it quieter"   -> VolumeDecrease
D  "it is a bit louder in here can you make it quieter"   -> VolumeDecrease
```

A and C use describing words the training data taught as *symptoms* of that
direction. B and D use words the training data taught as *requests for the
opposite* direction. **A−B and C−D isolate the effect of word association with
meaning, structure and label held constant.** A large gap is the finding; it
requires no interpretation.

**Probe 3 — what the data offers.** The table in §2, computed from whatever
training set is current.

Take the probe output to the review. "Our model retains X% of its accuracy on
scrambled sentences" is a fact. "I think it doesn't understand sentences" is not.

### Measured baseline — shipped model, 22 August 2026

`reports/structure_probe_BEFORE.json`

| | |
|---|---|
| accuracy, normal sentences | 0.9086 |
| accuracy, tokens shuffled | 0.9011 |
| accuracy, tokens reversed | 0.8937 |
| **accuracy retained when scrambled** | **99.2%** |

Order sensitivity is 0.0074. The 2σ noise floor on test accuracy is 0.0045, so
the model's use of word order is barely distinguishable from noise. It is, to a
close approximation, a bag of words.

| arm | describing words | accuracy | wrong AND accepted |
|---|---|---|---|
| A — up, symptom word | faint / soft / muffled | **1.000** | 0.000 |
| B — up, request word | quieter / lower / softer | **0.389** | 0.111 |
| C — down, symptom word | harsh / booming / loud | **1.000** | 0.000 |
| D — down, request word | louder / higher / raised | **0.389** | 0.278 |

**Lexical association effect: +0.611 in both directions.** Same request, same
frame, same gold label, 100% against 39%.

One finding nobody asked for, and the most diagnostic of the three:

| arm | description first | request first |
|---|---|---|
| B | 0.556 | **0.222** |
| D | 0.667 | **0.111** |

Accuracy halves when the request comes *first*. The model is taking the **last**
direction word, not the one in the request clause. That is consistent with probe
1 rather than contradicting it: ordinary sentences contain one direction word,
so order is irrelevant; when there are two, recency decides. There is no
syntactic notion of "which clause is the instruction" — only a weak positional
prior, and it is the wrong one.

## 4. The plan

Four steps, in leverage order. Steps 1 and 2 are cheap and independent — do them
first and re-measure before committing to step 3.

### Step 1 — Remove the shortcut from the data (half a day)

Not "add more words". Make the words *stop being predictive*.

Today "faint" is 100% Cmd.VolumeIncrease. Rebuild the symptom/request generator
so every direction word appears roughly equally as a description and as a
request, in both directions. Then `P(intent | word)` drops toward chance and the
model has no choice but to use position and syntax.

Success criterion, checked in the generator itself: no direction word exceeds
**0.65** share of any single intent in `train_augmented.csv`, asserted at build
time the way the existing leakage and length checks are.

Cost: rewrite of `symptom_pairs.py`, plus one training run. Risk: low. This
alone may substantially move probe 2, and it costs nothing architectural.

### Step 2 — Attention pooling instead of mean pooling (half a day)

Replace the mean with a small learned attention pool: one query vector scores
each token and the output is the weighted sum. It is a few hundred to a few
thousand parameters, it exports to ONNX cleanly, and it lets the model learn
*which tokens matter* instead of weighting all of them equally.

This attacks cause (a) directly and is the cheapest architectural change
available. It also gives a debugging handle: the attention weights show which
word the model actually used.

Cost: change in `encoders.py` and `export_onnx.py`, one distillation run.
Risk: low-medium — size impact negligible, but calibration will shift and
thresholds must be refitted.

### Step 3 — Unfreeze the encoder (one day, plus a noise floor)

`scripts/finetune_encoder.py` already exists. Train encoder and head jointly
with three loss terms:

- cross-entropy on the task
- KL against the teacher (keeps the distilled knowledge from washing out)
- **a margin loss on minimal pairs**: for a pair differing in one token,
  require `p(A | x_a) − p(A | x_b) > margin`. This is the term that makes the
  model use the differing token *in its context* rather than as a bag feature.

Sentence structure lives in the encoder's attention layers. While it is frozen,
no amount of data on top of it can teach structure — which is exactly what the
corrective-negation result already demonstrated.

Cost: fine-tune 5–10 min, then the full chain again (~45 min), plus a **new
noise floor** because a new encoder invalidates the old one (~2 hours).
Risk: real. Calibration will change, coverage may move again, and the size
budget must be re-checked. This is a day, not an afternoon.

### Step 4 — Re-measure against §3 and the noise floor

Pre-registered, before any numbers are seen:

| metric | now | target after |
|---|---|---|
| probe 2, A−B gap | **0.611** | ≤ 0.15 |
| probe 2, C−D gap | **0.611** | ≤ 0.15 |
| probe 2, wrong-and-accepted (worst arm) | **0.278** | ≤ 0.05 |
| probe 2, request-first vs description-first gap | **0.33–0.56** | ≤ 0.10 |
| accuracy retained when scrambled | **99.2%** | ≤ 90% |
| corrective held-out (3 families) | 0.48 | ≥ 0.65 |
| accepted precision | 0.9789 | ≥ 0.97 |
| coverage | 0.658 | ≥ 0.658 |
| INT8 size | 4.75 MB | ≤ 5 MB |

Any change smaller than the 2σ column in `reports/variance.json` is not
evidence. Current 2σ: accuracy 0.0045 · macro-F1 0.0072 · contextual 0.048 ·
minimal pairs 0.025 · hard negatives 0.015 · negation 0.007 · OOD 0.013.

## 5. Options considered and rejected

**Add the missing words to the symptom lists.** Fixes the reported sentence,
leaves the class of sentences broken. This is what was nearly shipped; it is in
this document so the review can see it was considered and why it was dropped.

**A rule: if two direction words appear, take the one after the request verb.**
Tempting and cheap. Rejected as a primary fix — it is a parser written in
regular expressions and will break on phrasings nobody anticipated. It is
defensible only as a *narrow safety net* on the five high-risk intents, where
being wrong is expensive; worth revisiting after step 3 if a residual gap
remains.

**Use a larger or instruction-tuned encoder.** Would very likely solve it. Not
available inside a 2–5 MB fully offline budget. Recorded so the review does not
spend time rediscovering it.

**More data of the current kind.** The corrective-negation result already
settles this: taught pairs 0.74, held-out pairs 0.48. There are thousands of
possible pairs and every untaught one sits near chance. Volume does not buy
structure.

## 6. Severity — corrected after measuring

An earlier draft of this document said the cost of this defect was "coverage,
not safety", because in the one reported example the gate refused the request
(0.6937 against a 0.925 threshold). **Measuring 72 sentences instead of one
shows that was wrong**, and the correction matters more than the original
claim:

```
wrong AND accepted by the gate:    arm B  0.1111      arm D  0.2778

0.934  ACCEPTED  Cmd.VolumeIncrease -> Cmd.VolumeDecrease
       | "can you make it louder, it is on the lower side in here"
```

The user asks for more volume, the aid gives less, and the gate does not stop
it. In arm D that happens to more than one sentence in four. Confidence on these
is high — 0.93 and above — because the model is not uncertain: it is confidently
answering a different question.

So this is a **wrong-action defect, not a refusal defect**. It does not block
the current release, which is safe on everything measured before today, but it
raises the priority of the work below and it should be stated plainly to the
team rather than softened.

## 6b. RESULT — the plan was run, and it answered the question

All four steps were executed. The outcome is not the one the plan expected, and
it is more useful than the one it expected.

### Step 1 (remove the data shortcut) — worked, partially

F16 balanced the six dual-role words. Measured on the frozen encoder:

| | before | after |
|---|---|---|
| lexical association effect | +0.611 / +0.611 | +0.333 / +0.389 |
| arm D wrong AND accepted | 0.278 | **0.000** |
| minimal pairs | 0.705 | 0.773 |
| OOD rejection | 0.9685 | 0.9860 |

The wrong-action defect from §6 was eliminated. Probe 1 did not move
(99.2% → 99.1%), exactly as predicted before the run.

### Step 3 (unfreeze the encoder) — worked, decisively, on the teacher

12 epochs on bge-small (12 layers, 384-dim):

| | frozen | fine-tuned |
|---|---|---|
| test accuracy | 0.9081 | **0.9266** |
| contextual | 0.538 | **0.855** |
| minimal pairs | 0.727 | **0.909** |
| hard negatives | 0.710 | **0.902** |
| accepted precision | 0.9787 | **0.9930** |
| corrective held-out | 0.48 | **0.869** |
| **lexical gap** | +0.611 | **0.000** |

The gap held at 0.000 from epoch 2 through epoch 11 while macro-F1 climbed to
0.9186, so it is not an artefact of an undertrained model. The diagnosis in §2
was correct: structure lives in the encoder, and the encoder was frozen.

### The finding that closes the question: it does not fit in 4 layers

Two independent attempts to get that capability into the 4.75 MB artefact:

**Distillation from the fine-tuned teacher.** Failed. Student: contextual 0.524,
hard negatives 0.627, lexical gap +0.222/+0.500, INT8 coverage **0.000**.
Distillation matches the teacher's OUTPUTS on the training distribution, not its
mechanism. The structure-critical rows are 3.5% of `train_augmented`, so the
student can match teacher logits nearly everywhere using the same lexical
shortcuts and does.

**Fine-tuning the student directly.** Also failed, and more informatively. Best
epoch was **epoch 0**; loss fell 3.13 → 0.01 while validation macro-F1 went
*down*. The gap never came below +0.222/+0.389 in twelve epochs.

```
12 layers x 384:   lexical gap 0.000    corrective held-out 0.869
 4 layers x 256:   lexical gap 0.278    corrective held-out 0.557
```

Same data, same objective, same recipe. **The limit is capacity, not data,
objective or pooling.** Attention pooling (step 2) was never run, and after this
result it is not the next thing to try — a pooling layer cannot supply
representational capacity the encoder does not have.

### What that costs in megabytes

Parameters = `V·H + P·H + L·(12H² + 13H)`, V=3267, P=64:

| config | params | INT8 | reads sentences? |
|---|---|---|---|
| 4 × 256 | 4.01 M | 4.75 MB | **no — measured** |
| 6 × 256 | 5.59 M | ~5.6 MB | untested |
| 8 × 256 | 7.17 M | ~7.2 MB | untested |
| 6 × 384 | 11.9 M | ~12 MB | untested |
| 12 × 384 (pruned vocab) | 22.6 M | ~22.6 MB | **yes — measured** |

Only the two ends are measured. The honest statement is that the capability
needs more than 4 layers and no more than 12; where between those it appears is
one experiment per row.

## 7. The decision this now comes down to

This stopped being an ML question. It is a size-budget question, and it belongs
to the product.

**Option 1 — keep 4.75 MB, accept word-matching.** Rebuild the pre-fine-tuning
student (~30 min: distil from the frozen teacher, `pick_seed`, export). Known
numbers: accepted precision 0.9789, coverage 0.658, contextual ~0.49. It will
keep getting "it's a bit quieter here, make it louder" wrong. F16 means it no
longer *executes* the opposite silently — arm D's wrong-and-accepted is 0.000 —
so the failure is a refusal, not a wrong action.

**Option 2 — raise the budget.** Test 6 × 256 (~5.6 MB) and 8 × 256 (~7.2 MB).
One fine-tune each, about 20 minutes per config, and the lexical gap in the
per-epoch output answers it immediately. This is the only open ML question left
and it is cheap.

**Option 3 — ship the 35 MB teacher.** It works: accuracy 0.9266, accepted
precision 0.9930, coverage 0.666, gap 0.000. Only viable if 35 MB and 4.4 ms are
acceptable on the target hardware.

### Also outstanding, and not optional if any of these ships

**Validation has been used for six nested selections** — teacher epoch, student
epoch, student fine-tune epoch, head seed, temperature, and every gate
threshold. It no longer estimates held-out behaviour, and the symptom is
everywhere in this session's logs: validation coverage 0.736 against test
coverage 0.143 on one run. Before shipping anything, calibration and thresholds
need a split that nothing else has selected on. Every coverage number in this
document is optimistic by an unmeasured amount because of this.

**Fine-tuning weakened the OOD signal.** Mahalanobis AUROC on held-out OOD fell
0.939 → 0.699 as the encoder was fine-tuned — expected, since training for
classification compacts the classes and flattens the distance geometry the score
relies on. `ood_rej` stayed high (0.993) because the reject class and confidence
carry it, but signal 3 of the gate is now much weaker than its documentation
claims.
