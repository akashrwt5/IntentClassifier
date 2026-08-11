# semantic_intent

On-device semantic intent classifier: **frozen MiniLM encoder + calibrated
linear head**, exported as one 24 MB ONNX file.

Replaces the TF-IDF bag-of-words classifier, whose failure mode was that a
word's *presence* decided the intent regardless of its role in the sentence:

```
"it's too quiet here, can you make it louder"   ->  device.volume.DECREASE   # wrong
```

`quiet` carries decrease mass in the corpus (88 decrease vs 19 increase rows),
so the sum of unigram weights lands on decrease even though the *action* word
is `louder`.

---

## Quick start

```bash
# train + export (writes models/semantic_intent.onnx)
python -m semantic_intent.train --data datasets/balanced_intents_final.xlsx

# evaluate the exported artifact
python -m semantic_intent.evaluate --data datasets/balanced_intents_final.xlsx

# predict
python -m semantic_intent.predict "it's too quiet here can you make it louder"
```

```python
from semantic_intent import SemanticIntentClassifier

clf = SemanticIntentClassifier()
r = clf.predict("it's too quiet here can you make it louder")
# Prediction(intent='device.volume.increase', confidence=1.00, ood_score=0.79, accepted=True)

r.routed_intent   # 'Default Fallback Intent' when the gates reject
clf.top_k("dial it back", k=3)
```

---

## Results

Corpus: `balanced_intents_final.xlsx` — 20,724 phrases, 11 intents, perfectly
balanced (1,884 each). Split by **core group**, seed 0. Both models get the
same split and the same contrastive augmentation, so the comparison isolates
the encoder. Reproduce with `python -m semantic_intent.baseline
--data ... --compare-semantic --augment 600`.

| metric | TF-IDF (1.6 MB) | semantic (24 MB) |
|---|---|---|
| held-out accuracy (grouped) | 0.9913 | 0.9932 |
| hard paraphrases, unseen wording (37) | 36/37 | 37/37 |
| — low vocabulary overlap (<70% content words seen) | 4/4 | 4/4 |
| antonym pairs (4) | 4/4 | 4/4 |
| mean confidence on hard set | 0.87 | **0.94** |
| out-of-scope rejection | **none** (16/42 pass a 0.7 gate) | **37/42** |
| ECE after temperature scaling | — | 0.0034 |
| latency per utterance (2 threads, CPU) | <1 ms | 0.8 ms (p95 1.2 ms) |

**Read this table honestly.** On accuracy the encoder buys almost nothing here
— +0.2pt held-out, +1 hard paraphrase. On a template-generated corpus with 11
well-separated intents, a bag-of-words model is genuinely competitive, and the
antonym bug is fixed by the *data*, not the encoder (see below).

What the 24 MB actually buys:

1. **Rejection.** TF-IDF has no way to say "not my job" — 16 of 42 clearly
   out-of-scope utterances clear a 0.7 confidence gate. The prototype gate
   rejects 37 of 42. For a hearing-aid app that routes rejects to a GenAI
   fallback, this is the difference between a sensible answer and muting
   someone's aids because they asked about the weather.
2. **Calibrated confidence.** Mean 0.94 vs 0.87 on unseen phrasings, with an
   ECE of 0.003. A correct prediction below the gate is still a fallback, so
   confidence quality is accuracy in practice.

If neither of those matters for a deployment, ship the TF-IDF model — it is
15x smaller.

### Gate thresholds are the weakest part

In-domain and out-of-scope OOD scores overlap: realistic in-domain bottoms out
at 0.52, out-of-scope tops out at 0.65. No single threshold separates them.
The shipped value (0.556) is Youden's J on 37 hard paraphrases vs 42
out-of-scope probes — a tiny sample standing in for real traffic. Five leaks
survive it (`"set a timer for ten minutes"`, `"call my daughter"`, `"turn on
the kitchen lights"`, `"start a workout"`, `"how do i get to the station"`),
all of which are arguably requests the app should support rather than reject.

Retune `ood_threshold` and `conf_threshold` on logged production utterances as
soon as they exist. They are stored in ONNX metadata and can be overridden at
construction time without retraining:

```python
SemanticIntentClassifier(ood_threshold=0.60, conf_threshold=0.50)
```

Fused-ONNX vs Python parity: `max |Δp| = 1.6e-06`, argmax agreement 100%.

---

## Design notes

### Grouped splits, not random splits

36.9% of corpus rows are politeness variants of another row ("turn it down",
"please turn it down", "can you turn it down"). A random split scatters
siblings across train and test and reports memorisation as accuracy.
`data.grouped_split` assigns whole *core groups* — the utterance with
politeness affixes stripped — to one split, and asserts no group straddles two.

### Contrastive augmentation is what fixes the antonym bug

Swapping TF-IDF for a semantic encoder **did not** fix it. Measured, with the
encoder alone the canonical pair still collapsed to decrease/decrease. The
corpus contained state+action phrasings in only 0.2% of rows; a model cannot
learn "the verb decides, the adjective is context" from that, whatever the
architecture.

`augment.py` generates ~600 state×action crossings per volume intent into the
**train split only** (dev and test stay untouched, so the score stays honest).
That single change moved the semantic model from 1/4 to 4/4 on antonym pairs
and 31/37 to 37/37 on hard paraphrases — and it fixes the TF-IDF model too.
**The augmentation is the fix; the encoder is a separate decision about
rejection and calibration.**

It also generates complaint-only phrasings with no action verb —
`"i can't make out a single word she is saying"` → `volume.increase`. These
were not merely misclassified before; they scored so far from every prototype
that the OOD gate rejected them outright, despite being among the most common
things a hearing-aid user actually says.

Deliberate overlaps in the lexicons force the rule rather than a shortcut:
`"too strong"` is a loud state while `"make it stronger"` is an up action, so
the adjective alone cannot carry the decision. Venue templates
(`"this restaurant is deafening, tone it down"`) separate volume requests from
`device.memory.change`, since venue words double as hearing-aid program names.

### Confidence cannot reject; only the prototype score can

A linear head always names a class, so softmax confidence cannot express
"none of these". This is not a theoretical worry — out-of-scope utterances are
routinely *confidently* wrong:

```
"turn on the kitchen lights"     -> reminders.task.create   conf 0.97
"set a timer for ten minutes"    -> reminders.task.create   conf 1.00
```

Top1-minus-top2 margin is no better: out-of-scope margin median 0.59 against a
hard-set 5th percentile of 0.28 — the two populations sit the wrong way round.

So the model exposes two independent signals answering different questions:

* `confidence` — given this is in scope, which intent? (temperature-scaled)
* `ood_score` — is it in scope at all? Max cosine to a k-means prototype of
  the training embeddings.

Measured AUROC in-domain vs out-of-scope: **0.998 for the prototype score**
against 0.968 for max softmax probability. 64 prototypes per class matches the
separation of keeping all 14.5k training vectors at 1/20th the size.

An utterance must clear both gates, but `conf_threshold` is deliberately set
below the weakest correct in-domain prediction (0.38) so it does no rejection
work. A tighter confidence gate does catch two more out-of-scope probes — by
luck, not by signal — at the cost of vetoing correct answers like
`"dial it back please"` (correct at 0.44). Rejecting a real command is worse
than leaking one that the app arguably ought to support anyway.

### Linear head, not an MLP

An `MLPClassifier(512, 256)` gains +0.1pt in-distribution, loses 2 points on
held-out paraphrases, and is far more overconfident on out-of-scope input
(mean 0.83 vs 0.52) — for 78x the parameters. The linear head is 4,235
parameters and both smaller and better behaved.

### Batch size 1 everywhere

The INT8 encoder uses *dynamic* quantisation, so `DynamicQuantizeLinear`
derives its scale from the whole input tensor. Padding a short sentence to a
longer neighbour's length changes its own embedding — cosine 0.982 between the
same sentence encoded alone vs. in a padded batch of four. The device runs one
sentence at a time, so training must too. Ignoring this cost 3 points of hard
paraphrase accuracy and broke ONNX parity (`max |Δp| = 0.16`). It is free:
throughput is ~1,270 utterances/s either way at these sequence lengths.

---

## Artifacts

| file | size | notes |
|---|---|---|
| `models/semantic_intent.onnx` | 24.0 MB | encoder + pooling + head + OOD, batch 1 |
| `models/minilm-vocab.txt` | 226 KB | WordPiece vocab for the runtime tokeniser |
| `models/semantic_intent_head.npz` | ~540 KB | head + prototypes, for retraining/inspection |

The encoder is 22.9 MB of that 24.0 MB. Everything task-specific — 11 intents,
thresholds, labels, prototypes — is ~1.1 MB. Retargeting to a new intent set
means retraining the head, not the encoder.

Labels and thresholds are stored in ONNX `metadata_props`, so the runtime
cannot drift out of sync with the weights.

---

## Multilingual (fr/de/da)

MiniLM-L6-v2 is English-only; this model is English-only. For the repo's four
languages you need a multilingual encoder, and size is dominated by the
embedding matrix, not the transformer:

| option | params | fp32 | INT8 |
|---|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | 118M | ~470 MB | ~120 MB |
| ↑ vocab pruned to en/fr/de/da (~32k tokens) | ~34M | ~135 MB | **~35 MB** |

250,002 × 384 embeddings = 96M of the 118M parameters (81%). Dropping the
~218k tokens the four languages never emit is the single largest size win, and
it is lossless for those languages. Everything else in this package — grouped
splits, augmentation, calibration, dual gates, fused export — is
language-agnostic and carries over unchanged.

---

## Known gaps

* Five out-of-scope probes clear the OOD gate, mostly landing on
  `reminders.task.create` — `"set a timer for ten minutes"`, `"call my
  daughter"`, `"turn on the kitchen lights"`, `"start a workout"`, `"how do i
  get to the station"`. All are plausibly features the app should have; adding
  intents is a better fix than tightening the threshold.
* No `reminders.task.list` intent exists — `"read out my pending to dos"` is
  correctly rejected rather than classified. Note that `help.reminder.show` is
  help *about* the reminder feature ("how do I set a reminder"), not a request
  to list reminders.
* Residual held-out confusions are almost all within the volume family
  (`mute` ↔ `unmute`, `decrease` ↔ `increase`), 4 cases each out of 3,087.

When a real utterance is misclassified, add it to `eval_sets.py`. Those lists
are the regression suite.
