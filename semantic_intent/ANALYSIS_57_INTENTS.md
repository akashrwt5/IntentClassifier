# What a semantic model can actually do on the 57-intent set

Measured on `datasets/04_GENERATED_MASTER_training_data.csv` (9,826 phrases,
57 intents) with the frozen MiniLM encoder + linear head from this package.
Evaluation uses 5-fold **grouped** CV (politeness variants never straddle a
fold) and `datasets/semantic_holdout_100.csv`, which is genuinely unseen.

---

## Headline numbers

| measure | TF-IDF | semantic |
|---|---|---|
| 5-fold grouped CV, 57 intents | — | acc 0.906 / macro-F1 0.905 |
| same, fallback removed from the class set | — | **acc 0.938 / macro-F1 0.928** |
| HARD holdout (100 unseen paraphrases) | 78% | **81%** |
| mean confidence when correct, HARD holdout | 0.59 | **0.77** |
| correct answers below 0.5 confidence | 25/100 | **10/100** |
| OOD AUROC vs 523 curated out-of-scope | none | 0.897 |

The accuracy gap is 3 points. The confidence gap is the real one: TF-IDF gets a
quarter of its correct answers right at under 50% confidence, which any
sensible gate would route to fallback anyway.

---

## Finding 1 — data volume is not the bottleneck

Cap phrases per intent, retrain, measure the HARD holdout:

| cap/intent | rows | holdout acc | mean conf | correct under 0.5 conf |
|---|---|---|---|---|
| 10 | 570 | 54% | 0.31 | 46 |
| 20 | 1,140 | 64% | 0.48 | 36 |
| 40 | 2,280 | **78%** | 0.55 | 31 |
| 80 | 4,076 | 78% | 0.67 | 21 |
| 150 | 5,575 | 79% | 0.70 | 16 |
| 300 | 6,701 | 83% | 0.71 | 16 |
| all | 9,826 | 81% | 0.77 | 10 |

**40 phrases per intent already reaches 78%. The remaining 7,500 phrases buy
3 points.** Accuracy plateaus around 40–80; only *confidence* keeps improving.

And across all 57 intents, `correlation(phrase_count, CV recall) = −0.26`.
Negative. `Cmd.MemoryChange` has 1,884 phrases and 50% holdout accuracy;
`Help_FindMyHearingAids` has 264 and scores 100%. More rows of the same
template shape teaches nothing new — `Cmd.MemoryChange` has a unique-core
ratio of 39%, meaning 61% of its rows are politeness variants of each other.

**Requirement: ~150 *distinct* phrasings per intent, then stop.** Budget
paraphrase diversity, not row count. `data.audit()` reports the unique-core
ratio; below ~70% you are inflating, not collecting.

---

## Finding 2 — 86% of errors are three structural problems

Breakdown of all 922 CV errors:

| error type | count | share |
|---|---|---|
| involves `Default Fallback Intent` | 455 | **49%** |
| command ↔ help act flip (`reminders.add` ↔ `Help_Reminder`) | 214 | 23% |
| `Help_X` ↔ `Help_Y` (33 help intents, adjacent topics) | 129 | 14% |
| everything else | 124 | 13% |

None of these is a modelling problem, and none is fixed by more data.

### 2a. Fallback must be a gate, not a class (biggest single win)

`Default Fallback Intent` is currently the 57th class, with 1,308 phrases and
a CV recall of 0.72 — the worst of all 57. It competes with real intents and
appears in half of every error made.

Removing it from the label space and rejecting via the prototype OOD gate:

```
acc      0.906 -> 0.938     (+3.2 pts)
macro-F1 0.912 -> 0.928     (on the 56 real intents)
```

This is the highest-value change available, and it costs nothing but a
retrain. "Not one of my intents" is a distance question, not a class question —
a discriminative head has no way to represent it.

### 2b. Every `Cmd.X` has a `Help_X` twin

33 `Help_*` intents against 21 `Cmd.*`. `"remind me to take my pills"` and
`"how do I set a reminder"` differ by speech act, not topic, and 23% of errors
are that flip.

**A hierarchical act→topic model does not fix it.** Measured:

```
stage 1  command vs help          95.5%
stage 2  topic within command     97.0%   (23 intents)
stage 2  topic within help        95.3%   (33 intents)
         combined ceiling         ~91.7%
```

That is *worse* than the flat 93.8%, because a stage-1 mistake is
unrecoverable. Keep the flat classifier. Fix this in the data instead: help
phrases need interrogative framing ("how do I", "what does", "can the app")
and command phrases need imperative framing, consistently, in every intent.

### 2c. 33 help intents are too fine-grained

`Help_Home` ↔ `Help_WhatsNew`, `Help_DeviceSettings` ↔ `Help_AppSettings`,
`Help_Battery` ↔ `Help_SelfCheck`. These boundaries are not reliably
distinguishable from a single utterance — `"give me a quick tour"` is
labelled `Help_Home` but `Help_WhatsNew` is a defensible reading.

Either merge them into ~12 help topics and disambiguate in a follow-up turn,
or accept that this cluster caps out around 95%.

---

## Finding 3 — rejection gets much harder at 57 intents

| intent set | OOD AUROC (in-domain vs out-of-scope) |
|---|---|
| 11 intents (`balanced_intents_final.xlsx`) | 0.998 |
| 56 intents, vs 523 curated OOS | 0.897 |
| 56 intents, vs 1,308 fallback phrases | 0.806 |

With 57 intents the model covers so much semantic ground that few utterances
are far from everything. This is the binding constraint on "confidently right
or confidently silent", not classification accuracy.

Mitigations, in order of expected value: more and more varied OOS examples for
threshold fitting; per-intent thresholds rather than one global value (a tight
intent like `Cmd.BatteryLevel` can afford a higher bar than a broad one like
`Cmd.MemoryChange`); and a second-stage verification turn for low-margin
accepts.

---

## Finding 4 — `semantic_benchmark_250.csv` is 100% leaked

All 249 rows appear verbatim in `04_GENERATED_MASTER_training_data.csv`. Any
score reported on it is a memorisation score. `semantic_holdout_100.csv` is
clean (0/100 leaked) but only covers 9 of 57 intents, 10 utterances each.

**Requirement: a real evaluation set before anything else.** ~20 unseen
paraphrases per intent, roughly 1,100 utterances, written by someone who has
not read the training data, with a leakage guard in CI. Without it there is no
way to tell whether a change helped.

---

## What to build, in order

1. **Drop fallback from the class set; reject with the OOD gate.**
   +3.2 pts accuracy, removes half of all errors. One retrain.
2. **Build a clean 1,100-utterance holdout.** Everything downstream is
   unmeasurable without it. Add a CI leakage guard.
3. **Fix the label space** — merge the over-fine help topics, and enforce
   consistent interrogative-vs-imperative framing across every Cmd/Help twin.
   This is where the remaining 37% of errors live.
4. **Rebalance rather than grow.** Cap the inflated intents, and add *distinct*
   phrasings to the ~55-phrase ones until every intent has ~150 unique cores.
   Ignore total row count.
5. **Fit OOD thresholds per intent** on real logged traffic.
6. Only then consider a larger encoder. Nothing measured here is
   encoder-limited.

### Realistic expectation

With steps 1–4, roughly **94–96% on a clean held-out set** and **85–88% on
deliberately hard paraphrases**, at 24 MB and under 1 ms per utterance.

"Say anything around the intent and always get it right with high confidence"
is not reachable with 57 intents where ~14 of them overlap by construction.
What *is* reachable: high accuracy on the ~40 well-separated intents, and
honest low confidence on the overlapping cluster, routed to a clarifying
question instead of a guess.

---

## Multilingual note

All of the above is English-only (MiniLM-L6-v2). For en/fr/de/da the encoder
becomes `paraphrase-multilingual-MiniLM-L12-v2` — ~120 MB at INT8, or **~35 MB**
with the vocabulary pruned to the four languages (250k of its 118M parameters
are token embeddings, 81% of the model). Every finding above is
language-agnostic and carries over; the label-space problems get worse, not
better, with translation.
