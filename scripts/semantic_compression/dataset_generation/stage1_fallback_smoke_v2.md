# Stage 1 Quality Report

1 intents, 49 utterances.

## 0. Coverage vs budget

An intent can end a run short of its budget without anything looking
wrong: batches that keep failing validation are abandoned, and API
errors do the same. Nothing else in this report compares output against
what was asked for, so a truncated intent is otherwise invisible -- and
truncation lands hardest where the budget is largest, which is Fallback,
the class that decides FAR.

| Intent | have | want | coverage |
|---|---:|---:|---:|
| `Default Fallback Intent` | 49 | 800 | 6%  ⚠ |

**1 intent(s) finished short of budget: `Default Fallback Intent`.** Rerunning `generator.py`
resumes from the store and will attempt the shortfall again.

## 0b. Quota compliance

**`Default Fallback Intent`** — the prompt states these as counts per batch; the
targets below are scaled to the 49 rows produced.

| constraint | asked | got | |
|---|---|---:|---|
| difficulty Easy | ~17 | 18 | |
| difficulty Medium | ~20 | 22 | |
| difficulty Hard | ~12 | 9 |  **OFF** |

## 1. Diversity vs seed  *(the headline metric)*

Mean pairwise token distance. The premise of this project is that the
legacy seeds are permutation-heavy, so `gain` should be clearly positive.
A gain near zero means the generator is rewording rather than extending.

| Intent | n | seed | generated | gain |
|---|---:|---:|---:|---:|
| `Default Fallback Intent` | 49 | 0.969 | 0.948 | -0.021  ⚠ |

**Mean gain across intents: -0.021**

## 1b. Saturation — is the budget right?

Two signals, both taken in generation order.

`new vocab per quarter` is how many previously-unseen words each
quarter contributed. A generator with semantic room left keeps
introducing vocabulary; one that has run dry recycles it.

`last-quarter redundancy` is the share of the final quarter that
restates something from the first three.

Together these answer the budget question empirically rather than by
guesswork. Note that mean pairwise distance does NOT work here -- on
short utterances it sits near 1.0 and barely moves, so it reads
saturated whatever the truth is.

| Intent | n | new vocab per quarter | last-quarter redundancy | read |
|---|---:|---|---:|---|
| `Default Fallback Intent` | 49 | 74 → 77 → 38 → 48 | 0% | **still climbing — raise budget** |

## 2. Internal redundancy

Share of utterances that overlap another in the same intent by ≥75% of
their tokens. This is what the avoid-list is supposed to prevent.

| Intent | near-dupes | vocab novelty | mean words | sd |
|---|---:|---:|---:|---:|
| `Default Fallback Intent` | 0.0% | 35.4% | 11.1 | 3.7 |

## 3. Type and difficulty mix

| Type | count | share |
|---|---:|---:|
| Observation | 20 | 40.8% |
| Conversation | 12 | 24.5% |
| ExplicitCommand | 8 | 16.3% |
| Question | 5 | 10.2% |
| Negation | 4 | 8.2% |

**Indirect share (Implicit + ObservationPlusCommand): 0.0%**

This is the number to watch. A run dominated by ExplicitCommand has not
produced the indirect phrasings the architecture asks for, however
diverse it looks lexically.

| Difficulty | share |   | Source | share |
|---|---:|---|---|---:|
| Medium | 44.9% |   | LLM-Generated | 100.0% |
| Easy | 36.7% |   |  |  |
| Hard | 18.4% |   |  |  |

## 4. Boundary leakage

For each generated utterance, which intent's SEED centroid is it
actually nearest? The misses are split by how far away they landed,
because the two halves mean opposite things.

**FAR** — nearest centroid sits in a different region of the space
(similarity below 0.65). Review these; do NOT assume they
are defects. Checked by hand on the first real run, all three FAR rows
were correctly labelled — they were simply the batch's hardest
utterances, sitting closest to a neighbouring region because one
clause of a compound pulled that way. That makes them the best
candidates for `dev_hard.csv`, not deletion candidates. What a FAR
figure genuinely catches is gross drift: an utterance about batteries
filed under a volume intent would land near 0.2 with a wide margin.

**NEAR** — nearest centroid is a neighbour sitting almost on top of
this one. Measured on this corpus, `Cmd.VolumeMute` and
`Cmd.VolumeUnmute` are 0.902 apart despite being opposites, and
`Cmd.VolumeIncrease` vs `Cmd.VolumeDecrease` 0.876, while an
unrelated intent (`Help_Battery`) sits at 0.38. A sentence embedding
captures what an utterance is ABOUT and barely captures direction or
speech act, so among such neighbours the winner is close to a coin
toss. A NEAR figure is not a data defect and chasing it wastes review
time.

The cut is centroid distance, not the `families` map: families are
generation scaffolding and do not track topic. `Cmd.VolumeIncrease`
and `Help_Volume` are in different families yet sit 0.822 apart.

What a high NEAR figure DOES say is that the boundary is not
separable by topic alone — which is precisely the case Stage 3's hard
negatives exist to teach, and `command_help_pairs` exists to name.

This metric answers *did the generator drift off-product?* It cannot
answer *is this row labelled correctly?* — the resolution is below
the granularity of this taxonomy, and it never could.

| Intent | FAR (review) | into | NEAR (noise) | into |
|---|---:|---|---:|---|
| `Default Fallback Intent` | 55.1% | `Cmd.VolumeDecrease` | 0.0% | `-` |

## 5. Rejections

9 rejected batches. Most common causes:

- duplicates an utterance already accepted — ×4
- intent is 'ImplicitCommand', must be 'Cmd.VolumeIncrease' — ×3
- labelled ImplicitCommand but states no need and requests not — ×2
- 22 words exceeds the 20-word limit for type Question — ×2
- intent is 'ExplicitCommand', must be 'Cmd.VolumeIncrease' — ×1
- intent is 'ObservationPlusCommand', must be 'Cmd.VolumeIncre — ×1
- 23 words exceeds the 20-word limit for type Question — ×1
- 21 words exceeds the 20-word limit for type Observation — ×1

---

## What this cannot tell you

No metric here catches an **invented capability**. An utterance like
"set a sleep timer on my hearing aids" scores as diverse, on-topic and
well-formed; it is only wrong because the product has no sleep timer.
That still needs a person — but on a sample of 30-50 utterances, chosen
from the Hard slice and the highest-leakage intents above, not on all of
them.
