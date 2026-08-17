# Deferred experiment — can a classifier trained on generated data handle real speech?

**Status:** NOT RUN. Deferred deliberately on 2026-08-17. Nothing in the pipeline
depends on it, but it answers a question that is currently being decided by
judgement instead of measurement.

**Owner:** unassigned.

---

## 1. The question

Will a classifier trained *only* on the synthetic Super Dataset recognise
`Cmd.VolumeIncrease` — and separate it from its siblings — when it is fed the
utterances real users actually said?

Two sub-questions, and they have different answers:

| | Confidence before the experiment |
|---|---|
| Separating the intent from distant intents (Battery, Pairing, Fallback) | high |
| Separating it from `Cmd.VolumeDecrease` / `VolumeMute` / `VolumeUnmute` / `Help_Volume` | **unknown** |

The second is the one that matters, because that is where False-Accept Rate is
won or lost, and it is currently unmeasured.

## 2. Why it is worth running

Three measurements from the first real Stage 1 batch (gpt-5.1 via Azure, 180
utterances for `Cmd.VolumeIncrease`) motivate it.

**(a) The generated length distribution does not resemble the real one.**

| utterance length | generated (180) | real seeds (68) |
|---|---:|---:|
| 1-3 words | 0.0% | 30.9% |
| 4 words | 0.6% | 26.5% |
| 5-7 words | 13.9% | 33.8% |
| 8-12 words | 70.0% | 8.8% |
| 13+ words | 15.6% | 0.0% |

Real users say "Louder" and "Turn it up" 57% of the time. One generated
utterance out of 180 is that short. The model would be fit almost entirely on
long sentences and then asked, most of the time, to classify very short ones.

**(b) Compound utterances are 65% of the generated set and 0% of the real one.**
Not one of the 68 real seeds pairs an observation with a command. This is also a
`Default Fallback Intent` problem, not only a frequency mismatch: Fallback is
*defined* by pure observations, so filling a command intent with
observation-bearing rows weakens the exact feature that separates them.

**(c) Sibling centroids are nearly coincident**, measured with
`all-mpnet-base-v2` over the seed corpus:

```
Cmd.VolumeMute     vs Cmd.VolumeUnmute     0.902   <- opposite intents
Cmd.VolumeIncrease vs Cmd.VolumeDecrease   0.876   <- opposite intents
Cmd.VolumeDecrease vs Help_Volume          0.850
Cmd.VolumeIncrease vs Help_Volume          0.822
Cmd.VolumeIncrease vs Help_Battery         0.379   <- control
```

A sentence embedding captures what an utterance is *about* and barely captures
direction or speech act. A trained head can still find a boundary in that
region, but the signal has to come from somewhere — which is what Stage 3's hard
negatives are for, and they do not exist yet.

**A prediction worth testing:** TF-IDF + logistic regression may beat the
embedding on *direction*, because `up/higher/louder` and `down/lower/quieter`
are distinct tokens and a linear model reads them directly. The same model
should be much weaker on ASR corruption, where `kwyet` and `here-ing aid` are
out-of-vocabulary and the feature vector collapses. If that split shows up, it
is an argument about the runtime architecture, not just the dataset.

## 3. Why the real seeds are a legitimate holdout

Architecture Section 8 asks for a Tier-2 set "outside the generator's prompt
lineage", and suggests human-authored text or a second model. A second model is
still synthetic and still anticipates.

The legacy Dialogflow export is real production ASR from real hearing-aid users.
It is the only material available that the generator demonstrably did not
invent, and it is already on disk. Two constraints:

- Only 8 seeds per intent ever reach a prompt (`generation.seed_reference_count`),
  and none of the corpus enters `train.csv`, so the overlap is small and known.
  For a strict result, exclude the 8 shown seeds per intent from the test set —
  `sample_seeds_for_intent(config, phrases, k=8)` returns exactly which.
- `Default Fallback Intent` seeds are raw production transcripts containing
  customer PII. They may be used locally. They must never be committed or sent
  to any API. Keep any test artefact gitignored.

## 4. Procedure

Five intents — one family plus its Help counterpart, the hardest available case.

**Step 1. Generate the four missing intents** (`Cmd.VolumeIncrease` is done):

```bash
cd scripts/semantic_compression/dataset_generation
python3 generator.py --only Cmd.VolumeDecrease Cmd.VolumeMute Cmd.VolumeUnmute Help_Volume
```

Budgets: VolumeDecrease 180, the rest 120 each → 32 LLM calls, roughly 110k
tokens total. Confirm with `--dry-run` first.

**Step 2. Build the two datasets.**
- TRAIN: the generated rows only, from `.checkpoints/stage1/*.jsonl`.
- TEST: `corpus.intents[i]` for the same five intents, minus the 8 seeds shown
  in each prompt. Roughly 300 real utterances.

**Step 3. Train two classifiers on TRAIN, score both on TEST.**
- `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)` + `LogisticRegression`
- `all-mpnet-base-v2` embeddings + `LogisticRegression`
  (mpnet is the dedup judge, deliberately a different family from the
  `bge-small` teacher being distilled — keep it that way here too)

**Step 4. Report, sliced.** Overall accuracy is the least interesting number.

- Accuracy on the TEST rows of **≤4 words** vs **5+ words**. This is the
  headline: it is the direct test of finding (a).
- Confusion matrix over the five labels. Watch two cells specifically:
  `VolumeIncrease -> VolumeDecrease` (direction), and
  `VolumeIncrease -> Help_Volume` (command vs question, i.e. FAR).
- FAR proxy: share of `Help_Volume` TEST rows predicted as any `Cmd.*`.
- TF-IDF vs embedding, per slice, not just overall.

## 5. Decision rules — write these down before looking

| Result | What it means | Action |
|---|---|---|
| ≤4-word accuracy is far below 5+-word accuracy | Finding (a) is real and costly | Length quotas in the prompt become mandatory, and the target is derived from this gap rather than guessed |
| ≤4-word accuracy holds up | The length skew is survivable | Do not spend prompt budget on length quotas; revisit the compound skew on its own merits |
| `VolumeIncrease -> VolumeDecrease` confusion is material | Direction is not being learned | Stage 3 hard negatives on direction pairs are a blocker for release, not a nice-to-have |
| `Help_Volume` rows predicted as `Cmd.*` | FAR is real and measurable now | Confirmation gate and threshold work move earlier; `command_help_pairs` sampling is load-bearing |
| TF-IDF beats the embedding on direction | Mean-pooled similarity is the wrong readout for this taxonomy | Feeds the runtime architecture decision, beyond this pipeline |

## 6. What this cannot tell you

- It covers five intents out of sixty, and the easiest family to reason about.
  A clean result here does not clear the `Help_*` siblings, which are more
  numerous and more alike.
- The real seeds are permutation-heavy. High accuracy on them may partly reflect
  that they are repetitive, not that the model generalises.
- It says nothing about the sealed Tier-2 holdout of Architecture Section 8,
  which still has no owner. This is a probe, not that gate.
- The generated data used for TRAIN was produced by the *current* prompt. If the
  prompt changes, this result expires.

## 7. Cost

32 LLM calls, roughly 110k tokens, cents. Training and scoring are local and
take minutes. `sentence-transformers` and `scikit-learn` are the only extra
dependencies, and `sentence-transformers` is already installed.
