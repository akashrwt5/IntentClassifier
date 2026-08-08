# Can MiniLM be the primary model?

**Question:** drop the keyword rules and the TF-IDF featurizer entirely, classify
straight off MiniLM embeddings — does that clear 90%, and is it production-ready?

**Answer: yes on accuracy, no on production-ready.** 91.7% on the honest
holdout, better macro-F1 than the shipped system, and it fixes the exact weakness
the keyword rules were papering over. It is 12x slower and 12x larger, and three
things stand between it and a shippable system.

Measured at commit `256e2e2c` on `holdout_honest.csv` (1470 rows, leak-guarded),
no keyword rules, no TF-IDF, no confidence gate — raw model accuracy, the same
quantity `nlu_training.evaluate` reports.

---

## The headline

| | accuracy | macro-F1 | OOS recall |
|---|---|---|---|
| **MiniLM + LR head** (C=15) | **0.9170** | **0.9187** | 0.7641 |
| TF-IDF today (shipped) | 0.9190 | 0.9060 | 0.8150 |

Accuracy is a tie. **Macro-F1 is 1.3 points better**, which is the more
informative number here: it weights all 57 intents equally, so it measures the
tail rather than the head, and the tail is where this product's help content
lives.

### The shipped semantic head is NOT this number

The head currently in `models/semantic_head.npz` scores **0.8116 / 0.8749 /
0.3128** — far worse. It is not a fair reading of MiniLM, because it was trained
from a different pipeline entirely: `datasets/01_source_base_training_data.csv`
rather than `language_packs/en/train.csv`, capped at **250** per intent, with its
own curated `semantic_oos.csv` instead of the training set's OOS rows.

The numbers above come from retraining the head on exactly the data the TF-IDF
model uses. Any comparison against the shipped head is measuring a stale
artifact, not the architecture.

---

## Where each model wins

MiniLM's gains are concentrated in the `help.*` tail — the intents that need
paraphrase understanding rather than keyword overlap:

```
intent                              MiniLM   TF-IDF   delta    n
help.transcribe.show                  100%      67%    +33%   12
help.clean_care.show                  100%      70%    +30%   10
help.battery.show                     100%      75%    +25%    8
help.home.show                         79%      58%    +21%   19
help.mask_mode.show                   100%      88%    +12%    8
```

Its losses are concentrated in short, formulaic device commands:

```
help.remote_programming.show           77%      95%    -18%   22
device.volume.unmute                   85%     100%    -15%   20
help.app_settings.show                 88%     100%    -12%    8
help.customize.show                    90%     100%    -10%   10
```

That split is the whole story. **The 28 keyword regexes exist because TF-IDF
cannot read a paraphrase** — they hand-encode "I keep asking people to repeat
themselves" → volume up. MiniLM does that natively and needs no rules for it.
But a bag of words is very good at `unmute`, and an embedding of a two-word
utterance carries less signal than a direct lexical match.

Neither model dominates. That is an argument for arbitration between them, not
for replacing one with the other — the same conclusion the keyword/model
arbitration already reached one level down.

---

## The cost

```
stage                              ms/turn
TF-IDF + keyword arbitration         0.080     <- today
MiniLM embed (INT8 ONNX)             0.915
  + 57x384 LR head                   0.003
MiniLM total                         0.918     <- 12x slower
```

```
artifact                            size
models/intent/en/model.onnx         1.76 MB    <- today
models/minilm-l6-v2.onnx           21.80 MB
models/minilm-vocab.txt             0.22 MB
models/semantic_head.npz            0.08 MB
MiniLM total                       22.10 MB    <- 12x larger
```

0.9 ms is not a latency problem next to 200–500 ms of ASR. **22 MB is the real
question**, and it is a device-budget decision rather than an ML one. See
`docs/PRODUCTION_TRACKER.md` — a ternary-quantised distilled encoder reaches
~7 MB at ~0.84 correlation with this teacher, which is the lever if 22 MB is too
much.

---

## What stands between this and production

### 1. The regularisation is not honestly fitted

Accuracy climbs steeply with `C`, well past the standard value:

```
    C   accuracy  macro-F1  OOS recall
   15    0.9170    0.9187     0.7641
   30    0.9231    0.9249     0.7846
   60    0.9327    0.9329     0.8308
  120    0.9361    0.9349     0.8615
  300    0.9401    0.9365     0.8821
```

At C=300 it beats the shipped system on **all three** measures — accuracy +2.1,
macro-F1 +3.1, OOS recall +6.7.

**Do not quote those numbers.** That table is `C` selected against the holdout,
which is blocker B9 exactly. The out-of-fold selection did not complete in the
environment this was measured in (high-`C` multinomial fits over 8400x384 are
slow), so **only the C=15 row is defensible** — and C=15 is a standard value, not
a tuned one.

Someone must run the OOF sweep before any of this ships. The honest reading right
now is "at least a tie, plausibly a clear win, unconfirmed".

### 2. There is no CoreML/TFLite parity story for the encoder

`SemanticHead.mlpackage` exists, but the whole export/parity apparatus
(`export_coreml_test`, the Tier-A numeric-equivalence gate, the dual-vocab
heads) is built around the TF-IDF graph. Making MiniLM primary means that
apparatus has to cover the encoder, on both platforms, with the same numeric
guarantees. That is the largest piece of work here and none of it exists.

### 3. Multilingual gets harder, not easier

`all-MiniLM-L6-v2` is an English model. fr/de/da would need a multilingual
encoder (bigger) or per-language encoders (much bigger). The repo has
`multilingual/SemanticSupport/` and a multilingual head trainer, so the intent
was there, but nothing is measured. Given fr/de/da currently have no packs at
all (tracker B3), this compounds an existing blocker rather than adding a new
one.

---

## What I would do

**Not a replacement — a third recogniser in the arbitration that already
exists.** The engine already arbitrates keyword-vs-model and already has
`agreement_threshold` for the case where two recognisers concur. MiniLM fits
that structure directly, and the per-intent table above says it should: it is
better exactly where TF-IDF is weak and worse exactly where TF-IDF is strong.

Concretely, in order:

1. **Run the OOF `C` sweep.** Everything below is conditional on it. Cheap, and
   it decides whether this is a tie or a win.
2. **Retrain and ship the semantic head from `language_packs/en/train.csv`.**
   The current artifact is stale regardless of what happens next — it is trained
   on a different corpus with a 250-row cap and scores 0.81.
3. **Turn the stage on and measure the ensemble**, rather than the replacement.
   `semantic_rescue_enabled` is `false` today; the rescue path already exists.
4. **Only then** consider dropping the keyword rules, and drop them by measuring
   which ones MiniLM makes redundant rather than all at once.

Replacing the primary model is the largest change available and the evidence
does not yet require it. Enabling a stage that is already written, already
exported, and already 12 points better on the help tail, does.

---

## A leak this investigation turned up

`normalize_text` expands contractions before featurisation. Five training rows
become byte-identical to holdout rows only AFTER that expansion:

```
"i can't remember where my aids are"  ->  "i cannot remember where my aids are"
```

`train.py`'s leakage guard runs on raw text at step 1b, so it does not see them.
Five rows out of 8430 (0.06%) is not material to any number in this document,
and it was excluded from every measurement above — but the guard should run on
the normalised form, since that is what the model actually trains on.
