# Approach, trade-offs, and where this model fails

**Scope:** the English Stage-3 semantic model in `new_semantic/`, as of 2026-08-10.
Every number here was measured on this repo's data; nothing is estimated unless
marked. Sources are named so each claim can be re-checked.

---

## 1. The approach

### 1.1 Where the model sits

It is **not** a standalone intent classifier. It is the third stage of a cascade:

```
Stage 0   keyword regex triggers          confidence 1.0, short-circuits
Stage 1   entity / datetime extraction    lexicon-driven
Stage 2   TF-IDF + LogisticRegression     conf >= 0.70  -> fires, DONE
Stage 3   semantic rescue (this model)    only runs when Stage 2 is unsure
```

Stage 2 handles the easy turns. This model only ever sees the leftovers — the
utterances where a bag-of-words classifier could not commit. Measured handover
rate: **15.1% on the locked test, 41.4% on novel phrasing (stress), 92.1% on
out-of-scope.**

That framing matters more than anything else in this document. It was also the
single biggest mistake in the project: the model was designed, trained and
selected as a standalone classifier for weeks before anyone measured it in the
position it actually occupies.

### 1.2 How the model is built

**Knowledge distillation from a sentence embedder into a tiny transformer.**

| | |
|---|---|
| Teacher | `intfloat/e5-small-v2` (offline only, never shipped) |
| Student | `TinyIntentClassifier`: `Embedding(1982, 64)` → 2-layer TransformerEncoder (d64, 4 heads, ff128) → LayerNorm → `Linear(64, 57)` |
| Tokenizer | word-level, punctuation discarded, closed vocabulary of 1,982 tokens built from the training text itself |
| Input | 24 tokens max (0% of the corpus truncates) |
| Loss | `0.70 × weighted CE(labels) + 0.30 × KL(student ‖ teacher, T=2)` |
| Class weights | inverse frequency — the data is 55× imbalanced by design |
| Augmentation | 921 synthetic rows: in-scope sentences with ≥70% of tokens replaced by unseen fillers, labelled fallback |
| Shipped artifact | ONNX FP32, static shape (1, 24), **0.953 MB** self-contained |

The teacher's job is to supply soft targets; its own size and latency are free
because it never ships. The student is what runs on device.

**Why distillation and not just supervised training:** the historical record in
`semantic_project/` shows the same 0.75 MB architecture scoring 79.6% with a
TF-IDF teacher and 98.75% with an E5 teacher. Teacher quality, not student
capacity, was the lever. (Both those numbers are inflated by a test-set leak —
see §4.1 — but the *direction* held up.)

### 1.3 How decisions were made

This changed mid-project, after three separate occasions where a confident
conclusion turned out to be noise.

| Protocol | Why it exists |
|---|---|
| Every config run on **3 seeds**, compared on means | Identical configs varied by **44 points** on OOD rejection across seeds |
| A gap must exceed **2× pooled sd** to count | Most of the v1–v5 "findings" did not |
| Eval sets split into **dev / test**, selection on dev only | 17 policies had been compared on the same rows that were then reported |
| Gates **pre-registered** before seeing results | The best-harmonic policy would have shipped a model that rejects 33% of real commands |
| Leak guards in every data script, failing loudly | 100% of the original "locked test" was inside the training file |

### 1.4 The combination policy

Stage 2's prediction is not discarded when Stage 3 declines:

```
answer = argmax of whichever stage is more confident
accept = stage3_conf >= 0.40  OR  stage2_conf >= 0.30
else   = GENAI fallback
```

---

## 2. Pros

**Size and cost.** 0.953 MB ONNX — 24× smaller than the 22.9 MB MiniLM
currently in the runtime. (An earlier draft said 0.166 MB / 138×; that read the
graph file and missed the 0.787 MB external-data sidecar.) No torch, no transformers on the inference path; the
device needs only `onnxruntime` and `numpy`.

**Numeric integrity.** ONNX vs PyTorch on 2,654 utterances: max logit delta
4.3e-06, **0 argmax mismatches, 0 gate disagreements**. Static shape (1, 24), no
dynamic axes — ANE-compatible by construction.

**Reproducible.** The shipped config has locked-accuracy sd of 0.0036 and OOD sd
of 0.0224 across seeds. The un-augmented baseline had OOD sd of **0.2317**. This
was an accident — UNK augmentation was added for OOD rejection and turned out to
stabilise training — but for a medical-adjacent product it may be the more
valuable property.

**Additive, not substitutive.** With the new policy the pipeline beats
Stage-2-alone on both axes: in-scope accuracy is unchanged within noise (0.8298
vs 0.8333) while OOD rejection goes from 0.4527 to **0.8590**.

**Honest measurement infrastructure.** Leak guards, seed protocol, dev/test
split, cascade evaluation and policy comparison are all scripts in the repo, not
conventions in someone's head.

---

## 3. Cons

**Closed vocabulary.** Any word absent from the 1,982-token vocabulary becomes
`[UNK]` and loses all meaning. "quieter" and "asdfghjkl" are the same token.
Subword tokenisation was tried and made things worse (§4.6).

**Trained on the wrong distribution.** The model is trained on all 24k rows but
only ever runs on the subset Stage 2 could not resolve — a harder, differently
shaped distribution it never optimised for. This is the largest known defect.

**Synthetic training data.** 921 of the training rows are machine-corrupted text
(`synthetic_text: true` in the manifest). They buy OOD rejection and stability
but are not real user language.

**Confidence is not a probability.** There is a sharp cliff: raising the gate
from 0.60 to 0.65 drops in-scope accuracy from 0.798 to 0.552 for +0.5% OOD
gain. A large mass of *correct* answers sits in a narrow band. No temperature
calibration has been fitted.

**Fixed label space.** 57 intents, closed set. A new intent means retraining and
re-exporting everything.

**English only.** Stage 3 does not exist for fr/de/da.

---

## 4. Where it fails

Ordered by how likely you are to hit it in production.

### 4.1 Volume commands phrased unusually — the most dangerous failure

On novel phrasing (stress test), the four volume intents are the **worst
performing classes in the entire model**:

| intent | recall on novel phrasing | n |
|---|---:|---:|
| `Cmd.VolumeUnmute` | **0.442** | 52 |
| `Cmd.VolumeIncrease` | 0.565 | 46 |
| `Cmd.VolumeMute` | 0.604 | 53 |
| `Cmd.VolumeDecrease` | 0.673 | 52 |

These four are semantic siblings sharing almost all of their vocabulary — the
difference between them is one polarity word ("up"/"down", "on"/"off"). A
bag-of-64-dimensional-embeddings model with 2 layers resolves that weakly.

**Why this one matters most:** mute and unmute are opposites, and they are the
two weakest classes. A hearing-aid user saying "I can't hear anything, turn the
sound back on" in an unusual way is exactly the person who most needs it to
work. The engine has polarity guards (`_apply_polarity_guards`) that catch some
of this, but they are regex rules, not model capability.

### 4.2 Any utterance containing a word the model has never seen

74% of out-of-scope utterances contain at least one unseen token; on novel
in-scope phrasing it is 17%. Every one of those words collapses to `[UNK]`.

This produces failures in **both directions**:
- a real command with one unfamiliar word (a brand, a contact name, a regional
  synonym) loses that word entirely;
- junk input and a genuine command with an unknown word look identical to the
  model.

The model cannot distinguish them because after tokenisation there is nothing
left to distinguish.

### 4.3 Near-domain out-of-scope — 14% still gets through

At the shipped operating point OOD rejection is 0.8590, so roughly **1 in 7**
out-of-scope utterances still produces a command. Where they land is diffuse —
`Help_Home` (8), `reminders.add` (4), `Cmd.BatteryLevel` (4), `Help_CleanCare`
(4) — i.e. there is no single hole to patch.

Worse: the OOD eval set is only **403 rows**, so that 0.859 has a wide
confidence interval. The true rate could be meaningfully different.

### 4.4 Intents with little training data

23 of 57 intents have fewer than 50 training rows; the smallest are
`Cmd.ActivityStep` (34), `Cmd.ActivityStand` (35), `Help_DemoMode` (35),
`Help_MaskMode` (35). Their measured recalls — `Help_AppSettings` 0.444,
`Help_Battery` 0.556 — are computed on **9 test rows each**, where the 95%
confidence interval is **±22 points**. So two things are true at once: these
classes are probably weak, and we cannot say how weak.

### 4.5 Compositional utterances that describe a state then request the opposite

"it's quieter now, can you make it a bit louder" — the sentence contains the
vocabulary of *decrease* and the intent of *increase*. Earlier work in
`semantic_project/` built a whole model variant (`v4_contextual`) for exactly
this failure, which is evidence it is real and recurring. Nothing in the current
architecture handles it: mean-pooled token embeddings have no mechanism for
"the second clause overrides the first".

### 4.6 Long or multi-intent utterances

Not a truncation problem — 0% of the corpus exceeds 24 tokens, so `MAX_LEN` is
not binding. The failure is structural: the model emits one label. "Turn up the
volume and remind me to charge them tonight" can only be answered wrong.

### 4.7 Anything non-English

Stage 3 is English-only. Non-English utterances that fall below Stage 2's gate
go straight to GenAI. For Danish this matters most — its Stage 2 macro-F1 is
0.74, the weakest of the four shipped languages, so it hands over most often and
has no rescue.

### 4.8 The confidence number itself

Do not treat the reported confidence as a calibrated probability. The 0.60→0.65
cliff means small threshold changes have large, non-linear effects. Any downstream
logic keying off specific confidence values will behave unpredictably.

---

## 5. What would fix what

| Failure | Fix | Effort |
|---|---|---|
| 4.1 volume siblings | Targeted hard-negative pairs across the four volume intents | Days |
| 4.2 unseen words | Not the tokenizer — that was tried (§6). Needs a student with real pretrained semantics, i.e. a bigger model | Weeks |
| 4.3 OOD leakage | More real OOD data — 403 → 1,000+ | Days, mostly collection |
| 4.4 small classes | 23 classes to 150+ rows each; enlarge their eval slices too | Weeks |
| 4.5 compositional | Architecture change; `v4_contextual` is prior art | Weeks |
| 4.6 multi-intent | Product decision (multi-label? segmentation?) before any modelling | — |
| 4.7 multilingual | `docs/semantic-tiny-model-plan.md` §6 — blocked on data | Months |
| 4.8 calibration | Fit a temperature on held-out data | Hours |

**The one with the best ratio of effort to value is none of these:** train the
student on the **handover distribution** — the rows where Stage 2 is actually
unsure — instead of on all 24k. That is where it runs, and it has never been
optimised for it.

---

## 6. Things that were tried and did NOT work

Recorded so they are not retried.

| Attempt | Result |
|---|---|
| Up-weighting the fallback class (it is only 4% of data) | No significant effect |
| Oversampling real fallback rows 3× | No significant effect |
| Subword tokenizer (WordPiece-style, 3,000 pieces) | `[UNK]` on OOD went 22.5% → **0%**, and the model got **worse**: stress 0.8265 → 0.7540. A 64-dim, 2-layer student on 24k sentences cannot reassemble word meaning from pieces, and loses the direct word→meaning mapping that was working |
| `--unk-robust` counter-examples to recover stress | No significant effect |
| INT8 quantization | Not shipped: **5 argmax flips and 9 gate disagreements**. (An earlier note also claimed it was 2x larger than FP32 — that was a measurement error: 0.166 MB was the graph without its external-data sidecar. INT8 at 0.349 MB is in fact 2.7x SMALLER than the 0.953 MB FP32, so revisiting it is worthwhile if the flips can be removed with per-channel scales.) |
| Dual gate keyed on Stage 2's own 0.70 threshold | Vacuous: the handover subset is *defined* by Stage 2 scoring below 0.70, so the clause was always false and the policy silently collapsed into the old one |

---

## 7. Numbers you should quote, and numbers you should not

**Quote these** (held-out TEST half, mean of 3 seeds, cascade position):

| metric | value |
|---|---:|
| in-scope accuracy, novel phrasing | **0.8298 ± 0.0094** |
| out-of-scope rejection | **0.8590 ± 0.0057** |
| model size | **0.953 MB** |

**Do not quote these:**

- **98.87%** — from `semantic_project/`. All 1,686 rows of that "locked test"
  are present verbatim in its training file. It measures memorisation.
- **0.8891 locked accuracy** — real, but the locked test is 70% recombined
  training phrasing (88.2% of its bigrams already appear in training). It
  overstates real-world performance. The stress number is the honest proxy.
- Any per-class recall for an intent with 9 test rows — the interval is ±22
  points.

---

## 8. Bottom line

A 0.953 MB model that adds 40 points of out-of-scope rejection to the pipeline
without costing in-scope accuracy is a good outcome for the size.

It is not a language understander. It is a closed-vocabulary pattern matcher
over 57 fixed intents, and it degrades exactly where a pattern matcher does: new
words, unusual phrasing, and semantically adjacent commands that differ by a
single polarity word. The most consequential of those — telling mute from unmute
when the user phrases it in a way the training data never saw — is currently its
weakest measured behaviour, and is being held up by regex guards rather than by
the model.
