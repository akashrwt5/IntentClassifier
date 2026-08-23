# Roadmap: one model that handles short commands AND full sentences

Written 23 August 2026, after the fine-tuning experiments. Everything marked
*measured* has a number behind it in `reports/`. Everything else is marked as an
estimate or an open question, and the difference matters — this project has been
burned more than once by treating a guess as a finding.

---

## 0. What "perfect" means, since it has to mean something

Perfect is not available at 4.75 MB, fully offline, on 5,173 unique training
examples across 57 intents. Anyone who says otherwise is selling something.

What *is* available is a model that passes a written bar. Here is the bar this
roadmap targets. It is deliberately stricter than the last release's, and every
line is measurable today with scripts that already exist.

| criterion | today (best 4.75 MB) | target | measured by |
|---|---|---|---|
| accepted precision | 0.9789 | ≥ 0.97 | `evaluate_onnx.py` |
| coverage | 0.658 | **≥ 0.80** | `evaluate_onnx.py` |
| lexical gap (A−B, C−D) | +0.33 / +0.39 | **≤ 0.10** | `structure_probe.py` |
| accuracy retained when scrambled | 99.1% | **≤ 92%** | `structure_probe.py` |
| contextual (long sentences) | 0.49 | **≥ 0.80** | `evaluate_onnx.py` |
| corrective held-out | 0.48 | **≥ 0.80** | `finetune_encoder.py` |
| wrong AND accepted, probe 2 | 0.000 | 0.000 | `structure_probe.py` |
| OOD rejection | 0.986 | ≥ 0.95 | `evaluate_onnx.py` |
| OOD AUROC (held-out) | 0.939 frozen / 0.699 fine-tuned | **≥ 0.85** | `train_classifier.py` |
| INT8 size | 4.75 MB | ≤ budget (see §3) | `export_onnx.py` |
| gate agreement | 0.996 | ≥ 0.99 | `parity_test.py` |

**Coverage ≥ 0.80 is the line that makes this a product rather than a demo.**
At 0.658, one request in three is answered with "say that again". At 0.80 it is
one in five, which people tolerate; below that they stop using the feature.

---

## 1. What is measured, and therefore not up for debate

**Fine-tuning the encoder works.** 12 epochs on bge-small (12 layers, 384-dim):

| | frozen | fine-tuned |
|---|---|---|
| test accuracy | 0.9081 | 0.9266 |
| contextual | 0.538 | **0.855** |
| minimal pairs | 0.727 | **0.909** |
| hard negatives | 0.710 | **0.902** |
| corrective held-out | 0.48 | **0.869** |
| lexical gap | +0.611 | **0.000** |
| accepted precision | 0.9787 | 0.9930 |

The gap held at 0.000 from epoch 2 to epoch 11 while macro-F1 climbed to 0.9186,
so it is not an artefact of an undertrained model.

**Distillation does not carry that capability into a small model.** The student
distilled from the fine-tuned teacher scored contextual 0.524, hard negatives
0.627, lexical gap +0.222/+0.500, and INT8 coverage 0.000. Distillation matches
the teacher's OUTPUTS on the training distribution, not its mechanism, and the
structure-critical rows are 3.5% of `train_augmented` — so the student can match
teacher logits nearly everywhere using the old lexical shortcuts, and it did.

**The data offers a shortcut that mostly works.** `P(intent | word present)`:
soft 1.000, faint 1.000, harsh 1.000, muffled 0.923. A model answering from
those words is fitting, not failing.

**Validation is exhausted.** It has been used for six nested selections: teacher
epoch, student epoch, student fine-tune epoch, head seed, temperature, and every
gate threshold. One run reported validation coverage 0.736 against test coverage
0.143. Every coverage figure in this repository is optimistic by an unmeasured
amount.

**Fine-tuning weakened the OOD signal.** Mahalanobis AUROC on held-out OOD:
0.939 frozen → 0.699 fine-tuned. Expected — training for classification compacts
the classes and flattens the distance geometry the score depends on — but it
means gate signal 3 is now much weaker than its documentation claims.

---

## 2. What is NOT settled, despite yesterday's conclusion

I wrote that 4 layers cannot hold the structure. That claim rests on **one
training recipe**, and the recipe shows clear signs of a different problem:

```
epoch 0:  loss 3.1296   val_macroF1 0.8613   <- best epoch
epoch 11: loss 0.0104   val_macroF1 0.8480
```

Loss fell 300× while validation went *down*, and the best epoch was the first
one. That is textbook overfitting, and none of the standard responses were
tried: no increased dropout, no weight decay tuning, no layer-wise learning-rate
decay, no freezing of the lower layers, no early stopping on the gap rather than
on macro-F1.

So the honest statement is: **4 layers × 256 did not learn the structure under
the one recipe tested.** Whether it *cannot* is Phase 1's job to establish. Do
not spend money on a bigger model before ruling out that the small one was
simply trained badly.

---

## 3. The phases

Each phase has a decision gate. Do not start the next one until the current
one's gate has an answer, and write the answer down before running the next
thing.

### Phase 0 — Repair the measurement foundation (half a day) — DO THIS FIRST

Nothing below is trustworthy until this is done. Every number in §1 carries an
unknown optimism.

**0a. Split the data four ways instead of three.**

```
train        model fitting
val-model    early stopping, epoch selection, seed selection
val-calib    temperature and ALL gate thresholds — nothing else touches it
test         read once, at the end
```

Currently `validation.csv` is doing the jobs of both val-model and val-calib.
Modify `split_dataset.py` to carve `val-calib` out of the existing validation
split using the same group-based leakage control (roughly 50/50 of the current
1,508 rows; each half stays leakage-free against train and test by construction,
since the groups were already disjoint).

**Gate:** `train_classifier.py` reports shipped-gate coverage on `val-calib`,
and that figure lands within ~0.05 of test coverage. If it does not, the split
is still leaking selection pressure and nothing further is measurable.

**0b. Re-measure the noise floor for the fine-tuned regime.**

```bash
python scripts/variance_check.py --encoder bge-small-en-v1.5-ft --seeds 5
```

The current `reports/variance.json` was measured on the frozen encoder. A
fine-tuned encoder is a different model family and its run-to-run spread is
unknown. Every "improvement" in Phase 1 and beyond is meaningless without this.

**Gate:** a 2σ column exists for the fine-tuned regime. ~2 hours, unattended.

---

### Phase 1 — Establish the real minimum capacity (one day)

Three experiments, each about 20 minutes, each answering one question. Run them
in this order because they get progressively more expensive.

**1a. Is 4 × 256 badly trained, or genuinely too small?**

Re-run the student fine-tune with regularisation appropriate to a small model:

```bash
python scripts/finetune_encoder.py --encoder student-h256-l4-ft \
  --epochs 12 --lr 3e-5 --dropout 0.3 --suffix=-reg
```

`finetune_encoder.py` will need two additions, both small:
- `--weight-decay` exposed (currently hardcoded to 0.01; try 0.1)
- `--freeze-layers N` to hold the bottom N transformer layers fixed, so the
  limited capacity is spent on the task-specific upper layers rather than
  re-learning general language

Also change the checkpoint criterion. Selecting on `val_macroF1` kept epoch 0,
whose gap was +0.222/+0.500. Select on the gap instead, or on
`val_macroF1 + (1 − gap)`, because macro-F1 is not the thing being bought.

**Gate 1a:** lexical gap ≤ 0.10 and corrective held-out ≥ 0.80 at 4 × 256.
If met — **stop. You are done at 4.75 MB.** Go to Phase 4.
If not met, continue to 1b.

**1b. What is the smallest config that works?**

```bash
python scripts/distill_student.py --teacher models/final_ft --hidden 256 --layers 6 \
  --tokenizer-from bge-small-en-v1.5-v --name student-h256-l6
python scripts/finetune_encoder.py --encoder student-h256-l6 --epochs 12 --lr 5e-5 --suffix=-task
```

Then the same for `--layers 8`. Sizes, from `V·H + P·H + L·(12H² + 13H)` with
V=3267, H=256, P=64:

| config | params | INT8 | status |
|---|---|---|---|
| 4 × 256 | 4.01 M | 4.75 MB | failed under one recipe |
| 6 × 256 | 5.59 M | ~5.6 MB | untested |
| 8 × 256 | 7.17 M | ~7.2 MB | untested |
| 12 × 384 | 22.6 M | ~22.6 MB | **works — measured** |

Note the distil-then-fine-tune order. Distillation gives a sensible
initialisation; fine-tuning supplies the task gradients that distillation cannot.
Doing only one of the two is what failed.

**Gate 1b:** the smallest config meeting gap ≤ 0.10 and corrective ≥ 0.80. That
number is your real size budget, and it is a fact you can take to whoever owns
the budget rather than an argument.

**1c. Is bge even the right base?**

Every encoder benchmarked so far — bge, e5, MiniLM — is a **contrastive
retrieval embedder**, trained so that topically similar sentences land near each
other. "turn it up" and "turn it down" are topically near-identical. That family
is known to be weak on antonymy and negation, and the comparison was therefore
never a fair one: three candidates from the same family is one candidate.

Test one MLM-pretrained encoder of similar size — a small BERT, ELECTRA-small,
or DeBERTa-v3-xsmall — through the same fine-tune path. Classification
fine-tuning from an MLM objective often beats a retrieval embedder on exactly
the distinctions this product needs.

**Gate 1c:** run only if 1b lands above your acceptable budget. A better base
may reach the bar with fewer layers, which is the cheapest possible way to buy
size back.

---

### Phase 2 — Make the structure learnable rather than memorisable (one day)

Distillation failed partly because structure-critical rows are 3.5% of training.
Even with enough capacity, that thin a signal is fragile.

**2a. Weight the loss toward the hard cases.** Rows from F1 (negation), F11
(corrective), F12 (symptom), F13 (long-form) and F16 (role balance) carry the
structure. Give them 3–5× weight in the cross-entropy. This is a two-line change
in `finetune_encoder.py` (`CrossEntropyLoss` accepts per-sample weights via
`reduction="none"` and a manual mean).

**2b. Add a minimal-pair margin loss.** For a pair `(x_a → A, x_b → B)`
differing by one token, add a term requiring
`p(A | x_a) − p(A | x_b) > margin`. This forces the differing token to be used
*in its context* rather than as a bag feature. The pairs already exist in
`data/challenge/minimal_pairs.csv` and in F16's output.

**2c. Stop generating from templates where it matters.** Two independent results
say template data teaches templates: corrective (taught 0.74, held-out 0.48) and
accessories (0.471 on unseen names). For the long-sentence capability
specifically, **collect real utterances**. 300–500 real two-clause requests will
do more than 5,000 generated ones. This is the single highest-value item in this
document and the only one that cannot be automated.

**Gate 2:** contextual ≥ 0.80 on the test suite, and the held-out accessory
suite above 0.70. If real data lifts the held-out numbers where generated data
did not, that settles the generation question for good.

---

### Phase 3 — Attention pooling (half a day)

Only now, and only if probe 1 is still near 99%.

Replace mean pooling with a learned attention pool: one query vector scores each
token, output is the weighted sum. A few thousand parameters, exports to ONNX
cleanly, and it gives a debugging handle — the attention weights show which word
the model actually used.

It was deferred because a pooling layer cannot supply capacity the encoder does
not have. Once capacity is settled it becomes the cheapest remaining lever on
word order.

Changes needed: `encoders.py` (the pooling in `HFEncoder`), `finetune_encoder.py`
(`Net.forward`), `export_onnx.py` (the graph builder). All three must agree, and
`parity_test.py` will catch it if they do not.

**Gate 3:** accuracy retained when scrambled ≤ 92%, with no regression beyond 2σ
on accepted precision.

---

### Phase 4 — Restore the OOD signal (half a day)

Fine-tuning cost the Mahalanobis score 0.24 AUROC. Three options, in order of
cost:

1. **Fit the OOD scorer on the PRE-fine-tune embeddings.** The geometry that
   made it work still exists in the frozen encoder. This means shipping two
   embedding extractions, which costs latency and size — probably too expensive.
2. **Switch to energy-based OOD**, which reads logits rather than embedding
   distance and is less affected by class compaction. `ood_score.py` already
   implements it: `--ood-method energy`. One run to compare.
3. **Accept the weaker signal** and lean on the reject class and confidence,
   documenting that signal 3 is now a weak contributor rather than the 0.92
   AUROC the README claims.

**Gate 4:** held-out OOD AUROC ≥ 0.85, or the README and the Android contract
are corrected to state the real figure. Shipping a documented 0.92 while the
model delivers 0.70 is the kind of thing that gets found in an audit.

---

### Phase 5 — Refit the gate honestly (one hour)

With Phase 0's clean split:

```bash
python scripts/pick_seed.py --encoder <winner> -- --target-precision 0.97
```

Two known defects to keep in view, both already fixed in code but worth
verifying in the output:

- `train_classifier.py` now prints **SHIPPED GATE on validation** separately from
  the operating point's own numbers. Only the former is real.
- The per-risk step applies a **second** precision target on top of the operating
  point's. In one run it cost 0.294 of coverage. If coverage is short, that is
  the first dial, not `--max-fallback-leak`.

**Gate 5:** shipped-gate coverage on `val-calib` within 0.05 of test coverage,
and the §0 bar met.

---

### Phase 6 — Acceptance on real speech (one day of recording)

Two things need the same recording session, so do them together.

**6a. Signal 6, the ASR confidence gate.** Protocol in
`reports/asr_confidence_protocol.md`. ~120 utterances. Skip entirely if the
product uses push-to-talk.

**6b. The question this whole roadmap exists to answer.** Of 100 real user
utterances, what fraction are **two-clause** — a situation described, then a
request? That number decides whether the long-sentence capability is worth its
size, and nothing in `en.csv` can tell you: the corpus is what somebody *wrote*,
not how people *speak*.

If real usage turns out to be 90% short commands, Phase 1's 4.75 MB model is the
right product and Phases 2–3 were insurance. If it is 40% two-clause, the size
budget has to move and now you have the evidence to move it.

**Do 6b before Phase 2's data collection**, since the same recordings serve both.

---

## 4. Order, cost, and what to do if it fails

| phase | cost | if the gate fails |
|---|---|---|
| 0 measurement | half a day | everything downstream is unmeasurable — do not skip |
| 6b acceptance | 1 day | — this reframes everything below |
| 1a small-model regularisation | 20 min | go to 1b |
| 1b capacity ladder | half a day | go to 1c, or move the budget |
| 1c different base encoder | half a day | the budget must move; you have the evidence |
| 2 data and losses | 1 day | collect more real data; do not generate more |
| 3 attention pooling | half a day | drop it — it is the smallest lever |
| 4 OOD | half a day | take option 3 and correct the docs |
| 5 gate | 1 hour | the per-risk target is the dial |

**Total: about a week of work, of which one day is recording and about two hours
is unattended compute.**

The fastest useful thing is not Phase 1. It is **Phase 6b** — one day of
recordings that may tell you the expensive half of this roadmap is unnecessary.

---

## 5. Risks worth stating before starting

**Fine-tuning may not survive quantisation as well as the frozen model did.**
INT8 parity has only ever been checked on frozen-encoder students. A fine-tuned
encoder has a different weight distribution and per-channel scales may behave
differently. `parity_test.py` will catch it; budget a day if it does.

**More capacity means more latency.** 4 × 256 runs at 1.0 ms p50; 12 × 384 at
4.4 ms. Both are fine on a phone. On a hearing aid's own DSP they may not be —
check the real target before committing to a size.

**The corpus is small in the way that matters.** 9,723 rows collapse to 5,173
leakage groups, roughly 90 unique examples per intent, with a tail near 25. No
architecture fixes a thin tail. `Help_DemoMode` has 53 rows; it will stay weak.

**"Perfect" will not arrive.** The bar in §0 is what success looks like. Write
the achieved numbers next to it when you get there, ship, and put the remainder
in the next document.
