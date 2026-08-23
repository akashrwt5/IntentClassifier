# semantic_intent — offline hearing-aid intent classifier

The robustness & trustworthy-confidence plan, implemented for the **57-intent**
English command set in `data/raw/en.csv`.

Shipped model: a **4.75 MB INT8 ONNX file** — a 256-dim, 4-layer student
distilled from bge-small-en-v1.5, with a pruned 3267-token vocabulary and
calibration fused into the graph.

## Where it stands

| | teacher (35 MB) | **student, SHIPPED (4.75 MB)** |
|---|---|---|
| accepted precision | 0.9787 | **0.9789** |
| coverage | 0.715 | 0.658 |
| test accuracy | 0.9095 | 0.9061 |
| test macro-F1 | 0.9007 | 0.8932 |
| calibration ECE | 0.0141 | 0.0244 |
| contextual (145) | 0.538 | 0.490 |
| STT noise (1496) | 0.842 | **0.852** |
| hard negatives (916) | 0.710 | **0.715** |
| minimal pairs (88) | 0.727 | 0.705 |
| negation (156) | 0.628 | 0.590 |
| **OOD rejection (286)** | 0.969 | **0.9685** |
| accessories (85) | — | 0.471 |
| latency p50 | 4.18 ms | **1.08 ms** |

**Accepted predictions reach 97.9% precision at 65.8% coverage.** The gate was
built to a 97% precision promise and it keeps it, at 7.4x smaller and 3.9x
faster than the teacher.

Coverage is the one criterion that missed. The ship bar was 0.68 and this is
0.658 — about two requests in a hundred more get "say that again" than intended.
It shipped anyway because the alternative was another round of tuning against a
number that moves 0.035 between two fits of the same model, and because a
refusal is the recoverable failure. It is the first thing to look at next time,
not a settled result.

**Accepted predictions reach 98.3% precision at 70.8% coverage** on the
leakage-controlled held-out test set — and the student clears that bar more
comfortably than the teacher it was distilled from, at 7.4x smaller and 4.4x
faster.

**The head fit is a lottery, and nothing said so for months.** The encoder is
distilled once; the head, temperature, OOD scorer and every threshold are
re-fitted on top of it in about six seconds. Across five seeds of that six-second
fit, validation ECE ranged 0.0109–0.0193 and TEST coverage moved from 0.623 to
0.658 — larger than most of the effects this project spent weeks chasing. Every
single-fit reading before now was one draw from that distribution reported as a
fact. `scripts/pick_seed.py` now fits across seeds and selects on validation ECE.

Worth recording honestly: the mechanism was not what was predicted. The
expectation was that better-calibrated heads would show higher validation
coverage, and they did not — the winning seed had *lower* validation coverage
than the one it replaced (0.8444 vs 0.8482) and still gained 3.5 points on test.
Validation coverage spread only 0.0137 while test coverage moved 0.035. So
selecting on ECE worked, but not for the stated reason, and validation coverage
is not a usable proxy for the test number.

The student wins because distillation transfers the teacher's whole probability
distribution, not just its answers. Being told "0.7 increase, 0.2 decrease,
0.05 Help_Volume" teaches which intents sit near which; a small model cannot
memorise that structure, so it has to learn it, and that acts as a regulariser.

INT8 costs nothing measurable: fp32 parity is exact on gate decisions
(agreement 1.00000) and INT8 reproduces accepted precision to four decimal
places.

Where the student is genuinely behind the teacher: negation (−0.038) and
coverage (−0.057). Contextual (−0.048) sits right at the teacher's 2σ of 0.048
and should not be called a gap.

**Accessories — the product's first priority — score 0.471, and that number is
the most useful thing in this table.** The suite holds out accessory names the
model never saw, and the failures are all of one kind: "my neck loop will not
stay connected" goes to Fallback at 0.946 confidence, "what should i do about my
audio dongle" goes to Help_EdgeMode. The model handles the accessory names it
was trained on and does not generalise to new ones. That is not a tuning
problem — the corpus has 21 rows for the TV streamer, 43 for the remote mic, and
**zero for Auracast**. This suite exists so that gap is a number instead of an
impression.

Suite sizes are in brackets for a reason. `negation` and `contextual` were 26
and 25 rows and carried ±9-10 points of run-to-run noise, which made every
reading from them meaningless; they are now 156 and 145. `hard_negatives` went
36 → 916 and its noise fell from ~±9 to ±2.1.

`ood_test` was the last one left at 45 rows, and enlarging it turned up
something the size explanation had been hiding. The stated reason it had no
noise floor was "the suite is too small" — but a small suite explains why a
floor would be *wide*, not why one was never computed. It was never computed
because `variance_check.py` did not have OOD in its metric list at all. Both are
fixed: 286 rows, measured every seed.

Enlarging it also found four rows in the original 45 that were not OOD.
`"turn the television volume down"` is the clearest: `Help_Accessories`
contains, verbatim, *"how do i turn up the tv sound?"*, and `television` appears
54 times in the corpus, 48 of them `Cmd.MemoryChange`. The model was being
marked wrong for handling a request it is supposed to handle. See
`scripts/ood_generate.py` for the two checks every row now passes and
`data/challenge/ood_rejected.csv` for what they refused.

## The safety gate

Six reasons to refuse. Four of them read the same softmax — which compares the
57 known classes against each other and can be completely confident about input
resembling none of them. Only the OOD score and the recognizer confidence are
independent of it, and only the recognizer confidence does not look at the words
at all.

| signal | catches | measured |
|---|---|---|
| **ASR confidence** | audio the recognizer was unsure of, and speech never aimed at the device | not fitted — device-specific, see below |
| reject class | requests the model knows are unsupported | — |
| **OOD score** (Mahalanobis, embedding space) | input unlike anything in training | AUROC 0.92 on real OOD, 0.70 on STT noise |
| confidence, per risk tier | half-recognised requests | normal ≥0.845, high ≥0.925 |
| top1−top2 margin | 0.51 / 0.48 coin flips between opposite commands | — |
| corrective structure | "not X, I meant Y" — accuracy there is 0.48–0.74 | fires on 0/1513 real rows, 100% of corrective cases |

Signal 6 ships **inert**. There is no default threshold and there should not be
one: Android returns 0..1, Whisper returns a negative `avg_logprob`, Vosk
returns a per-word mean, so a number that rejects 5% of good input on one stack
rejects 60% on another. An app that passes no confidence behaves exactly as it
did before. To turn it on, record ~120 utterances per
`reports/asr_confidence_protocol.md` and run
`scripts/fit_asr_threshold.py --apply <onnx dir>`, which writes an `asr` block
into `runtime_config.json` and does not touch the model. If the recognizer's
score does not separate addressed from unaddressed speech (AUROC < 0.60) the
fitter refuses to write a threshold and says so.

Five intents carry the stricter threshold because a mistake there is one the
user may not notice or be able to undo: `Cmd.VolumeMute`, `Cmd.SendMessage`,
`reminders.complete`, `Cmd.StreamingStop`, `Cmd.MemoryChange`.
`Cmd.VolumeUnmute` is deliberately excluded — it is the recovery action.

## What is solved and what is not

**Solved.** Symptom words no longer decide direction: "it is too loud … make it
quieter" is a Decrease, and the model holds that on vocabulary it never saw in
training (0.95 on held-out symptom words). The dataset explained the original
failure — only 54 rows used symptom phrasing, split 31 Increase to 14 Decrease,
so "a complaint about sound" had been learned as evidence for turning it up.

**Not solved — corrective negation.** On intent pairs explicitly taught the
corrective frame the model scores 0.74; on a held-out control group of three
families it scores **0.48**, which is a coin flip. It learned the pairs, not the
rule. More data cannot fix this: there are thousands of possible pairs and every
untaught one sits at chance. The gate now refuses these outright, which costs
nothing on real traffic. The real fix is encoder fine-tuning — sentence
structure lives in the encoder, and it is currently frozen.

**Not solved, but now wired — ASR fragments.** Text like "and push it down for
dramatics", produced while the user was talking about something else, is
ordinary English that genuinely sits near volume commands. It is not
out-of-distribution (OOD AUROC on STT noise is 0.70 versus 0.92 on real OOD) and
no text-only signal separates it — which is why every attempt to fix it inside
the model failed. The signal that does separate it was never in the text: the
recognizer's own utterance confidence, discarded between the ASR and this model.

The gate now has a branch for it, the fitting script exists, and the recording
protocol is written. What is missing is an hour of recordings from the target
hardware, because the threshold is a property of that audio stack and cannot be
derived from this dataset. **If the product can use push-to-talk, none of that
is necessary** — a button removes the failure mode completely and costs no model
work.

**Solved — long conversational requests.** Both models were classifying long
sentences as unsupported, and the corpus said why: of 6702 training rows only
23 had 18+ words, and 20 of those 23 were `Default Fallback Intent`. The model
had correctly learned that long text means "unsupported". Adding long-form
training data with train/test-disjoint scaffolds moved the student from 0.117
to 0.538. This was visible in the phase-1 audit (p99 = 15 words) and was not
acted on for far too long.

**Solved — size.** 35 MB → 4.75 MB, inside the 2-5 MB budget, with accepted
precision slightly better than the teacher. Pruning bge could not get there: at
hidden size 384 each layer costs 1.77M parameters, leaving room for one layer
inside 5 MB. Distillation into a narrower student works because per-layer cost
is 12H² — quadratic in width — so halving the width buys four times the depth.

## Run order

```bash
python scripts/audit_dataset.py            # phase 1
python scripts/build_taxonomy.py           # phase 2 — 57 intents, 17 families, 11 policies
python scripts/ood_generate.py --report    # phase 9 — 286 rows + what was refused
python scripts/build_challenge_sets.py     # phases 4-9 (calls ood_generate itself)
python scripts/split_dataset.py            # phases 3 + 10
python scripts/build_targeted_training.py  # phase 17/23 — F1..F12
python scripts/benchmark_classifiers.py --train train_augmented
python scripts/variance_check.py --seeds 5     # measure noise BEFORE trusting anything
python scripts/train_classifier.py --encoder bge-small-en-v1.5 --classifier mlp \
       --train train_augmented --out models/final
python scripts/error_analysis.py models/final final
python scripts/export_onnx.py --model models/final --out models/final/onnx --quantize-embeddings
python scripts/parity_test.py --model models/final --onnx-dir models/final/onnx
python scripts/evaluate_onnx.py --model models/final --onnx models/final/onnx/intent_int8.onnx
python scripts/make_reports.py
python scripts/pick_seed.py --encoder student-h256-l4   # fit the head across seeds
python scripts/predict.py                      # try it by hand, ONNX only
```

Use `pick_seed.py` rather than a bare `train_classifier.py` for the student. A
single head fit varies enough to move test coverage by 0.035, which is larger
than most differences worth acting on.

Optional, and only if the product is always-listening rather than
push-to-talk — see `reports/asr_confidence_protocol.md` for how to record the
input:

```bash
python scripts/fit_asr_threshold.py --data data/asr_samples.csv \
       --apply models/final_student_256/onnx
```

Encoders live in `models/encoders/<name>/` and are auto-discovered by any
directory containing a `config.json`. Nothing is downloaded at inference time.

## Design decisions worth knowing

**Leakage control is group-based.** Two sentences share a split if they share a
normalized form, a content-word key, or ≥92 token_sort similarity within an
intent. 9723 rows collapse into 5173 groups — roughly half the corpus is
near-duplicate material, so a random split would have measured memorisation.
Verified: 0 shared keys and 0 shared groups between train and test.

**Nothing is selected on a challenge suite.** Selection uses validation macro-F1
and validation ECE. The challenge suites are reported for every candidate so
failure modes stay visible, but they never feed back into the choice.

**Generated training data is checked against the corpus.** Any generated row
whose nearest real sentence carries a different label is dropped — unless the
two differ by a polarity word, in which case it is a minimal pair and the
different label is the point. Without that exemption the validator deletes
exactly the data the model most needs.

**Calibration is inside the ONNX graph; thresholds are outside.** Temperature
belongs with the weights it was fitted to. Thresholds are a product decision.

**fp32 and INT8 are judged on different criteria.** fp32 must be numerically
faithful — it is meant to be the same model. INT8 is a different model by
construction, so it is judged on decision parity and suite equivalence.
Per-channel quantization scales were the fix that mattered: without them INT8
changed 7% of decisions.

**The baseline comparison is not like-for-like.** The 0.236 MB student was
measured on 11 intents. This is 57 intents with a 35× class imbalance on a
leakage-controlled split. Re-run the baseline against these suites before
claiming either one wins.
