# Closing the wrong-action budget — options, measured

**Status:** analysis for owner decision. **Nothing here is implemented.**
Companion to charter B5 (`ENGLISH-PRODUCTION-STATUS.md` § B5).
Measured 2026-07-26 against `datasets/en/holdout_honest.csv` (1470 turns,
T = 0.657336, semantic off): **11 wrong actions, budget ≤5.**

---

## Read this first: every number below is holdout-fitted

Each option was scored on the honest holdout. That is the right way to learn
**which mechanism works** and the wrong way to choose **its parameter value**.
Picking `below_confidence = 0.90` because it scores well on this table would fit
a threshold to the one set that is supposed to be an unbiased estimate — the
same class of error as blocker B9, one layer up, and it would leave us with no
honest measurement left.

**Method that must be used instead:**

1. Fit every parameter on out-of-fold predictions over `train.csv` (the B2
   machinery already produces these — no data is sacrificed and the shipped
   model is not retrained).
2. Confirm **once** on `holdout_honest.csv`, and report that number whatever it
   is.
3. If step 2 disappoints, do not re-tune and re-confirm. That is how a holdout
   dies.

Read the tables as *"this mechanism has this shape"*, never as *"set it here"*.

---

## What the failures actually are

| | |
|---|---|
| Total wrong actions | 11 / 1470 turns |
| `truth == sys.oos.fallback` | **10** |
| Genuine intent confusion | 1 (`"set me up for conversation"` → `device.volume.increase`) |
| Produced by the keyword stage | **0** — all 11 come from the TF-IDF path |
| Already fire an intent in `uncertain_confirm` | **8** (but at conf ≥ 0.80, so the gate never engaged) |
| Fire above 0.95 confidence | 5 (one at 1.000) |

Two consequences. The keyword rules are not implicated and should not be
touched. And the budget is a **failure to abstain**, not a failure to
discriminate — the engine is not mixing commands up, it cannot say "I don't
know" to `"iphone"` or `"play festival"`.

OOS deflection is already at 92.3% (181/196). The budget is decided by the last
8%.

---

## The options, measured

### The distinction that decides this

Two different costs get conflated as "recall loss", and they are not
comparable in a hearing-aid app:

- **Deflection** — the command is dropped; the user must notice and rephrase.
- **Confirmation** — one extra turn ("Did you mean mute?"); the command still
  executes.

A confirmation is a small friction. A deflection is a failed interaction for a
user who may not have heard that it failed. Options are ranked accordingly.

### 1. Extend the confirmation gate — best measured trade

`uncertain_confirm` currently lists 11 intents; there are 15 state-changing
intents. `messaging.message.listen` is **not** gated, and 3 of the 11 failures
are exactly that — reading a message aloud unbidden is a privacy event and
belongs in the gate on its own merits.

Extending the list to all 15 state-changing intents:

| `below_confidence` | wrong caught | correct fires → confirmation |
|---|---|---|
| 0.90 | **6 / 11** | 15 / 492 (**3.0%**) |
| 0.95 | 6 / 11 | 55 / 492 (11.2%) |
| unconditional | 11 / 11 | 492 / 492 (100%) |

Cost is paid in confirmations, not deflections. Unconditional confirmation does
close the budget outright — at the price of confirming every volume change,
which is the wrong trade for a device people adjust constantly.

### 2. Add the unused OOS training data — free, and targets the survivors

`datasets/en/oos_2.csv` holds **355 out-of-scope utterances that are not in
`train.csv`** and — checked — **not in the honest holdout** (1 row overlaps and
must be dropped). `oos.csv` is a subset of it and adds nothing.

Training OOS coverage would go **1112 → 1467 rows (+32%)** at zero UX cost and
zero policy change. This is the only option with no downside, and it aims at the
exact residue the gate cannot reach: the 5 survivors are all high-confidence OOS
misses.

### 3. Abstention signals that do not need a new model — mostly dead ends

Tested on logits the engine already computes:

| Signal | Result |
|---|---|
| top1−top2 logit margin | margin < 1.5 catches 5/11, **deflects 4.0%** |
| OOS class probability | > 0.01 catches 6/11, deflects 6.1% |
| max logit (energy-style OOD) | **no separation at all** — medians 6.33 wrong vs 7.30 correct, overlapping. Dead |
| OOS class rank | rank ≤ 2 catches 7/11 but deflects **26.6%**. Dead |

The margin is the only one worth keeping in reserve. It is strictly worse than
option 1 (fewer caught, and the cost is deflections rather than confirmations),
so it is a fallback, not a first move.

### 4. Raising the fire threshold — the worst instrument, confirmed

| threshold | wrong caught | correct **deflected** |
|---|---|---|
| 0.70 (current) | 0 / 11 | 0 |
| 0.80 | 1 / 11 | 49 (4.8%) |
| 0.90 | 6 / 11 | 89 (8.7%) |
| 0.99 | 9 / 11 | 297 (28.9%) |

Strictly dominated by option 1: same catch rate at 0.90, but the cost is
deflections and it is ~3× larger. **Do not raise the fire threshold.**

---

## Recommended sequence

Order matters — adding training data changes the model, which changes every
confidence, which invalidates any gate parameter fitted before it. Doing gates
first means fitting them twice.

1. **Add the 355 OOS rows** to `train.csv` (excluding the 1 holdout overlap).
   Retrain, refit `T` (the calibration provenance hash will force this),
   re-measure. No policy touched. The honest holdout stays frozen and disjoint.
2. **Re-measure the budget.** The residue may be materially smaller, which
   changes how much the gate has to do.
3. **Fit the gate on out-of-fold predictions** — the intent list and
   `below_confidence` — never on the holdout.
4. **Confirm once** on `holdout_honest.csv`. Report it whatever it says.
5. Only if that still misses: consider unconditional confirmation on a narrow,
   genuinely high-cost subset (`messaging.message.send`, `find.phone.locate`),
   not on volume.

**If data plus gates cannot close it,** the principled fix is a dedicated binary
in-scope/out-of-scope head instead of making `sys.oos.fallback` win a 57-way
argmax — a class competing inside the softmax is a weak abstention mechanism. It
is real work (a second artifact through ONNX, CoreML and the pack manifest), so
it should be the answer to a demonstrated shortfall, not the opening move.

## What is a policy decision and what is not

| Step | Nature | Needs owner |
|---|---|---|
| Add OOS training rows | data | no |
| Retrain + refit `T` | build | no |
| Extend the gated intent list | **policy** | **yes** |
| Change `below_confidence` | **policy** | **yes** |
| Raise the fire threshold | **policy** | **yes** — and not recommended |
| Unconditional confirmation | **policy** | **yes** |

Step 1 can proceed immediately. Steps 3–5 cannot.
