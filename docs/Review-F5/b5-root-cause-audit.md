# Wrong actions — root-cause audit, and a correction to B5

**Status:** analysis for owner decision. Nothing implemented, no policy changed.
Written in response to the challenge: *"confirm the behaviour you said — that
wrong action is the result of wrong dataset."*

**Short answer: I could not confirm it, and the claim as I made it was wrong.**
The dataset is implicated, but in close to the opposite way, and the dominant
cause is neither the data nor the thresholds.

---

## 1. What I claimed, and what is actually true

B5 said: *10 of 11 failures have `truth == sys.oos.fallback`, therefore this is
an abstention problem, therefore grow OOS coverage.*

The first clause is a fact. **The rest was inference presented as finding.** I
observed a correlation with the label and named a cause without testing it.
Tested now:

| Claim | Verdict |
|---|---|
| 10/11 failures carry `truth == sys.oos.fallback` | ✅ true |
| The OOS class is under-resourced | ❌ **not supported** — 1112 rows, 13.3% of train, already deflecting 92.3% of held-out OOS |
| The OOS training data is contaminated | ❌ **largely not** — only 11 of 1112 rows (1.0%) sit within cosine 0.7 of an in-scope utterance |
| More OOS data will reduce wrong actions | ⚠️ **untested.** Plausible, unproven. It is an ablation, not a deduction |

So "wrong dataset" is not confirmed as the cause of the count.

## 2. Where the dataset *is* wrong — the test labels, not the training volume

Auditing all 10 OOS-truth failures against their nearest training neighbours:

**`"can you read last text message"`** → labelled `sys.oos.fallback`; model fired
`messaging.message.listen` at 0.974. Training data contains `"can you play last
message"` and `"can you hear last message."`, both labelled
`messaging.message.listen`. These are the same request. **The label is wrong and
the model is right.** This is a false positive in the budget.

**`"can you stream, youtube music."` / `"please stream, youtube music."`** →
labelled OOS; model fired `streaming.session.start` at 0.977 / 0.874. But
`"can you stream"` and `"can you stream tv"` are labelled
`streaming.session.start`. The intent *is* streaming; what is unsupported is
YouTube as a source. This conflates **capability** with **intent** — see §5.

The other 7 are defensible OOS and genuine model errors.

**Consequence: the honest count is ~8 real wrong actions, not 11.** Still over
budget. But three of the eleven were measuring a labelling error, and my
recommendation to *train on more data of that kind* would have taught the model
that "read my last message" is out of scope — breaking a working command.

The lexical contamination scan did not catch these, because the contradictions
are **semantic, not lexical** (`"read last text message"` vs `"play last
message"` share almost no tokens). No automated guard in this repo would find
them. Only reading them does.

**On the 355 candidate additions:** they survive this audit well. Their
high-similarity cases (`"how do i clean my airpods"` vs `"how do i clean my
hearing aid?"`, `"what's the best way to clean windows"`) are *correctly*
labelled OOS and are exactly the hard negatives an OOS class needs. Roughly 3
rows are genuinely contradictory (`"set an alarm for six am"` vs
`reminders.task.create`, `"text my landlord"`, `"translate this into spanish for
fun"` — the last two name supported capabilities). The pool is worth adding
after a small audit. It is just not the *fix*.

## 3. The actual dominant cause: the model cannot represent the distinctions

Direct probes of the shipped classifier:

**Repetition manufactures confidence.**

| utterance | confidence |
|---|---|
| `can you phone` | 0.773 |
| `can you phone phone` | 0.988 |
| `can you phone phone phone` | 0.998 |
| `can you phone phone phone phone phone` | **0.999** |

Nothing was learned between row 1 and row 4. Repeating a token inflates the
TF-IDF weight, `sublinear_tf` only dampens it. **The 0.999 that made this
utterance a wrong action is an artifact of the featurizer.** No threshold, gate,
margin or calibration touches this — the confidence is high and wrong on
purpose.

**Word order is invisible, and scrambling can *raise* confidence.**

| utterance | intent | confidence |
|---|---|---|
| `turn the volume up` | device.volume.increase | 0.750 |
| `up volume the turn` | device.volume.increase | **0.997** |

A word-salad string beats the grammatical one. Bag-of-words plus bigrams has no
syntax; it cannot.

**Polarity is not represented.**

| utterance | predicted | truth | confidence |
|---|---|---|---|
| `turn mute on` | `device.volume.`**`unmute`** | `device.volume.mute` | 0.537 |
| `turn mute off` | `device.volume.unmute` | ✅ | 0.997 |
| `do not mute` | `device.volume.`**`mute`** | negated | 0.433 |

Two of three inverted. For a hearing-aid user, unexpected muting is a genuine
safety event — not a UX blemish.

This is where the residue lives. It is a **representation ceiling**, not a
threshold-tuning problem.

## 4. A regression that B9 caused and nobody has connected

> **CORRECTED 2026-07-26 by `b9-decision-sweep.md`.** Two claims in this section
> did not survive re-derivation on the honest holdout:
>
> - **"The removal rests on void evidence" → the evidence base was indeed void,
>   but the decision was right anyway.** Restoring the six rules costs **+3
>   wrong actions and −3 correct fulfils**; applied directly they fix 1 case and
>   break 5 (`"take the mute off"` → mute, `"make it less loud"` → increase).
>   The bare `\bmute\b` pattern cannot distinguish `"turn mute on"` from
>   `"take the mute off"`. Do **not** restore them.
> - **"Live regression" overstates it.** `"turn mute on"` scores 0.537, below
>   the 0.70 fire threshold, so the engine deflects rather than misfires. The
>   mis-prediction is real; the safety consequence is not.
>
> What holds: the *evidence base* for the directive was invalid and had to be
> re-derived, which is the point of §8 item 1. The sweep also found the one
> decision that is genuinely wrong — `semantic_rescue_enabled: false`. Read
> `b9-decision-sweep.md` for the re-derived numbers.


`content/platform.yaml` records an owner directive of 2026-07-24:

> `polarity_guards: REMOVED. The model resolves volume/mute polarity itself;`
> `replaying the 1,461-utterance holdout with these OFF was as good or better`
> `(3 vs 4 wrong state-changing actions). The rules only ever flipped CORRECT`
> `answers.`

**That 1,461-utterance holdout is `multilingual/test/en_holdout.csv` — the set
that is 99.9% training data (blocker B9).**

The chain is exact and checkable:

- `"turn mute on"` is labelled `device.volume.mute` in that holdout.
- It was therefore also in `train.csv` at the time (B1 later carved the honest
  holdout *out of* train.csv, so everything now in the holdout was in train
  before).
- The model had memorised the string, answered it correctly, and the polarity
  guard consequently looked like it was overriding a correct prediction.
- Today, properly held out: **`"turn mute on"` → `device.volume.unmute` @ 0.537.**
  The model gets it wrong. The guard would have caught it.

**The decision to remove polarity guards was made on invalid evidence.** It is
the same failure as the shipped temperature, in a file nobody re-examined after
B9 was found. The `help_marker_guard` was decided on the same tainted holdout —
it was *kept*, so the risk is lower, but its stated justification is equally
unsupported and should be re-derived.

## 5. Architectural finding: `sys.oos.fallback` conflates three different things

| Type | Example | Correct response |
|---|---|---|
| Not a command | `"my ears sweat"`, garbled text | "Sorry, I didn't catch that" |
| Understood, capability unsupported | `"call my dentist"`, `"turn off toshiba"` | "I can't make calls" |
| **Intent supported, parameter unsupported** | `"stream youtube music"` | "I can stream, but not YouTube" |

Type 3 is actively harmful as training signal: it teaches the classifier that
`"stream youtube music"` is **not** a streaming intent, which is false, and it
degrades the `streaming.session.start` boundary for every legitimate phrasing.
It also makes the right UX response impossible — you cannot say "I can't do
YouTube" if you have refused to recognise the streaming intent.

Type 3 belongs as a **fulfilment-time capability check on a recognised intent**,
not as a classification label.

## 6. Is the budget measuring the right thing?

The wrong-action budget is described as a medical-safety control on firing the
wrong **state-changing** action. It currently counts all state changes equally.
But in this residue:

- `device.volume.mute` fired on `"turn off toshiba"` — a hearing aid goes
  silent unexpectedly. **Genuine safety event.**
- `streaming.session.start` fired on `"stream youtube music"` — the user asked
  to stream; streaming started. **Not a safety event in any medical sense.**

A flat count of 5 treats these as equivalent. If the budget is a safety control,
it should be weighted by harm and reversibility — muting a hearing aid is not
the same as starting music. Recommending this as a **definition** change, to be
settled before the number is chased, because a mis-specified budget will drive
the wrong engineering.

## 7. So — are we doing the correct things?

**Right, and worth keeping:**

- The measurement discipline. Frozen holdout, provenance on the temperature,
  OOF fitting, refusing to tune on the holdout. B1–B4 are sound work and the
  reason any of §3 and §4 is visible at all.
- Language neutrality (A1–A10). Independent of this and genuinely done.
- Treating the previous 41 as untrustworthy rather than as a baseline.

**Wrong, and this is my error:**

- **Priority order.** B5 and my options memo pointed at thresholds, gates and
  OOS volume — all of which sit *downstream* of a confidence signal that §3
  shows is partly manufactured. Tuning a gate on `0.999` produced by token
  repetition is tuning on noise. I ranked the options carefully and did not
  question the layer.
- **I asserted a cause without an ablation.** "Grow OOS coverage" was reasoning
  from a label distribution, not evidence, and it would have been mildly
  harmful.
- **Nobody re-audited the guard decisions after B9.** The temperature was
  re-derived; `platform.yaml` was not. Every decision justified by that holdout
  is suspect and should have been swept at the time.

## 8. Revised plan

Ordered by evidence strength, not by cost:

1. **Re-audit every decision justified by the leaked holdout.** `platform.yaml`
   is the known one. This is a sweep, not a guess — grep for the 1,461 figure
   and re-derive each on the honest holdout. *No new policy; restoring the
   evidence base for existing policy.*
2. ~~**Restore `polarity_guards`**~~ — **withdrawn.** Re-derived in
   `b9-decision-sweep.md`: restoring them costs +3 wrong actions and −3 correct
   fulfils, because the bare lexical patterns break `"take the mute off"` and
   `"make it less loud"`. The decision to remove them holds. Replaced by: *if*
   the mute mis-prediction is judged worth a rule, design one that handles the
   `off|remove|cancel` construction and validate it against those 6 cases.
3. **Fix the mislabelled holdout rows** (§2). Requires care: the holdout is
   frozen, so correcting labels changes every number measured against it. Do it
   once, deliberately, with the manifest hash updated and the change recorded.
4. **Normalise repeated tokens before featurisation.** Deterministic, cheap,
   kills a 0.999-confidence failure mode outright. Not a policy change.
5. **Split type-3 OOS out of the classification label** into a fulfilment-time
   capability check (§5). Architectural; needs design.
6. **Then** the OOS pool (audited), then gates, then re-measure. These are still
   worth doing — they are simply not first, and they should be fitted against a
   confidence signal that has stopped lying.
7. **Re-open the semantic path — now the highest-value open item.** Measured in
   `b9-decision-sweep.md`: enabling it gives **+50 correct fulfils and 100 fewer
   deflections at +0 wrong actions**, against a recorded trade-off of "+5 wrong"
   that does not reproduce. **But the semantic head predates B1 by one day and
   therefore trained on what is now the holdout**, so that figure is an upper
   bound, not a measurement. Retrain the head on post-B1 `train.csv` with
   provenance, then re-measure.
8. **Extend the B4 provenance contract to every model artifact.**
   `semantic_head.json` carries `embedder`, `labels`, `weights`, `bias` and
   nothing else — its leakage had to be inferred from a file mtime. Only
   `calibration.json` is traceable, because it is the one artifact B8 forced us
   to fix.

**What I would not do now:** raise the fire threshold, add OOS data unaudited,
or fit gate parameters — until items 1–4 land, all four are tuning against a
confidence signal known to be unreliable in characterised ways.
