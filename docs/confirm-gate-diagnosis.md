# Why "increase volume" asks for confirmation

Diagnosis of the confirmation-gate regression, the decision ladder as actually
implemented, and the fix. All numbers reproduced against
`language_packs/en/holdout_honest.csv` (1470 rows), `models/intent/en/model.onnx`,
schema `language_packs/en/nlu_schema.json`.

---

## 1. The reproduction

```
utterance                stage     tier            intent                  conf    result
'increase volume'        keyword   regex           device.volume.increase  0.750   CONFIRM
'decrease volume'        keyword   regex           device.volume.decrease  0.750   CONFIRM
'turn up the volume'     keyword   regex           device.volume.increase  0.750   CONFIRM
'volume up'              keyword   regex           device.volume.increase  0.750   CONFIRM
'raise the volume'       keyword   regex           device.volume.increase  0.750   CONFIRM
'turn it down'           keyword   regex           device.volume.decrease  0.750   CONFIRM
'start transcription'    keyword   regex_guarded   transcription.start     0.900   CONFIRM
'send a message to john' keyword   regex_guarded   messaging.message.send  0.900   CONFIRM
'mute'                   keyword   exact           device.volume.mute      0.970   FULFILL
'make it louder'         tfidf     —               device.volume.increase  0.955   FULFILL
```

Note the pattern: **every confirmed utterance has confidence 0.750 or 0.900,
exactly.** Those are not probabilities. They are constants.

## 2. The model learned it perfectly

Bypassing the keyword stage and asking the trained model directly:

| utterance | model intent | calibrated confidence |
|---|---|---|
| increase volume | device.volume.increase | **0.9992** |
| decrease volume | device.volume.decrease | **0.9784** |
| turn up the volume | device.volume.increase | **0.9997** |
| volume up | device.volume.increase | **0.9998** |
| turn it down | device.volume.decrease | **0.9923** |
| raise the volume | device.volume.increase | **0.9755** |
| start transcription | transcription.session.start | **0.9976** |
| send a message to john | messaging.message.send | **1.0000** |

So the answer to "does our system not know this sentence means increase the
volume?" is: **it knows, at 0.999.** It is never asked. The classifier
short-circuits before inference runs, and the number it substitutes is what
trips the gate.

## 3. Root cause — a unit mismatch between two independently-correct components

Three pieces, each defensible alone, that are wired together on incompatible scales.

**(a) `classifier.py` short-circuits and invents a confidence.**

```python
def classify(self, text):
    kw_intent, kw_conf = self._keyword_match(text)
    if kw_intent:
        self.last_stage = "keyword"
        return kw_intent, kw_conf     # <-- model never runs
    ...
```

with

```python
KEYWORD_CONFIDENCE = {"exact": 0.97, "contains": 0.85,
                      "regex_guarded": 0.90, "regex": 0.75}
```

These four numbers are hand-chosen. The docstring calls them "honest,
match-type-calibrated" — they are neither fitted nor validated against anything.

**(b) `fit_confirm_gate.py` fits the band on a completely different distribution.**

The band is swept over out-of-fold **TF-IDF softmax** confidences at T=0.6537.
It produced `below_confidence = 0.91`. That fit is methodologically sound and I
would not change how it is done — but it never observes a single keyword-stage
turn, because OOF logits come from `oof_logits(X, y, folds)`, which is pure
model. The band therefore describes the geometry of a 57-class calibrated
softmax and nothing else.

**(c) `engine.py` compares (a) against (b).**

```python
if intent in self._confirm_intents and conf < self._confirm_below:   # 0.91
    ... return CONFIRM
```

`conf` here is whichever of the two arrived. A hardcoded `0.75` is compared
against a threshold fitted on softmax probabilities. **0.75 < 0.91 is
arithmetically true and semantically meaningless.**

The band moved 0.80 → 0.91 in the B8 temperature fix. Nobody re-derived the
keyword constants, because nothing ties them together. Every `regex`-tier rule
in the schema silently became permanently un-fireable at that moment.

This is the whole bug. It is not overengineering in the sense of too many
components — it is a **missing arbitration layer** between them.

### 3.1 The constants are also inverted relative to measured precision

Measured on the honest holdout:

| tier | rules | n (holdout) | **measured precision** | **assigned confidence** |
|---|---|---|---|---|
| `exact` | 4 | — | — | 0.97 |
| `regex` | 28 | 108 | **0.954** | **0.75** ❌ |
| `regex_guarded` | (subset) | 30 | **0.767** | **0.90** ❌ |
| `contains` | 0 | 0 | dead path | 0.85 |

The most reliable rule tier is assigned the lowest confidence, and the least
reliable is assigned the second-highest. The system is confirming its *accurate*
rules and firing its *inaccurate* ones. Precisely backwards.

(`contains` has zero rules in the shipped schema — the `_is_negated` guard and
its per-language cue tables are dead code today.)

## 4. Blast radius

The gate is doing far more harm than good in production shape:

```
Honest holdout, 1470 turns, current configuration
  FULFILL   1005   (976 correct, 29 wrong actions)
  CONFIRM    165   (24 caught a wrong action,  141 were pure friction)
  FALLBACK   257
  PROMPT      88
```

**85% of every confirmation shown to a user is friction on a correct
prediction.** 68 of 120 gated confirmations originate from the keyword stage,
i.e. from a constant rather than a measurement.

Per-intent share of *correct* predictions that get an unnecessary "are you sure?":

| intent | correct-but-confirmed |
|---|---|
| device.volume.increase | **26/42 — 61.9%** |
| translation.session.start | **10/13 — 76.9%** |
| device.volume.decrease | **24/50 — 48.0%** |
| messaging.message.send | 11/28 — 39.3% |
| reminders.task.create | 40/130 — 30.8% |
| streaming.session.stop | 3/10 — 30.0% |
| device.volume.mute | 1/19 — 5.3% |

The distribution of confidence over *correct* gated predictions gives the tell:

```
5th percentile  = 0.750
10th percentile = 0.750
25th percentile = 0.921
median          = 0.991
```

The entire low-confidence tail of correct predictions is one constant. There is
no probability mass down there — just the `regex` literal.

**Deployment view:** volume up/down are the two highest-frequency commands in a
hearing-aid app, and they are the two most-penalised intents. A user raising the
volume in a noisy restaurant is being made to say "yes" to a device they cannot
hear well. This is the single worst place in the product to spend a confirmation
turn.

## 5. The decision ladder as actually implemented

You asked for exactly when the engine returns, confirms, or rejects. Here it is
read off `engine.py`, in evaluation order. Nothing below is from the docs.

### 5.1 Turn-level dispatch — `handle()` (L535)

| # | Condition | Path |
|---|---|---|
| 1 | `session.pending_confirm` set | `_handle_uncertain_confirmation` |
| 2 | active followup context | `_handle_confirmation` |
| 3 | `session.pending_intent` set | `_handle_slot_filling` |
| 4 | back-reference pattern matches | `_try_back_reference` |
| 5 | otherwise | `_handle_new_intent` |

Input is truncated to 500 chars before any inference.

### 5.2 Fresh turn — `_handle_new_intent` (L954)

```
classify → polarity guard → help-marker guard → cfg lookup
```

Effective fire threshold:

| intent shape | threshold | source |
|---|---|---|
| has slots | **0.50** | `slot_confidence_threshold` |
| no slots | **0.70** | `confidence_threshold` |

Then:

| band | outcome |
|---|---|
| `conf ≥ threshold` and intent **not** gated | **FULFILL** — fire immediately |
| `conf ≥ threshold` and intent gated and `conf ≥ 0.91` | **FULFILL** |
| `conf ≥ threshold` and intent gated and `conf < 0.91` | **CONFIRM** ← where volume dies |
| `conf < threshold` → semantic rescue accepts | fulfil via MiniLM path |
| `conf < threshold`, gated, `conf ≥ 0.55` (`confirm_floor`), no slots | **CONFIRM** |
| otherwise | **FALLBACK** → `genai.fallback` (the "reject") |

Semantic rescue acceptance (`self.semantic` is not None):
- standard: `sem_conf ≥ 0.40` (`semantic_threshold`), or
- agreement gate: TF-IDF and MiniLM name the same real intent → bar drops to
  `AGREEMENT_THRESHOLD = 0.50`.

### 5.3 Slot flows — `_advance_slots` (L780)

- Any required slot missing → **PROMPT**.
- All slots present, `entry_conf` given (i.e. the flow completed on the
  classifying turn), intent gated, `entry_conf < 0.91` → **CONFIRM**.
- Otherwise → **FULFILL** with `confidence = 1.0`.

Continuation turns pass `entry_conf=None` and therefore skip the gate by design
— once the user has answered a prompt the flow is established.

### 5.4 Resolving a held action — `_handle_uncertain_confirmation` (L927)

| reply | outcome |
|---|---|
| affirmative | **FULFILL** with the parked parameters, `confidence = 1.0` |
| negative | **FULFILL** `sys.confirm.cancelled`, no action |
| unparseable | drop the held action (never fire on ambiguity), reprocess the turn fresh |

`_yes_no` neutralises the `_NO_IDIOMS` table ("no worries", "no problem") before
polarity scanning, and returns `None` on `_UNCERTAIN` ("maybe", "not sure").
Ambiguous both-polarity replies resolve to **False** — fail-safe. This part is
correct.

### 5.5 Interruption and abandonment

- Re-classify every slot-filling turn; switch topic if `conf ≥ 0.68`
  (`interrupt_threshold`), unless the turn is a weak `contains` keyword hit or
  is a valid answer to the awaited slot (`≥ 0.90` strict, non-fuzzy).
- Pure cancel cue, or bare refusal ≤ 2 tokens → **FULFILL** `sys.slot.cancelled`.
- 3 failed attempts on one slot (`MAX_SLOT_ATTEMPTS`) → **FALLBACK**.

### 5.6 Two structural observations about the ladder

**`confidence_threshold` is dead for all 14 gated intents.** The gated set has a
fire bar of 0.91 and a confirm floor of 0.55; the 0.70 threshold is never the
operative bound for any of them. The confirmation band is **36 points wide** —
[0.55, 0.91). For a 57-class problem with a well-calibrated head (ECE 0.0119),
that band is far wider than the model's actual uncertainty region.

**Coverage is defined by exclusion, not by cost.** `test_confirm_gate.py`
asserts that *every* state-changing intent is gated, so the set is "everything
that isn't `help.*`, `sys.*`, or a `.query`". That makes an irreversible
`messaging.message.send` and a trivially-reversible `device.volume.increase`
carry identical friction. Cost-asymmetry is the whole justification for a
confirmation gate; erasing it makes the gate a tax rather than a safety net.

## 6. Verdict on "did we overengineer this?"

Component by component:

| component | verdict |
|---|---|
| Temperature calibration (OOF, eval sets excluded, provenance) | **Correct.** ECE 0.1206 → 0.0119. Keep. |
| Band fit out-of-fold, not on holdout (`fit_confirm_gate.py`) | **Correct methodology.** Keep. |
| Polarity guards / help-marker guard | **Correct.** High-precision, abstain-on-contradiction. Keep. |
| Slot-flow gate escape fix (`entry_conf`) | **Correct.** Real bug, properly fixed. |
| `_yes_no` idiom handling, cancel purity check | **Correct.** Keep. |
| **Keyword `KEYWORD_CONFIDENCE` constants** | **Wrong.** Unfitted, unvalidated, inverted vs. measured precision. |
| **Keyword short-circuit — no arbitration** | **Wrong.** Discards a 0.999 signal to return a 0.75 literal. |
| **Gate coverage = all state-changing intents** | **Wrong shape.** Ignores action cost and reversibility. |
| **Band applied to non-calibrated scores** | **Wrong.** The category error. |

Not overengineered — **under-integrated**. Each stage was built and validated in
isolation, and no test compares a keyword-stage confidence to a model-stage
confidence, because temperature scaling is rank-preserving and every accuracy
test passes either way. The defect lives exactly in the blind spot the existing
test suite was designed around.

## 7. Fix

### Fix 1 — arbitrate, don't short-circuit (the actual repair)

Always run the statistical head. Treat the rule and the model as two independent
recognizers and combine them. This is what a production assistant does.

```python
def classify(self, text: str):
    kw_intent, kw_tier_conf = self._keyword_match(text)
    kw_tier = self.last_keyword_tier
    model_intent, model_conf = self._model_classify(text)   # always

    if not kw_intent:
        self.last_stage = "tfidf"
        return model_intent, model_conf

    self.last_stage = "keyword"
    if model_intent == kw_intent:
        # Corroborated: two independent recognizers agree. Strongest evidence
        # the system can produce. Report the CALIBRATED confidence.
        self.last_arbitration = "corroborated"
        return kw_intent, max(model_conf, kw_tier_conf)

    # Contested: rule and model disagree. THIS is genuine ambiguity — the
    # condition the confirmation gate exists to catch. The rule still wins the
    # label (it is deliberate, hand-authored intent), but it must not claim
    # calibrated certainty it does not have.
    self.last_arbitration = "contested"
    return kw_intent, min(kw_tier_conf, model_conf)
```

Cost, measured on this repo's ONNX graph: **0.065 ms/utterance** for the extra
inference (keyword scan is 0.013 ms). The short-circuit buys nothing on device.

Result on the honest holdout — friction drops by two thirds for one additional
wrong action in 1470 turns:

| configuration | wrong actions | confirmations | useful | **friction** |
|---|---|---|---|---|
| **current** | 29 | 165 | 24 | **141** |
| **arbitration, band 0.91** | 30 | 70 | 22 | **48** — −66% |
| arbitration, band 0.85 | 31 | 51 | 21 | 30 |
| arbitration, band 0.80 | 33 | 41 | 19 | 22 |

`increase volume` / `decrease volume` return **FULFILL at 0.999 / 0.978** under
arbitration.

I recommend arbitration at the currently-fitted band (0.91) as the first change
— it is the pure bug fix and does not touch a fitted parameter. Band retuning is
a separate, later decision.

### Fix 2 — make keyword reliability empirical

Replace the four hand-written constants with per-tier (ideally per-rule)
precision measured out-of-fold, written to an artifact with provenance, exactly
as `calibration.json` and the band already are. Measured today: `regex` 0.954,
`regex_guarded` 0.767. This also correctly *starts* gating `regex_guarded`,
which is genuinely unreliable and currently fires unchecked at 0.90.

### Fix 3 — tier the gate by action cost

Split the gated set by reversibility rather than gating everything
state-changing:

- **`never`** — trivially reversible, self-evident from device feedback:
  `device.volume.increase`, `.decrease`, `.unmute`. The user hears the result
  instantly and can undo it in one word.
- **`when_ambiguous`** — current behaviour: `device.memory.change`,
  `streaming.*`, `transcription.*`, `translation.*`, `find.phone.locate`,
  `reminders.*`.
- **`always`** — irreversible or externally-visible, confirm regardless of
  confidence: `messaging.message.send`.

`policies.json` already has the `never | when_ambiguous | always` vocabulary and
`content_bundle.py` already emits it — the taxonomy exists, it is just uniformly
populated. Note this contradicts `test_confirm_gate.py::
test_every_state_changing_intent_is_gated`, so it is an explicit owner decision,
not a silent change. Review-F5 B1 already flags `always_confirm` as open.

### Fix 4 — a test that would have caught this

Nothing in the suite compares scores across stages. Add:

1. **Cross-stage scale invariant** — for a sample of utterances that trip a
   keyword rule, assert the reported confidence is within a tolerance of the
   model's calibrated confidence when the two stages agree. Any hardcoded
   constant fails this immediately.
2. **Head-command smoke test** — the N highest-frequency commands per language
   must return `FULFILL`, not `CONFIRM`. `increase volume` belongs in it.
3. **Friction budget** — assert correct-but-confirmed on the honest holdout stays
   under a declared ceiling, alongside the existing `WRONG_ACTION_BUDGET = 5`.
   The suite currently budgets only one side of the trade, which is why the
   friction side could regress to 85% unnoticed.

## 8. Secondary findings

- **`dist/bundle-en/runtime/policies.json` is stale.** It omits
  `uncertain_confirm_below` / `uncertain_confirm_floor`, which
  `content_bundle.py:398` does emit. BUG-007 is fixed in source; the checked-in
  artifact predates it. A device loading that bundle knows *which* intents are
  gated but not *when* — so it confirms always or never. Rebuild, and add a
  freshness assertion.
- **`routing.json` carries `below_confidence: 0.7`** under `ladder.reprompt` —
  same key name, different meaning from the confirm band. Two distinct concepts
  sharing a name in one spec is a live foot-gun; rename one.
- **`NLUEngine` never reads `policies.json`.** It loads
  `language_packs/<lang>/nlu_schema.json` directly. The compiled bundle is
  produced and validated but is not the runtime's source of truth, so schema and
  bundle can diverge with nothing failing.
- **The `contains` keyword tier is dead** (0 rules shipped). `_is_negated`, the
  per-language negation cue tables, and the `weak_keyword` interrupt guard that
  depends on it are all currently unreachable.

## 9. Recommended order

1. **Fix 1** — arbitration. Pure defect repair, no fitted parameter changes.
   Resolves the reported symptom.
2. **Fix 4.2** — head-command smoke test. Locks the symptom out.
3. **Fix 2** — empirical keyword reliability. Removes the remaining unfitted
   numbers from the confidence path.
4. **Fix 4.1 / 4.3** — cross-stage invariant + friction budget.
5. **Fix 3** — cost-tiered gate. Owner decision; changes an existing test.
6. Re-run `fit_confirm_gate --lang en` *after* 1–3, since the confidence
   distribution the band is fit against will have changed shape.
7. Secondary findings — rebuild `dist/`, rename the `routing.json` key, decide
   whether the bundle or the schema is the runtime's source of truth.

---

*Reproduced against `models/intent/en/` @ T=0.653712 and
`language_packs/en/holdout_honest.csv` (1470 rows), semantic stage disabled
(the keyword short-circuit precedes it, so it does not affect these paths).*
