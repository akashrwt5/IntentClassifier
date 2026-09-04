# Confirmation-gate remediation plan

Companion to `docs/confirm-gate-diagnosis.md`, which establishes the defect.
This file is the plan of record: what changes, in what order, and what each step
is blocked on.

## The change in one paragraph

Confidence is currently written by two components on two incompatible scales — a
hand-typed constant from the keyword stage and a temperature-calibrated softmax
from the model — and then compared against a band (`0.91`) that was fitted on
only one of them. Separately, that band sits **above** the fire threshold, so it
converts commands that would have fired into questions. The remedy is to make
the model the sole author of confidence, and to move the only legitimate
confirmation band **below** the fire threshold, where it recovers commands that
would otherwise be rejected instead of taxing commands that would have worked.

## Decision ladder AS SHIPPED

The plan originally proposed a three-way ladder with a recovery band below the
fire threshold. **The owner decided against it**: confirmation is removed
entirely, matching Dialogflow and this product's own legacy behaviour.

```
conf >= confidence_threshold   ->  the intent fires
conf <  confidence_threshold   ->  the fallback intent
```

One number, `confidence_threshold = 0.70`. Removed outright:
`uncertain_confirm.below_confidence`, `uncertain_confirm.confirm_floor`,
`uncertain_confirm.intents` (the hand-curated 14-intent list),
`slot_confidence_threshold`, and the four `KEYWORD_CONFIDENCE` constants.

Two things are deliberately NOT part of the binary rule:

- **`agreement_threshold` (0.50)** — when two independent recognisers name the
  same intent (a keyword rule and the model, or the model and the semantic
  head), the bar drops. This is evidence strength, not a confidence: the
  reported number stays the model's calibrated probability. It reuses the
  relaxation this engine already applied to TF-IDF/MiniLM agreement, and is now
  content-owned rather than a class constant. Measured: corroborated turns are
  99.2% correct (n=118), and 100% correct in the 0.50–0.70 band (n=4).
- **An authored `followup` on `messaging.message.send`** — the one irreversible,
  externally-visible action confirms on *every* turn regardless of confidence.
  This is dialogue design, the same mechanism Dialogflow used, and it is what
  keeps the `Cmd.SendMessage - yes/no` app contract alive. It also makes that
  contract deterministic; it previously appeared only when the classifier
  happened to land inside the old uncertainty band.

### Cost of removing the band

Honest holdout (1470 turns), before → after:

| | before | after |
|---|---|---|
| wrong actions | 27 | 36 |
| correct fires | 1028 | 1036 |
| confirmations | 119 (16 useful / 103 friction) | 27 (send-message only) |
| valid rejected | 98 | 115 |

The band was catching 16 wrong predictions; without it they reach the action.
That is recorded in `WRONG_ACTION_BUDGET = 13` (harness definition) rather than
absorbed quietly. **Lower it by improving the model or the threshold, never by
reintroducing a confidence-triggered ask.**

### Why there is no band above FIRE

Measured on `holdout_honest.csv` (1470 rows), with arbitration applied:

| configuration | wrong actions | correct fires | rejections |
|---|---|---|---|
| pure binary, T=0.70 | 41 | 1128 | 127 |
| ladder, FIRE=0.70 / FLOOR=0.55 | 41 | 1128 | 71 |

Identical wrong actions, identical fires. The band below FIRE changes only the
rejection count — it converts 56 rejections into questions. It is a **recovery**
mechanism, not a safety mechanism; the fire threshold is what controls wrong
actions.

A band above FIRE does the opposite: it converts would-be fires into questions.
That is the entire source of the 85%-friction figure in the diagnosis, and it is
a behaviour the product never asked for — the Dialogflow system this replaces
had no confidence-triggered confirmation at all (`confirm_compound` in
`legacy_label_map.json` carries exactly one entry, an explicitly authored
send-message dialogue).

## Steps

### Step 1 — arbitration in `classifier.classify()`  ✅ ships now

Run the model on every turn. On a keyword hit:

- model agrees with the rule → return the model's calibrated probability
- model disagrees → return `CONTESTED_CONFIDENCE`

The rule keeps deciding the **label** (it is deliberate, hand-authored product
intent). The model becomes the sole author of the **confidence**, which puts the
number back on the same scale as every threshold it is compared against.

`KEYWORD_CONFIDENCE` leaves the confidence path. `last_keyword_tier` stays — the
`weak_keyword` interrupt check in `engine._handle_slot_filling` reads it.

Measured basis (holdout, 972 keyword-stage rows across holdout+train):

```
model agrees with rule     n=807   rule correct 99.13%
model disagrees with rule  n=165   rule correct  6.67%   (holdout-only: 45%, n=20)
```

Cost: one extra inference on the 9.4% of turns that hit a keyword rule. The
other 90.6% already run the model. Measured 0.0634 ms per inference, 0.0059 ms
averaged across all traffic; end-to-end p50 0.098 → 0.106 ms.

This step is a pure defect repair — it changes no fitted parameter — and is
mergeable on its own.

### Step 2 — regression tests  ✅ ships now

Nothing in the suite compares a score across stages, which is precisely why a
hardcoded constant could sit in the confidence field undetected. Add:

1. **Cross-stage scale invariant.** When the keyword stage and the model agree,
   the reported confidence must match the model's calibrated confidence. Any
   constant written into that field fails this immediately.
2. **Head-command smoke test.** The highest-frequency commands must return
   `FULFILL`, not `CONFIRM`. `increase volume` belongs in it.

### Step 3 — revive or remove the semantic stage  ⛔ blocks step 5

`NLUEngine.semantic` is `None` — the MiniLM artifacts do not load. Every
rejection figure in this plan is therefore a worst case: in production the
semantic stage would rescue an unknown share of sub-threshold turns. FIRE and
FLOOR cannot be fitted honestly until this is either working or deliberately
removed.

### Step 4 — remove the band above FIRE

- delete the `intent in self._confirm_intents and conf < self._confirm_below`
  block in `_fulfill_intent`
- delete the equivalent `entry_conf` gate in `_advance_slots`
- keep the sub-threshold recovery confirm in `_handle_new_intent`
- schema: drop `below_confidence` and `intents`; rename the floor to
  `recovery_floor` — it is not a gate and the name should not imply one

**Open decision:** the recovery band currently only applies to intents in the
14-intent list. With the list gone, does it apply to everything? A low-confidence
read-only `help.*` intent could equally be shown directly or rejected; asking is
the most expensive of the three. Recommendation: apply to all, drop the list
(simpler, and a question beats a rejection). This is a product call.

Tests that must change — note the first one, which currently enforces the defect
as an invariant:

| test | action |
|---|---|
| `test_confirm_gate.py::test_band_sits_above_the_fire_threshold` | **invert** — asserts `below_confidence > confidence_threshold`, i.e. asserts the friction band exists |
| `test_confirm_gate.py::test_every_state_changing_intent_is_gated` | delete — wrong invariant |
| `test_confirm_gate.py::test_confirmation_band_matches_the_fitted_value` | rewrite against the new fitted values |
| `test_confirm_gate.py::test_a_slot_flow_completing_on_entry_still_hits_the_gate` | rework |
| `test_content_bundle.py::test_confirmation_policy_matches_the_uncertainty_gate` | reads the 14-intent list |
| `test_wrong_action_mitigations.py::test_quiet_request_is_at_least_gated` | review |

### Step 5 — fit FIRE and FLOOR  ⛔ blocked on step 3

`fit_confirm_gate.py` sweeps one threshold and does not count rejections at all,
which is why the friction/rejection side of the trade was never visible. Extend
it to a joint `(FIRE, FLOOR)` sweep reporting all four turn outcomes:
correct-fire, wrong-fire, confirm, reject.

Out-of-fold on `train.csv`. The holdout is used once, to confirm.

Do not hand-pick these values. Fitting a constant is what the last two years of
this file have been; the point of the exercise is that the numbers come from
data with provenance attached.

Also add a **friction budget** alongside `WRONG_ACTION_BUDGET = 5`. Budgeting one
side of a two-sided trade is how friction reached 85% with a green suite.

### Step 6 — bundle, compiler, spec

- `content_bundle.py:398` emits `uncertain_confirm_below` / `_floor` — update to
  the new shape
- `spec/bundle/3.0/policies.schema.json` — rename keys
- the per-intent `confirmation: when_ambiguous` map in `policies.json` becomes
  meaningless once the gated list is gone
- rebuild `dist/bundle-en`; the checked-in copy predates BUG-007's fix and omits
  the band entirely
- `routing.json` uses the key `below_confidence` for the reprompt ladder — a
  different concept with the same name. Rename one.

### Step 7 — regenerate the legacy conformance fixtures  ⚠️ client-visible

`tests/fixtures/legacy_label_parity_en.csv` asserts `CONFIRM` for four volume
utterances. It is generated by running the engine, so it captured the defect as
the cross-platform contract iOS and Android are told to reproduce
(`docs/confirmation-contract.md`, ADR-011). Those four rows flip to `FULFILL`.

This is a contract change and must be communicated to the client teams, not
silently regenerated. Rebuild `dist/bundle-en` first — the generator loads it.

Add a test that consumes the CSV. Today only the generator touches it, so the
engine and the published contract can drift with nothing failing.

## Order of execution

```
1  arbitration                            defect repair, no fitted params
2  cross-stage invariant + smoke test      locks the defect out
3  semantic stage: revive or remove        unblocks 5
4  remove the band above FIRE, invert tests
5  fit FIRE / FLOOR out-of-fold + friction budget
6  bundle / compiler / spec, rebuild dist
7  regenerate fixtures, notify clients
```

Steps 1–2 land together and change no fitted parameter. Everything from step 4
onward alters shipped decision behaviour and should not start before step 3
makes the rejection numbers honest.

## Appendix — pending `.claude/memory/inference.md` update

This change set altered runtime behaviour, so per CLAUDE.md the matching memory
file must move with the code. It could not be written from the session that made
the change (`.claude/` was not writable).

**What to do:** open `.claude/memory/inference.md`, replace the
`3. **Classify**` bullet with the one below, and insert the two sections
immediately before the existing `## Confidence + calibration` heading.

````markdown
3. **Classify** — keyword/model arbitration -> single fire threshold ->
   **semantic rescue** if below it -> entity/datetime extraction -> slot prompts.

## The decision ladder

    conf >= confidence_threshold (0.70)  ->  the intent fires
    conf <  confidence_threshold         ->  the fallback intent

One threshold, two outcomes. There is **no confidence-triggered confirmation**.

`uncertain_confirm` (a 0.55–0.91 band over a hand-curated 14-intent list) was
REMOVED, not retuned: it sat ABOVE the fire threshold, so it turned commands
that would have fired into questions — 103 friction turns against 16 useful
catches on the honest holdout. `slot_confidence_threshold` (0.50) went too; a
slot-bearing intent whose slots are all filled by the classifying utterance
completes immediately, so the lower bar applied to a live action.

Two things sit outside the plain rule, both deliberate:

* **`agreement_threshold` (0.50)** — when two INDEPENDENT recognisers name the
  same intent, the bar drops to this. Evidence strength, not a confidence: the
  reported number stays the model's calibrated probability. Content-owned in
  `platform.yaml`. Corroborated turns measure 99.2% correct (n=118).
* **An authored `followup`** — `messaging.message.send` confirms on EVERY turn
  regardless of confidence. Dialogue design, not a classifier artifact, and what
  keeps `Cmd.SendMessage - yes/no` deterministic for the app.

Cost of removing the band: wrong actions 27 -> 36 on the honest holdout, pinned
as `WRONG_ACTION_BUDGET = 13` in `tests/test_decision_ladder.py`. **Lower it by
improving the model or the threshold, never by reintroducing an ask.**

## Keyword/model arbitration (`classifier.classify`)

The model runs on **every** turn and is the **sole author of confidence**. A
keyword rule, when one fires, is the sole author of the **label**.

| case | label | confidence |
|---|---|---|
| no keyword hit | model's argmax | model's calibrated probability |
| keyword hit, model agrees (`corroborated`) | rule | model's calibrated probability |
| keyword hit, model disagrees (`contested`) | rule | `CONTESTED_CONFIDENCE` (0.60) |

Why: a rule is deterministic and cannot express a probability. The keyword stage
used to short-circuit the model and return a hardcoded constant
(`KEYWORD_CONFIDENCE`, `regex` = 0.75), then compared it against thresholds
fitted on temperature-calibrated softmax — two scales, one comparison. Every
`regex` rule became un-fireable when the band moved 0.80 -> 0.91 (B8), and
"increase volume" asked for confirmation while the model scored it 0.9992.

This is also what `spec/bundle/3.0/keywords.schema.json` always required: tier 2
is "pattern boost **requiring classifier agreement**". The engine was the
component out of spec, not the clients.

**Do not put a constant back in the keyword path.**
`tests/test_confidence_scale.py` asserts the reported confidence tracks the model
when the stages agree. `KEYWORD_TIER_ORDER` survives for telemetry only;
`last_keyword_tier` still feeds the `weak_keyword` interrupt check.

Cost: one extra inference on the ~9% of turns that hit a keyword rule (the rest
already ran the model). 0.06 ms; end-to-end p50 0.098 -> 0.106 ms.

**Guard re-derivation.** When `_apply_polarity_guards` / `_apply_help_guard`
change the reported intent, `_handle_new_intent` re-reads the confidence via
`classifier.calibrated_confidence(new_intent)`. Confidence must describe the
intent actually being returned, not the one that was blocked.

**Known gaps.** `CONTESTED_CONFIDENCE = 0.60` is the one unfitted constant left
— fit it with `nlu_training.fit_decision_ladder`. Tier 1 should short-circuit
per spec but is currently arbitrated like tier 2 (0 cases on the holdout, so
latent). The semantic-rescue path applies guards without re-deriving confidence.
````

`known-issues.md` should also record that `NLUEngine.semantic` is `None` in this
tree (MiniLM artifacts absent), so every rejection figure is a worst case.

## What this plan does not do

It does not introduce the per-intent cost model (fire threshold derived from the
cost of a wrong action, rather than one global number). That remains the right
end state — `device.volume.increase` is trivially reversible and
`messaging.message.send` is not, and a single global FIRE cannot serve both. The
two-number ladder here is the correct first step toward it, not a substitute.
Real cost ratios are a product and clinical input, not an ML one.
