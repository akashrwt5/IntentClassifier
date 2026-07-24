# B1 Remediation Plan — Meeting the Wrong-Action Safety Budget

**Owner of this doc:** Principal engineer (NLU safety)
**Date:** 2026-07-24
**Blocker addressed:** B1 from `production-readiness-review.md` — the medical
wrong-action budget is violated ~8× and the residue is high-confidence.
**Status:** Plan for owner review. **No shipped data, thresholds, calibration, or
policy have been changed** — every item below that touches training data, the label
set, thresholds, or the confirmation policy is an approval gate under charter §7 and is
written here as a proposal with expected impact and a measurement protocol.

---

## 1. The problem, precisely

From the committed replay (`tests/parity/oracle_post_migration/wrong_action_system_report.json`):

```
budget_met = False
wrong_actions_shipped_langs = 41   (en 11, fr 16, de 14)   |   da 11 (waived)
budget = 5
```

I clustered the 40 recorded failure examples by root cause. **This is the map the whole
plan is built on** — the budget is not one bug, it's four different problems that need
four different levers:

| Cluster | Count | Example | Why it fires high-confidence | Right lever |
|---|---:|---|---|---|
| **help → action** | 18 | "translate user guide" → `translation.session.start` (0.90) | `help.X.show` and `X.session.start` share the dominant content word | Guard expansion + **data** + confirm on recording actions |
| **volume polarity** | 7 | "lower how loud it is" → `device.volume.increase` (0.90) | decrease/increase/mute/unmute are lexically adjacent; noisy labels | **Data** + polarity-guard expansion (NOT confirmation) |
| **OOS → action** | 5 | "iphone" → `find.phone.locate` (0.999); "sende en besked" → `messaging.message.send` (1.00) | model has no strong reject basin; OOS recall is only 0.68 | **Data (negatives)** + confirm on irreversible actions |
| **activity confusion** | 4 | "how many calories jogging" → `activity.calories.query` | run/walk/stand/calories queries overlap | **Reclassify: read-only, not a harmful action** |
| cross-domain misc | 6 | streaming↔device, device→activity | tail confusions | Data / accept as residual |

**33 of 40 fire at ≥0.90 confidence.** This is the single most important fact: the
existing confidence-based confirmation gate (`conf < 0.80` → ask first) is *architecturally
blind* to this residue. Turning its threshold up would just add friction to the 7 correct
predictions between 0.80 and 0.90 without catching the 0.999 cases. **The fix is not a
threshold — it is (a) reduce confusability at the data/model layer, (b) add a
confidence-independent confirmation tier for the few genuinely irreversible actions, and
(c) stop counting read-only query errors as safety events.**

---

## 2. What the engine already does (so we build on it, not around it)

Verified in `packages/runtime/nlu_engine/engine.py`. The safety stack is real and
config-driven — this plan extends it, it does not replace it:

- **Polarity guards** (`_apply_polarity_guards`, 6 rules) — redirect a prediction
  contradicted by explicit polarity words (`\bmute\b(?!\s+off)` blocks unmute, etc.).
  Fire only on an unambiguous single match.
- **Help-marker guard** (`_apply_help_guard`, ND-14, 11 action→help pairs + a per-language
  `markers` regex) — if the utterance carries help/question markers *and* the predicted
  action has a `help.*` sibling, redirect to the read-only help intent.
- **Uncertainty-confirmation gate** (`uncertain_confirm`, 11 intents, `below_confidence:
  0.80`) — a flagged fire-and-forget intent *below 0.80* asks first instead of firing.

The 11 `uncertain_confirm` intents are already the correct high-cost set:
`device.volume.{decrease,increase,mute,unmute}`, `find.phone.locate`,
`messaging.message.send`, `reminders.task.complete`,
`{streaming,transcription,translation}.session.{start,stop}`.

**The gap is exactly the `< 0.80` condition.** Everything in §1 fires above it.

---

## 3. The plan — four workstreams

Ordered so the highest-harm, lowest-friction, fastest wins land first.

### WS-1 — Reframe the budget: per-domain, and harm-tiered (do this first)

The charter itself says the budget is "≤5 global, **moving toward per-domain budgets**."
Before changing any model, fix what we are counting, because it changes where effort goes.

1. **Separate harmful actions from wrong read-only queries.** The 4 activity-confusion
   cases (`activity.*.query`) change nothing on the device and cost the user nothing —
   confusing "calories" with "run" is a *query-accuracy* miss, not a safety event. They
   should be tracked under an accuracy metric, not the wrong-**action** budget. This is a
   definition change to the harness/report, not a model change.
2. **Define per-domain, harm-tiered budgets.** Proposed tiers:
   - **Tier-0 irreversible / privacy-invasive** (send a message, start recording):
     budget **0** — these must never fire wrong. Enforced by WS-2 confirmation.
   - **Tier-1 state-changing reversible** (volume, streaming, reminders): small per-domain
     budget (e.g. ≤2 per domain per language), driven down by WS-3 data work.
   - **Tier-2 read-only query**: not in the action budget; tracked as query F1.
3. **Deliverable:** update `wrong_action_harness` + report schema to emit
   `per_domain` and `per_tier` counts with tier-specific budgets and a single
   `budget_met` boolean per tier. (Code-only change; no model retrain.)

> **Approval note:** redefining the budget is a safety-accounting change — it must be
> owner-approved so no one can later claim the bar was moved to make a number pass. The
> justification is harm, not convenience: a wrong volume nudge is recoverable in one word;
> a wrongly-sent message or a silent recording is not.

### WS-2 — Confidence-independent confirmation for irreversible actions (the policy lever)

This is the only lever that catches the 0.90–1.00 residue. Keep it **surgical** — apply it
*only* where a wrong fire is genuinely irreversible or privacy-invasive, because every
intent added here is permanent friction.

**Proposed Tier-0 always-confirm set (3 intents):**

| Intent | Why it must confirm regardless of confidence |
|---|---|
| `messaging.message.send` | Irreversible — a wrongly-sent message can't be recalled. "sende en besked" fired at 1.00. |
| `transcription.session.start` | Starts recording the user — privacy/medical-sensitive; a silent start is a consent violation. |
| `translation.session.start` | Same — starts capturing audio; the dominant help→action failure (8 German cases) lands here. |

`find.phone.locate`, `reminders.task.complete`, `streaming.*`, and `device.volume.*` stay
**confidence-gated only** — they are reversible/low-harm and unconditional confirmation
would be intrusive on common actions.

**Design (minimal, config-first — no new code path if possible):**

- Add a schema key `uncertain_confirm.always_confirm: [<intent>, …]` (a subset of
  `intents`). Semantics: for an intent in `always_confirm`, the fulfill path asks first
  **regardless of confidence** (as long as it has no open slots).
- One-line engine change in `_fulfill_intent`, guarded so behavior is unchanged when the
  list is empty:

```python
# _fulfill_intent, replacing the single uncertainty check:
gate = intent in self._confirm_intents and (
    intent in self._always_confirm or conf < self._confirm_below
)
if gate:
    session.pending_confirm = {...}
    return NLUResult(type="CONFIRM", ...)
```

- Localized `confirm_prompt` per Tier-0 intent already flows through the overlay system
  (`content/localization/*`), so this is data, not code, in every language.

**Expected impact:** eliminates the messaging + recording-start wrong actions entirely
(they become an ask-first turn), i.e. removes the highest-harm cluster and a large slice
of the help→action and OOS→action counts that terminate in these three intents.
**Friction cost:** one extra confirmation on send/record — acceptable and arguably
*desirable* for a medical device (explicit consent before recording is a feature).

### WS-3 — Data & model work on the confusable clusters (the durable fix)

Confirmation is a safety net; the real fix is making the model stop confusing these. All
of this is an approval gate (retrains shipped models) and needs the datasets, which are
DVC-hosted (see B2 — this work is **blocked until a real DVC remote exists and models can
be regenerated + re-replayed in CI**).

1. **Volume polarity (7).** Audit `device.volume.*` training rows for the residual
   decrease/increase/mute confusions (the fr and da label-conflict passes already found
   this pattern — extend it to en/de). Add hard-negative contrastive pairs ("lower how
   loud it is" labeled decrease next to "make it louder" labeled increase). Expand the 6
   polarity-guard rules to cover the missed surface forms. **Target:** volume domain ≤2
   wrong per language.
2. **OOS rejection (5).** OOS recall is 0.68 — a third of out-of-scope utterances leak in.
   Augment the `sys.oos.fallback` class with adversarial negatives resembling the false
   fires ("iphone", short brand words, profanity, fragments). Consider a per-intent
   minimum-margin reject for Tier-0/Tier-1 targets. **Target:** OOS recall ≥0.80, no
   confident OOS→Tier-0 fire.
3. **help → action (18).** Two sub-fixes: (a) **guard** — extend the ND-14 marker set to
   *declarative* help asks that carry no interrogative ("translate user guide", "stream
   from an accessory mic", "battery questions"), which currently slip the regex; (b)
   **data** — the German translate cluster (8) is really **ND-13**: those `help.*` and
   `translation` German rows are untranslated English placeholders, so the model never
   learned German help phrasing. This needs native German authoring, not another guard.
4. **Re-calibrate** after every retrain (temperature scaling is already in place; keep ECE
   ≤0.03) and **re-replay** the wrong-action harness. No metric may regress vs the recorded
   baseline.

### WS-4 — Make the budget a permanent, enforced gate (depends on B2)

A safety number that isn't in CI will drift back. Once B2 (real DVC remote + `dvc pull` in
CI) is done:

- Wire the wrong-action replay as a **blocking** CI job with the per-tier budgets from
  WS-1: **Tier-0 = 0**, Tier-1 per-domain ceilings, produced as the release gate artifact.
- Fail the build (do not skip) when models/data are present in the release context.
- Add the WS-2 always-confirm behavior and the WS-3 contrastive pairs to the golden
  conversation corpus so a regression is caught as a unit test, not only in the full
  replay.

---

## 4. Sequencing & expected trajectory

| Step | Workstream | Gate | Blocks on | Effect on the 41 |
|---|---|---|---|---|
| 1 | WS-1 budget reframe (code + report) | owner sign-off on accounting | — | −4 (activity queries leave the action budget) |
| 2 | WS-2 always-confirm 3 Tier-0 intents | owner sign-off (policy/UX) | — | removes messaging + recording-start wrong fires (highest-harm) |
| 3 | WS-3a guard expansion (help declarative + polarity surface) | owner sign-off (behavior) | — | cuts a chunk of help→action + volume without a retrain |
| 4 | WS-3b data/model (volume, OOS, German) + recalibrate | owner sign-off (retrain) | **B2** (DVC remote) | drives Tier-1 domains toward their per-domain budgets |
| 5 | WS-4 blocking CI budget gate | — | **B2** | prevents regression forever |

Steps 1–3 need no retrain and no dataset access — they are executable now on owner
approval and should meaningfully cut the count and remove the worst harm. Steps 4–5 are
where the number actually reaches budget, and they are **gated on B2**, which is why the
review recommended treating B1 and B2 as one hardening milestone.

**Honest expectation:** WS-1+2+3a will take the *harmful* count down sharply and zero out
Tier-0, but hitting a strict global ≤5 across all four languages almost certainly requires
WS-3b (the data work) — especially the German native-authoring for the translate cluster,
which no amount of guarding or gating can fix. Set owner expectations that "budget met" is
a data milestone, not a policy toggle.

---

## 5. What I need from the owner to proceed

Concrete decisions, each small:

1. **WS-1:** approve the harm-tiered / per-domain budget accounting (Tier-0 = 0, Tier-1
   per-domain ceiling, read-only queries excluded from the action budget).
2. **WS-2:** approve unconditional confirmation on the 3 Tier-0 intents (accept the one
   extra tap on send/record). Confirm the Tier-0 set — in particular whether
   `find.phone.locate` should join it.
3. **WS-3:** approve a retrain cycle once B2 lands, and green-light the German
   native-authoring task (ND-13) — this is the true blocker on the translate cluster.
4. Confirm the **strict target**: global ≤5, or per-tier (Tier-0 = 0, Tier-1 ≤ N/domain)?
   The latter is more honest for a multi-domain product and is what the charter is moving
   toward.

On approval of 1–3, I can implement WS-1, WS-2, and WS-3a immediately (code + config +
tests, no model retrain, no dataset access), re-run the replay against the committed
fixtures, and report the new per-tier counts before we touch a single model.

---

### Appendix — files this plan touches
- Policy lever: `packages/runtime/nlu_engine/engine.py` (`_fulfill_intent`), schema key
  `uncertain_confirm.always_confirm`, per-language `content/localization/*` confirm prompts.
- Guards: `content/nlu_schema.json` `help_marker_guard.markers` / `polarity_guards`
  (+ per-language overlays).
- Budget accounting: the wrong-action harness + `report_card`/wrong-action report schema
  (`spec/bundle/3.0/report_card.schema.json`), CI gate in `.github/workflows/ci.yml`.
- Data (WS-3, gated on B2): `datasets/multilingual/{en,fr,de,da}.csv`, OOS class,
  German `help.*`/`translation` authoring (ND-13).
