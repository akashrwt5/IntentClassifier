# Sweep: every decision justified by the leaked holdout

**Status:** measurement complete, for owner decision. No policy file modified.
Reproduce: `PYTHONPATH=packages/buildtime:packages/runtime python
scripts/analysis/guard_ablation_honest.py`
Raw output: `tests/parity/oracle_honest_en/guard_ablation.json`

After blocker B9 the calibration temperature was re-derived, but nothing else
was. Three English guard decisions cite replays of the 1,461-utterance
`multilingual/test/en_holdout.csv` — the set that is 99.9% training data. This
re-runs each through the full engine against `datasets/en/holdout_honest.csv`.

## Result

All figures on 1470 honest turns, `T = 0.657336`.

| configuration | wrong actions | correct fulfils | confirm-gated wrong | fallback |
|---|---|---|---|---|
| **baseline (as shipped)** | **11** | **1026** | 8 | 266 |
| + polarity guards restored | 14 | 1023 | 10 | 266 |
| − help_marker_guard | 18 | 1016 | 11 | 266 |
| + semantic rescue | 11 | 1076 | 22 | 166 |
| + polarity + semantic | 14 | 1073 | 23 | 165 |
| + polarity, − help_marker | 21 | 1013 | 13 | 266 |

## Verdicts

### `polarity_guards: REMOVED` — decision CORRECT. My challenge was wrong.

I argued in `b5-root-cause-audit.md` §4 that this removal rested on void
evidence and should be reversed. **Re-derived on honest data it does not
reverse.** Restoring the six rules costs **+3 wrong actions and −3 correct
fulfils.**

Applied directly to every honest-holdout utterance, the guards **fix 1 and break
5**:

| | utterance | model | guard redirects to |
|---|---|---|---|
| fixed | `turn mute on` (0.537) | unmute ❌ | **mute** ✅ |
| broke | `take the mute off` (0.998) | unmute ✅ | mute ❌ |
| broke | `take my aids off mute` (0.868) | unmute ✅ | mute ❌ |
| broke | `make it less loud` (0.995) | decrease ✅ | increase ❌ |
| broke | `it's too quiet` (0.750) | increase ✅ | decrease ❌ |
| broke | `the tv is too quiet for me` (0.750) | increase ✅ | decrease ❌ |

The rules are bare lexical matches: `\bmute\b` blocking `unmute` cannot tell
`"turn mute on"` from `"take the mute off"`. The original directive said exactly
this — *"the rules only ever flipped CORRECT answers ('cancel the mute' → mute,
'less loud' → increase)"* — and it was right about the mechanism as well as the
outcome.

**Also correcting myself on severity:** I called this a *live* safety
regression. It is not. `"turn mute on"` scores 0.537, below the 0.70 fire
threshold, so the engine deflects it — it never becomes a wrong action. The
mis-prediction is real; the safety consequence is not.

What survives: the *concept* is sound and the *implementation* is too crude. A
polarity rule that understood `off|remove|cancel|take … off` around `mute` would
fix the one case without breaking the five. That is a new rule to design and
validate, not a restoration.

### `help_marker_guard: KEPT` — decision CORRECT

Removing it costs **+7 wrong actions and −10 correct fulfils** (11→18, 1026→1016)
on honest data. The original claim (44 misfires fixed, 0 correct diverted) was
measured on the leaked set, but the direction and the safety rationale hold.
Keep it.

### `semantic_rescue_enabled: false` — UNRESOLVED, and the recorded trade-off is wrong

The memory note records the trade as *"+150 recovered valid commands per 1,461
turns at +5 wrong actions (23→28)"*. On the honest holdout, enabling it gives:

- **+50 correct fulfils** (1026 → 1076)
- **−100 deflections** (fallback 266 → 166)
- **+0 wrong actions** (11 → 11)
- +14 confirm-gated wrong guesses, +21 wrong-but-read-only `help.*` responses

The **cost side of the recorded trade-off does not reproduce.** The safety
budget is untouched; the recovered commands are real users who currently get
"sorry, I didn't understand" and would instead get their action.

**But this number cannot be used yet, and the reason matters:**

`models/semantic_head.json` was built **2026-07-25**. B1 created the honest
holdout on **2026-07-26**, by carving it out of `train.csv`. So every row of
`holdout_honest.csv` was in the semantic head's training data. **Measuring the
semantic path on this holdout is leaked for that component** — precisely the
error this whole sweep exists to correct, and I nearly recommended acting on it.

The +50 is an **upper bound**, not a measurement.

**Required before the flag is even a decision:** retrain the English semantic
head on the post-B1 `train.csv`, then re-measure. If the recovery survives at
anything near +50 for +0 wrong actions, enabling it is the largest single
quality gain available — 100 fewer deflections for a hearing-aid user is worth
more than most of what B1–B5 achieved.

## The meta-finding: only one artifact carries provenance

`models/semantic_head.json` contains exactly four keys — `embedder`, `labels`,
`weights`, `bias`. **No training source, no date, no data hash, no fit method.**
There is no way to tell from the artifact what it was trained on. That is why
its leakage had to be inferred from a file mtime.

Charter B4 made provenance mandatory for `calibration.json` and enforced it by
test. Nothing else in `models/` has it — not `model.onnx`, not `labels.pkl`, not
the semantic head. The temperature was the artifact that happened to get caught;
the discipline was never generalised.

**Recommendation: extend the B4 provenance contract to every model artifact** —
source path, source sha256, fit date, method — and extend
`tests/test_calibration.py`'s staleness check to match. Without it, the next
retrain silently produces the same class of untraceable claim.

## Summary

| Decision | Original evidence | Re-derived | Verdict |
|---|---|---|---|
| polarity_guards REMOVED | leaked | +3 wrong / −3 correct | **holds** (my challenge was wrong) |
| help_marker_guard KEPT | leaked | −7 wrong / +10 correct | **holds** |
| semantic rescue OFF | leaked | +50 correct / +0 wrong, **but head is also leaked** | **unresolved — retrain then re-measure** |

Two of three tainted decisions were right anyway. The third is the one worth
the work, and it needs a retrained head before it can be judged.

## Next

1. **Retrain the English semantic head** on post-B1 `train.csv` with provenance;
   re-run this ablation. *No policy change; this is a build step.*
2. **Extend the provenance contract** to every model artifact (B4 generalised).
3. **Design a correct polarity rule** for the `off|remove|cancel` construction,
   validated against the 6 cases above, if the mute mis-prediction is judged
   worth a rule at all — it currently deflects rather than misfires.
4. Leave `polarity_guards` empty and `help_marker_guard` in place. Both
   confirmed.
