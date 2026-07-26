STATUS: IN PROGRESS — Track A COMPLETE (A1–A10); Track B COMPLETE through B5; B5 result is OVER BUDGET and awaits an owner policy decision

# English Production Cycle — Status

- **Run date:** 2026-07-26
- **Branch:** `claude/nlu-production-readiness-dqyl38`
- **Driven from:** an interactive session, not the scheduled Routine (see
  *Blockers*). Charter: `docs/Review-F5/ENGLISH-PRODUCTION-ROUTINE.md`.

## Data-gate state — **AUTHORITATIVE (English)**

**DVC was removed on 2026-07-26** (owner decision: no third-party tooling for
6.7 MB of CSV that a local-only remote made unreachable from every machine but
one). `datasets/` is now tracked in git — see `datasets/README.md`.

The authoritative tree was recovered from git history at `a6cbb81c` (byte-exact
against the old `datasets.dvc` fingerprints) and committed in `aeb1cdcf`, then
relaid out per-language in `0e1080a6`. **The bootstrap gate is lifted for
English:** English metrics below are authoritative, not provisional.

fr/de/da remain closed — they have no honest holdout, no retrained model on the
per-language layout, and no fitted temperature. Nothing in this document makes a
claim about them.

## Completed

| Step | State | Acceptance gate |
|---|---|---|
| **A1** English datetime golden corpus | ✅ | 130 cases / 29 branches; `pytest tests/test_datetime_parity_en.py` 133 passed; `entities.py` untouched |
| **A2** Correct the calibration-chain docs | ✅ | No memory file claims the runtime reads `config/calibration.json`; banner present; `test_smoke` 3 passed |
| **A3** Language-neutrality guard (ratchet) | ✅ | Guard exits 0 with the allowlist; mutation-verified both directions; `test_neutrality` passed |
| **A4** Language-aware negation suppression | ✅ | 21 negation tests; allowlist 6 → 5; ruff clean |
| **A5** Normalised leakage matching + declared deps | ✅ | 14 leakage tests; bundle build/lifecycle/spec 27 passed with no import errors; suite 208 → 262 passed as declared deps unlocked previously-erroring tests |
| **A6** Language Pack contract on `spec/bundle/3.0` | ✅ | 27 contract + boundary tests; loads the real golden bundles; contract imports nothing heavy |
| **A7** Evict English behaviour into pack tables | ✅ | Golden 130 + 27 strip cases; fr/de/da parity 25 unchanged; **neutrality allowlist 6 → 0** |
| **A8** Hostile-language proof | ✅ | 8 tests; a made-up `zz` runs the full pipeline with no engine edit; static scan mutation-verified both directions |
| **A9** Neutrality guard blocking in CI | ✅ | Blocking step in `ci.yml`, before pytest |
| **A10** `release-pack.yml` (dev-signed, channel-gated) | ✅ | 13 tests; assemble → sign → verify → load proven end to end; key id + channel are workflow inputs |

### Track B

| Step | State | Result |
|---|---|---|
| **B1** Honest holdout + baseline-v2 | ✅ | 1470-row holdout, disjoint by normalised text. Accuracy 0.9007 / macro-F1 0.8872 vs the leaked set's 0.907. **Accuracy barely moved; ECE was 2.4× worse (0.0441 vs 0.018).** The leakage was concealing *miscalibration*, not inflating accuracy |
| **B2** Out-of-fold temperature fit | ✅ | 5-fold OOF, evaluation sets excluded, provenance recorded. **T = 0.657336**; OOF ECE 0.1170 → 0.0099 |
| **B3** Wire the runtime to the fitted T | ✅ | `calibration.json` now takes precedence over the device `weights.json`. Honest-holdout ECE 0.0441 → 0.0315; accuracy bit-identical (the rank-preservation check) |
| **B4** Calibration hygiene tests | ✅ | Provenance required by test; source-hash staleness check; fitter-vs-trainer featurizer drift check; planted-leak mutation test |
| **B5** Honest wrong-action re-measure | ✅ measured | **11 wrong actions vs a budget of ≤5.** Over budget. No policy changed — see below |

Suite at end of run: **348 passed, 58 skipped, 0 failed.** The 26 skips are all
model-dependent (`trained artifacts not present`) — expected in the bootstrap
grade, and the subject of charter step B0. The count rose from 208/30 because
A5's declared dependencies let previously-erroring bundle tests actually run.

Neutrality ratchet: **CLOSED at 0** (6 → 5 at A4 → 0 at A7). Review-F5 blocker
**B10 is closed** — the engine has no language branches and no embedded match
vocabulary. Adding a language is now shipping files, not editing code. A8 is
what *proves* that rather than asserting it.

## Findings this run

1. **`_NEGATIONS` was dead code, not a live bug — correction to the round-2
   review.** The review called it a live fr/de/da safety defect. In fact the
   shipped schema declares 28 `regex` and 4 `exact` keyword triggers and **zero
   `contains` rules**, and `_is_negated` is only reachable from the `contains`
   branch. So it was unreachable for *all four* languages. Latent, not live —
   it activates the moment a `contains` rule is authored. Fixed regardless,
   since it also blocked neutrality.

2. **Word-number relative durations do not work** (recorded in the golden
   corpus as `KNOWN_GAPS`, not fixed). `"in five minutes"` returns `None`: §1
   matches only `\d+` and runs *before* the §3 word-number normaliser. The
   normaliser does produce `"in 5 minutes"`, and that parses correctly — only
   the ordering is wrong. Word-number *clock* times (`"nine pm"`) work because
   §6 runs after the normaliser. Worse, it is environment-dependent: with
   `dateparser` installed the §8 fallback likely absorbs these, so the feature
   appears to work in a dev environment and silently fails in a lean container.
   Fixing it is a behaviour change and needs its own step.

3. **A pre-existing English negation false positive**, fixed as part of A4:
   substring matching meant `"another translate"` contained `"not"` and was
   silently suppressed. Now word-boundary matched.

4. **The reference branch's grammar table silently dropped a variant.** The
   original regex was `half\s+an?\s+hour`, so `"in half a hour"` — ungrammatical
   but common ASR output — resolved. The ref's `half_an_hour` list contained
   only `"half an hour"`. Caught by the golden corpus during the port, not by
   review. This is the clearest evidence that porting the ref's tables needed a
   parity net rather than trust.

5. **`strip_datetime` had zero test coverage**, so a whole function was about to
   be refactored unguarded. Before trusting the new implementation it was
   diffed against the pre-refactor version from git across 27 topic-strip cases
   — zero differences — and only then captured. The corpus pins the original
   behaviour, not the new implementation's opinion of it.

6. **`multilingual/models/<lang>/` is a legacy layout the pack architecture
   cannot produce — and the engine still depends on it.** Owner observation,
   confirmed: there is no "train multilingual" step in the target design. Each
   Language Pack carries its own model at `models/intent/<lang>/model.onnx`
   inside the bundle; a combined multilingual model has no place in it.

   Yet three supposedly-current modules still resolve models from the legacy
   tree, and only `multilingual/train_multilingual.py` writes it:

   - `packages/runtime/nlu_engine/engine.py:390` (`_load_classifier`)
   - `packages/buildtime/nlu_training/calibrate_languages.py:37`
   - `packages/buildtime/nlu_training/evaluate.py:35`

   This is not a neutrality violation — the guard is right that no language
   literal remains — but it is the same class of problem one layer down: a
   hardcoded per-language model path that the pack is meant to own.

   Consequence for the 58 skipped tests: they gate on
   `multilingual/models/en/en_intent_model.onnx`, so they can only ever run by
   invoking the legacy trainer. Running it to "clear the skips" would deepen a
   dependency we are removing. The right fix is to retarget model resolution at
   the pack — the "wire the engine to nlu_langpack" item already on the list —
   and repoint these tests at pack-produced artifacts.

7. **Three recovered files still carry the OLD 59-label space**, all 59 of their
   labels absent from the shipped 57-intent schema: `semantic_holdout_2.csv`,
   `semantic_holdout_expansion_template.csv` and
   `multilingual/pending/pva_intent_{danish,german}.csv`. The training masters
   are correctly migrated (57/57). `semantic_holdout_2.csv` is the one that
   matters — `train.py` uses it as the leakage-guard holdout, so it is being
   compared against labels that no longer exist. Needs migrating through
   `capability-map.json` (the same map the bootstrap used) or explicitly
   retiring.

8. **The A3 guard flagged prose.** The ported implementation split on `#`,
   leaving docstrings live, so a comment *describing* the forbidden pattern
   tripped it. Now tokenised — comments and string literals are blanked.

## B5 — the honest wrong-action number, and what it means

**English: 11 wrong actions on 1470 honest-holdout turns. Budget is ≤5. Not met.**

Artifacts: `tests/parity/oracle_honest_en/wrong_action_honest.json` (the run) and
`wrong_action_attribution.json` (the four-way comparison below). Semantic rescue
off. Per domain: messaging 4, device 3, find 2, streaming 2. Two wrong read-only
queries, not budget-charged. Eight confirmation-gated wrong guesses — the gate
did its job on those.

Before this run the harness read `multilingual/test/en_holdout.csv`, which is
99.9% training data (blocker B9). It now reads `datasets/en/holdout_honest.csv`
and **has no fallback to the old file** — a missing honest holdout is a hard
error, because quietly measuring the wrong thing is how the previous number
survived.

### Finding 1 — correcting the calibration made the count worse, and that is expected

Same engine, same data, only the temperature varies:

| Holdout | T | Wrong actions | Confirm-gated wrong |
|---|---|---|---|
| honest | **0.657** (correct, shipped) | **11** | 8 |
| honest | 0.796 (pre-B3 device value) | 5 | 12 |
| leaked | 0.796 | 4 | 4 |

`T = 0.657 < 0.796` sharpens the distribution, so every confidence rises. Six
turns that previously landed in the 0.70–0.80 CONFIRM band now clear 0.80 and
fire without asking. Nothing got less accurate — **the thresholds moved relative
to the confidence scale.**

The conclusion is *not* that B3 was wrong. B3 was right: OOF ECE went 0.1170 →
0.0099, and confidence now means what it says. The conclusion is that **0.70 and
0.80 were tuned against a miscalibrated scale, and the miscalibration was
accidentally acting as a safety margin.** Re-deriving those two thresholds on the
corrected scale is now a required step, and it is a *policy* change — charter
STOP rule 1 puts it with the owner.

### Finding 2 — this is an abstention problem, not a confusion problem

**10 of the 11 failures have `truth == sys.oos.fallback`.** Only one is a genuine
intent confusion (`"set me up for conversation"` → `device.volume.increase`).
The engine is not mixing commands up. It is failing to say "I don't know" on
out-of-scope speech: `"iphone"`, `"play festival"`, `"can you phone phone phone
phone phone"`, `"please my ears sweat"`.

This reframes the remediation. The B1 plan treats the budget as a
confusion/threshold problem; the honest data says it is an **out-of-scope
rejection** problem. `sys.oos.fallback` is a trained class competing on equal
footing with real intents inside one softmax, which is a weak abstention
mechanism — it has to *win* an argmax rather than act as a floor.

### Finding 3 — a threshold raise cannot close this

The residue is high-confidence: **5 of 11 fire above 0.95**, one at 1.000
(`"iphone"` → `find.phone.locate`), and only one sits in 0.70–0.80. Lifting the
fire threshold high enough to catch them would deflect a large share of correct
in-scope commands — 1061 of 1470 turns currently fulfil. The budget cannot be
bought with a constant.

### Recommendation (for owner decision — nothing here was implemented)

1. **Re-derive the 0.70/0.80 thresholds on the corrected confidence scale**, by
   sweeping them against the honest holdout for the wrong-action/recall frontier.
   This is the cheapest real gain and it is now *necessary*, because the current
   values were fitted to a scale that no longer exists.
2. **Add an explicit abstention margin** rather than relying on
   `sys.oos.fallback` to win an argmax — e.g. require a minimum top-1/top-2
   logit gap before firing a state-changing intent. Targets the 10.
3. **Unconditional confirmation on high-cost state-changing intents** (the
   standing B1 policy question). On this run it would convert wrong *actions*
   into wrong *confirmations*, which the budget does not charge.
4. **Grow OOS coverage.** 196 OOS rows against 1274 in-scope is a thin decision
   boundary for the one thing the budget is actually measuring.

Options 1–3 are all policy changes. Per STOP rule 1, **none were made**: this
step modified only the harness's data source and wrote reports.

## Blockers

| # | Blocker | Owner | Blocks |
|---|---|---|---|
| 1 | **Scheduled Routine cannot push.** Fired sessions get `403 Not authorized` on `receive-pack` while `upload-pack` works. This session pushes fine, so it is a per-session credential difference, not a repo setting. The trigger's stored config names an auto-generated branch (`claude/bold-bardeen`), suggesting the write grant is scoped to that rather than to the charter's fixed work branch. `trig_018ygxy3X9EgNX48wtquNeky` is **disabled** to stop hourly wasted runs. | owner / env config | unattended execution only — the work itself proceeds interactively |
| 2 | ~~Commit the full `datasets/` tree~~ — **RESOLVED.** Recovered from git history at `a6cbb81c` and committed (`aeb1cdcf`, relaid out in `0e1080a6`). | — | — |
| 3 | ND-8 production signing keys / KMS | owner | promoting releases past `channel: dev` |
| 4 | **Wrong-action budget: 11 vs ≤5.** Needs the threshold re-derivation *and* the unconditional-confirm decision — see *B5* above. Both are policy. | owner | shipping English at all |
| 5 | Authorisation for the French pack trial | owner | P3 / the neutrality proof |

## Next

**Track A (A1–A10) and Track B (B1–B5) are complete as specified.** B5 was the
last step either track authorises without an owner decision.

- **B6 — first green release run.** Not blocked by the budget: it publishes a
  `channel: dev` pack, and dev is precisely the channel for an artifact that is
  not yet cleared to ship. Ready to run.
- **The wrong-action budget** — blocker 4. Three of the four recommended moves
  are policy; the charter forbids making them here. This is the only thing
  standing between English and a shippable pack.
- **Stale paths in `release-pack.yml`.** The coreml-export step still names
  `dl/models/intent_model.onnx` and `models/intent_classifier_weights.json`,
  both retired by the per-language relayout. Must be fixed before B6 or the
  export step fails.
- **P3 French trial** — the acceptance test for language-independence.
  Owner-gated.
- **Revisit the removed polarity guards.** `content/platform.yaml` records their
  removal as deliberate, with evidence. Post-B1 retrain, `"turn mute on"` →
  `device.volume.unmute` at 0.512, caught only by the fire threshold. Worth
  re-examining now that confidence is trustworthy — but it is a policy file, so
  not in this cycle.
- **Legacy readers of the leaked holdout.** `multilingual/`'s own pipeline,
  `calibrate_languages.py`, `scripts/analyze_polarity_guard_failures.py` and
  `scripts/experiment_guards_off_holdout.py` still read
  `multilingual/test/*_holdout.csv`. None is on the English production path, so
  they were left alone — but any number they print is a memorisation score.
