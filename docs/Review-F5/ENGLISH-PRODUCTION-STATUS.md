STATUS: IN PROGRESS — Track A COMPLETE (A1–A10); Track B awaiting the full datasets in git

# English Production Cycle — Status

- **Run date:** 2026-07-26
- **Branch:** `claude/nlu-production-readiness-dqyl38` @ `7b85e3f4`
- **Driven from:** an interactive session, not the scheduled Routine (see
  *Blockers*). Charter: `docs/Review-F5/ENGLISH-PRODUCTION-ROUTINE.md`.

## Data-gate state — **BOOTSTRAP (PROVISIONAL)**

**DVC was removed on 2026-07-26** (owner decision: no third-party tooling for
6.7 MB of CSV that a local-only remote made unreachable from every machine but
one). `datasets/` is now tracked in git — see `datasets/README.md`.

The authoritative 29-file tree has **not been committed yet**: those bytes exist
only on the owner's machine, so this checkout has just the English bootstrap
snapshot (`python scripts/ci/bootstrap_en_data.py`, 5 files, source commit
`0089b894`).

**Every metric below is therefore PROVISIONAL.** No authoritative baseline may
be recorded, and no bootstrap-fitted calibration may be wired to the runtime,
until the full datasets are committed. fr/de/da model work stays closed.

No model metrics were produced this run — A1–A4 are all structural, so nothing
here depends on the data grade.

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

## Blockers

| # | Blocker | Owner | Blocks |
|---|---|---|---|
| 1 | **Scheduled Routine cannot push.** Fired sessions get `403 Not authorized` on `receive-pack` while `upload-pack` works. This session pushes fine, so it is a per-session credential difference, not a repo setting. The trigger's stored config names an auto-generated branch (`claude/bold-bardeen`), suggesting the write grant is scoped to that rather than to the charter's fixed work branch. `trig_018ygxy3X9EgNX48wtquNeky` is **disabled** to stop hourly wasted runs. | owner / env config | unattended execution only — the work itself proceeds interactively |
| 2 | **Commit the full `datasets/` tree** (29 files, 6.7 MB) from the machine that holds it — `git add datasets/ && git commit && git push`. DVC is gone and the path is un-ignored, so this is now a plain commit. | owner | fr/de/da entirely; promoting any English result from PROVISIONAL to authoritative |
| 3 | ND-8 production signing keys / KMS | owner | promoting releases past `channel: dev` |
| 4 | B1 unconditional-confirm policy decision | owner | closing the wrong-action budget |
| 5 | Authorisation for the French pack trial | owner | P3 / the neutrality proof |

## Next — Track A is done; everything left needs an owner

**Track A (A1–A10) is complete.** Nothing further in it can be done without one
of the blockers below.

Remaining engineering, all gated:

- **Track B** (honest holdout, OOF calibration, runtime temperature, wrong-action
  re-measure) — needs the full datasets committed to produce anything
  authoritative.
  The machinery can be built against the bootstrap corpus first; the numbers
  cannot be published as a baseline.
- **Wire the engine to `nlu_langpack`.** A6 built the loader and A7 made the
  engine neutral, but the engine still reads loose files — `grep -rn load_pack
  packages/runtime/nlu_engine/` returns nothing. This is the join that makes a
  pack actually load at runtime, and it retires the one interim rule A7 left
  behind: semantic-artifact selection currently infers the encoder+head pair
  from localization-file presence, where the pack manifest already states it
  outright (`models.semantic_head.<lang>.artifact`).
- **P3 French trial** — the acceptance test for all of the above. Owner-gated.
