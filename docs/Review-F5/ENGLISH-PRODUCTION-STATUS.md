STATUS: IN PROGRESS — Track A, steps A1–A4 complete

# English Production Cycle — Status

- **Run date:** 2026-07-26
- **Branch:** `claude/nlu-production-readiness-dqyl38` @ `59e7c32f`
- **Driven from:** an interactive session, not the scheduled Routine (see
  *Blockers*). Charter: `docs/Review-F5/ENGLISH-PRODUCTION-ROUTINE.md`.

## Data-gate state — **BOOTSTRAP (PROVISIONAL)**

`dvc` is not installed and the remote is still the local path
`../../dvc-store`, so `dvc pull` is unavailable.
`python scripts/ci/bootstrap_en_data.py` materialised the tracked English
snapshot (5 files, source commit `0089b894`).

**Every metric below is therefore PROVISIONAL.** No authoritative baseline may
be recorded, and no bootstrap-fitted calibration may be wired to the runtime,
until a shared DVC remote exists. fr/de/da model work stays closed.

No model metrics were produced this run — A1–A4 are all structural, so nothing
here depends on the data grade.

## Completed

| Step | State | Acceptance gate |
|---|---|---|
| **A1** English datetime golden corpus | ✅ | 130 cases / 29 branches; `pytest tests/test_datetime_parity_en.py` 133 passed; `entities.py` untouched |
| **A2** Correct the calibration-chain docs | ✅ | No memory file claims the runtime reads `config/calibration.json`; banner present; `test_smoke` 3 passed |
| **A3** Language-neutrality guard (ratchet) | ✅ | Guard exits 0 with the allowlist; mutation-verified both directions; `test_neutrality` passed |
| **A4** Language-aware negation suppression | ✅ | 21 negation tests; allowlist 6 → 5; full suite 208 passed / 30 skipped; ruff clean |

Suite at end of run: **208 passed, 30 skipped, 0 failed.** The 30 skips are all
model-dependent (`trained artifacts not present`) — expected in the bootstrap
grade, and the subject of charter step B0.

Neutrality ratchet: **5 remaining**, all in `engine.py`/`entities.py`, all
scheduled for removal by A7.

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

4. **The A3 guard flagged prose.** The ported implementation split on `#`,
   leaving docstrings live, so a comment *describing* the forbidden pattern
   tripped it. Now tokenised — comments and string literals are blanked.

## Blockers

| # | Blocker | Owner | Blocks |
|---|---|---|---|
| 1 | **Scheduled Routine cannot push.** Fired sessions get `403 Not authorized` on `receive-pack` while `upload-pack` works. This session pushes fine, so it is a per-session credential difference, not a repo setting. The trigger's stored config names an auto-generated branch (`claude/bold-bardeen`), suggesting the write grant is scoped to that rather than to the charter's fixed work branch. `trig_018ygxy3X9EgNX48wtquNeky` is **disabled** to stop hourly wasted runs. | owner / env config | unattended execution only — the work itself proceeds interactively |
| 2 | Shared DVC remote (S3/GCS) + repo secret | owner | fr/de/da entirely; promoting any English result from PROVISIONAL to authoritative |
| 3 | ND-8 production signing keys / KMS | owner | promoting releases past `channel: dev` |
| 4 | B1 unconditional-confirm policy decision | owner | closing the wrong-action budget |
| 5 | Authorisation for the French pack trial | owner | P3 / the neutrality proof |

## Next

**A5** — normalised leakage matching in `train.py` + lock the compiler/signing
deps (`referencing`, `cryptography`, `jsonschema>=4.18`). Then **A6** (pack
contract), **A7** (the datetime eviction, ported from the reference branch per
the charter), **A8** (hostile-pack test), **A9** (guard blocking), **A10**
(`release-pack.yml`).
