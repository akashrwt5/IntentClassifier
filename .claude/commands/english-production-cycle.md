---
description: Drive the English production cycle — evidence chain, English-as-a-pack, signed release (resumable, gated)
---

Load and follow `docs/Review-F5/ENGLISH-PRODUCTION-ROUTINE.md` as a standing charter.

1. Read that routine in full — it defines the role, the fixed work branch, the data
   gate, the STOP rules, the baseline policy, both task tracks with their acceptance
   gates, and the definition of green.
2. Switch to the fixed work branch `claude/nlu-production-readiness-dqyl38`. Never
   create a new auto-named branch — resume depends on reusing this exact name.
3. Run the §Data gate check first to decide which track is open. Track A is
   data-independent and always open; Track B unlocks only when `dvc pull` succeeds.
   Track B being blocked is a normal outcome, not a failure.
4. Work top to bottom through every open step you can complete — do not stop after
   one. Decide "done" by running each step's acceptance gate, not by reading commit
   messages. Commit and push after each step.
5. End every run by refreshing `docs/Review-F5/ENGLISH-PRODUCTION-STATUS.md` and
   writing a 2–3 line progress note.

Rationale for the whole plan is `docs/Review-F5/production-readiness-review-round2.md`.

Do not change confirmation policy or any confidence threshold, do not start French or
any other new language, and do not regenerate a golden fixture to make a failing test
pass. Those need explicit owner sign-off — see the routine's STOP rules.
