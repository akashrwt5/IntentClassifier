---
description: Drive the Review-F5 Language Pack plan one gated increment at a time
---

The plan is `docs/Review-F5/IMPLEMENTATION-PLAN.md`; live status is
`docs/Review-F5/IMPLEMENTATION-PROGRESS.md` (newest checkpoint at the top).
Load `.claude/memory/roadmap.md` and `.claude/memory/langpack.md` first.

1. Reconcile before doing anything: confirm what is actually done against the
   repo rather than trusting the status doc, and correct it if it has drifted.
2. **All 6 plan steps (§9) are complete.** The next checkpoint is the big-bang
   move `scripts/nlu` → `packages/nlu_engine` plus the residual lexicon eviction
   — it is the first step that edits existing engine code and **requires
   explicit human go-ahead** (ADR-009). Do not start it unprompted; ask.
3. Work one increment at a time. After each, verify the parity baselines in
   `langpack.md` still hold (classifier 37/37, datetime 77/77 both paths, strip
   20/20, fr/de/da unchanged, hostile `zz` end-to-end) and that the suite is at
   exactly the 8 tracked failures.
4. Keep `.claude/memory/` and `IMPLEMENTATION-PROGRESS.md` in sync each
   iteration; end with a 2-3 line progress note.

If the move happens: `git mv scripts/nlu packages/nlu_engine`, update the ~18
importers, then update `ENGINE_DIR` in `scripts/ci/check_language_neutral.py` and
the paths in `.github/workflows/pr.yml`.

Stop and ask before anything security-, data-, or release-critical.
