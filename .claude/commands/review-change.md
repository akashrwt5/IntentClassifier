---
description: Review the current diff before merge
---

Invoke the **reviewer** agent on the working diff.

1. Gather the diff: `git diff` (unstaged) and `git diff --staged`.
2. Use the **Code Graph Memory MCP** for impact analysis — callers, affected
   modules, and whether the change crosses the pack boundary or a shape contract.
3. Check against `.claude/memory/known-issues.md` (don't re-flag the 8 tracked
   test failures or the deferred-deprecation fallbacks) and the constraints in
   `architecture.md` / `langpack.md`.
4. Run the language-neutrality guard and the affected tests:
   `python scripts/ci/check_language_neutral.py`
   `pytest` — confirm the failure count is still **exactly the 8 tracked ones**.
5. Report findings as `[blocker] / [should-fix] / [nit]` with file:line and a
   concrete fix.

Treat these as blockers: a new `if language` / English literal in `scripts/nlu/`,
a weakened assertion or new `--deselect` line, loosened
`config/gate_thresholds.json`, or `export_weights.py` no longer deriving from
`models/intent_pipeline.pkl`. Confirm the diff is minimal and doesn't rewrite
working code.
