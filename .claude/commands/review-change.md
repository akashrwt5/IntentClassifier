---
description: Review the current diff before merge
---

Invoke the **reviewer** agent on the working diff.

1. Gather the diff: `git diff` (unstaged) and `git diff --staged`.
2. Use the **Code Graph Memory MCP** for impact analysis — callers, affected
   modules, and whether the change crosses the ONNX static-batch-1 boundary.
3. Check against `.claude/memory/known-issues.md` (don't re-flag known issues)
   and `architecture.md` constraints.
4. Run `ruff check`, `mypy`, and `pytest` on the affected areas.
5. Report findings as `[blocker] / [should-fix] / [nit]` with file:line and a
   concrete fix. Confirm the diff is minimal and doesn't rewrite working code.
