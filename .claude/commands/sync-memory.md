---
description: Sync .claude/memory/ with the current code and recent changes
---

Reconcile the living memory in `.claude/memory/` with the actual repository.

1. Determine what changed recently: `git log --oneline -20` and
   `git diff --name-only HEAD~10..HEAD` (adjust range as needed).
2. For each changed area, use the **Code Graph Memory MCP** to confirm current
   symbols, call flow, and boundaries — do not scan the whole repo.
3. Update ONLY the affected memory file(s), keeping each single-responsibility:
   architecture, datasets, training, inference, mobile, roadmap, decisions,
   known-issues. Record new architectural choices as ADRs in `decisions.md`;
   move fixed bugs out of `known-issues.md`.
4. Keep `CLAUDE.md` small — if project knowledge crept in, move it to memory.
5. Report a short diff summary of what you changed and why. Make no code changes.
