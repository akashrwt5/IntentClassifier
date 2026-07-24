---
description: Continuously drive the Review-F5 production roadmap (safe, incremental, gated)
---

Load and follow `docs/Review-F5/PRODUCTION-EXECUTION-PROMPT.md` as a standing charter.

1. Read that prompt in full — it defines the mission, source-of-truth precedence,
   guardrails, phased backlog, the continuous loop, approval gates, and exit gates.
2. Bootstrap per its §11: reconcile what's already done, refresh
   `docs/Review-F5/EXECUTION_STATUS.md`, and queue approval-gated items with crisp
   questions.
3. Run the §6 loop one increment at a time until the §8 exit gates are green or only
   approval-gated / blocked work remains. Respect every §4 STOP rule and §7 approval gate.
4. Keep `.claude/memory/` and `EXECUTION_STATUS.md` in sync each iteration; end each with
   a 2–3 line progress note.

Do not start Phase 2 (Rust core / Android) or anything security-, data-, or safety-critical
without explicit human sign-off.
