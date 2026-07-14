# Privacy stance: `data/unknown_data.csv` (low-confidence utterance log)

Status: **Documented stance — enforcement change pending approval** (Review-F5
Appendix A #10, risk RK6). This file is the source of truth for what the
product is allowed to collect from low-confidence turns.

## The stance

Users of a hearing-aid app speak in a medical context. Raw utterances are
potentially sensitive personal data. Therefore:

1. **Default = aggregate counters only.** Out of the box, the system may
   record only non-identifying aggregates about low-confidence turns: counts
   per language, per confidence bucket, per predicted-intent, per day. No raw
   text, no timestamps precise enough to re-identify, no device identifiers.
2. **Raw text is opt-in only.** Verbatim utterance logging requires an
   explicit, revocable user consent flag. Absent that flag, raw text must
   never be written to `unknown_data.csv`, logs, or telemetry.
3. **Opt-in data is quarantined for review, not training.** Even consented raw
   text goes to a review queue; it enters training data only after human
   review and the holdout-leakage guard (never via automated labeling —
   `scripts/auto_label.py` is quarantined for this reason).
4. **Retention:** consented raw text is deleted after review or after a fixed
   retention window, whichever comes first (window to be set with legal —
   see "Needs decision").

## Current implementation vs. the stance (gap)

`scripts/predict.py::save_unknown()` (legacy CLI path) currently appends
`[text, confidence, timestamp]` — raw text by default. This predates the
stance. It is a **known deviation**, tracked in
`.claude/memory/known-issues.md`. The fix (counters by default, raw text
behind an opt-in flag) changes data-collection behavior and is therefore
**approval-gated** (charter §7: privacy). See the "Needs decision" queue in
`docs/Review-F5/EXECUTION_STATUS.md`.

Mitigations already in place:

- The NLU engine never embeds the raw utterance in `NLUResult` (it would leak
  into caller logs); utterance logging in the engine is opt-in via
  `NLU_LOG_UTTERANCES` and off by default.
- The GenAI escalation path (ADR-004) sends the current utterance only, never
  context/history/device state, and the placeholder endpoint is rejected at
  startup.
- `unknown_data.csv` is not consumed by any automated training path
  (`auto_label.py` refuses to run).

## Related

- Review-F5 roadmap Appendix A #10, risk RK6
- ADR-004 (GenAI routing — egress boundaries)
- `.claude/memory/known-issues.md`
