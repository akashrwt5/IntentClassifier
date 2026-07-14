# Privacy stance: `data/unknown_data.csv` (low-confidence utterance log)

Status: **Enforced** (approved ND-5, 2026-07-14; Review-F5 Appendix A #10,
risk RK6). This file is the source of truth for what the product is allowed
to collect from low-confidence turns.

Implementation: `scripts/unknown_log.py` — counters by default to
`data/unknown_counters.csv` (date, confidence bucket, count); raw text to
`data/unknown_data.csv` only when `NLU_COLLECT_RAW_UNKNOWN` is set (the app
layer must map this to real, revocable user consent). `scripts/predict.py`
routes through it. Tests: `tests/test_phase0_guards.py`.

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

## Enforcement history

`scripts/predict.py::save_unknown()` (legacy CLI path) previously appended
`[text, confidence, timestamp]` — raw text by default. Fixed 2026-07-14
under ND-5 approval: it now delegates to `scripts/unknown_log.py`
(counters by default, raw text opt-in). Open follow-up: the retention
window for consented raw text still needs a legal decision.

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
