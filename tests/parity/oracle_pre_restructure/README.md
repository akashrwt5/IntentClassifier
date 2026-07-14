# Frozen oracle — pre-restructure baseline (ND-2 M0, 2026-07-14)

Captured immediately before the packages/ restructure (move plan M0).
Every restructure phase must reproduce these outputs EXACTLY:

- `evaluate_report.json` — unified evaluate (en/fr/de/da) on the shipped
  artifacts: the byte-comparison target for `python -m nlu_training`.
- Test suite state at capture: 60 passed, 0 failed (full pytest).
- Captured at commit: see the M0 commit that adds this directory.

Any non-empty diff after a move phase aborts that phase (revert the commit).
