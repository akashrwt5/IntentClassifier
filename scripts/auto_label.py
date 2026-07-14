#!/usr/bin/env python3
"""QUARANTINED — DO NOT RUN. (Appendix A #1, risk RK5)

This script auto-labeled ``data/unknown_data.csv`` rows using keyword rules
that emit the RETIRED flat uppercase label taxonomy (VOLUME, REMINDER, ...).
The shipping multilingual pipeline uses a different label space; appending
these rows to ``data/01_source_base_training_data.csv`` would silently poison
training data and corrupt the next retrain.

It is kept only as a historical reference. It refuses to run unconditionally.

If a keyword-bootstrap labeler is ever needed again it must be rewritten to:
  1. target the CURRENT intent taxonomy (see .claude/memory/datasets.md),
  2. write to a review queue, never directly into training data,
  3. pass the holdout-leakage guard before any retrain.

See docs/Review-F5/production-architecture-review-and-roadmap.md, Appendix A #1.
"""

import sys

BANNER = """
!! QUARANTINED SCRIPT — REFUSING TO RUN !!

scripts/auto_label.py writes a retired label taxonomy and would poison
training data (Review-F5 Appendix A #1, risk RK5). It has been disabled.

Nothing was read or written. See the module docstring for the rewrite
requirements if this functionality is ever needed again.
"""


def main() -> None:
    sys.exit(BANNER)


# --- Historical reference only (dead code, guarded by the exit above) -------
# KEYWORD_RULES mapped substrings to retired labels, e.g.:
#   "volume" -> "VOLUME", "remind" -> "REMINDER", "weather" -> "WEATHER_FORECAST"
# and appended matches from data/unknown_data.csv to
# data/01_source_base_training_data.csv without review.
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
