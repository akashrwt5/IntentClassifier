"""nlu_training — training-side buildtime package (roadmap §9.2).

v0 ships the UNIFIED EVALUATE: one command, one JSON report, shaped to
validate against spec/bundle/3.0/report_card.schema.json — so the evaluation
artifact that gates CI today is byte-compatible with the report card the
bundle compiler will ingest tomorrow (ADR-005 stage 13: "a bundle cannot be
built from an evaluation that didn't run").

Like nlu_compiler, this is a NEW package in the target layout; it imports
the existing trained artifacts in place and moves nothing (ND-2 pending).
"""

from .evaluate import evaluate_all

__all__ = ["evaluate_all"]

__version__ = "0.1.0"
