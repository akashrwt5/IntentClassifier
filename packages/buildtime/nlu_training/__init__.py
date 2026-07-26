"""nlu_training — training-side buildtime package (roadmap §9.2).

v0 ships the UNIFIED EVALUATE: one command, one JSON report, shaped to
validate against spec/bundle/3.0/report_card.schema.json — so the evaluation
artifact that gates CI today is byte-compatible with the report card the
bundle compiler will ingest tomorrow (ADR-005 stage 13: "a bundle cannot be
built from an evaluation that didn't run").

Like nlu_compiler, this is a NEW package in the target layout; it imports
the existing trained artifacts in place and moves nothing (ND-2 pending).
"""

# --- runtime-package bootstrap ------------------------------------------------
# Build-time code legitimately imports from the RUNTIME package: the trainer and
# every threshold fitter must featurise text with the exact function inference
# applies (`nlu_engine.text_norm.normalize_text`), or the fitted values describe
# a featurizer the shipped model does not use — which is blocker B8.
#
# That import has to work without the caller setting PYTHONPATH. It previously
# relied on each module inserting the path itself; train.py did, the fitters did
# not, and CI (which sets only `PYTHONPATH=packages/buildtime`) died with
# `ModuleNotFoundError: No module named 'nlu_engine'` AFTER training had already
# succeeded. Doing it once here means the next fitter cannot forget it.
import sys as _sys
from pathlib import Path as _Path

_RUNTIME = _Path(__file__).resolve().parents[2] / "runtime"
if _RUNTIME.is_dir() and str(_RUNTIME) not in _sys.path:
    _sys.path.insert(0, str(_RUNTIME))

from .evaluate import evaluate_all  # noqa: E402  (must follow the bootstrap)

__all__ = ["evaluate_all"]

__version__ = "0.1.0"
