"""nlu_compiler — the Bundle Compiler (ADR-005).

v0 ships the SHARED VALIDATOR LIBRARY: compiler stages 1–10 exposed as
importable checks (ADR-005 AI#3 — "one validator": Studio, CI, and the
compiler must all import THIS, never reimplement it). Packaging/signing
(stages 11–15) land later; signing is approval-gated (ND-8).

Layout note: this is a NEW package placed per the target layout
(restructure-move-plan.md); it moves nothing and imports nothing from the
legacy tree, so it does not preempt the ND-2 decision.
"""

from .diagnostics import Diagnostic, Severity
from .validator import validate_bundle_dir

__all__ = ["Diagnostic", "Severity", "validate_bundle_dir"]

__version__ = "0.1.0"
