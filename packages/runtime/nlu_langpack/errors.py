"""Loud, specific Language Pack error taxonomy.

Every failure mode gets its own type so a caller can distinguish "this bundle is
malformed" from "this bundle is fine but too new for me" — the two need very
different operator responses.
"""
from __future__ import annotations

__all__ = [
    "LangPackError", "PackManifestError", "PackCompatibilityError",
    "PackResourceError", "PackLanguageError",
]


class LangPackError(Exception):
    """Base for every Language Pack failure."""


class PackManifestError(LangPackError):
    """bundle.json is missing, unparseable, or structurally invalid."""


class PackCompatibilityError(LangPackError):
    """The bundle requires a runtime contract or feature this runtime lacks."""


class PackResourceError(LangPackError):
    """A declared resource or model artifact is missing or unreadable."""


class PackLanguageError(LangPackError):
    """The requested language is absent, or incomplete on a production channel."""
