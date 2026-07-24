"""
Language Pack error taxonomy.

All pack failures are surfaced loudly and specifically — a bad pack must fail at
load time with a diagnostic naming the problem, never silently degrade on a
user's device (the ADR-005 "move failure from runtime to load/compile" principle).
"""

from __future__ import annotations


class PackError(Exception):
    """Base class for every Language Pack failure."""


class PackManifestError(PackError):
    """`pack.json` is missing, malformed, or missing required fields."""


class PackCompatibilityError(PackError):
    """The pack's declared engine_compat range does not include this runtime's
    contract version, or it requires a feature the runtime lacks."""


class PackResourceError(PackError):
    """A resource or model artifact the manifest declares is absent or invalid."""
