"""
Runtime-contract versioning.

The RUNTIME CONTRACT is the set of interfaces + resource shapes the engine and a
pack agree on (the Protocols in `interfaces.py` plus the manifest/config schema).
It is versioned INDEPENDENTLY of any package version, model version, or app
version — a pack declares the contract range it was built for (`engine_compat` in
`pack.json`), and a runtime refuses a pack whose range does not include the
runtime's own contract version.

Versioning rule (semver-lite, "major.minor"):
- major bump = breaking change to an interface or a required resource shape.
  A runtime MUST refuse a pack whose min contract major exceeds its own.
- minor bump = additive (new optional component/field). Old runtimes ignore
  unknown optional additions; new runtimes still load old packs.

This is the ONLY structural gate between engine and pack (ADR-005 Part 8 —
"runtimes gate on exactly three things"; here that is contract-compat, plus
integrity and, later, signature).
"""

from __future__ import annotations

RUNTIME_CONTRACT_VERSION = "1.0"


def parse_version(v: str) -> tuple[int, int]:
    """Parse a 'major.minor' string into a comparable (major, minor) tuple.

    The literal 'x' as a minor (e.g. '1.x') means "any minor of that major" and
    is represented as a very large minor so range checks treat it as open-ended.
    """
    major_str, _, minor_str = v.strip().partition(".")
    major = int(major_str)
    if minor_str in ("", "x", "*"):
        return (major, 1_000_000)
    return (major, int(minor_str))
