"""Runtime contract version + the compatibility gate.

There is exactly ONE version anchor in this system:
`spec/contracts/runtime-contract-v1.md` §7, which bundles gate against via
`bundle.json:engine_compat` (ADR-005 Part 3). This module restates that number
for the loader; it does not introduce a second axis.
"""
from __future__ import annotations

from .errors import PackCompatibilityError

__all__ = ["RUNTIME_CONTRACT_VERSION", "SUPPORTED_FEATURES", "check_compatibility"]

# Integer, matching bundle.schema.json's engine_compat (min/max are integers).
RUNTIME_CONTRACT_VERSION = 1

# Optional runtime features this contract version implements. A bundle listing
# anything outside this set is refused by declaration rather than by crashing
# somewhere downstream.
SUPPORTED_FEATURES: frozenset[str] = frozenset()


def check_compatibility(min_contract: int, max_tested: int,
                        required_features: frozenset[str]) -> list[str]:
    """Gate this runtime against a bundle's declared compatibility.

    Returns a list of non-fatal ISSUES; raises on anything fatal.

    The asymmetry is deliberate and follows the spec's own wording. The bundle
    declares `max_TESTED_runtime_contract` — the newest runtime it was verified
    against, not a hard ceiling. Running a newer runtime is therefore untested
    rather than forbidden, and is reported as an issue so an operator can see
    it. `min_runtime_contract` IS a hard floor: below it the bundle genuinely
    depends on seams this runtime does not have.
    """
    issues: list[str] = []

    if RUNTIME_CONTRACT_VERSION < min_contract:
        raise PackCompatibilityError(
            f"bundle needs runtime contract >= {min_contract}; this runtime is "
            f"{RUNTIME_CONTRACT_VERSION}. Update the engine, or load an older bundle."
        )
    if RUNTIME_CONTRACT_VERSION > max_tested:
        issues.append(
            f"bundle was tested only up to runtime contract {max_tested}; this "
            f"runtime is {RUNTIME_CONTRACT_VERSION} (allowed, but unverified)"
        )

    unsupported = required_features - SUPPORTED_FEATURES
    if unsupported:
        raise PackCompatibilityError(
            f"bundle requires runtime feature(s) this runtime does not implement: "
            f"{sorted(unsupported)}"
        )
    return issues
