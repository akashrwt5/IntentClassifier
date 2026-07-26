"""
Language-neutrality guard, run as a test so it fails PRs locally too.

The engine must contain no language-specific branches and no hardcoded
natural-language match vocabulary — language behaviour belongs in a Language
Pack. The engine is not there yet (Review-F5 blocker B10), so the guard carries
a `KNOWN_OFFENDERS` ratchet that can only shrink. See
`scripts/ci/check_language_neutral.py`.

The hostile-pack test that proves a new language needs zero engine edits arrives
with charter step A8, once the pack contract (A6) and the eviction (A7) land.
"""

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_GUARD = _ROOT / "scripts" / "ci" / "check_language_neutral.py"


def _load_guard():
    spec = importlib.util.spec_from_file_location("check_language_neutral", _GUARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_guard_passes():
    """No NEW language coupling, and no stale allowlist entries."""
    assert _load_guard().main() == 0, (
        "language-neutrality guard failed — run "
        "`python scripts/ci/check_language_neutral.py` for the detail"
    )


def test_allowlist_matches_reality_exactly():
    """The ratchet must be exact in both directions.

    An entry that no longer matches means something was fixed without shrinking
    the list; a violation missing from the list means new coupling landed.
    Either way the list has stopped describing the code.
    """
    mod = _load_guard()
    from collections import Counter

    found = mod.collect()
    allowed = Counter(mod.KNOWN_OFFENDERS)
    assert not (found - allowed), f"unallowlisted coupling: {dict(found - allowed)}"
    assert not (allowed - found), f"stale allowlist entries: {dict(allowed - found)}"


def test_allowlist_only_shrinks():
    """A tripwire on the ratchet itself.

    6 is the count recorded when the guard landed (A3): four `if language`
    branches in engine.py, one in entities.py, and the English-only `_NEGATIONS`
    table in classifier.py. A4 and A7 drive it to zero, at which point A9 makes
    the guard blocking in CI. This number must never go up.
    """
    mod = _load_guard()
    assert sum(mod.KNOWN_OFFENDERS.values()) <= 6, (
        "KNOWN_OFFENDERS grew — new language coupling must be moved into the "
        "Language Pack, never allowlisted"
    )
