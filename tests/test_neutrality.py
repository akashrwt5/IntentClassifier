"""
Language-neutrality guard, run as a test so it fails PRs locally too.

The engine contains no language-specific branches and no hardcoded
natural-language match vocabulary — language behaviour lives in data tables a
Language Pack overrides. The `KNOWN_OFFENDERS` ratchet reached zero at A7
(Review-F5 blocker B10 closed). See `scripts/ci/check_language_neutral.py`.

The hostile-pack test that PROVES a new language needs zero engine edits — as
opposed to asserting it — arrives with charter step A8.
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


def test_allowlist_is_empty_and_stays_empty():
    """The ratchet is closed.

    It landed at A3 with 6 (four `if language` branches in engine.py, one in
    entities.py, the English-only `_NEGATIONS` table in classifier.py), dropped
    to 5 at A4, and reached 0 at A7. Any non-zero value now means language
    coupling was re-introduced and allowlisted instead of moved into the pack.
    """
    mod = _load_guard()
    assert mod.KNOWN_OFFENDERS == {}, (
        f"language coupling was re-allowlisted: {dict(mod.KNOWN_OFFENDERS)}. "
        f"It belongs in the Language Pack as data."
    )
