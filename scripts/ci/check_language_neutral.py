#!/usr/bin/env python3
"""
CI guard: the NLU engine must contain NO language-specific branches, and no
hardcoded natural-language match vocabulary.

Language behaviour belongs in a Language Pack, never in engine code. Adding a
language must mean authoring a pack plus training data and nothing else. Ported
from the reference branch and retargeted at `packages/runtime/nlu_engine`.

TWO CHECKS
----------
1. **No language branches.** Any `if language == …` / `language != …` /
   `language in (…)` construct in the engine.

2. **No embedded match vocabulary.** A module-level collection of
   natural-language phrases that is not named `_DEFAULT_*`.

   `_DEFAULT_` is the convention for a fallback DATA table a pack overrides
   (`_DEFAULT_DT_GRAMMAR`, `_DEFAULT_NEGATIONS`). Anything else holding
   user-facing words is vocabulary the engine can never override for a new
   language.

   Canonical role keys are deliberately NOT flagged — `_WD_ORDER` ("Monday"),
   `_ANCHOR_OFFSET` ("day_after_tomorrow"), tier names ("regex_guarded"). Those
   are stable identifiers a pack maps its own words onto, not text matched
   against user speech. The heuristic separating them: match vocabulary is
   lowercase and contains a space or apostrophe ("do not", "don't"); role keys
   are single tokens or Capitalized.

   Check 2 exists because `_NEGATIONS` — an English-only tuple in classifier.py
   — silently made negation suppression a no-op for every non-English language,
   and check 1 could not see it.

THE ALLOWLIST IS A RATCHET
--------------------------
The engine is not neutral yet (Review-F5 blocker B10), so landing this guard
"clean" would mean landing it as a no-op. Instead `KNOWN_OFFENDERS` records
exactly today's violations, and the guard fails on:

  - any violation NOT on the list  -> new coupling, rejected immediately;
  - any list entry that no longer matches -> you fixed something, so shrink the
    list in the same change.

Both directions are enforced so the list can only ever move toward empty and
cannot silently drift. Charter step A9 flips this guard blocking in CI once
`KNOWN_OFFENDERS` is empty.

USAGE
    python scripts/ci/check_language_neutral.py
"""
from __future__ import annotations

import ast
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = ROOT / "packages" / "runtime" / "nlu_engine"

# Today's violations, keyed by (file, exact source line) -> count.
# Keyed on the CODE TEXT rather than a line number so ordinary edits elsewhere
# in the file do not invalidate the list. Two identical branches in the same
# file are distinguished by the count.
#
# Every entry here is scheduled for removal by charter steps A4 (_NEGATIONS) and
# A7 (the language branches). Shrink this list; never grow it.
KNOWN_OFFENDERS: dict[tuple[str, str], int] = {
    ("engine.py", 'elif self.language in ("en", "", "multilingual"):'): 1,
    ("engine.py", 'if language in ("en", ""):'): 1,
    ("engine.py", 'if language in ("en", "", "multilingual"):'): 2,
    ("entities.py", 'if language and language != "en":'): 1,
    ("classifier.py", "_NEGATIONS"): 1,
}

_PATTERNS = [
    re.compile(r"language\s*==|language\s*!="),
    re.compile(r"language\s+in\s*[\(\[{]"),
]


def _strip_comment(line: str) -> str:
    """Drop everything after the first '#'. Good enough for a guard: the engine
    never puts '#' inside a string on a line that also compares `language`."""
    return line.split("#", 1)[0]


def _is_match_vocabulary(s: str) -> bool:
    """True if `s` looks like a natural-language phrase matched against user
    text, rather than a canonical role key / identifier / path."""
    return s.islower() and (" " in s or "'" in s) and "/" not in s


def _branch_offenders(path: Path) -> list[tuple[str, str]]:
    """(file, source line) for each language-branch violation."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if any(p.search(_strip_comment(line)) for p in _PATTERNS):
            out.append((path.name, line.strip()))
    return out


def _vocabulary_offenders(path: Path) -> list[tuple[str, str]]:
    """(file, variable name) for each module-level match-vocabulary table that
    is not a `_DEFAULT_*` fallback."""
    out = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or target.id.startswith("_DEFAULT"):
                continue
            value = node.value
            if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
                items = value.elts
            elif isinstance(value, ast.Dict):
                items = value.keys
            else:
                continue
            if any(isinstance(e, ast.Constant) and isinstance(e.value, str)
                   and _is_match_vocabulary(e.value) for e in items):
                out.append((path.name, target.id))
    return out


def collect() -> Counter:
    found: Counter = Counter()
    for f in sorted(ENGINE_DIR.glob("*.py")):
        found.update(_branch_offenders(f))
        found.update(_vocabulary_offenders(f))
    return found


def main() -> int:
    if not ENGINE_DIR.is_dir():
        print(f"FAIL: engine directory not found: {ENGINE_DIR}")
        return 1

    found = collect()
    allowed = Counter(KNOWN_OFFENDERS)

    new = found - allowed        # present, over the allowed count
    fixed = allowed - found      # allowlisted, no longer present

    if new:
        print("FAIL: language coupling that is NOT on the allowlist:")
        for (fname, text), n in sorted(new.items()):
            print(f"  {fname}: {text}" + (f"  (x{n})" if n > 1 else ""))
        print("\nThe engine must be language-neutral: move this behaviour into "
              "the Language Pack as data. Do NOT add it to KNOWN_OFFENDERS — "
              "that list only shrinks (see the module docstring).")
        return 1

    if fixed:
        print("FAIL: KNOWN_OFFENDERS lists violations that no longer exist:")
        for (fname, text), n in sorted(fixed.items()):
            print(f"  {fname}: {text}" + (f"  (x{n})" if n > 1 else ""))
        print("\nGood news — you fixed these. Remove them from KNOWN_OFFENDERS "
              "in this same change so the ratchet stays accurate.")
        return 1

    remaining = sum(found.values())
    if remaining:
        print(f"OK: {ENGINE_DIR.relative_to(ROOT)} has {remaining} KNOWN language "
              f"coupling(s), all allowlisted, none new:")
        for (fname, text), n in sorted(found.items()):
            print(f"  {fname}: {text}" + (f"  (x{n})" if n > 1 else ""))
        print("\nThese are scheduled for removal by charter steps A4 and A7. "
              "The guard becomes blocking at A9, when this list is empty.")
    else:
        print(f"OK: {ENGINE_DIR.relative_to(ROOT)} is language-neutral "
              f"(no language branches, no embedded vocabulary). "
              f"KNOWN_OFFENDERS is empty — the ratchet is closed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
