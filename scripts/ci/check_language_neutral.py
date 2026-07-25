#!/usr/bin/env python3
"""
CI guard: the NLU engine must contain NO language-specific branches, and no
hardcoded natural-language match vocabulary.

Enforces IMPLEMENTATION-PLAN §4 — language behaviour lives in a Language Pack,
never in engine code. Adding a language must mean authoring a pack plus training
data, and nothing else.

Two checks:

1. **No language branches.** Fails if any `if language == …` / `language != …`
   / `language in (…)` construct appears in the engine package.

2. **No embedded match vocabulary.** Fails on a module-level collection of
   natural-language phrases that is not named `_DEFAULT_*`.

   The `_DEFAULT_` prefix is the convention for a fallback DATA table that a
   pack overrides (`_DEFAULT_DT_GRAMMAR`, `_DEFAULT_NEGATIONS`). Anything else
   holding user-facing words is vocabulary the engine can never override for a
   new language.

   Canonical role keys are deliberately NOT flagged — `_WD_ORDER` ("Monday"),
   `_ANCHOR_OFFSET` ("day_after_tomorrow"), tier names ("regex_guarded"). Those
   are stable identifiers that a pack maps its own words onto, not text matched
   against user speech. The heuristic that separates them: match vocabulary is
   lowercase and contains a space or an apostrophe ("do not", "no need to",
   "don't"); role keys are single tokens or Capitalized.

   This check exists because `_NEGATIONS` — an English-only tuple in
   classifier.py — silently made negation suppression a no-op for every
   non-English pack, and check 1 could not see it.

Comments and docstring lines are ignored by check 1 (only executable code is
checked), so a comment that *mentions* the old pattern does not trip the guard.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The engine package. (When the engine is relocated to packages/nlu_engine,
# update this path — the check is location-agnostic otherwise.)
ENGINE_DIR = ROOT / "scripts" / "nlu"

_PATTERNS = [
    re.compile(r"language\s*==|language\s*!="),
    re.compile(r"language\s+in\s*[\(\[{]"),
]


def _strip_comment(line: str) -> str:
    # Good enough for a guard: drop everything after the first '#'. (Our engine
    # never puts '#' inside a string on a line that also compares `language`.)
    return line.split("#", 1)[0]


def _is_match_vocabulary(s: str) -> bool:
    """True if `s` looks like a natural-language phrase matched against user
    text, rather than a canonical role key / identifier / path."""
    return s.islower() and (" " in s or "'" in s) and "/" not in s


def _vocabulary_offenders(path: Path) -> list[str]:
    """Module-level string collections holding match vocabulary, not named
    `_DEFAULT_*`."""
    out: list[str] = []
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
            words = [e.value for e in items
                     if isinstance(e, ast.Constant) and isinstance(e.value, str)
                     and _is_match_vocabulary(e.value)]
            if words:
                out.append(f"{path.name}:{node.lineno}: {target.id} = {words[:4]}…")
    return out


def main() -> int:
    offenders: list[str] = []
    for f in sorted(ENGINE_DIR.glob("*.py")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            code = _strip_comment(line)
            if any(p.search(code) for p in _PATTERNS):
                offenders.append(f"{f.relative_to(ROOT)}:{i}: {line.strip()}")

    vocab: list[str] = []
    for f in sorted(ENGINE_DIR.glob("*.py")):
        vocab.extend(_vocabulary_offenders(f))

    if offenders:
        print("FAIL: language-specific branch(es) found in the engine:")
        for o in offenders:
            print("  " + o)
        print(
            "\nThe engine must be language-neutral. Move this behaviour into the "
            "Language Pack (see docs/Review-F5/IMPLEMENTATION-PLAN.md §4)."
        )
        return 1

    if vocab:
        print("FAIL: hardcoded natural-language vocabulary found in the engine:")
        for o in vocab:
            print("  " + o)
        print(
            "\nThis vocabulary can never be overridden for a new language. Move it "
            "into the pack (a table in lexicons.json / the grammar) and keep only a "
            "`_DEFAULT_*` fallback for the no-pack path — see _DEFAULT_NEGATIONS."
        )
        return 1

    print(f"OK: {ENGINE_DIR.relative_to(ROOT)} is language-neutral "
          f"(no language branches, no embedded vocabulary).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
