#!/usr/bin/env python3
"""
CI guard: the NLU engine must contain NO language-specific branches.

Enforces IMPLEMENTATION-PLAN §4 — language behaviour lives in a Language Pack,
never in engine code. Fails (exit 1) if any `if language == …` / `language != …`
/ `language in (…)` construct appears in the engine package.

Comments and docstring lines are ignored (only executable code is checked), so a
comment that *mentions* the old pattern does not trip the guard.
"""
from __future__ import annotations

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


def main() -> int:
    offenders: list[str] = []
    for f in sorted(ENGINE_DIR.glob("*.py")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            code = _strip_comment(line)
            if any(p.search(code) for p in _PATTERNS):
                offenders.append(f"{f.relative_to(ROOT)}:{i}: {line.strip()}")

    if offenders:
        print("FAIL: language-specific branch(es) found in the engine:")
        for o in offenders:
            print("  " + o)
        print(
            "\nThe engine must be language-neutral. Move this behaviour into the "
            "Language Pack (see docs/Review-F5/IMPLEMENTATION-PLAN.md §4)."
        )
        return 1

    print(f"OK: {ENGINE_DIR.relative_to(ROOT)} is language-neutral (no language branches).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
