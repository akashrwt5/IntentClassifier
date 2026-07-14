"""Portable regex subset conformance — compiler stage 10.

Normative definition: spec/bundle/portable-regex.md. This module is the
machine check: a pattern passes iff it (a) compiles under Python `re` and
(b) contains no construct outside the subset.
"""

from __future__ import annotations

import re

MAX_PATTERN_LENGTH = 512
MAX_QUANTIFIER_BOUND = 32
MAX_GROUP_DEPTH = 4

# Constructs explicitly outside the subset (see portable-regex.md "Forbidden").
_FORBIDDEN = [
    (re.compile(r"\\[1-9]"), "backreference"),
    (re.compile(r"\(\?="), "lookahead"),
    (re.compile(r"\(\?!"), "negative lookahead"),
    (re.compile(r"\(\?<[=!]"), "lookbehind"),
    (re.compile(r"\(\?P"), "named group"),
    (re.compile(r"\(\?<[a-zA-Z]"), "named group"),
    (re.compile(r"\(\?[aiLmsux]+[):]"), "inline flags"),
    (re.compile(r"\(\?#"), "comment group"),
    (re.compile(r"\(\?>"), "atomic group"),
    (re.compile(r"[*+?]\?"), "lazy quantifier"),
    (re.compile(r"[*+?]\+"), "possessive quantifier"),
    (re.compile(r"\{\d+(,\d*)?\}[?+]"), "lazy/possessive bounded quantifier"),
    (re.compile(r"\\p\{"), "unicode property escape"),
    (re.compile(r"\\[0-7]{2,3}"), "octal escape"),
]

_QUANT_BOUND = re.compile(r"\{(\d+)(?:,(\d*))?\}")


def check_pattern(pattern: str) -> list[str]:
    """Return a list of problems (empty = conformant)."""
    problems: list[str] = []
    if len(pattern) > MAX_PATTERN_LENGTH:
        problems.append(f"pattern longer than {MAX_PATTERN_LENGTH} chars")
    for rx, name in _FORBIDDEN:
        if rx.search(pattern):
            problems.append(f"forbidden construct: {name}")
    for m in _QUANT_BOUND.finditer(pattern):
        for g in m.groups():
            if g and int(g) > MAX_QUANTIFIER_BOUND:
                problems.append(f"quantifier bound {g} exceeds {MAX_QUANTIFIER_BOUND}")
    depth = 0
    max_depth = 0
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "\\":
            i += 2
            continue
        if c == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif c == ")":
            depth = max(0, depth - 1)
        i += 1
    if max_depth > MAX_GROUP_DEPTH:
        problems.append(f"group nesting {max_depth} exceeds {MAX_GROUP_DEPTH}")
    try:
        re.compile(pattern)
    except re.error as exc:
        problems.append(f"does not compile: {exc}")
    return problems
