#!/usr/bin/env python3
"""Shared definitions for the evaluation instruments: normalisation, near-duplication, power.

WHY THIS MODULE EXISTS
----------------------
Three numbers in the plan of record steered decisions and none of them had a
script behind it: the 44% near-duplicate share of ``holdout_honest.csv``, the
823-row size of the ``dev_hard`` set derived from it, and the minimum
detectable effect of that set. When they were re-measured on 2026-08-22 the
first came back 657 rather than 647, and the difference could not be
attributed -- the original method was unrecoverable. A value with no script is
not a measurement, it is a memory.

Everything here is deterministic. There is no seed because there is no
randomness: the same inputs produce the same outputs on any machine, which is
the property that makes ``dev_hard`` usable as a frozen ruler across P2-P8.

THE VENDORED NORMALISER
-----------------------
``normalize_text`` below is a byte-for-byte copy of
``packages/buildtime/nlu_training/leakage.py``. It is duplicated rather than
imported for one reason: this directory imports nothing from the repository so
that it can be lifted into a separate project (see README). A copy is a drift
risk, so ``test_instruments.py`` asserts the two implementations agree on a
corpus of adversarial cases whenever the repository is present, and skips when
it is not. That test is the reason the copy is allowed to exist.

The normaliser matters more than it looks. ``holdout_honest.csv`` was built by
grouping on this exact function, so a different definition of "the same
utterance" here would partition the holdout along a different seam than the one
it was built with, and the resulting sets would not mean what their names say.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

__all__ = [
    "normalize_text",
    "token_set",
    "near_duplicate_flags",
    "minimum_detectable_effect",
    "read_rows",
    "write_rows",
    "sha256_file",
]

# --------------------------------------------------------------------------
# normalisation -- vendored from packages/buildtime/nlu_training/leakage.py
# --------------------------------------------------------------------------
# Apostrophes are ELIDED, not spaced: "what's" -> "whats", so contraction
# variants collapse together. Every other punctuation mark becomes a SPACE, so
# "volume,please" -> "volume please" rather than "volumeplease". One rule for
# both gets one of the two cases wrong.
_APOSTROPHE = re.compile(r"['’ʼ]")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize_text(text: object) -> str:
    """Canonical form for utterance comparison.

    Casefold, strip punctuation, collapse whitespace, NFKC-normalise.
    Deliberately does NOT strip accents: 'ou' and 'ou`' are different French
    words and folding them would create false leak reports that mask real ones.
    """
    s = unicodedata.normalize("NFKC", str(text)).casefold().strip()
    s = _APOSTROPHE.sub("", s)
    s = _PUNCT.sub(" ", s)
    return _SPACE.sub(" ", s).strip()


def token_set(text: object) -> frozenset[str]:
    return frozenset(normalize_text(text).split())


# --------------------------------------------------------------------------
# near-duplication
# --------------------------------------------------------------------------
def _prefix_length(size: int, threshold: float) -> int:
    """Tokens of a set that must be indexed for prefix filtering to stay exact.

    If J(A,B) >= t then |A n B| >= ceil(t*|A|), so B must contain at least one
    of A's |A| - ceil(t*|A|) + 1 rarest tokens. Indexing that prefix under a
    globally consistent token order therefore loses no true pair -- this is an
    exact filter, not an approximation, and the Jaccard is still computed in
    full for every surviving candidate.
    """
    return max(size - math.ceil(threshold * size) + 1, 1)


def near_duplicate_flags(
    eval_texts: Sequence[object],
    train_texts: Iterable[object],
    threshold: float = 0.8,
) -> list[str | None]:
    """For each eval row, a train utterance it near-duplicates, or None.

    Near-duplicate means token-set Jaccard >= threshold: shared words, which is
    precisely what a lexical model scores on. Returning the matched training
    text rather than a bare boolean means a human can look at a pair and judge
    whether the threshold is behaving.
    """
    train_list = [str(t) for t in train_texts]
    train_sets = [token_set(t) for t in train_list]

    df: Counter[str] = Counter()
    for s in train_sets:
        df.update(s)

    def ordered(s: frozenset[str]) -> list[str]:
        # Rarest first, ties broken lexicographically so the order is total and
        # identical on every run and every machine.
        return sorted(s, key=lambda tok: (df.get(tok, 0), tok))

    index: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(train_sets):
        if not s:
            continue
        for tok in ordered(s)[: _prefix_length(len(s), threshold)]:
            index[tok].append(i)

    out: list[str | None] = []
    for text in eval_texts:
        s = token_set(text)
        if not s:
            out.append(None)
            continue
        candidates: set[int] = set()
        for tok in ordered(s)[: _prefix_length(len(s), threshold)]:
            candidates.update(index.get(tok, ()))
        best_i, best_j = None, 0.0
        for i in candidates:
            union = len(s | train_sets[i])
            if not union:
                continue
            j = len(s & train_sets[i]) / union
            if j >= threshold and j > best_j:
                best_i, best_j = i, j
        out.append(train_list[best_i] if best_i is not None else None)
    return out


# --------------------------------------------------------------------------
# statistical power
# --------------------------------------------------------------------------
# z for alpha=0.05 two-sided and for 80% power. Hardcoded rather than pulled
# from scipy so this module has no dependency beyond the standard library.
_Z_ALPHA_2 = 1.959964
_Z_BETA_80 = 0.841621


def minimum_detectable_effect(
    n_rows: int, discordance: float, z_alpha_2: float = _Z_ALPHA_2, z_beta: float = _Z_BETA_80
) -> float:
    """Smallest accuracy difference McNemar's test can reliably detect.

    Two encoders scored on the SAME rows disagree on a fraction ``discordance``
    of them; McNemar's test asks whether those disagreements split evenly. With
    n_d = n * discordance discordant pairs, detecting a split of psi vs 1-psi
    needs |psi - 0.5| ~ (z_a + z_b) / (2 * sqrt(n_d)), and the accuracy
    difference that corresponds to is discordance * (2*psi - 1). Substituting:

        MDE = (z_a + z_b) * sqrt(discordance / n)

    A difference below this is not a small win. It is a number the instrument
    cannot tell from zero, and reporting it as a win is how a plan starts
    believing things that are not true.
    """
    if n_rows <= 0 or discordance <= 0:
        return float("inf")
    return (z_alpha_2 + z_beta) * math.sqrt(discordance / n_rows)


# --------------------------------------------------------------------------
# csv / hashing helpers
# --------------------------------------------------------------------------
def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if "text" not in fields or "intent" not in fields:
        raise SystemExit(f"{path}: expected 'text' and 'intent' columns, found {fields}")
    return fields, rows


def write_rows(path: Path, fields: Sequence[str], rows: Iterable[dict[str, str]]) -> None:
    """Write a CSV with LF endings, matching every other data file in the pack.

    ``lineterminator`` is explicit because csv.writer defaults to CRLF. The
    default produced files that differed from their own source holdout only in
    line endings -- invisible in every diff and every editor, but enough to make
    the derived sets fail the repository's mixed-line-ending hook and, worse, to
    change their sha256 after the hook silently rewrote them, so the manifest
    they had just been pinned against no longer matched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
