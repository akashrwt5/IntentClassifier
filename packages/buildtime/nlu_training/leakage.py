"""
Train/eval leakage detection — one normaliser, shared by every consumer.

WHY THIS MODULE EXISTS
----------------------
The English "holdout" turned out to be 99.9% training data (Review-F5 blocker
B9), which invalidated the English accuracy, ECE and fitted temperature all at
once. The training guard that was supposed to prevent exactly that compared RAW
strings, so any pair differing only by punctuation or spacing sailed through:

    "turn up the volume"   vs   "turn up the volume?"
    "set a reminder"       vs   "Set  a reminder."

Both are the same utterance for a TF-IDF model. Normalised comparison catches
them; raw comparison reports clean.

Leakage has to be checked in more than one place — `train.py` before training,
the calibration fitter before fitting (charter B2), holdout construction after
partitioning (B1). Each having its own near-miss notion of "the same utterance"
is how the discrepancy survived. This module is the single definition.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

__all__ = ["normalize_text", "find_leaks", "leak_report"]

# Apostrophes are ELIDED, not spaced: "what's" -> "whats", "don't" -> "dont",
# so contraction variants collapse together (the training data carries both
# spellings — see _DEFAULT_NEGATIONS, which lists "don't" and "dont").
# Every other punctuation mark becomes a SPACE, so "volume,please" -> "volume
# please" rather than "volumeplease". Doing both with one rule gets one of the
# two cases wrong.
_APOSTROPHE = re.compile(r"['’ʼ]")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize_text(text: object) -> str:
    """Canonical form for utterance comparison.

    Casefold, strip punctuation, collapse whitespace, and NFKC-normalise so
    Unicode variants of the same character compare equal. Deliberately does NOT
    strip accents: 'ou' and 'où' are different French words, and folding them
    would create false leak reports that mask real ones.
    """
    s = unicodedata.normalize("NFKC", str(text)).casefold().strip()
    s = _APOSTROPHE.sub("", s)
    s = _PUNCT.sub(" ", s)
    return _SPACE.sub(" ", s).strip()


def find_leaks(train_texts: Iterable[object],
               eval_texts: Iterable[object]) -> list[str]:
    """Training utterances that also appear in an evaluation set.

    Returns the ORIGINAL training strings (not the normalised forms) so the
    caller can point a human at rows they will recognise.
    """
    eval_norm = {normalize_text(t) for t in eval_texts}
    eval_norm.discard("")
    seen, leaks = set(), []
    for original in train_texts:
        norm = normalize_text(original)
        if norm and norm in eval_norm and norm not in seen:
            seen.add(norm)
            leaks.append(str(original))
    return leaks


def leak_report(leaks: list[str], n_eval: int, *, source: str = "evaluation set",
                limit: int = 5) -> str:
    """Human-readable summary for a raised error or a log line."""
    if not leaks:
        return f"Leakage guard: 0 leaks ({n_eval} {source} utterances checked)."
    sample = "\n  ".join(repr(x) for x in sorted(leaks)[:limit])
    more = f"\n  ...and {len(leaks) - limit} more" if len(leaks) > limit else ""
    return (f"Leakage detected — {len(leaks)} training utterance(s) also appear "
            f"in the {source} (normalised comparison: case, punctuation and "
            f"spacing ignored).\n  {sample}{more}")
