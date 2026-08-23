"""Shared helpers: text normalization, hashing, clustering keys."""

from __future__ import annotations

import hashlib
import re
import unicodedata

# --- normalization ------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s']")
_WS = re.compile(r"\s+")

CONTRACTIONS = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'m": " am",
    "'d": " would",
    "let's": "let us",
    "it's": "it is",
    "that's": "that is",
    "what's": "what is",
    "who's": "who is",
    "there's": "there is",
    "how's": "how is",
    "i'm": "i am",
    "don't": "do not",
}

# Words with zero discriminative value for intent, safe to drop when building
# the *leakage* key (NOT used for model input).
_LEAK_STOP = {
    "the",
    "a",
    "an",
    "my",
    "please",
    "can",
    "you",
    "could",
    "would",
    "hey",
    "ok",
    "okay",
    "so",
    "just",
    "um",
    "uh",
    "well",
    "do",
    "i",
    "to",
    "it",
    "is",
    "are",
    "for",
    "of",
    "me",
    "and",
    "that",
    "this",
}


def normalize(text: str) -> str:
    """Light normalization used for model input and duplicate detection."""
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text)).lower().strip()
    for k, v in CONTRACTIONS.items():
        t = t.replace(k, v)
    t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    return t


def leakage_key(text: str) -> str:
    """Aggressive key: sorted content words. Two sentences sharing this key are
    near-certain paraphrase/duplicate and must never straddle a split."""
    toks = [w for w in normalize(text).split() if w not in _LEAK_STOP]
    # crude stemming: drop trailing 's'/'ing'/'ed' so "volume"/"volumes" collide
    stems = []
    for w in toks:
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > 4 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        stems.append(w)
    return " ".join(sorted(set(stems)))


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def df_to_markdown(df, index: bool = False) -> str:
    """Markdown table without the optional `tabulate` dependency.

    pandas' own .to_markdown() imports tabulate lazily and raises at report-time
    if it is missing — which means a full training run can finish and then fail
    while writing the report. Not worth an extra dependency.
    """
    import pandas as pd  # local import keeps this module import-light

    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]

    def cell(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        if isinstance(v, float):
            return f"{v:.4f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:.4g}"
        return str(v).replace("|", "\\|")

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in row.tolist()) + " |")
    return "\n".join(lines)
