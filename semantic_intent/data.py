"""
Dataset loading, normalisation and *leakage-safe* splitting.

Why grouped splits matter here
------------------------------
The corpus is template-inflated: "turn it down", "please turn it down" and
"can you turn it down" are three rows describing one phrasing. A random split
puts siblings on both sides and the reported accuracy is then measuring
memorisation, not generalisation. On balanced_intents_final.xlsx that is
20,724 rows collapsing to 14,584 distinct cores — roughly 30% sibling rows.

`grouped_split` therefore assigns whole *core groups* to a split, so no
politeness variant of a training phrase can appear in dev or test.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Leading politeness / framing wrappers, applied repeatedly ("please can you ...").
_PREFIX = re.compile(
    r"^(?:(?:please|pls|hey|hi|ok|okay|can you|could you|would you|will you|"
    r"i want to|i need to|i would like to|i'd like to|i wanna)\s+)+"
)
# Trailing politeness.
_SUFFIX = re.compile(r"(?:\s+(?:please|pls|for me|thanks|thank you))+[.!?]*$")
_WS = re.compile(r"\s+")

DEFAULT_TEXT_COL = "text"
DEFAULT_LABEL_COL = "intent"


def normalize(text: str) -> str:
    """Lowercase, unify apostrophes, drop punctuation, squeeze whitespace.

    Kept deliberately simple: the encoder's WordPiece tokeniser handles the
    rest, and aggressive normalisation here would only hide real variation.
    """
    t = str(text).lower().strip().replace("’", "'")
    t = re.sub(r"[^a-z0-9'\s]", " ", t)
    return _WS.sub(" ", t).strip()


def core_of(text: str) -> str:
    """Strip politeness affixes down to the semantic core of an utterance."""
    t = normalize(text)
    prev = None
    while prev != t:
        prev = t
        t = _SUFFIX.sub("", _PREFIX.sub("", t)).strip()
    return t or normalize(text)


def load(
    path: str | Path, text_col: str = DEFAULT_TEXT_COL, label_col: str = DEFAULT_LABEL_COL
) -> pd.DataFrame:
    """Load a CSV/TSV, or a multi-sheet XLSX where each sheet is one intent.

    Returns a frame with columns: text, intent, core.
    """
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        xl = pd.ExcelFile(path)
        frames = []
        for sheet in xl.sheet_names:
            d = xl.parse(sheet)
            d.columns = [str(c).strip().lower() for c in d.columns]
            if label_col not in d.columns:
                d[label_col] = sheet
            if text_col not in d.columns:
                raise ValueError(f"sheet {sheet!r} has no {text_col!r} column")
            frames.append(d[[text_col, label_col]])
        df = pd.concat(frames, ignore_index=True)
    else:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        df.columns = [str(c).strip().lower() for c in df.columns]

    df = df.rename(columns={text_col: "text", label_col: "intent"})
    df["text"] = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df[(df.text.str.len() > 0) & (df.intent.str.len() > 0)]
    df = df.drop_duplicates(subset=["text", "intent"]).reset_index(drop=True)
    df["core"] = df.text.map(core_of)
    return df


def audit(df: pd.DataFrame) -> dict:
    """Cheap dataset health report. Returns the numbers, prints a summary."""
    dupes = df.text.duplicated().sum()
    ambiguous = df.groupby("text").intent.nunique()
    inflation = 1 - df.core.nunique() / len(df)
    counts = df.intent.value_counts()

    print(
        f"  rows={len(df)}  intents={df.intent.nunique()}  " f"distinct cores={df.core.nunique()}"
    )
    print(f"  template inflation: {inflation:.1%} of rows are politeness variants")
    print(f"  duplicate texts: {dupes}   texts with >1 intent: {(ambiguous > 1).sum()}")
    print(
        f"  class balance: min={counts.min()} max={counts.max()} "
        f"ratio={counts.max() / counts.min():.2f}x"
    )

    # Cross-intent core collisions: same core phrase labelled two ways.
    collide = df.groupby("core").intent.nunique()
    n_collide = int((collide > 1).sum())
    if n_collide:
        print(f"  !! {n_collide} core phrases carry conflicting labels — inspect these")
    return {
        "rows": len(df),
        "cores": int(df.core.nunique()),
        "inflation": float(inflation),
        "ambiguous": int((ambiguous > 1).sum()),
        "core_collisions": n_collide,
    }


def grouped_split(
    df: pd.DataFrame, dev: float = 0.15, test: float = 0.15, seed: int = 0
) -> pd.DataFrame:
    """Assign whole core-groups to train/dev/test, stratified per intent."""
    if not 0 <= dev + test < 1:
        raise ValueError("dev + test must be in [0, 1)")
    rng = np.random.default_rng(seed)
    part = np.full(len(df), "train", dtype=object)
    idx = np.arange(len(df))

    for intent in df.intent.unique():
        sel = df.intent.values == intent
        cores = pd.unique(df.core.values[sel])
        rng.shuffle(cores)
        n_dev, n_test = int(len(cores) * dev), int(len(cores) * test)
        dev_set = set(cores[:n_dev])
        test_set = set(cores[n_dev : n_dev + n_test])
        in_core = df.core.values
        part[idx[sel & np.isin(in_core, list(dev_set))]] = "dev"
        part[idx[sel & np.isin(in_core, list(test_set))]] = "test"

    out = df.copy()
    out["split"] = part
    _assert_no_leak(out)
    return out


def _assert_no_leak(df: pd.DataFrame) -> None:
    groups = {s: set(df[df.split == s].core) for s in df.split.unique()}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if a in groups and b in groups:
            overlap = groups[a] & groups[b]
            if overlap:
                raise AssertionError(f"{len(overlap)} core groups leak between {a} and {b}")


def polarity_report(df: pd.DataFrame, probes: Iterable[str] = ()) -> None:
    """Show, per probe word, how the label mass is distributed.

    A word whose mass sits mostly on an intent it does not semantically belong
    to is exactly the trap that makes bag-of-words models answer 'decrease' to
    "it's too quiet, make it louder".
    """
    probes = list(probes) or [
        "loud",
        "louder",
        "quiet",
        "quieter",
        "soft",
        "softer",
        "up",
        "down",
        "mute",
        "silence",
    ]
    lower = df.text.str.lower()
    for word in probes:
        hit = df[lower.str.contains(rf"\b{re.escape(word)}\b", regex=True)]
        if hit.empty:
            continue
        counts = hit.intent.value_counts()
        purity = counts.iloc[0] / counts.sum()
        summary = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in counts.head(3).items())
        flag = "  <-- ambiguous" if purity < 0.85 else ""
        print(f"  {word:9s} {summary}{flag}")
