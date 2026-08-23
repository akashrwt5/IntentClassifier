#!/usr/bin/env python3
"""Measure every English evaluation set and emit INSTRUMENTS.md. No hand-typed numbers.

WHY
---
The plan of record steered by three instrument numbers that had no script behind
them; one of them (the 44% near-duplicate share) could not be reproduced when it
was re-measured. This script exists so that no charter in this project ever
again carries a number a human typed from memory.

For each set it reports what it is, what it may be used for, and the four facts
that decide whether a result from it means anything:

  rows            how much evidence there is
  intents         how much of the taxonomy it can speak about
  exact leak      share of rows whose normalised text is already in train.csv
  near-duplicate  share within token-set Jaccard 0.8 of a training utterance
  MDE             smallest accuracy difference McNemar can resolve on it

An instrument with a high leak share is not useless -- it detects regressions
fine -- but it cannot answer "did this model generalise better", and the charter
says so per set rather than leaving each reader to work it out.

    python3 inventory_instruments.py            # print
    python3 inventory_instruments.py --write     # regenerate INSTRUMENTS.md
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from instruments import (  # noqa: E402
    minimum_detectable_effect,
    near_duplicate_flags,
    normalize_text,
    sha256_file,
)

HERE = Path(__file__).resolve().parent

# --- Outside this directory ---------------------------------------------
REPO = HERE.parents[1]
PACK = REPO / "language_packs" / "en"
TRAIN = PACK / "train.csv"
# ------------------------------------------------------------------------

DISCORDANCE = 0.15

# path relative to the pack, and the charter -- what it answers, what it must not
# be asked. Charters are editorial and belong here; every number beside them is
# measured.
INSTRUMENTS = [
    (
        "dev_hard.csv",
        "PRIMARY DECISION INSTRUMENT (P2-P8). Holdout rows with no near-duplicate in "
        "train.csv. Decide with McNemar on discordant items, never by comparing two "
        "accuracy numbers. Frozen for the duration of the plan.",
    ),
    (
        "dev_near.csv",
        "REGRESSION DETECTION ONLY. A high score is evidence of memorisation, not of "
        "generalisation. Never gate on it; never report it without dev_hard beside it.",
    ),
    (
        "holdout_honest.csv",
        "The parent of the two above, kept for continuity with earlier numbers. Report "
        "it only to compare against history; decide on dev_hard.",
    ),
    (
        "holdout_leakage_guard.csv",
        "Broad, clean, independent of the train/holdout split. The best available "
        "second opinion on dev_hard, and the natural cross-check when a dev_hard "
        "result is close to its MDE.",
    ),
    (
        "extras/semantic_holdout_2.csv",
        "Broad and clean. Same role as the leakage guard. Carries two malformed labels "
        "(see the defect note below) which must be excluded before scoring.",
    ),
    (
        "extras/holdout_paraphrase.csv",
        "Deep paraphrases on a narrow slice of the taxonomy. Strong evidence about "
        "those intents, silent about the rest -- never read as an overall figure.",
    ),
    (
        "extras/benchmark_250.csv",
        "Historical benchmark. Heavily overlapped with training data and narrow in "
        "taxonomy: a memorisation measure. Do not use it to compare encoders.",
    ),
    ("extras/oos.csv", "Out-of-scope inputs: measures false-accept rate, not accuracy."),
    ("extras/oos_2.csv", "Larger out-of-scope set. Same charter as oos.csv."),
]

TEXT_COLUMNS = ("text", "utterance")
LABEL_COLUMNS = ("intent", "expected_intent", "label")


def _column(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def _read(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def measure(train_texts, train_norm, path: Path) -> dict | None:
    if not path.exists():
        return None
    cols, rows = _read(path)
    tcol = _column(cols, TEXT_COLUMNS)
    lcol = _column(cols, LABEL_COLUMNS)
    if tcol is None or not rows:
        return None
    texts = [r[tcol] for r in rows]
    labels = [str(r[lcol]).strip() for r in rows] if lcol else []
    exact = sum(1 for t in texts if normalize_text(t) in train_norm)
    near = sum(1 for m in near_duplicate_flags(texts, train_texts, 0.8) if m is not None)
    return {
        "rows": len(rows),
        "intents": len(set(labels)),
        "exact": exact / len(rows),
        "near": near / len(rows),
        "mde": minimum_detectable_effect(len(rows), DISCORDANCE),
        "sha256": sha256_file(path)[:12],
        "labels": set(labels),
    }


def build() -> str:
    _, train_rows = _read(TRAIN)
    train_texts = [r["text"] for r in train_rows]
    train_norm = {normalize_text(t) for t in train_texts}
    train_labels = {r["intent"].strip() for r in train_rows}

    out = [
        "# English evaluation instruments",
        "",
        "Generated by `inventory_instruments.py --write`. Do not edit by hand: every",
        "number below is measured, and a hand-edit is how a charter starts describing",
        "a file that no longer exists in that shape.",
        "",
        f"Measured {datetime.now(timezone.utc).strftime('%Y-%m-%d')} against "
        f"`language_packs/en/train.csv` ({len(train_rows):,} rows, "
        f"{len(train_labels)} intents).",
        "",
        "**Exact leak** is the share of rows whose normalised text already appears in",
        "training data. **Near-dup** adds rows within token-set Jaccard 0.8 of a training",
        "utterance. **MDE** is the smallest accuracy difference McNemar's test can resolve",
        f"at {DISCORDANCE:.0%} discordance, 80% power -- a result smaller than this is not a",
        "small win, it is a number the instrument cannot tell from zero.",
        "",
        "| Instrument | Rows | Intents | Exact leak | Near-dup | MDE |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    measured = {}
    for rel, _charter in INSTRUMENTS:
        m = measure(train_texts, train_norm, PACK / rel)
        measured[rel] = m
        if m is None:
            out.append(f"| `{rel}` | — | — | — | — | *missing* |")
            continue
        out.append(
            f"| `{rel}` | {m['rows']:,} | {m['intents']} | {m['exact']:.1%} | "
            f"{m['near']:.1%} | {m['mde']:.3f} |"
        )

    out += ["", "## Charters", ""]
    for rel, charter in INSTRUMENTS:
        m = measured[rel]
        head = f"**`{rel}`**" + (f" — {m['rows']:,} rows, {m['intents']} intents" if m else "")
        out += [head, "", charter, ""]

    # --- observations that are measured, not editorial ---
    out += ["## What the measurements say", ""]
    notes = []
    for rel, _ in INSTRUMENTS:
        m = measured[rel]
        if not m:
            continue
        if m["exact"] >= 0.5:
            notes.append(
                f"- `{rel}` is **{m['exact']:.0%} exact-leaked** into `train.csv`. Accuracy on it "
                "is close to a memorisation score; it cannot rank two encoders."
            )
        stray = sorted(lab for lab in m["labels"] if lab and lab not in train_labels)
        if stray:
            shown = ", ".join(f"`{s}`" for s in stray[:6])
            notes.append(
                f"- `{rel}` uses {len(stray)} label(s) absent from the training taxonomy: "
                f"{shown}. Rows carrying them are unscoreable — every model is wrong on them "
                "by construction."
            )
    notes.append(
        "- The two broad clean sets — `holdout_leakage_guard.csv` and "
        "`extras/semantic_holdout_2.csv` — are independent of the train/holdout split "
        "and of each other's construction. They are the natural cross-check whenever a "
        "`dev_hard` result lands near its MDE."
    )
    out += notes
    # Exactly one trailing newline: the repo's end-of-file-fixer hook rewrites
    # anything else, which would change this file after it was generated.
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="regenerate INSTRUMENTS.md")
    args = ap.parse_args(argv)
    text = build()
    if args.write:
        (HERE / "INSTRUMENTS.md").write_text(text, encoding="utf-8")
        print(f"wrote {HERE / 'INSTRUMENTS.md'}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
