#!/usr/bin/env python3
"""Score the shipped encoder on every English instrument. This is the P1 baseline.

WHY THIS EXISTS
---------------
Until now this project had one number for the 9 MB encoder: 0.8578 on
``holdout_honest.csv``. That set is 44.7% near-duplicate of the training data,
so the number is part generalisation and part recall, in an unknown ratio. No
decision after P1 can rest on it.

This script produces the table that replaces it: the same encoder and the same
shipped head, scored on each instrument separately, with each result printed
next to the smallest difference that instrument can actually resolve. The
``dev_hard`` row is the baseline every later phase is judged against.

WHAT IT WRITES, AND WHY THAT MATTERS MORE THAN THE TABLE
-------------------------------------------------------
``baseline_predictions.csv`` carries one row per scored utterance: instrument,
text, gold label, prediction, correct. The plan decides with McNemar's test on
discordant items, which needs to know WHICH rows each model got right -- not
just how many. Two accuracy numbers cannot be turned into a McNemar test after
the fact. Keeping the per-row predictions is what makes P2 comparable to P1
without re-running P1.

UNSCOREABLE ROWS
----------------
A row whose gold label is not among the head's classes cannot be got right by
any model. Those rows are excluded from the accuracy and counted separately
rather than silently scored as wrong -- otherwise an instrument's ceiling sits
below 100% and every number from it reads a little low for a reason nobody can
see (B15).

    python3 score_instruments.py              # score everything, write the report
    python3 score_instruments.py --only dev_hard
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from artifact import encode, head_path, load_encoder  # noqa: E402
from instruments import minimum_detectable_effect  # noqa: E402

HERE = Path(__file__).resolve().parent

# --- Inside this directory ----------------------------------------------
MODEL_PATH = HERE / "output_models" / "stage2_contrastive_bge_small_onnx" / "model_quantized.onnx"
REPORT_PATH = HERE / "BASELINE.md"
PREDICTIONS_PATH = HERE / "baseline_predictions.csv"

# --- Outside this directory ---------------------------------------------
REPO = HERE.parents[1]
PACK = REPO / "language_packs" / "en"
# ------------------------------------------------------------------------

DISCORDANCE = 0.15

# name, path relative to the pack, and what the result may be used for.
# `decision` marks the instruments a later phase is allowed to gate on.
INSTRUMENTS = [
    (
        "dev_hard",
        "dev_hard.csv",
        True,
        "PRIMARY. Clean rows. This is the baseline P2-P8 is judged against.",
    ),
    ("dev_near", "dev_near.csv", False, "Memorisation. Regression detection only."),
    (
        "holdout_honest",
        "holdout_honest.csv",
        False,
        "The parent set. Reported only for continuity with earlier numbers.",
    ),
    (
        "leakage_guard",
        "holdout_leakage_guard.csv",
        True,
        "Independent cross-check, built separately from the train/holdout split.",
    ),
    (
        "semantic_holdout_2",
        "extras/semantic_holdout_2.csv",
        True,
        "Second independent cross-check.",
    ),
    (
        "paraphrase",
        "extras/holdout_paraphrase.csv",
        False,
        "Deep on a narrow slice of the taxonomy. Never read as an overall figure.",
    ),
    (
        "benchmark_250",
        "extras/benchmark_250.csv",
        False,
        "B14: 85.5% leaked, 10 of 57 intents. Reported to show what it is worth.",
    ),
]

TEXT_COLUMNS = ("text", "utterance")
LABEL_COLUMNS = ("intent", "expected_intent", "label")


def _column(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    raise SystemExit(f"no column among {candidates} in {cols}")


def load_instrument(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        cols = [c.strip() for c in (reader.fieldnames or [])]
        rows = list(reader)
    tcol, lcol = _column(cols, TEXT_COLUMNS), _column(cols, LABEL_COLUMNS)
    # Lowercased to match how the head was fitted; stated rather than assumed.
    texts = [str(r[tcol]).lower().strip() for r in rows]
    gold = [str(r[lcol]).strip() for r in rows]
    return texts, gold


def score(encoder, clf, texts, gold):
    known = set(clf.classes_)
    keep = [i for i, g in enumerate(gold) if g in known]
    dropped = sorted({g for g in gold if g not in known})

    start = time.time()
    vectors = encode(encoder, [texts[i] for i in keep])
    predictions = clf.predict(vectors)
    elapsed = time.time() - start

    correct = np.array([predictions[k] == gold[i] for k, i in enumerate(keep)])

    # Macro-F1 is averaged over the classes PRESENT IN GOLD, not over the union
    # of gold and predicted classes.
    #
    # sklearn's average="macro" uses the union, which is the right convention
    # when an instrument covers the whole taxonomy. It is badly misleading when
    # it does not: `holdout_paraphrase` has 10 gold intents scored by a 57-class
    # head, so every stray prediction adds a class with F1 = 0 and drags the
    # mean toward zero. Under the union convention that set scores 0.397 against
    # an accuracy of 0.750 -- a gap that reads as a broken model and is really
    # just an instrument narrower than the head. Classes the model invented are
    # reported separately, as a count, where they cannot be mistaken for recall.
    gold_labels = sorted({gold[i] for i in keep})
    invented = sorted(set(predictions) - set(gold_labels))
    per_class = []
    for lab in gold_labels:
        tp = sum(1 for k, i in enumerate(keep) if predictions[k] == lab and gold[i] == lab)
        fp = sum(1 for k, i in enumerate(keep) if predictions[k] == lab and gold[i] != lab)
        fn = sum(1 for k, i in enumerate(keep) if predictions[k] != lab and gold[i] == lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_class.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)

    return {
        "n_total": len(gold),
        "n_scored": len(keep),
        "unscoreable_labels": dropped,
        "n_unscoreable": len(gold) - len(keep),
        "accuracy": float(correct.mean()) if len(keep) else float("nan"),
        "macro_f1": float(np.mean(per_class)) if per_class else float("nan"),
        "n_gold_classes": len(gold_labels),
        "n_invented_classes": len(invented),
        "ms_per_query": (elapsed / len(keep) * 1000) if keep else float("nan"),
        "rows": [
            {
                "text": texts[i],
                "gold": gold[i],
                "pred": str(predictions[k]),
                "correct": int(correct[k]),
            }
            for k, i in enumerate(keep)
        ],
    }


def unpaired_difference(a: dict, b: dict) -> tuple[float, float, float]:
    """Difference between two accuracies measured on DIFFERENT rows, with its 95% CI.

    dev_hard and dev_near are disjoint sets, so McNemar does not apply -- it needs
    the same rows scored twice. This is the two-proportion comparison instead.
    Reported because the gap between them is the single most useful number in this
    table: it is how much of the old holdout_honest figure was recall rather than
    generalisation, and it is the reason dev_hard replaces it as the baseline.
    """
    p1, n1 = a["accuracy"], a["n_scored"]
    p2, n2 = b["accuracy"], b["n_scored"]
    se = ((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2)) ** 0.5
    return p2 - p1, se, 1.959964 * se


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", help="score a single instrument by name")
    args = ap.parse_args(argv)

    if not MODEL_PATH.exists():
        raise SystemExit(f"encoder not found: {MODEL_PATH}")
    encoder = load_encoder(MODEL_PATH)
    with head_path(MODEL_PATH).open("rb") as fh:
        clf = pickle.load(fh)

    print(f"encoder: {encoder.summary()}")
    print(f"head:    {head_path(MODEL_PATH).name}  ({len(clf.classes_)} classes)\n")

    results = {}
    for name, rel, decision, charter in INSTRUMENTS:
        if args.only and name != args.only:
            continue
        path = PACK / rel
        if not path.exists():
            print(f"skip  {name}: {path} not found")
            continue
        texts, gold = load_instrument(path)
        r = score(encoder, clf, texts, gold)
        r.update(rel=rel, decision=decision, charter=charter)
        r["mde"] = minimum_detectable_effect(r["n_scored"], DISCORDANCE)
        results[name] = r
        flag = "*" if decision else " "
        note = f"   ({r['n_unscoreable']} unscoreable)" if r["n_unscoreable"] else ""
        print(
            f"{flag} {name:<20} n={r['n_scored']:>5}  acc={r['accuracy']:.4f}  "
            f"macroF1={r['macro_f1']:.4f}  classes={r['n_gold_classes']:>2}"
            f"(+{r['n_invented_classes']})  MDE={r['mde']:.3f}{note}"
        )

    if not results:
        raise SystemExit("nothing scored")

    with PREDICTIONS_PATH.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["instrument", "text", "gold", "pred", "correct"], lineterminator="\n"
        )
        w.writeheader()
        for name, r in results.items():
            for row in r["rows"]:
                w.writerow({"instrument": name, **row})

    write_report(encoder, clf, results)
    print(f"\nwrote {REPORT_PATH.name} and {PREDICTIONS_PATH.name}")
    return 0


def write_report(encoder, clf, results) -> None:
    lines = [
        "# P1 baseline — the shipped encoder on every instrument",
        "",
        f"Generated by `{Path(__file__).name}`. Do not edit by hand.",
        "",
        "```",
        encoder.summary().rstrip(),
        f"head: {head_path(MODEL_PATH).name}, {len(clf.classes_)} classes, as shipped "
        f"(not re-fitted)",
        "```",
        "",
        "`MDE` is the smallest accuracy difference McNemar's test can resolve on that "
        f"instrument at {DISCORDANCE:.0%} discordance and 80% power. A later phase that moves "
        "a number by less than this has not been shown to move it at all.",
        "",
        "Macro-F1 is averaged over the intents present in each instrument's gold labels, "
        "not over the union with whatever the head predicted — see the note in the script. "
        "`Invented` counts intents the model emitted that the instrument does not contain: "
        "on a narrow instrument that is a false-accept signal, not a recall failure.",
        "",
        "| Instrument | Gate? | Scored | Unscoreable | Intents | Invented | Accuracy | "
        "Macro-F1 | MDE |",
        "|---|:-:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        lines.append(
            f"| `{name}` | {'**yes**' if r['decision'] else 'no'} | {r['n_scored']:,} | "
            f"{r['n_unscoreable']} | {r['n_gold_classes']} | {r['n_invented_classes']} | "
            f"{r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['mde']:.3f} |"
        )
    lines += ["", "## What each row may be used for", ""]
    for name, r in results.items():
        lines.append(f"- **`{name}`** — {r['charter']}")
        if r["unscoreable_labels"]:
            labs = ", ".join(f"`{x}`" for x in r["unscoreable_labels"])
            lines.append(
                f"  - {r['n_unscoreable']} row(s) excluded: gold label(s) {labs} are not "
                f"among the head's classes, so no model can score them."
            )
    if "dev_hard" in results and "dev_near" in results:
        diff, se, ci = unpaired_difference(results["dev_hard"], results["dev_near"])
        hh = results.get("holdout_honest")
        lines += [
            "",
            "## How much of the old number was memorisation",
            "",
            f"`dev_near` scores **{results['dev_near']['accuracy']:.4f}** and `dev_hard` "
            f"**{results['dev_hard']['accuracy']:.4f}** — a gap of **{diff:+.4f}** "
            f"(95% CI ±{ci:.4f}, {abs(diff / se):.1f} SE). The two sets are disjoint, so this "
            "is a two-proportion comparison, not McNemar.",
            "",
        ]
        if hh:
            lines.append(
                f"`holdout_honest` sits between them at {hh['accuracy']:.4f}, which is what a "
                f"{results['dev_near']['n_scored'] / hh['n_scored']:.0%} / "
                f"{results['dev_hard']['n_scored'] / hh['n_scored']:.0%} blend of the two "
                "should look like. That figure was never wrong — it was answering a different "
                "question than the one it was being asked. **`dev_hard` is the baseline from "
                "here.**"
            )
            lines.append("")
    lines += [
        "",
        "## Per-row predictions",
        "",
        f"`{PREDICTIONS_PATH.name}` holds one row per scored utterance: instrument, text, "
        "gold, prediction, correct. McNemar's test compares *which* rows two models get "
        "right, which cannot be recovered from two accuracy figures — so this file, not the "
        "table above, is what makes P2 comparable with P1.",
        "",
    ]
    # rstrip before the final newline: the section builders above leave trailing
    # blanks, and the repo's end-of-file-fixer rewrites anything but exactly one.
    REPORT_PATH.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
