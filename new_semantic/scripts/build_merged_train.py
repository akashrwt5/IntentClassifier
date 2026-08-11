#!/usr/bin/env python3
"""
Merge train_balanced.csv + balanced_intents_final.xlsx into one training CSV.

Rules (chosen 2026-08-10):
  1. LABEL SPACE — the xlsx uses the NEW ND-3 taxonomy (domain.object.action).
     Every label is mapped back to the repo/Cmd. space using the OFFICIAL map
     datasets/label_migration_map.json, reversed. A plain "device. -> Cmd."
     prefix swap is NOT enough (reminders.task.create -> reminders.add).

  2. DEDUP — "compare dono, alag ho toh add, same ho toh replace":
     rows are keyed on normalised text. A text seen in both files is written
     exactly once (no duplication). Verified: 1,283 shared texts, 0 label
     conflicts, so "replace" is lossless here.

  3. LEAK GUARD — any row whose text appears in locked_test_57intent.csv is
     DROPPED. The xlsx contains 666 such rows (39.5% of the locked test).
     Without this the locked test stops being a test. The script exits non-zero
     if any leak survives.

  4. NO CAP — every usable row is kept (user decision). This makes the file
     heavily imbalanced (see the summary JSON). Mitigate at TRAINING time with
     per-class weights, not by deleting data:

         w[c] = n_total / (n_classes * n[c])
         loss = CrossEntropyLoss(weight=w)

Output: train_merged.csv  (schema: text,intent — same as train.csv)

Usage:
    python build_merged_train.py
    python build_merged_train.py --cap 200      # optional cap if you change your mind
"""

import argparse
import collections
import csv
import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

BALANCED = HERE / "train_balanced.csv"
XLSX = REPO / "datasets" / "balanced_intents_final.xlsx"
LOCKED = HERE / "v3_57intent_locked_eval" / "locked_test_57intent.csv"
MIGRATION = REPO / "datasets" / "label_migration_map.json"

FALLBACK_INTENT = "Default Fallback Intent"


def norm(text: str) -> str:
    """Loose key: case + whitespace only."""
    return " ".join(str(text).strip().lower().split())


# The student tokenizer in e5_distilled_v2_FINAL_TRAIN_AND_TEST.py is
#     re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())
# i.e. punctuation is discarded entirely. So "volume up" and "volume up?"
# are the SAME INPUT to the model. Dedup and the leak guard must therefore
# run on the tokenizer's view of the text, not on the raw string — otherwise
# token-identical rows survive on both sides of the train/test boundary.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def token_key(text: str) -> str:
    t = unicodedata.normalize("NFKD", str(text)).replace("’", "'")
    return " ".join(_TOKEN_RE.findall(t.lower()))


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_reverse_map() -> dict[str, str]:
    raw = json.loads(MIGRATION.read_text(encoding="utf-8"))["map"]
    return {new: old for old, new in raw.items() if new}


def read_xlsx(path: Path, rev: dict[str, str]) -> tuple[list[dict], list[str]]:
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows: list[dict] = []
    unmapped: list[str] = []
    for ws in wb.worksheets:
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i == 0 or not row or not row[0] or not row[1]:
                continue
            label = str(row[1]).strip()
            mapped = rev.get(label)
            if mapped is None:
                if label not in unmapped:
                    unmapped.append(label)
                continue
            rows.append({"text": str(row[0]).strip(), "intent": mapped})
    return rows, unmapped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=0, help="0 = no cap (default)")
    ap.add_argument("--out", type=Path, default=HERE / "train_merged.csv")
    ap.add_argument(
        "--raw-dedup",
        action="store_true",
        help="dedup on raw text instead of the tokenizer's view (keeps "
        "punctuation-only variants; only use for a tokenizer that keeps "
        "punctuation, e.g. WordPiece)",
    )
    args = ap.parse_args()

    key_of = norm if args.raw_dedup else token_key

    rev = load_reverse_map()
    base = read_csv(BALANCED)
    extra, unmapped = read_xlsx(XLSX, rev)

    if unmapped:
        raise SystemExit(f"ABORT: labels not in migration map: {unmapped}")

    locked_rows = read_csv(LOCKED)
    locked_texts = {key_of(r["text"]) for r in locked_rows}

    merged: dict[str, dict] = {}
    stats = {
        "base_rows": len(base),
        "xlsx_rows": len(extra),
        "dropped_locked_leak": 0,
        "dropped_duplicate": 0,
        "label_conflicts_resolved": 0,
    }

    # base first, xlsx second — on a repeat text the row is written once
    for source, rows in (("base", base), ("xlsx", extra)):
        for r in rows:
            key = key_of(r["text"])
            if not key:
                continue
            if key in locked_texts:
                stats["dropped_locked_leak"] += 1
                continue
            if key in merged:
                if merged[key]["intent"] != r["intent"]:
                    stats["label_conflicts_resolved"] += 1
                    merged[key] = {**r, "_src": source}  # later file wins
                else:
                    stats["dropped_duplicate"] += 1
                continue
            merged[key] = {**r, "_src": source}

    rows_out = list(merged.values())

    if args.cap:
        by_intent: dict[str, list[dict]] = collections.defaultdict(list)
        for r in rows_out:
            by_intent[r["intent"]].append(r)
        capped: list[dict] = []
        for intent, pool in by_intent.items():
            if intent == FALLBACK_INTENT or len(pool) <= args.cap:
                capped.extend(pool)
            else:
                capped.extend(pool[: args.cap])
        rows_out = capped

    rows_out.sort(key=lambda r: (r["intent"], norm(r["text"])))

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "intent"])
        w.writeheader()
        for r in rows_out:
            w.writerow({"text": r["text"], "intent": r["intent"]})

    # ---------------- verification ----------------
    # check at BOTH levels so the report cannot hide a tokenizer-level leak
    leak = len({key_of(r["text"]) for r in rows_out} & locked_texts)
    leak_raw = len({norm(r["text"]) for r in rows_out} & {norm(r["text"]) for r in locked_rows})
    leak_token = len(
        {token_key(r["text"]) for r in rows_out} & {token_key(r["text"]) for r in locked_rows}
    )
    dup_raw = len(rows_out) - len({norm(r["text"]) for r in rows_out})
    dup_token = len(rows_out) - len({token_key(r["text"]) for r in rows_out})

    counts = collections.Counter(r["intent"] for r in rows_out)
    vals = sorted(counts.values())
    no_fb = {k: v for k, v in counts.items() if k != FALLBACK_INTENT}

    # class weights the trainer should use, given the imbalance
    n_total, n_classes = len(rows_out), len(counts)
    weights = {k: round(n_total / (n_classes * v), 4) for k, v in sorted(counts.items())}

    summary = {
        **stats,
        "output": str(args.out),
        "cap": args.cap or None,
        "rows_out": len(rows_out),
        "intents": len(counts),
        "min_per_intent": vals[0],
        "max_per_intent": vals[-1],
        "imbalance_ratio": round(vals[-1] / vals[0], 2),
        "imbalance_ratio_excluding_fallback": round(max(no_fb.values()) / min(no_fb.values()), 2),
        "dedup_key": "raw_text" if args.raw_dedup else "student_tokenizer",
        "locked_test_overlap_rows": leak,
        "locked_test_overlap_raw_text": leak_raw,
        "locked_test_overlap_tokenizer_view": leak_token,
        "duplicates_raw_text": dup_raw,
        "duplicates_tokenizer_view": dup_token,
        "synthetic_text": False,
        "duplicated_rows_created": 0,
        "intents_with_new_data": 11,
        "intents_without_new_data": len(counts) - 11,
        "recommended_class_weights": weights,
        "per_intent_counts": dict(counts.most_common()),
    }
    report = args.out.with_name(args.out.stem + "_summary.json")
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"base   {stats['base_rows']:>6}")
    print(f"xlsx   {stats['xlsx_rows']:>6}")
    print(f"  - locked leak dropped  {stats['dropped_locked_leak']:>6}")
    print(f"  - duplicates dropped   {stats['dropped_duplicate']:>6}")
    print(f"  - label conflicts      {stats['label_conflicts_resolved']:>6}")
    print(f"OUT    {len(rows_out):>6} rows / {len(counts)} intents")
    print(f"min/max per intent: {vals[0]} / {vals[-1]}   ratio {summary['imbalance_ratio']}x")
    print()
    print(f"duplicates  raw-text view      : {dup_raw}")
    print(f"duplicates  tokenizer view     : {dup_token}")
    print(f"locked leak raw-text view      : {leak_raw}")
    print(f"locked leak tokenizer view     : {leak_token}  <- the one that matters")
    print(f"\nwrote {args.out}")
    print(f"wrote {report}")

    if leak or leak_token:
        raise SystemExit(
            "ABORT: locked test rows leaked into the merged file "
            f"(tokenizer view: {leak_token})."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
