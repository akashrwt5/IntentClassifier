#!/usr/bin/env python3
"""One-time relabel of utterances wrongly marked out-of-scope.

WHY
---
Charter B5 found 11 wrong actions on the honest holdout, 10 of them carrying
``truth == Default Fallback Intent``. Auditing those against their nearest in-scope
training neighbours showed that some were not model errors at all: the same
request appears under an in-scope label elsewhere in the corpus, so the model
answered correctly and the TEST was wrong.

Only utterances where an in-scope sibling makes the label a CONTRADICTION are
touched. Ambiguous capability-boundary cases are deliberately left alone,
because relabelling them asserts a product capability rather than fixing data
(see NOT_TOUCHED below).

THE RELABELS
------------
``read last text message`` -> ``Cmd.ListenMessage``
    In-scope training already contains ``read my new texts aloud``,
    ``read my unread messages``, ``read out who texted me`` and
    ``play the text i just got``, all labelled Cmd.ListenMessage. Same
    verb, same object. The model fired Cmd.ListenMessage at 0.974 and
    was charged a wrong action for being right.

``translate hello to french`` -> ``Cmd.TranslationStart``
    In-scope training contains ``translate how much do i owe you to spanish``
    and ``start translating in french`` under Cmd.TranslationStart —
    identical "translate <phrase> to <language>" shape.

``where do i find the mute button`` -> ``Help_Volume``
    In-scope Help_Volume contains ``how do i mute the sound?``,
    ``where do i change loudness`` and ``where is the loudness control``. A
    read-only help intent, so this one does not affect the wrong-action budget;
    it is corrected because it is wrong.

NOT TOUCHED (deliberately)
--------------------------
``stream, youtube music`` — the INTENT is streaming; YouTube is the unsupported
    part. Relabelling it to Cmd.StreamingStart would assert that YouTube
    streaming is supported. That is a product decision, and the deeper issue is
    that Default Fallback Intent conflates "not a command" with "a command whose
    parameter we cannot fulfil" (see docs/Review-F5/b5-root-cause-audit.md §5).
``how do i set up the automatic car memory`` — may name a feature that does not
    exist.
``turn off alarm``, ``text levitar``, ``thunder message``,
``got to getting a text message at boston`` — correctly out of scope.

THE HOLDOUT IS FROZEN
---------------------
One of these rows lives in ``holdout_honest.csv``. Changing it changes every
number measured against that file, so the manifest sha256 is refreshed and the
edit is recorded there. This is the deliberate, once-only correction the B1
manifest anticipates — not a re-split.

AFTER RUNNING, all of these must be redone:
    python -m nlu_training.train --lang en
    python -m nlu_training.fit_calibration --lang en --write
    python -m nlu_training.fit_confirm_gate --lang en --max-friction 0.15
    python -m nlu_training.wrong_action_harness --langs en

USAGE
    python scripts/ci/fix_oos_mislabels.py --dry-run
    python scripts/ci/fix_oos_mislabels.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# (compiled matcher, new label, why) — matched against the raw utterance.
RELABELS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\bread\b.*\b(text\s+)?message", re.I),
     "Cmd.ListenMessage",
     "in-scope siblings: 'read my new texts aloud', 'read my unread messages'"),
    (re.compile(r"\btranslate\b.+\bto\s+(french|spanish|german|danish|english)\b", re.I),
     "Cmd.TranslationStart",
     "in-scope sibling: 'translate how much do i owe you to spanish'"),
    (re.compile(r"\bwhere\b.*\b(mute|volume|loudness)\b.*\bbutton\b"
                r"|\bwhere\b.*\bbutton\b.*\b(mute|volume)\b", re.I),
     "Help_Volume",
     "in-scope siblings: 'where do i change loudness', 'how do i mute the sound?'"),
]

TARGETS = ("train.csv", "holdout_honest.csv")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply(lang: str, dry_run: bool) -> int:
    ds = REPO / "datasets" / lang
    changed_all: list[dict] = []

    for name in TARGETS:
        path = ds / name
        if not path.exists():
            print(f"skip {name} (absent)")
            continue
        with path.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            fields, rows = list(reader.fieldnames or []), list(reader)

        changed = []
        for row in rows:
            if row["intent"].strip() != "Default Fallback Intent":
                continue
            for pat, new_label, why in RELABELS:
                if pat.search(row["text"]):
                    changed.append({"file": name, "text": row["text"],
                                    "from": "Default Fallback Intent", "to": new_label,
                                    "why": why})
                    row["intent"] = new_label
                    break

        print(f"{name}: {len(changed)} row(s) relabelled")
        for c in changed:
            print(f"   {c['text']!r}  ->  {c['to']}")
        changed_all += changed

        if changed and not dry_run:
            with path.open("w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)

    if dry_run:
        print("\n--dry-run: nothing written.")
        return 0
    if not changed_all:
        print("\nnothing to do (already applied).")
        return 0

    # The holdout is frozen by hash; refresh it and record why it moved.
    man_path = ds / "holdout_honest.manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        man["sha256"] = {"train.csv": _sha256(ds / "train.csv"),
                         "holdout_honest.csv": _sha256(ds / "holdout_honest.csv")}
        man.setdefault("amendments", []).append({
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "by": "scripts/ci/fix_oos_mislabels.py",
            "what": "Relabelled utterances wrongly marked Default Fallback Intent where "
                    "an in-scope sibling made the label a contradiction. This is "
                    "a label correction, NOT a re-split — the train/holdout "
                    "partition is unchanged, so the sets remain disjoint.",
            "rows": changed_all,
        })
        man_path.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        print(f"\nupdated {man_path.relative_to(REPO)} (hashes + amendment record)")

    print("\nNOW REDO, in order — every downstream number depends on train.csv:")
    print("  python -m nlu_training.train --lang en")
    print("  python -m nlu_training.fit_calibration --lang en --write")
    print("  python -m nlu_training.fit_confirm_gate --lang en --max-friction 0.15")
    print("  python -m nlu_training.wrong_action_harness --langs en")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return apply(a.lang, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
