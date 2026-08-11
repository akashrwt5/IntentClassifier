#!/usr/bin/env python3
"""
Type an utterance, see exactly which stage answered it — and what the OTHER
backstop setting would have done.

    Stage 2  TF-IDF          conf >= 0.70  -> fires, done
    Stage 3  semantic rescue conf >= 0.40  -> rescues
    backstop TF-IDF again    conf >= 0.30  -> keeps Stage 2's answer  [NEW]
    else                                   -> GENAI fallback

Both engines are the REAL NLUEngine; the only difference between them is
`_stage2_backstop`. When the two columns disagree, that turn is exactly what the
change does.

STAGE 3 IS FORCED ON HERE. The shipped English pack has
`semantic_rescue_enabled: false`, so the engine would otherwise run without it.
The constructor parameter overrides the schema, which keeps this test honest
without editing — and risking shipping — that flag.

Which model answers as Stage 3 depends on what is installed:

    models/semantic_student/en/   present  -> StudentSemantic (the new student)
                                  absent   -> SemanticFallback (legacy MiniLM)

The header prints which one loaded. Read it before trusting anything below it.

Usage:
    python scripts/try_it.py
    python scripts/try_it.py "turn it up a bit"
    python scripts/try_it.py --file my_phrases.txt
    python scripts/try_it.py --no-semantic       # Stage 2 + backstop only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _p in ("packages/buildtime", "packages/runtime"):
    sys.path.insert(0, str(REPO / _p))

MIGRATION = REPO / "datasets" / "label_migration_map.json"


def _old_space():
    raw = json.loads(MIGRATION.read_text(encoding="utf-8"))["map"]
    return {new: old for old, new in raw.items() if new}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*", help="one utterance; omit for a prompt loop")
    ap.add_argument("--file", type=Path, help="a file with one utterance per line")
    ap.add_argument("--backstop", type=float, default=0.30)
    ap.add_argument(
        "--no-semantic",
        action="store_true",
        help="leave Stage 3 off (the shipped default) to isolate " "what the backstop alone does",
    )
    args = ap.parse_args()

    from nlu_engine.engine import NLUEngine

    print("loading engine...", flush=True)
    # Stage 3 is disabled in the shipped English pack
    # (`semantic_rescue_enabled: false`). The constructor parameter wins over
    # the schema, so testing the semantic stage does not require editing — and
    # therefore accidentally shipping — that flag.
    sem = None if args.no_semantic else True
    off = NLUEngine(language="en", semantic_enabled=sem)
    on = NLUEngine(language="en", semantic_enabled=sem)
    off._stage2_backstop = 0.0
    on._stage2_backstop = args.backstop
    rev = _old_space()

    s2_gate = getattr(on, "threshold", 0.70)
    print(
        f"Stage 2 gate {s2_gate}   Stage 3 gate {on.semantic_threshold}   "
        f"backstop {args.backstop}"
    )
    if on.semantic is None:
        print("Stage 3: NOT LOADED — every result below is Stage 2 + backstop only")
    else:
        backend = type(on.semantic).__name__
        extra = ""
        if backend == "StudentSemantic":
            extra = (
                f"  ({len(on.semantic.labels)} intents, "
                f"{len(on.semantic.vocab)} vocab, "
                f"gate {on.semantic.threshold})"
            )
        elif backend == "SemanticFallback":
            extra = "  (legacy MiniLM + LogReg head, 23 MB)"
        print(f"Stage 3: {backend}{extra}")
    print()

    def show(text: str) -> None:
        text = text.strip()
        if not text:
            return

        i2, c2 = on.classifier.classify(text)
        i2 = rev.get(i2, i2)
        handed = c2 < s2_gate
        line = f"  Stage 2   {i2:<32} {c2:.4f}"
        line += "   -> hands over" if handed else "   -> FIRES (done)"
        print(line)

        if on.semantic is not None:
            try:
                i3, c3 = on.semantic.classify(text)
                gate3 = getattr(on.semantic, "threshold", on.semantic_threshold)
                if not handed:
                    # Shown for inspection only. Stage 2 already fired, so this
                    # opinion had no effect on the turn — printing it without
                    # saying so would misrepresent what the pipeline did.
                    note = "   (NOT CONSULTED — Stage 2 already fired)"
                elif c3 >= gate3 and i3 != "Default Fallback Intent":
                    note = "   -> accepts"
                else:
                    note = "   -> declines"
                print(f"  Stage 3   {i3:<32} {c3:.4f}{note}")
            except Exception as e:  # noqa: BLE001
                print(f"  Stage 3   unavailable ({type(e).__name__})")

        r_off = off.handle("try-off", text)
        r_on = on.handle("try-on", text)

        def fmt(r):
            name = rev.get(r.intent, r.intent)
            tag = "GENAI" if r.type == "FALLBACK" and r.intent == "GENAI" else name
            src = "rescue" if getattr(r, "semantic_rescue", False) else "stage2"
            if tag == "GENAI":
                src = "reject"
            return f"{tag:<30} ({src}, {r.confidence:.3f})"

        same = r_off.intent == r_on.intent
        print(f"  OFF       {fmt(r_off)}")
        print(f"  ON        {fmt(r_on)}" + ("" if same else "     <<< DIFFERENT"))
        print()

    if args.file:
        for ln in args.file.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                print(f"> {ln.strip()}")
                show(ln)
        return 0

    if args.text:
        t = " ".join(args.text)
        print(f"> {t}")
        show(t)
        return 0

    print("type an utterance, blank line or Ctrl-D to quit\n")
    while True:
        try:
            t = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not t.strip():
            return 0
        show(t)


if __name__ == "__main__":
    raise SystemExit(main())
