#!/usr/bin/env python3
"""Generate legacy-label conformance fixtures from the reference engine.

Runs the Python reference NLU engine with the app-compat shim ON
(NLU_LEGACY_LABELS=1) over representative inputs and records the legacy label it
emits at the boundary. Native clients (iOS/Android) that read
``runtime/legacy_labels.json`` and translate at their own boundary must produce
the SAME strings for these inputs — this CSV is the cross-platform conformance
vector (ADR-011: parity by fixtures, not a shared binary).

    python scripts/gen_legacy_label_fixtures.py

Output: tests/fixtures/legacy_label_parity_en.csv
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

os.environ["NLU_LEGACY_LABELS"] = "1"  # boundary translation ON
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "runtime"))
OUT = REPO / "tests" / "fixtures" / "legacy_label_parity_en.csv"

# Single-turn commands: text -> expected legacy intent (each in its own session).
SINGLE_TURN = [
    "increase volume",
    "turn up the volume",
    "decrease volume",
    "mute",
    "unmute the volume",
    "what's my battery level",
    "find my phone",
    "how do i change my memories",   # a Help_* topic
    "start streaming",
]


def main() -> None:
    import warnings
    warnings.filterwarnings("ignore")
    from nlu_engine import NLUEngine
    from nlu_langpack import load_pack

    eng = NLUEngine(pack=load_pack(str(REPO / "dist" / "bundle-en")))
    rows: list[dict] = []

    for i, text in enumerate(SINGLE_TURN):
        sid = f"single-{i}"
        r = eng.handle(sid, text)
        rows.append({"session": sid, "turn": 1, "text": text,
                     "expected_type": r.type, "expected_intent": r.intent or ""})

    # Two-turn send-confirmation flow -> the compound dialogue-act labels.
    #
    # This used to force the uncertainty gate (`eng._confirm_below = 1.01`) so
    # the confirmation would happen "regardless of model confidence". That line
    # was the defect in miniature: the confirmation genuinely DID depend on
    # confidence, so the fixture had to lie to produce a stable vector, and the
    # app's dialogue act appeared for "send a message" but vanished for "send a
    # message to john".
    #
    # `Cmd.SendMessage` now declares a schema `followup`, so the
    # confirmation is authored dialogue and fires every turn on its own. No
    # forcing, and the fixture records what the engine actually does.
    for polarity_text, sid in [("yes", "send-yes"), ("no", "send-no")]:
        eng.reset(sid)
        r1 = eng.handle(sid, "send a message")
        rows.append({"session": sid, "turn": 1, "text": "send a message",
                     "expected_type": r1.type, "expected_intent": r1.intent or ""})
        r2 = eng.handle(sid, polarity_text)
        rows.append({"session": sid, "turn": 2, "text": polarity_text,
                     "expected_type": r2.type, "expected_intent": r2.intent or ""})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["session", "turn", "text",
                                          "expected_type", "expected_intent"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT.relative_to(REPO)}  ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['session']:10} t{r['turn']} {r['text']:24} -> "
              f"{r['expected_type']:8} {r['expected_intent']}")


if __name__ == "__main__":
    main()
