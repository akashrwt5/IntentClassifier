"""Test the model the way a person would. No ML knowledge needed to read this.

    python scripts/acceptance_test.py

Runs a fixed list of sentences through the shipped model and prints, for each
one, whether the hearing aid would have done the right thing.

WHAT THE THREE RESULTS MEAN
---------------------------
    OK        the aid does the right thing. This is what we want.

    ASKS      the aid does nothing and asks the user to repeat. Annoying, not
              dangerous. The user says it again.

    WRONG     the aid DOES SOMETHING, and it is the wrong thing. This is the
              only result that can hurt someone — the volume drops when they
              asked for more, or a message goes to the wrong person. Every
              WRONG is worth reporting; a few ASKS are normal.

A tester should read the WRONG count first and everything else second.

ADDING YOUR OWN SENTENCES
-------------------------
Make a CSV with two columns, `text` and `expected`, and run:

    python scripts/acceptance_test.py --file my_tests.csv

`expected` is the intent name the aid should act on — copy them from
configs/intents.yaml — or the word `reject` if the aid should refuse
(for example, a request meant for the television rather than the aid).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import Runtime  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REJECT = "reject"

# ---------------------------------------------------------------------------
# The test list, grouped by what each group is checking. Groups exist so a
# failure points somewhere: "long sentences fail, short ones pass" is a finding,
# "62% passed" is not.
# ---------------------------------------------------------------------------
CASES: list[tuple[str, str, str]] = [
    # ---- everyday short commands. These are the bread and butter. ----------
    ("short commands", "turn it up", "Cmd.VolumeIncrease"),
    ("short commands", "turn it down a bit", "Cmd.VolumeDecrease"),
    ("short commands", "make it louder", "Cmd.VolumeIncrease"),
    ("short commands", "mute my aids", "Cmd.VolumeMute"),
    ("short commands", "turn the sound back on", "Cmd.VolumeUnmute"),
    ("short commands", "switch to the restaurant program", "Cmd.MemoryChange"),
    ("short commands", "how much battery is left", "Cmd.BatteryLevel"),
    ("short commands", "where is my phone", "Cmd.FindMyPhone"),
    ("short commands", "start streaming the tv", "Cmd.StreamingStart"),
    ("short commands", "stop the streaming", "Cmd.StreamingStop"),
    ("short commands", "how many steps have i done", "Cmd.ActivityStep"),
    ("short commands", "read me my messages", "Cmd.ListenMessage"),

    # ---- help questions. Asking ABOUT a feature, not asking to use it. -----
    ("help questions", "how do i change the volume", "Help_Volume"),
    ("help questions", "how do i pair my aid to my phone", "Help_Pairing"),
    ("help questions", "how do i clean my hearing aids", "Help_CleanCare"),
    ("help questions", "what is the tinnitus feature", "Help_Tinnitus"),
    ("help questions", "how do i put them in properly", "Help_InsertDevice"),

    # ---- full sentences: situation first, then the request. ----------------
    # This is the KNOWN WEAK AREA. See STATUS.md.
    ("full sentences", "it is very noisy in this cafe, can you switch me to the restaurant setting", "Cmd.MemoryChange"),
    ("full sentences", "i am about to go into a meeting so please turn the volume down", "Cmd.VolumeDecrease"),
    ("full sentences", "the television has been on all evening and my ears are tired, turn it down", "Cmd.VolumeDecrease"),
    ("full sentences", "my daughter is visiting later, could you remind me to take my tablets at six", "reminders.add"),
    ("full sentences", "i cannot hear the speech very well in here, could you turn it up", "Cmd.VolumeIncrease"),

    # ---- the specific failure this model is known for ----------------------
    # A quiet room described with a word the model learnt as a REQUEST.
    ("describe then ask", "it is a bit quieter in here can you make it louder", "Cmd.VolumeIncrease"),
    ("describe then ask", "it is rather faint in here can you make it louder", "Cmd.VolumeIncrease"),
    ("describe then ask", "it is a bit louder in here can you make it quieter", "Cmd.VolumeDecrease"),
    ("describe then ask", "it is rather harsh in here can you make it quieter", "Cmd.VolumeDecrease"),

    # ---- meant for something ELSE in the room. Must NOT act. ---------------
    ("not the aid", "turn the dishwasher down", REJECT),
    ("not the aid", "the microwave is too loud", REJECT),
    ("not the aid", "lower the thermostat", REJECT),
    ("not the aid", "turn on the hallway lights", REJECT),
    ("not the aid", "set the oven to one eighty", REJECT),

    # ---- nothing to do with the product at all. Must NOT act. -------------
    ("out of scope", "what is the weather tomorrow", REJECT),
    ("out of scope", "tell me a joke about cats", REJECT),
    ("out of scope", "who won the football last night", REJECT),
    ("out of scope", "what is my blood pressure", REJECT),
    ("out of scope", "give my daughter a ring", REJECT),

    # ---- accessories: the TV streamer and remote mic. ----------------------
    ("accessories", "how do i remove my remote mic", "Help_Accessories"),
    ("accessories", "how do i unpair my tv streamer", "Help_Accessories"),
    ("accessories", "stream from the remote microphone", "Cmd.StreamingStart"),
    ("accessories", "i am done with the tv sound", "Cmd.StreamingStop"),

    # ---- the person changes their mind mid-sentence -----------------------
    ("changed mind", "not quieter, louder", "Cmd.VolumeIncrease"),
    ("changed mind", "turn it down, no i meant up", "Cmd.VolumeIncrease"),
    ("changed mind", "stop streaming, actually start it", "Cmd.StreamingStart"),

    # ---- speech the aid mis-heard, or that was not aimed at it ------------
    ("misheard", "turn up my right ade", "Cmd.VolumeIncrease"),
    ("misheard", "swtich to the resturant progam", "Cmd.MemoryChange"),
    ("misheard", "and push it down for dramatics", REJECT),
]


def verdict(row_expected: str, result: dict) -> tuple[str, str]:
    """Returns (code, plain-English explanation)."""
    acted = result["accepted"]
    got = result["intent"]

    if row_expected == REJECT:
        if not acted:
            return "OK", "correctly did nothing"
        return "WRONG", f"ACTED on it — did '{got}'"

    if acted and got == row_expected:
        return "OK", f"did '{got}'"
    if acted and got != row_expected:
        return "WRONG", f"did '{got}' instead of '{row_expected}'"
    if got == row_expected:
        return "ASKS", f"knew it was '{got}' but was not confident enough"
    return "ASKS", f"asked the user to repeat (its best guess was '{got}')"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/final_student_256/onnx")
    ap.add_argument("--quant", default="int8", choices=["int8", "fp32"])
    ap.add_argument("--file", default=None,
                    help="your own CSV with columns text,expected")
    ap.add_argument("--show-ok", action="store_true",
                    help="also list the sentences that worked")
    args = ap.parse_args()

    rt = Runtime(ROOT / args.model, args.quant)

    cases = CASES
    if args.file:
        import pandas as pd
        df = pd.read_csv(args.file)
        missing = {"text", "expected"} - set(df.columns)
        if missing:
            raise SystemExit(f"{args.file} needs columns: text, expected")
        cases = [("your tests", str(r.text), str(r.expected))
                 for r in df.itertuples()]

    print(f"\nTesting {rt.model_path.name} "
          f"({rt.model_path.stat().st_size/1e6:.2f} MB) on {len(cases)} "
          f"sentences.\n")

    if rt.warnings:
        print("  " + "!" * 68)
        for w in rt.warnings:
            print(f"  !! {w}\n")
        print("  This model is broken, not cautious. The results below will be")
        print("  almost all ASKS and they say nothing about the design.")
        print("  " + "!" * 68 + "\n")

    groups: dict[str, list] = {}
    for group, text, expected in cases:
        r = rt(text)
        code, why = verdict(expected, r)
        groups.setdefault(group, []).append(
            (code, text, why, r["confidence"]))

    order = ["OK", "ASKS", "WRONG"]
    totals = {c: 0 for c in order}
    print(f"{'group':18} {'OK':>4} {'ASKS':>6} {'WRONG':>7}")
    print("-" * 40)
    for group, rows in groups.items():
        c = {k: sum(1 for x in rows if x[0] == k) for k in order}
        for k in order:
            totals[k] += c[k]
        flag = "   <-- look here" if c["WRONG"] else ""
        print(f"{group:18} {c['OK']:>4} {c['ASKS']:>6} {c['WRONG']:>7}{flag}")
    print("-" * 40)
    n = sum(totals.values())
    print(f"{'TOTAL':18} {totals['OK']:>4} {totals['ASKS']:>6} "
          f"{totals['WRONG']:>7}   of {n}")

    print(f"\n  did the right thing        {totals['OK']}/{n} "
          f"({totals['OK']/n:.0%})")
    print(f"  asked the user to repeat   {totals['ASKS']}/{n} "
          f"({totals['ASKS']/n:.0%})   annoying, not dangerous")
    print(f"  DID THE WRONG THING        {totals['WRONG']}/{n} "
          f"({totals['WRONG']/n:.0%})   report every one of these")

    if totals["WRONG"]:
        print("\nWRONG — the aid acted, and acted wrongly:")
        for group, rows in groups.items():
            for code, text, why, conf in rows:
                if code == "WRONG":
                    print(f"  [{group}] \"{text}\"")
                    print(f"      {why}  (confidence {conf:.2f})")

    asks = [(g, t, w) for g, rows in groups.items()
            for c, t, w, _ in rows if c == "ASKS" for g in [g]]
    if asks:
        print(f"\nASKS — the aid did nothing and asked again ({len(asks)}):")
        for g, t, w in asks[:12]:
            print(f"  [{g}] \"{t}\"")
            print(f"      {w}")
        if len(asks) > 12:
            print(f"  ... and {len(asks) - 12} more")

    if args.show_ok:
        print("\nOK:")
        for group, rows in groups.items():
            for code, text, why, _ in rows:
                if code == "OK":
                    print(f"  [{group}] \"{text}\" -> {why}")

    print("\nRead the WRONG count first. A model that ASKS often is tiring to "
          "use; a model that acts WRONGLY is the one that has to be fixed "
          "before shipping.")


if __name__ == "__main__":
    main()
