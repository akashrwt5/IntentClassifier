"""Is the model wrong, or is the LABEL wrong?

Before accepting 'more OOS data' as the fix, check the premise: that the 10
failures labelled Default Fallback Intent are genuinely out of scope. If any are real
commands mislabelled OOS, then the model is right, the TEST is wrong, and
training on more of that data would actively break working commands.

Method: for each failure, find the nearest neighbours in train.csv by token
overlap and show what THEY are labelled. A near-identical utterance carrying an
in-scope label is a direct label contradiction.
"""
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path("/home/user/IntentClassifier")
sys.path.insert(0, str(REPO / "packages" / "buildtime"))

FAILURES = [
    ("turn off toshiba", "Cmd.VolumeMute"),
    ("please got to getting a text message at boston", "Cmd.SendMessage"),
    ("can you read last text message", "Cmd.ListenMessage"),
    ("can you phone phone phone phone phone", "Cmd.FindMyPhone"),
    ("please stream, youtube music.", "Cmd.StreamingStart"),
    ("can you stream, youtube music.", "Cmd.StreamingStart"),
    ("please my ears sweat", "Cmd.VolumeDecrease"),
    ("hear it as they got to go.", "Cmd.ListenMessage"),
    ("iphone", "Cmd.FindMyPhone"),
    ("play festival", "Cmd.ListenMessage"),
]

train = list(csv.DictReader(open(REPO / "datasets/en/train.csv",
                                 encoding="utf-8-sig", newline="")))


def toks(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


rows = [(toks(r["text"]), r["text"], r["intent"]) for r in train]

print("=" * 78)
print("LABEL AUDIT — nearest training neighbours of each 'OOS' failure")
print("=" * 78)
for text, fired in FAILURES:
    q = toks(text)
    scored = sorted(
        ((len(q & t) / max(len(q | t), 1), txt, lab) for t, txt, lab in rows),
        reverse=True)[:4]
    print(f"\n  UTTERANCE : {text!r}")
    print(f"  labelled  : Default Fallback Intent      model fired: {fired}")
    for j, (s, txt, lab) in enumerate(scored):
        flag = "  <== in-scope near-duplicate" if (
            s >= 0.5 and lab != "Default Fallback Intent") else ""
        print(f"     {s:.2f}  {lab:<28} {txt!r}{flag}")

# --- systemic check: how often does the SAME token-set carry both labels? ----
print("\n" + "=" * 78)
print("SYSTEMIC — utterances whose normalised token set appears under BOTH "
      "Default Fallback Intent and an in-scope intent")
print("=" * 78)
by_tokens = defaultdict(set)
for t, txt, lab in rows:
    by_tokens[frozenset(t)].add(lab)
contradictions = {k: v for k, v in by_tokens.items()
                  if "Default Fallback Intent" in v and len(v) > 1}
print(f"  {len(contradictions)} token-sets carry Default Fallback Intent AND an in-scope label")
for k, v in list(contradictions.items())[:10]:
    ex = next(txt for t, txt, lab in rows if frozenset(t) == k)
    print(f"    {sorted(v)}  e.g. {ex!r}")

# --- what IS the OOS pool made of? ------------------------------------------
print("\n" + "=" * 78)
print("OOS POOL COMPOSITION — does oos_2.csv contain real commands?")
print("=" * 78)
inscope_tokens = defaultdict(Counter)
for t, txt, lab in rows:
    if lab != "Default Fallback Intent":
        for w in t:
            inscope_tokens[w][lab] += 1

CMD = {"volume", "louder", "quieter", "mute", "unmute", "stream", "streaming",
       "message", "text", "call", "phone", "translate", "transcribe", "battery",
       "reminder", "read", "send", "play"}
pool = list(csv.DictReader(open(REPO / "datasets/en/oos_2.csv",
                                encoding="utf-8-sig", newline="")))
suspicious = [r for r in pool if toks(r["text"]) & CMD]
print(f"  oos_2.csv: {len(pool)} rows, {len(suspicious)} contain a command keyword")
for r in suspicious[:20]:
    print(f"    {sorted(toks(r['text']) & CMD)}  {r['text']!r}")
