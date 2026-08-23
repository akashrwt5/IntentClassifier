"""Does this model read sentences, or bags of words? Measure it.

WHY THIS EXISTS
---------------
A user found this, on the shipped model:

    "it's a bit quieter here can you make it louder"
      -> Cmd.VolumeDecrease  0.6937   (runner-up: Cmd.VolumeIncrease 0.2894)

The description says "quieter", the request says "louder", and the model
followed the description. The obvious response is to add "quieter" to a symptom
word list — but that fixes one word and leaves the next unseen comparative
exactly as broken. The real question is whether the model uses sentence
structure at all, and until now nothing in this repository measured that.

Three probes, none of which need a label anyone had to invent:

  1. WORD ORDER      Shuffle the tokens of a real test sentence. The label is
                     unchanged only if the model reads word order; a pure
                     bag-of-words model scores identically on shuffled input.
                     The accuracy DROP is the amount of structure being used.

  2. CONFLICT        Sentences containing a symptom word pointing one way and a
                     request pointing the other, in both orders. The request
                     decides. An AGREEING control (both point the same way)
                     separates "cannot handle two direction words at once" from
                     "cannot resolve a conflict between them".

  3. SHORTCUT        For each direction word, how predictive is its bare
                     presence in the training data? This is not a model
                     measurement — it is what the DATA offers a lazy learner.
                     If "quieter" is 93/145 Cmd.VolumeDecrease, a model that
                     latches onto it is not malfunctioning; it is fitting.

Probe 3 usually explains probes 1 and 2, and it is the cheapest to fix.

    python scripts/structure_probe.py
    python scripts/structure_probe.py --model models/final_student_256/onnx
    python scripts/structure_probe.py --out reports/structure_probe.json

Everything is measured through the SHIPPED ONNX artefact, via predict.Runtime —
the same path the phone takes, not the training pipeline.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from predict import Runtime  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# ---------------------------------------------------------------------------
# Probe 2 vocabulary. DISJOINT from symptom_pairs.py's training lists, which
# use loud/blaring/deafening/harsh/booming and quiet/faint/soft/low/muffled.
# Comparatives are used on purpose: they are the form that failed, and the form
# the training data only ever shows as a REQUEST.
# ---------------------------------------------------------------------------
# The experiment is a controlled swap. Every sentence below has the SAME
# meaning, the SAME frame and the SAME gold label; the only thing that changes
# is which word describes the room — and specifically, what the training data
# taught that word to mean.
#
#   arm A  the describing word is one training uses as a SYMPTOM of this
#          direction  ("faint" appears 35 times, 100% Cmd.VolumeIncrease)
#   arm B  the describing word is one training uses as a REQUEST for the
#          OPPOSITE direction ("quieter" appears 143 times, 64%
#          Cmd.VolumeDecrease)
#
# "it is rather faint in here, can you make it louder" and
# "it is a bit quieter in here, can you make it louder" are the same request in
# the same shape. If A scores well and B does not, the model is not reading the
# sentence — it is reacting to a word's training association. That is the whole
# claim, tested directly, with no interpretation needed.
UP, DOWN = "Cmd.VolumeIncrease", "Cmd.VolumeDecrease"

ARMS = {
    # (describing phrases, request phrases, gold)
    "A_up_symptom_word":  (["rather faint", "on the soft side", "quite muffled"],
                           ["can you make it louder", "could you turn it up",
                            "please raise the volume"], UP),
    "B_up_request_word":  (["a bit quieter", "on the lower side", "quite softer"],
                           ["can you make it louder", "could you turn it up",
                            "please raise the volume"], UP),
    "C_down_symptom_word": (["rather harsh", "quite booming", "on the loud side"],
                            ["can you make it quieter", "could you turn it down",
                             "please lower the volume"], DOWN),
    "D_down_request_word": (["a bit louder", "on the higher side", "quite raised"],
                            ["can you make it quieter", "could you turn it down",
                             "please lower the volume"], DOWN),
}

CONFLICT_FRAMES = [
    "it is {sym} in here {req}",       # description first — the failing shape
    "{req}, it is {sym} in here",      # request first
]

# A bare description, no request clause at all. Policy P4b says the symptom
# implies the action: a room described as quiet means turn it UP.
#
# Added after a two-word test found the same failure with no sentence to parse:
#
#     "it's quiet"  -> Cmd.VolumeDecrease  0.9351   WRONG
#     "it's faint"  -> Cmd.VolumeIncrease  0.8584   right
#
# Same meaning, two words each, opposite answers. This matters because it
# corrects a claim made earlier in this project — that the model is reliable on
# short input and only fails on long sentences. It is not a length problem. The
# word's training association wins whether the sentence is two words or twenty,
# and a two-word probe is the cheapest possible way to see it.
BARE_ARMS = {
    "E_bare_symptom_word": (["quite faint", "rather soft", "quite muffled",
                             "rather harsh", "quite booming", "rather loud"],
                            [UP, UP, UP, DOWN, DOWN, DOWN]),
    "F_bare_request_word": (["quite quiet", "rather lower", "quite softer",
                             "rather louder", "quite higher", "rather sharper"],
                            [UP, UP, UP, DOWN, DOWN, DOWN]),
}
BARE_FRAMES = ["it is {d}", "it is {d} in here", "everything is {d}"]

DIRECTION_WORDS = ["quieter", "louder", "quiet", "loud", "softer", "soft",
                   "faint", "muffled", "harsh", "up", "down", "raise",
                   "lower", "mute", "unmute"]


def assert_disjoint() -> None:
    """Report, per arm, how much of its wording the training data already has.

    The head words are shared on purpose — generalising them is the whole point.
    What must NOT be shared is the surrounding phrasing and the frames, and
    `role_balance.assert_disjoint_from_probe()` enforces that from the other
    side. This function is the visible half: it prints which arms reuse trained
    vocabulary so a reader knows which numbers measure recall.
    """
    import symptom_pairs as sp
    trained = {w.lower() for w in
               sp.LOUD_TRAIN + sp.QUIET_TRAIN + sp.DOWN_TRAIN + sp.UP_TRAIN}
    for arm, (descs, reqs, _) in ARMS.items():
        words = {w for phrase in descs for w in phrase.lower().split()}
        clash = sorted(words & trained)
        if clash:
            print(f"  note: {arm} reuses training words {clash} — its score is "
                  f"partly recall, which is why it is the CONTROL arm.")


# ---------------------------------------------------------------------------
# Probe 1 — word order
# ---------------------------------------------------------------------------
def probe_word_order(rt: Runtime, df: pd.DataFrame, rng: random.Random,
                     min_words: int = 4) -> dict:
    """Accuracy on real sentences vs the same sentences with tokens shuffled.

    Only sentences of >= min_words are used: shuffling "turn it up" cannot
    change much, and including it would dilute the effect toward zero and make
    the model look more structural than it is.
    """
    sub = df[df["text"].str.split().str.len() >= min_words]
    rows = []
    for text, gold in zip(sub["text"], sub["intent"]):
        toks = text.split()
        shuf = toks[:]
        while len(toks) > 1 and shuf == toks:
            rng.shuffle(shuf)
        rows.append((text, " ".join(shuf), " ".join(reversed(toks)), gold))

    def acc(idx: int) -> float:
        ok = sum(rt(r[idx])["intent"] == r[3] for r in rows)
        return ok / len(rows)

    normal, shuffled, reversed_ = acc(0), acc(1), acc(2)
    return dict(n=len(rows), min_words=min_words,
                accuracy_normal=round(normal, 4),
                accuracy_shuffled=round(shuffled, 4),
                accuracy_reversed=round(reversed_, 4),
                order_sensitivity=round(normal - shuffled, 4),
                retained_when_scrambled=round(shuffled / normal, 4)
                if normal else None)


# ---------------------------------------------------------------------------
# Probe 2 — conflicting vs agreeing direction words
# ---------------------------------------------------------------------------
def build_conflict() -> list[dict]:
    rows = []
    for arm, (syms, reqs, gold) in ARMS.items():
        for sym, req, frame in itertools.product(syms, reqs, CONFLICT_FRAMES):
            rows.append(dict(text=" ".join(frame.format(sym=sym, req=req).split()),
                             intent=gold, arm=arm,
                             order="description_first"
                                   if frame.startswith("it is") else "request_first"))
    for arm, (descs, golds) in BARE_ARMS.items():
        for (desc, gold), frame in itertools.product(zip(descs, golds),
                                                     BARE_FRAMES):
            rows.append(dict(text=" ".join(frame.format(d=desc).split()),
                             intent=gold, arm=arm, order="bare"))
    return rows


def probe_conflict(rt: Runtime) -> dict:
    rows = build_conflict()
    dec = {r["text"]: rt(r["text"]) for r in rows}
    out = {}
    for arm in list(ARMS) + list(BARE_ARMS):
        sub = [r for r in rows if r["arm"] == arm]
        ok = [dec[r["text"]]["intent"] == r["intent"] for r in sub]
        acc = [dec[r["text"]]["accepted"] for r in sub]
        d = dict(n=len(sub), accuracy=round(sum(ok) / len(sub), 4),
                 gated_coverage=round(sum(acc) / len(sub), 4),
                 wrong_and_accepted=round(
                     sum(a and not o for a, o in zip(acc, ok)) / len(sub), 4))
        for order in ("description_first", "request_first"):
            s2 = [r for r in sub if r["order"] == order]
            if s2:
                d[f"accuracy_{order}"] = round(
                    sum(dec[r["text"]]["intent"] == r["intent"] for r in s2)
                    / len(s2), 4)
        out[arm] = d

    # The headline: same meaning, same frame, only the describing word's
    # training association differs.
    out["lexical_association_effect"] = dict(
        up=round(out["A_up_symptom_word"]["accuracy"]
                 - out["B_up_request_word"]["accuracy"], 4),
        down=round(out["C_down_symptom_word"]["accuracy"]
                   - out["D_down_request_word"]["accuracy"], 4),
        bare=round(out["E_bare_symptom_word"]["accuracy"]
                   - out["F_bare_request_word"]["accuracy"], 4))
    out["failures"] = [
        dict(text=r["text"], gold=r["intent"], arm=r["arm"],
             pred=dec[r["text"]]["intent"],
             confidence=round(dec[r["text"]]["confidence"], 4),
             accepted=dec[r["text"]]["accepted"])
        for r in rows if dec[r["text"]]["intent"] != r["intent"]
    ][:14]
    return out


# ---------------------------------------------------------------------------
# Probe 3 — what shortcut does the DATA offer?
# ---------------------------------------------------------------------------
def probe_shortcut(train: pd.DataFrame) -> dict:
    low = train["text"].astype(str).str.lower()
    out = {}
    for w in DIRECTION_WORDS:
        hit = train[low.str.contains(rf"\b{w}\b", na=False, regex=True)]
        if len(hit) < 5:
            continue
        c = Counter(hit["intent"])
        top, n = c.most_common(1)[0]
        out[w] = dict(rows=len(hit), top_intent=top,
                      share=round(n / len(hit), 3))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["share"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/final_student_256/onnx")
    ap.add_argument("--quant", default="int8", choices=["int8", "fp32"])
    ap.add_argument("--train", default="train_augmented")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="reports/structure_probe.json")
    args = ap.parse_args()

    rt = Runtime(ROOT / args.model, args.quant)
    rng = random.Random(args.seed)
    assert_disjoint()

    print("\n=== PROBE 1: word order ===")
    p1 = probe_word_order(rt, pd.read_csv(DATA / "test.csv"), rng)
    print(f"  {p1['n']} test sentences of {p1['min_words']}+ words")
    print(f"  normal    {p1['accuracy_normal']:.4f}")
    print(f"  shuffled  {p1['accuracy_shuffled']:.4f}")
    print(f"  reversed  {p1['accuracy_reversed']:.4f}")
    print(f"  order sensitivity (normal - shuffled): {p1['order_sensitivity']:.4f}")
    print(f"  accuracy retained when scrambled:      {p1['retained_when_scrambled']:.1%}")
    print("  Read this as: the share of the model's accuracy that survives with")
    print("  word order destroyed is the share it never needed order for.")

    print("\n=== PROBE 2: same sentence, different describing word ===")
    p2 = probe_conflict(rt)
    for arm in list(ARMS) + list(BARE_ARMS):
        d = p2[arm]
        print(f"  {arm:22} n={d['n']:3}  accuracy {d['accuracy']:.4f}   "
              f"coverage {d['gated_coverage']:.4f}   "
              f"wrong+accepted {d['wrong_and_accepted']:.4f}")
        print(f"  {'':22} description-first {d.get('accuracy_description_first')}"
              f"   request-first {d.get('accuracy_request_first')}")
    e = p2["lexical_association_effect"]
    print(f"\n  lexical association effect:  up {e['up']:+.4f}   "
          f"down {e['down']:+.4f}   bare {e['bare']:+.4f}")
    print("  'bare' is two-word input with no request clause at all. A gap there")
    print("  means this is not a long-sentence problem — the word wins whatever")
    print("  the length.")
    print("  A and B are the same request in the same shape. So is C vs D. The only")
    print("  difference is whether the describing word was taught as a symptom of")
    print("  this direction or as a request for the opposite one. A large positive")
    print("  number means the model is answering on word association, not meaning.")
    if p2["failures"]:
        print("  failures:")
        for f in p2["failures"][:8]:
            print(f"    {f['confidence']:.3f} {'ACCEPTED' if f['accepted'] else 'rejected'}  "
                  f"{f['gold']} -> {f['pred']}  | {f['text'][:54]}")

    print("\n=== PROBE 3: shortcut strength in the training data ===")
    p3 = probe_shortcut(pd.read_csv(DATA / f"{args.train}.csv"))
    print(f"  {'word':10} {'rows':>5}  {'P(top intent | word present)':>28}")
    for w, d in list(p3.items())[:12]:
        print(f"  {w:10} {d['rows']:5}  {d['share']:.3f}  {d['top_intent']}")
    print("  These are what the data offers a model that only looks at words.")
    print("  A word at 0.9+ is a shortcut the training set is actively teaching.")

    out = dict(model=str(args.model), quant=args.quant,
               word_order=p1, conflict=p2, data_shortcuts=p3)
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
