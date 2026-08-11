#!/usr/bin/env python3
"""
Build the eval set that measures GENERALISATION TO UNSEEN WORDS.

None of the existing eval sets test this directly. Stress covers novel phrasing
but only 17% of its rows contain a word outside the vocabulary, so a model that
generalises well and one that merely memorised the corpus score about the same.

Every row here contains at least one word the current 1,982-token vocabulary
does NOT have, while the intent stays unambiguous to a human. That isolates one
question: does the model understand the word, or only recognise it?

    "elevate the volume"    <- elevate is OOV; a human has no doubt
    "heighten the sound"    <- ditto

PROVENANCE — READ THIS
----------------------
These phrases are AUTHORED, not collected from users. They are a diagnostic
probe, not a substitute for real data, and they must never be used for training
or for headline accuracy claims. The script verifies every row is genuinely OOV
against the vocabulary you point it at and drops any that is not, so the file
stays honest as the vocabulary changes.

Usage:
    python scripts/build_oov_testset.py
    python scripts/build_oov_testset.py --vocab models/en/vocab_base_s1.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.common import load_rows, load_vocab, token_key, tokenize  # noqa: E402

# (utterance, intent). Each is meant to carry at least one out-of-vocabulary
# word while remaining obvious to a person.
PHRASES: list[tuple[str, str]] = [
    # --- volume up -------------------------------------------------------
    ("elevate the volume", "Cmd.VolumeIncrease"),
    ("heighten the sound", "Cmd.VolumeIncrease"),
    ("intensify the audio", "Cmd.VolumeIncrease"),
    ("magnify the sound please", "Cmd.VolumeIncrease"),
    ("strengthen the audio level", "Cmd.VolumeIncrease"),
    ("augment the volume a bit", "Cmd.VolumeIncrease"),
    ("escalate the sound level", "Cmd.VolumeIncrease"),
    ("i need the sound elevated", "Cmd.VolumeIncrease"),
    ("can you heighten what i hear", "Cmd.VolumeIncrease"),
    ("please magnify the speech", "Cmd.VolumeIncrease"),
    # --- volume down -----------------------------------------------------
    ("diminish the volume", "Cmd.VolumeDecrease"),
    ("dampen the sound", "Cmd.VolumeDecrease"),
    ("subdue the audio", "Cmd.VolumeDecrease"),
    ("attenuate the volume", "Cmd.VolumeDecrease"),
    ("lessen the sound level", "Cmd.VolumeDecrease"),
    ("can you dampen that noise", "Cmd.VolumeDecrease"),
    ("i want the volume diminished", "Cmd.VolumeDecrease"),
    ("subdue what i am hearing", "Cmd.VolumeDecrease"),
    # --- mute ------------------------------------------------------------
    ("deaden the sound completely", "Cmd.VolumeMute"),
    ("quell the audio", "Cmd.VolumeMute"),
    ("i want the sound deadened", "Cmd.VolumeMute"),
    # --- unmute ----------------------------------------------------------
    ("reinstate the sound", "Cmd.VolumeUnmute"),
    ("please reinstate my audio", "Cmd.VolumeUnmute"),
    # --- memory ----------------------------------------------------------
    ("swap to the restaurant setting", "Cmd.MemoryChange"),
    ("toggle to my outdoor program", "Cmd.MemoryChange"),
    ("transition to the car memory", "Cmd.MemoryChange"),
    ("swap my hearing program", "Cmd.MemoryChange"),
    ("toggle the listening mode", "Cmd.MemoryChange"),
    # --- find phone ------------------------------------------------------
    ("trace my phone", "Cmd.FindMyPhone"),
    ("pinpoint where my phone is", "Cmd.FindMyPhone"),
    ("can you trace my mobile", "Cmd.FindMyPhone"),
    # --- streaming -------------------------------------------------------
    ("cast the tv audio to my aids", "Cmd.StreamingStart"),
    ("cast sound from the television", "Cmd.StreamingStart"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=Path, default=config.MODELS / "vocab_unkaug_s1.json")
    ap.add_argument("--out", type=Path, default=config.DATA / "eval" / "oov_test_en.csv")
    args = ap.parse_args()

    vocab, mode = load_vocab(args.vocab)
    if mode != "word":
        print(f"! vocab is '{mode}'; OOV is defined against a word vocabulary")

    labels = set()
    for p in (config.TRAIN_CSV,):
        labels |= {g for _, g in load_rows(p)}

    # never allow a probe phrase to collide with training or another eval set
    seen = {token_key(t) for t, _ in load_rows(config.TRAIN_CSV)}
    for p in (config.LOCKED_TEST, config.STRESS_TEST, config.OOD_TEST):
        if p.exists():
            seen |= {token_key(t) for t, _ in load_rows(p)}

    kept, dropped = [], []
    for text, intent in PHRASES:
        if intent not in labels:
            dropped.append((text, f"unknown intent {intent}"))
            continue
        if token_key(text) in seen:
            dropped.append((text, "already in training or an eval set"))
            continue
        oov = [w for w in tokenize(text) if w not in vocab]
        if not oov:
            dropped.append((text, "no OOV word — does not test anything"))
            continue
        kept.append((text, intent, " ".join(oov)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "intent", "oov_words"])
        w.writerows(kept)

    by_intent: dict[str, int] = {}
    for _, i, _ in kept:
        by_intent[i] = by_intent.get(i, 0) + 1

    print(f"vocabulary : {args.vocab.name} ({len(vocab)} tokens)")
    print(f"kept       : {len(kept)} rows, all containing >=1 OOV word")
    for i, n in sorted(by_intent.items()):
        print(f"   {i:<24} {n}")
    if dropped:
        print(f"dropped    : {len(dropped)}")
        for t, why in dropped:
            print(f"   {t[:44]!r:<48} {why}")
    print(f"\nwrote {args.out}")
    print("\nPROBE SET — authored, not user data. Never train on it; never quote")
    print("it as headline accuracy. It answers one question only: does the model")
    print("understand a word it has not seen?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
