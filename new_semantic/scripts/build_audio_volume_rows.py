#!/usr/bin/env python3
"""
Teach the model that a bare "audio" means the hearing aid's own sound.

THE PRODUCT RULE (from the product owner, 2026-08-10)
-----------------------------------------------------
    "make the audio stop"      -> Cmd.VolumeMute
    "make the streaming stop"  -> Cmd.StreamingStop
    in a streaming context, "mute" means stop the stream

So `audio` on its own refers to what the aids are producing. Streaming is marked
by an explicit token: stream / streaming / phone / tv / bluetooth / from ...

WHAT THE DATA CURRENTLY TEACHES
-------------------------------
The opposite, and by a wide margin:

    training rows containing 'audio'      volume      streaming
      total                                   12            724
      WITHOUT a streaming marker              12             83

The model learned the 60:1 prior it was shown. That is not a model defect — on
the stress set 23.2% of volume rows use 'audio' while training uses it for
volume in 0.2% of them, so the eval set was written to one convention and the
training data to another.

Note the streaming side is NOT the problem: 88.5% of its 'audio' rows also carry
an explicit marker, so they stay unambiguous under the rule above and are left
alone. Only the volume side is missing.

WHAT THIS WRITES
----------------
Volume-intent rows phrased with 'audio' and NO streaming marker, so the word
stops being a streaming signal on its own. Output goes to its own CSV for review
— it is NOT merged into train.csv automatically.

Every row is checked to be:
  * free of streaming markers          (else it would teach the opposite)
  * absent from the training corpus    (no duplicates, on token_key)
  * absent from EVERY eval set         (no leak, on token_key)

Usage:
    python scripts/build_audio_volume_rows.py
    python scripts/build_audio_volume_rows.py --out data/en/audio_volume.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.common import load_rows, token_key, tokenize  # noqa: E402

# Anything here means the utterance is about a STREAM, not the aids' own output.
STREAM_MARKERS = {
    "stream", "streaming", "streamed", "streams", "phone", "tv", "television",
    "bluetooth", "music", "podcast", "movie", "video", "call", "meeting",
    "laptop", "computer", "tablet", "ipad", "iphone", "connect", "connected",
    "connection", "disconnect", "route", "redirect", "beam", "cast", "from",
}

PHRASES: dict[str, list[str]] = {
    "Cmd.VolumeIncrease": [
        "turn the audio up", "turn up the audio", "raise the audio",
        "raise the audio level", "increase the audio", "increase the audio level",
        "boost the audio", "boost the audio a bit", "make the audio louder",
        "the audio is too quiet", "the audio is too soft", "the audio is too low",
        "i need more audio", "i need louder audio", "give me more audio",
        "bring the audio up", "bring the audio up a little", "louder audio please",
        "make the audio stronger", "strengthen the audio", "amplify the audio",
        "the audio needs to be louder", "can you raise the audio",
        "can you turn the audio up", "please increase the audio",
        "put the audio up", "push the audio up a bit", "crank the audio up",
        "audio up", "more audio please", "the audio is barely there",
        "i can hardly hear the audio", "the audio is faint",
        "lift the audio a little", "step the audio up",
    ],
    "Cmd.VolumeDecrease": [
        "turn the audio down", "turn down the audio", "lower the audio",
        "lower the audio level", "decrease the audio", "reduce the audio",
        "reduce the audio level", "make the audio quieter", "make the audio softer",
        "the audio is too loud", "the audio is too strong", "the audio is too harsh",
        "i need less audio", "less audio please", "bring the audio down",
        "bring the audio down a bit", "quieter audio please", "soften the audio",
        "dim the audio a little", "tone the audio down", "ease the audio down",
        "can you lower the audio", "can you turn the audio down",
        "please reduce the audio", "put the audio down", "audio down",
        "the audio is uncomfortable", "the audio is hurting my ears",
        "take the audio down a notch", "drop the audio a little",
        "the audio needs to be softer", "turn the audio down slightly",
    ],
    "Cmd.VolumeMute": [
        "make the audio stop", "stop the audio", "cut the audio",
        "cut off the audio", "mute the audio", "silence the audio",
        "kill all audio", "no audio please", "i want no audio",
        "turn the audio off", "turn off the audio", "shut the audio off",
        "shut off the audio", "audio off", "stop all the audio",
        "i need silence not audio", "end the audio", "halt the audio",
        "can you mute the audio", "can you stop the audio",
        "please turn the audio off", "give me zero audio",
        "make there be no audio", "quiet the audio completely",
        "silence all audio now", "block the audio", "the audio should stop",
        "no more audio", "drop the audio entirely",
    ],
    "Cmd.VolumeUnmute": [
        "bring the audio back", "start the audio again", "turn the audio on",
        "turn on the audio", "audio on", "i want audio again",
        "give me the audio back", "restore the audio", "enable the audio",
        "unmute the audio", "put the audio back on", "let the audio through",
        "i need the audio back", "switch the audio on",
        "can you turn the audio on", "can you unmute the audio",
        "please bring the audio back", "the audio should come back",
        "audio back on please", "resume the audio", "reactivate the audio",
        "i want to hear audio again", "turn the audio back on",
        "let me have audio again", "bring back my audio",
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=config.DATA / "en" / "audio_volume_rows.csv")
    args = ap.parse_args()

    train = load_rows(config.TRAIN_CSV)
    labels = {g for _, g in train}
    seen = {token_key(t) for t, _ in train}
    for p in (config.LOCKED_TEST, config.STRESS_TEST, config.OOD_TEST,
              config.DATA / "eval" / "oov_test_en.csv"):
        if p.exists():
            seen |= {token_key(t) for t, _ in load_rows(p)}

    kept, dropped = [], []
    for intent, phrases in PHRASES.items():
        if intent not in labels:
            dropped += [(t, f"unknown intent {intent}") for t in phrases]
            continue
        for text in phrases:
            toks = set(tokenize(text))
            bad = toks & STREAM_MARKERS
            if bad:
                dropped.append((text, f"streaming marker {sorted(bad)}"))
            elif token_key(text) in seen:
                dropped.append((text, "already in training or an eval set"))
            else:
                kept.append((text, intent))
                seen.add(token_key(text))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "intent"])
        w.writerows(kept)

    by = collections.Counter(i for _, i in kept)
    print(f"kept    : {len(kept)} rows")
    for i, n in sorted(by.items()):
        print(f"   {i:<24}{n:>4}")
    if dropped:
        print(f"dropped : {len(dropped)}")
        for t, why in dropped[:12]:
            print(f"   {t[:40]!r:<44}{why}")

    bare_v = sum(1 for t, g in train
                 if g.startswith("Cmd.Volume") and "audio" in tokenize(t)
                 and not (set(tokenize(t)) & STREAM_MARKERS))
    bare_s = sum(1 for t, g in train
                 if g.startswith("Cmd.Streaming") and "audio" in tokenize(t)
                 and not (set(tokenize(t)) & STREAM_MARKERS))
    print(f"\nbare 'audio' rows (no streaming marker), before -> after:")
    print(f"   volume     {bare_v:>4}  ->  {bare_v + len(kept):>4}")
    print(f"   streaming  {bare_s:>4}  ->  {bare_s:>4}   (untouched — 88.5% of the")
    print(f"                            streaming 'audio' rows carry a marker)")
    print(f"\nwrote {args.out}")
    print("\nNOT merged into train.csv. Review it, then merge deliberately —")
    print("this is authored text, and the repo tracks that as `synthetic_text`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
