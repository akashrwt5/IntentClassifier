#!/usr/bin/env python3
"""
Generate targeted hard negatives and polarity contrastive pairs for volume intents:
  - Cmd.VolumeUnmute
  - Cmd.VolumeMute
  - Cmd.VolumeIncrease
  - Cmd.VolumeDecrease

This script creates contrastive, high-impact volume utterances to address polarity confusion
(e.g., mute vs unmute, turn up vs turn down, bare "audio" volume) while enforcing leak guards
against all eval sets (locked, stress, ood, typo, oov).

Output: data/en/volume_hard_negatives.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.common import load_rows, token_key, tokenize  # noqa: E402

VOLUME_HARD_NEGATIVES: dict[str, list[str]] = {
    "Cmd.VolumeUnmute": [
        "unmute my hearing aids", "unmute the audio", "turn the mute off", "take off the mute",
        "turn the sound back on", "bring the sound back", "let the sound through",
        "i want to hear again", "restore my hearing aid volume", "turn on the hearing aid audio",
        "unmute sound", "enable the volume again", "start hearing aid sound again",
        "i can't hear anything turn the sound back on", "turn volume on", "unmute",
        "un-mute the hearing aids", "switch the mute off", "cancel mute", "remove the mute",
        "stop the mute", "end mute mode", "unmute the sound please", "bring back sound",
        "turn off the mute mode", "restore volume", "enable audio output again",
        "i need the sound turned back on", "unmute my device", "audio back on",
        "turn sound back on please", "make the sound come back", "reactivate sound",
    ],
    "Cmd.VolumeMute": [
        "mute my hearing aids", "mute the audio", "turn the mute on", "put on mute",
        "turn the sound off", "cut all sound", "silence the hearing aids",
        "make it silent", "i need quiet now", "kill the sound", "stop all audio output",
        "mute sound", "disable the volume", "shut off hearing aid audio",
        "make the hearing aids quiet", "mute", "silence",
        "turn off all sound", "quiet down completely", "mute my device",
        "put the sound on mute", "switch mute on", "enable mute mode",
        "cut off the sound", "silence the audio completely", "hush the hearing aids",
        "turn the volume completely off", "mute all audio now", "zero volume please",
        "stop sound output", "shut off all sound", "quiet the hearing aid audio",
    ],
    "Cmd.VolumeIncrease": [
        "turn the volume up", "increase the hearing aid volume", "make it louder",
        "raise the volume level", "turn up the audio", "boost the volume",
        "the sound is too quiet", "the sound is too soft", "i can't hear well make it louder",
        "give me more volume", "crank up the volume", "volume up please",
        "louder please", "strengthen the hearing aid audio", "raise hearing aid volume",
        "boost hearing aid volume", "i need more volume", "volume is too low",
        "the audio level is too soft", "turn up the hearing aid sound", "make the audio louder",
        "step up the volume", "lift the volume level", "volume up a bit",
        "can you turn the volume up", "please increase the sound level", "more volume please",
        "make my hearing aids louder", "turn up sound level", "increase audio output",
    ],
    "Cmd.VolumeDecrease": [
        "turn the volume down", "decrease the hearing aid volume", "make it quieter",
        "lower the volume level", "turn down the audio", "reduce the volume",
        "the sound is too loud", "the sound is hurting my ears", "make it softer",
        "give me less volume", "turn down the volume a bit", "volume down please",
        "quieter please", "soften the hearing aid audio", "lower hearing aid volume",
        "reduce hearing aid volume", "i need less volume", "volume is too high",
        "the audio level is too loud", "turn down the hearing aid sound", "make the audio quieter",
        "step down the volume", "drop the volume level", "volume down a bit",
        "can you turn the volume down", "please decrease the sound level", "less volume please",
        "make my hearing aids quieter", "turn down sound level", "reduce audio output",
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=config.DATA / "en" / "volume_polarity_hard_negatives.csv",
    )
    args = ap.parse_args()

    train_rows = load_rows(config.TRAIN_CSV)
    seen = {token_key(t) for t, _ in train_rows}

    # Add all eval sets to leak guard
    for p in (
        config.LOCKED_TEST,
        config.STRESS_TEST,
        config.OOD_TEST,
        config.OOV_TEST,
        config.DATA / "eval" / "typo_test_en.csv",
    ):
        if p.exists():
            seen |= {token_key(t) for t, _ in load_rows(p)}

    kept, dropped = [], []
    for intent, phrases in VOLUME_HARD_NEGATIVES.items():
        for text in phrases:
            tk = token_key(text)
            if tk in seen:
                dropped.append((text, intent, "already in train or eval set"))
            else:
                kept.append((text, intent))
                seen.add(tk)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "intent"])
        writer.writerows(kept)

    print(f"Kept    : {len(kept)} volume hard negative rows")
    if dropped:
        print(f"Dropped : {len(dropped)} rows (leak guard)")
        for t, i, why in dropped[:10]:
            print(f"   {t!r:<40} -> {i:<20} ({why})")

    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
