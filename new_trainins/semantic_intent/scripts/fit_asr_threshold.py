"""Fit the recognizer-confidence threshold — signal 6 of the safety gate.

WHAT THIS FIXES
---------------
"and push it down for dramatics", spoken while the user was talking to someone
else, is ordinary English that sits genuinely close to a volume command. Every
signal already in the gate reads TEXT, and as text that sentence is fine:

  * it is not the reject class — it looks like a real request
  * it is not out of distribution — OOD AUROC on this kind of input is 0.70,
    against 0.92 on real OOD, because it is not unusual English
  * confidence and margin are computed over 57 classes that all assume the user
    was addressing the device

The signal that separates it is not in the text at all. It is upstream, in the
recognizer: an utterance the ASR itself was unsure it heard correctly. That
number exists in every ASR API and this pipeline currently throws it away.

WHY THERE IS NO DEFAULT THRESHOLD IN THIS FILE
----------------------------------------------
Confidence scales are not comparable between recognizers. Android
SpeechRecognizer returns a 0..1 score whose distribution depends on the model;
Whisper exposes avg_logprob (negative, unbounded below) and no_speech_prob;
Vosk returns a per-word average. A number that rejects 5% of good input on one
of these can reject 60% on another. Picking a plausible-looking 0.6 here would
produce a gate that silently drops requests on your device and looks fine in
this repository.

So: you record, this fits. The protocol is in
reports/asr_confidence_protocol.md and takes about an hour.

    python scripts/fit_asr_threshold.py --data data/asr_samples.csv
    python scripts/fit_asr_threshold.py --data data/asr_samples.csv \\
           --max-command-loss 0.02 --apply models/final_student_256/onnx

Input CSV needs three columns:
    text            what the recognizer produced
    asr_confidence  the recognizer's own score for that utterance
    is_command      1 if the person was addressing the device, 0 if this was
                    background speech, another conversation, or the TV

is_command is a judgement about INTENT TO ADDRESS THE DEVICE, not about whether
the text looks like a command. "and push it down for dramatics" is is_command=0
even though it reads like one. That row is the entire point of the exercise; a
dataset without rows like it will fit a threshold that does nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = ["text", "asr_confidence", "is_command"]
MIN_PER_CLASS = 25


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Rank-based AUROC. Ties get average rank, which matters here because
    some recognizers quantise confidence to two decimal places."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = np.sort(scores)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def sweep(df: pd.DataFrame) -> pd.DataFrame:
    conf = df["asr_confidence"].to_numpy(dtype=float)
    is_cmd = df["is_command"].to_numpy().astype(bool)
    rows = []
    for t in np.unique(conf):
        passes = conf >= t
        rows.append(dict(
            threshold=float(t),
            commands_kept=float(passes[is_cmd].mean()),
            commands_lost=float(1 - passes[is_cmd].mean()),
            noise_blocked=float(1 - passes[~is_cmd].mean()),
            noise_passed=float(passes[~is_cmd].mean()),
        ))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/asr_samples.csv")
    ap.add_argument("--max-command-loss", type=float, default=0.02,
                    help="largest share of genuine commands you accept losing. "
                         "This is a product decision, not a statistical one: "
                         "a lost command is a button the user pressed that did "
                         "nothing, and they will press it again.")
    ap.add_argument("--apply", default=None,
                    help="onnx dir whose runtime_config.json to update")
    ap.add_argument("--out", default="reports/asr_threshold.json")
    args = ap.parse_args()

    path = ROOT / args.data
    if not path.exists():
        raise SystemExit(
            f"{path} not found.\n"
            "This script cannot invent the data — recognizer confidence scales\n"
            "are device-specific. See reports/asr_confidence_protocol.md for\n"
            "how to record it (about an hour).")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns: {missing}")
    df = df.dropna(subset=REQUIRED)
    df["is_command"] = df["is_command"].astype(int)

    n_cmd = int(df["is_command"].sum())
    n_noise = len(df) - n_cmd
    print(f"{len(df)} utterances: {n_cmd} commands, {n_noise} not-addressed")
    if min(n_cmd, n_noise) < MIN_PER_CLASS:
        raise SystemExit(
            f"need at least {MIN_PER_CLASS} of each class to fit anything "
            f"meaningful; got {n_cmd}/{n_noise}. A threshold fitted on fewer "
            f"rows will move with the next handful you record.")

    conf = df["asr_confidence"].to_numpy(dtype=float)
    is_cmd = df["is_command"].to_numpy().astype(bool)
    a = auroc(conf, is_cmd)
    print(f"AUROC of asr_confidence vs is_command: {a:.3f}")
    print(f"  commands      median {np.median(conf[is_cmd]):.3f}  "
          f"IQR {np.percentile(conf[is_cmd],25):.3f}-{np.percentile(conf[is_cmd],75):.3f}")
    print(f"  not-addressed median {np.median(conf[~is_cmd]):.3f}  "
          f"IQR {np.percentile(conf[~is_cmd],25):.3f}-{np.percentile(conf[~is_cmd],75):.3f}")

    if not np.isfinite(a) or a < 0.60:
        print("\nSTOP. This recognizer's confidence does not separate the two "
              "classes (AUROC < 0.60).")
        print("That is a real finding, not a failure of this script — it means "
              "the score you exported is not informative about whether the "
              "person was addressing the device.")
        print("Before fitting anything, check:")
        print("  * are you reading per-UTTERANCE confidence, or a per-word "
              "average? per-word averages are usually flat")
        print("  * does your API expose a separate no-speech / endpointing "
              "score? that is often the informative one")
        print("  * if neither: push-to-talk removes this problem entirely and "
              "costs no model work at all")
        Path(ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / args.out).write_text(json.dumps(
            dict(auroc=a, fitted=False,
                 reason="asr_confidence does not separate the classes"), indent=2))
        return

    curve = sweep(df)
    ok = curve[curve["commands_lost"] <= args.max_command_loss]
    if ok.empty:
        raise SystemExit(
            f"no threshold loses <= {args.max_command_loss:.1%} of commands. "
            f"The distributions overlap too much; raise --max-command-loss "
            f"only if you have decided that trade is acceptable.")
    best = ok.loc[ok["noise_blocked"].idxmax()]

    print(f"\nchosen threshold {best['threshold']:.4f}")
    print(f"  keeps  {best['commands_kept']:.1%} of genuine commands")
    print(f"  blocks {best['noise_blocked']:.1%} of not-addressed speech")
    print(f"  lets through {best['noise_passed']:.1%} of not-addressed speech, "
          f"which then still has to clear the other five signals")

    print("\ntrade-off around the chosen point:")
    near = curve.iloc[(curve["threshold"] - best["threshold"]).abs()
                      .argsort()[:9]].sort_values("threshold")
    for _, r in near.iterrows():
        mark = " <-" if r["threshold"] == best["threshold"] else ""
        print(f"  t={r['threshold']:.3f}  keep {r['commands_kept']:.3f}  "
              f"block {r['noise_blocked']:.3f}{mark}")

    result = dict(auroc=round(a, 4), fitted=True,
                  threshold=round(float(best["threshold"]), 4),
                  commands_kept=round(float(best["commands_kept"]), 4),
                  noise_blocked=round(float(best["noise_blocked"]), 4),
                  max_command_loss=args.max_command_loss,
                  n_commands=n_cmd, n_not_addressed=n_noise,
                  source=str(args.data))
    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.out).write_text(json.dumps(result, indent=2))
    curve.to_csv(ROOT / "reports/asr_threshold_curve.csv", index=False)
    print(f"\nwrote {args.out} and reports/asr_threshold_curve.csv")

    if args.apply:
        cfg_path = ROOT / args.apply / "runtime_config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["asr"] = dict(enabled=True,
                          min_confidence=result["threshold"],
                          fitted_on=str(args.data),
                          n_samples=len(df), auroc=result["auroc"],
                          note="Device-specific. Refit if the recognizer, its "
                               "model version, or the microphone changes.")
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"updated {cfg_path}")


if __name__ == "__main__":
    main()
