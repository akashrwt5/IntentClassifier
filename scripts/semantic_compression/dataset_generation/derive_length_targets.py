#!/usr/bin/env python3
"""Derive a per-intent short-utterance target from deployed speech. No API, no guessing.

WHY
---
One number governs utterance length for all 33 Help intents: ``min_short: 0.28``
in the help profile. Measured against what users actually say, that number is
wrong nearly everywhere and wrong in both directions:

    Help_Translate      97% of deployed rows are <= 7 words   -- ask was 28%
    Help_IntelliVoice   96%                                   -- ask was 28%
    Help_Volume         36%                                   -- ask was 28%
    Help_Tinnitus       26%                                   -- ask was 28%

The median Help intent sits at 68%. A single floor cannot serve a range that
runs from 26% to 97%, and the generated corpus shows it: measured across five
runs, generation lands 28 to 55 points below deployed on short-utterance share.

WHY THIS IS WORTH CHANGING, GIVEN IT WAS DELIBERATE
---------------------------------------------------
The pre-flight recorded a deliberate choice: "quotas are not meant to chase seed
proportions; hard/long material is over-weighted by design." That choice was made
without the measurement below.

Scoring the shipped encoder on ``dev_hard`` by utterance length:

    1-4 words     296 rows    79.1% accuracy
    5-7 words     381 rows    85.3%
    8-12 words    133 rows    88.7%

**Short utterances are where the model fails**, by nearly ten points. Over-weighting
long material spends the generation budget on the part the model already handles.
The tilt toward hard material is right; tying it to LENGTH is what turned out to be
backwards.

WHAT THIS SCRIPT DOES
---------------------
For every intent it measures the deployed share of utterances at or under the
profile's short cap and writes that share as the target. Four intents have no
deployed rows -- the three EdgeMode commands and Help_Activity -- and take their
family's median instead, recorded as such rather than silently defaulted.

The target is the deployed share, not a discounted one. Floors are approximately
honoured, not met: generation lands below the ask, so asking for reality is what
lands near reality. There is no fudge factor to explain or maintain.

    python3 derive_length_targets.py            # print the table
    python3 derive_length_targets.py --write    # write length_targets.yaml
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Ceiling on any target. Some intents are ~100% short in deployment
# (reminders.complete 99%, Help_Translate 97%), and asking for that would leave
# no room for the longer, harder material this corpus exists to ADD -- the
# pre-flight's "extend beyond the seeds" principle, which is right. The floor
# fixes the gross mismatch; the ceiling keeps the tilt.
MAX_TARGET = 0.85

HERE = Path(__file__).resolve().parent
SPECS = HERE / "intent_specs.yaml"
CONFIG = HERE / "generator_config.yaml"
OUT = HERE / "length_targets.yaml"

# --- Outside this directory ---------------------------------------------
DEPLOYED = HERE.parents[2] / "language_packs" / "en" / "train.csv"
# ------------------------------------------------------------------------


def profile_for(config: dict, intent: str) -> tuple[str, dict]:
    """Longest matching prefix wins; an exact intent name beats any prefix."""
    quotas = (config.get("generation") or {}).get("quotas") or {}
    assign = quotas.get("assign") or {}
    profiles = quotas.get("profiles") or {}
    best, best_len = None, -1
    for key, name in assign.items():
        if key == intent:
            best, best_len = name, 10**6
        elif intent.startswith(key) and len(key) > best_len:
            best, best_len = name, len(key)
    return best or "", profiles.get(best or "", {})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    specs = yaml.safe_load(SPECS.read_text(encoding="utf-8"))["intents"]
    family = {s["name"]: s.get("intent_family", "?") for s in specs}

    lengths: dict[str, list[int]] = defaultdict(list)
    with DEPLOYED.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            lengths[row["intent"].strip()].append(len(row["text"].split()))

    measured: dict[str, dict] = {}
    for spec in specs:
        name = spec["name"]
        profile_name, profile = profile_for(config, name)
        cap = int(profile.get("short_max_words", 4)) if profile else 4
        rows = lengths.get(name)
        if not rows:
            continue
        share = sum(1 for w in rows if w <= cap) / len(rows)
        measured[name] = {
            "profile": profile_name,
            "short_max_words": cap,
            "deployed_rows": len(rows),
            "deployed_short_share": round(share, 2),
            "target_short_share": round(min(share, MAX_TARGET), 2),
            "deployed_mean_words": round(statistics.mean(rows), 1),
            "source": "measured" + (f" (capped from {share:.0%})" if share > MAX_TARGET else ""),
        }

    # Intents with no deployed rows take their family's median rather than a default.
    by_family: dict[str, list[float]] = defaultdict(list)
    for name, rec in measured.items():
        by_family[family.get(name, "?")].append(rec["target_short_share"])

    inferred = []
    for spec in specs:
        name = spec["name"]
        if name in measured:
            continue
        fam = family.get(name, "?")
        peers = by_family.get(fam)
        profile_name, profile = profile_for(config, name)
        cap = int(profile.get("short_max_words", 4)) if profile else 4
        # Fall back one level at a time and record which level answered. The three
        # EdgeMode commands are a whole family with no deployed rows -- the feature
        # postdates the export -- so the family median has nothing to average and
        # the profile median stands in. A default written without saying so is how
        # a guessed number ends up looking measured.
        if peers:
            target = round(statistics.median(peers), 2)
            source = f"family median ({fam}, {len(peers)} peers)"
        else:
            same_profile = [
                r["target_short_share"]
                for r in measured.values()
                if r["profile"] == profile_name and r["deployed_rows"]
            ]
            if not same_profile:
                print(
                    f"WARNING: {name} has no deployed rows, no family peers and no "
                    f"profile peers; skipped"
                )
                continue
            target = round(statistics.median(same_profile), 2)
            source = f"profile median ({profile_name}, {len(same_profile)} intents)"
        measured[name] = {
            "profile": profile_name,
            "short_max_words": cap,
            "deployed_rows": 0,
            "deployed_short_share": None,
            "target_short_share": min(target, MAX_TARGET),
            "deployed_mean_words": None,
            "source": source,
        }
        inferred.append(name)

    ordered = dict(sorted(measured.items()))
    payload = {
        "_note": (
            "Per-intent short-utterance targets, derived from language_packs/en/train.csv "
            "by derive_length_targets.py. Do not edit by hand -- re-run the script. "
            "target_short_share is the DEPLOYED share of utterances at or under "
            "short_max_words for that intent's quota profile; floors are approximately "
            "honoured, so asking for reality is what lands near reality."
        ),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "language_packs/en/train.csv",
        "intents": ordered,
    }

    caps = sorted({r["short_max_words"] for r in ordered.values()})
    print(f"{'intent':<28}{'profile':<10}{'cap':>4}{'rows':>7}{'target':>9}  source")
    for name, r in ordered.items():
        src = "" if r["source"] == "measured" else f"  <- {r['source']}"
        print(
            f"{name:<28}{r['profile']:<10}{r['short_max_words']:>4}{r['deployed_rows']:>7}"
            f"{r['target_short_share']:>8.0%}{src}"
        )

    print(f"\n{len(ordered)} intents; {len(inferred)} inferred from family median: {inferred}")
    for cap in caps:
        vals = [r["target_short_share"] for r in ordered.values() if r["short_max_words"] == cap]
        print(
            f"  cap <= {cap} words: {len(vals):>2} intents, "
            f"median target {statistics.median(vals):.0%}, "
            f"range {min(vals):.0%} - {max(vals):.0%}"
        )

    if args.write:
        OUT.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100),
            encoding="utf-8",
        )
        print(f"\nwrote {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
