#!/usr/bin/env python3
"""Turn 60 intent specifications into the short list a human actually has to read.

WHY
---
The specs are the sole source of truth for every downstream label, so they gate
Stage 1. But "review 60 specs" is not a reviewable task: it is 60 x 6 free-text
fields, and the defects that matter are not inside any one spec -- they are
*between* two of them. Reading them one at a time is exactly the reading order
that cannot find a boundary collision, because the sibling is 40 pages away.

So this does the pairwise work mechanically and hands back only what needs
judgement, in descending order of "the machine is sure":

  MECHANICAL   provably wrong, no judgement needed (an unknown neighbour name,
               a one-directional neighbour link). Fix these first; they are
               cheap and they change what the other sections say.

  UNGUARDED    two intents whose trigger conditions claim overlapping ground,
               where NEITHER spec names the other in a rule field. An overlap
               with a stated boundary is a hard problem; an overlap nobody
               noticed is an unlabelable one, and the architecture doc's Section
               2 argues that is a defect no amount of generation quality repairs.
               Reported in tiers, because "names the sibling" and "mentions its
               subject" are different strengths of guard and the gap between them
               is itself a finding.

  CMD/HELP     every Cmd intent that has a Help counterpart, checked both ways.
               Separated from the general overlap check because it is the
               product's costliest error class: an information question misread
               as a command physically changes a hearing aid, which is why FAR
               outranks accuracy. The pairing is already declared in
               generator_config.yaml, so this only has to verify it.

  INTERNAL     one spec arguing with itself: a trigger_condition and a
               do_not_trigger line that say close to the same thing. Help_Volume
               shipped exactly this defect -- its first bullet sent a phrasing to
               Help and its second called the same phrasing a command -- and it
               survived because nobody reads one spec's two lists against each
               other.

WHAT THIS IS NOT
----------------
It is not a judge. Similarity of wording is evidence that two specs are talking
about the same thing; it is not proof that they conflict, and two intents can
legitimately share vocabulary. Every row below is a QUESTION for a human. The
counts are not a score and a zero is not a pass.

No API, no network, no dependencies beyond PyYAML. Reads only spec files --
never seed files.

    python3 spec_review.py                       # print
    python3 spec_review.py --markdown SPEC_REVIEW.md
    python3 spec_review.py --top 40              # more/fewer unguarded pairs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SPECS = HERE / "intent_specs.yaml"
CONFIG = HERE / "generator_config.yaml"

# Words that carry no discriminating signal in this domain. "hearing", "aid" and
# "user" appear in nearly every spec by construction, so leaving them in makes
# every pair look similar and buries the real overlaps.
STOP = set("""
a an and are as at be been being but by can could do does for from had has have
he her him his how i if in into is it its me my not of on or our out she should
so than that the their them then there these they this those to too up us was
we were what when where which who why will with would you your
user users utterance utterances intent hearing aid aids assistant device devices
request requests asks ask wants want asking needs need
""".split())

# A trigger_conditions bullet and a do_not_trigger bullet at or above this cosine
# are reported as one spec arguing with itself. Set high on purpose: the two
# lists share sentence frames by design ("User asks ..." / "User asks ..."), so a
# moderate score is normal and only a near-restatement is interesting.
INTERNAL_THRESHOLD = 0.45

REVIEWED_INTERNAL = {
    "Help_Accessories": "Accessory/TV volume is Help; hearing-aid volume is Cmd.Volume*.",
    "Cmd.VolumeDecrease": (
        "A request to go as low as possible is still a reduction; only explicit "
        "silence is Cmd.VolumeMute. The product applies a fixed step and sets no level."
    ),
    "Help_CleanCare": (
        "Wax-attributed problems stay here; unexplained device failure is Help_SelfCheck."
    ),
    "Help_AppSettings": "App version is AppSettings; new features/changes are Help_WhatsNew.",
    "Cmd.MemoryChange": (
        "Named memory/program/setting changes are distinct from unnamed "
        "hearing-difficulty complaints."
    ),
    "Cmd.ActivityStep": (
        "Step-goal progress is distinct from configuring/editing the step goal in Help_Activity."
    ),
}


def internal_review_key(spec: dict) -> str:
    payload = {
        "trigger_conditions": list(spec.get("trigger_conditions") or []),
        "do_not_trigger": list(spec.get("do_not_trigger") or []),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# Bound to the exact reviewed trigger/do_not text. If either changes, the approval
# automatically becomes stale and the report requires a fresh review.
REVIEWED_INTERNAL_HASHES = {
    "Help_Accessories": "79e58f2a8273c9d4bbc9989b88e9269207c77b5ddb1de7c4bb57a854bff3bc52",
    "Cmd.VolumeDecrease": "d1aefc6c6669786489876b045be7832846715e14a1cba50d07b33d8b442b0f48",
    "Help_CleanCare": "130f166cf6671ff8326eab129d43ed6cb6bf2fd36d6aabe8bcb6d84f0cbde477",
    "Help_AppSettings": "60656b0c1c3da74d3ba72bb7a49bdd1eecab92697fc0f1ba8a76f79c242d959a",
    "Cmd.MemoryChange": "8644ac4e4d2addc9e4fba261290e2ce985c35fdfd3049b90b5f3ab084e76062d",
    "Cmd.ActivityStep": "bc857620b9c642dae090d7099a1c6e6a87f78cef0786199e1ece6ca6627bfb10",
}


# Pair-level overlap worth a human's time. Not a hard line -- pairs are ranked
# and the top N are shown regardless, so this only controls where the tail is cut.
PAIR_THRESHOLD = 0.20


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", str(text).lower()) if w not in STOP and len(w) > 2]


def tfidf(docs: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """Plain TF-IDF with cosine-ready L2 normalisation. 60 docs; no sklearn needed."""
    n = len(docs)
    df: Counter = Counter()
    for words in docs.values():
        df.update(set(words))
    out = {}
    for key, words in docs.items():
        tf = Counter(words)
        vec = {w: (1 + math.log(c)) * math.log(n / (1 + df[w])) for w, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        out[key] = {w: v / norm for w, v in vec.items()}
    return out


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(w, 0.0) for w, v in a.items())


def load_specs(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"{path.name} not found -- run bootstrap_specs.py first")
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("intents", [])


# A boundary counts as stated only if it appears somewhere the generator reads as
# a rule. business_description is deliberately excluded: describing a neighbouring
# topic in prose is not the same as excluding it.
GUARD_FIELDS = ("do_not_trigger", "boundary_cases", "neighbor_intents")

# ...but these three are NOT interchangeable, and treating them as one set has
# already cost a round of work. `neighbor_intents` renders into the live Stage 1
# prompt as "Neighbour intents (most likely confusions)" and is the field
# bootstrap_specs.py names as the hard-negative sampling source. `do_not_trigger`
# is an exclusion rule. Moving a relationship from the first to the second keeps
# the boundary documented while silently dropping it out of confusion sampling --
# a review that scores them as equivalent reports that trade as a pass.
NEIGHBOUR_FIELD = "neighbor_intents"
PROSE_GUARD_FIELDS = ("do_not_trigger", "boundary_cases")


def is_neighbour(spec: dict, other: str) -> bool:
    return other in (spec.get(NEIGHBOUR_FIELD) or [])


def named_in_prose(spec: dict, other: str) -> list[str]:
    needle = other.lower()
    return [f for f in PROSE_GUARD_FIELDS if needle in " ".join(spec.get(f) or []).lower()]


def mentions(spec: dict, other: str, *, by_name_only: bool) -> list[str]:
    """Where, if anywhere, ``spec`` acknowledges ``other``. Two readings.

    ``by_name_only`` -- the sibling is named outright. Unambiguous, and the only
    form the generator can act on without inferring anything.

    Otherwise the sibling's subject word also counts ("TV or phone volume
    requests" guarding against a volume intent). Closer to how specs are actually
    written, and much too generous on its own.

    Why both readings, historically: on the pre-review specs (2026-08-23) the
    loose reading called 50 of 51 overlapping pairs guarded while the strict
    reading called 30. Reporting only the loose number would have hidden fourteen
    pairs where the sibling was never actually named. Those fourteen have since
    been fixed, so the two readings now agree -- but they agree because the specs
    were repaired, not because the distinction stopped mattering. Keep both.
    """
    needles = [other.lower()]
    if not by_name_only:
        tail = other.split(".")[-1].split("_")[-1]
        needles += [p.lower() for p in re.findall(r"[A-Z][a-z]+|[a-z]+", tail) if len(p) > 2]
    found = []
    for field in GUARD_FIELDS:
        value = spec.get(field)
        blob = " ".join(value).lower() if isinstance(value, list) else str(value or "").lower()
        if any(nd in blob for nd in needles):
            found.append(field)
    return found


def analyse(specs: list[dict], top: int) -> dict:
    names = [s["name"] for s in specs]
    known = set(names)
    by_name = {s["name"]: s for s in specs}
    family = {s["name"]: s.get("intent_family", "?") for s in specs}

    # --- mechanical -----------------------------------------------------
    # Fallback is excluded from the symmetry check in both directions. It is a
    # neighbour of every intent by definition and cannot list 59 of them, so
    # every "X names Fallback, Fallback does not name X" row is correct
    # behaviour.
    #
    # This is not a cosmetic filter. On the pre-review specs (2026-08-23) 53 of
    # 114 asymmetric links were exactly that, and listing them buried the 61 that
    # were real. The 53 have not gone away and are still excluded today; what has
    # changed is that the 61 real ones were fixed. The count is reported in the
    # section text so the exclusion stays visible rather than silent.
    fallback = next((s["name"] for s in specs if s.get("intent_family") == "Fallback"), None)

    unknown_neighbours = []
    asymmetric = []
    fallback_links = 0
    for spec in specs:
        me = spec["name"]
        for nb in spec.get("neighbor_intents") or []:
            if nb not in known:
                unknown_neighbours.append((me, nb))
            elif me not in (by_name[nb].get("neighbor_intents") or []):
                if fallback in (me, nb):
                    fallback_links += 1
                else:
                    asymmetric.append((me, nb))

    self_listed = [s["name"] for s in specs if s["name"] in (s.get("neighbor_intents") or [])]

    # --- pairwise overlap of the CLAIMS -------------------------------
    claim_docs = {
        s["name"]: tokens(
            " ".join(s.get("trigger_conditions") or []) + " " + s.get("business_description", "")
        )
        for s in specs
    }
    vecs = tfidf(claim_docs)

    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            score = cosine(vecs[a], vecs[b])
            if score < PAIR_THRESHOLD:
                continue
            a_named = mentions(by_name[a], b, by_name_only=True)
            b_named = mentions(by_name[b], a, by_name_only=True)
            a_topic = mentions(by_name[a], b, by_name_only=False)
            b_topic = mentions(by_name[b], a, by_name_only=False)
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "score": score,
                    "a_named": a_named,
                    "b_named": b_named,
                    "a_topic": a_topic,
                    "b_topic": b_topic,
                    "named_both": bool(a_named) and bool(b_named),
                    "named_one": bool(a_named) != bool(b_named),
                    "named_neither": not a_named and not b_named,
                    "topic_neither": not a_topic and not b_topic,
                    "same_family": family[a] == family[b],
                }
            )
    pairs.sort(key=lambda p: -p["score"])
    # Tier A: nothing on either side, under either reading. Tier B: sibling never
    # named but its subject appears. Tier C: one side names it, the other is silent.
    tier_a = [p for p in pairs if p["topic_neither"]][:top]
    tier_b = [p for p in pairs if p["named_neither"] and not p["topic_neither"]][:top]
    half = [p for p in pairs if p["named_one"]][:top]

    # --- the Command/Help boundary, checked pair by pair -------------------
    # This is the product's costliest error class, not just another overlap: an
    # information question misread as a command physically changes a hearing aid.
    # generator_config.yaml already names every Cmd intent that has a Help
    # counterpart, so the pairing needs no inference -- only checking.
    ch_pairs = []
    try:
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    except OSError:
        config = {}
    for cmd, help_intent in (config.get("command_help_pairs") or {}).items():
        if cmd not in by_name or help_intent not in by_name:
            ch_pairs.append({"cmd": cmd, "help": help_intent, "missing": True})
            continue
        ch_pairs.append(
            {
                "cmd": cmd,
                "help": help_intent,
                "missing": False,
                "cmd_names_help": mentions(by_name[cmd], help_intent, by_name_only=True),
                "help_names_cmd": mentions(by_name[help_intent], cmd, by_name_only=True),
                "cmd_nb": is_neighbour(by_name[cmd], help_intent),
                "help_nb": is_neighbour(by_name[help_intent], cmd),
                "cmd_prose": named_in_prose(by_name[cmd], help_intent),
                "help_prose": named_in_prose(by_name[help_intent], cmd),
            }
        )

    # --- one spec arguing with itself ------------------------------------
    internal = []
    for spec in specs:
        trig = spec.get("trigger_conditions") or []
        dont = spec.get("do_not_trigger") or []
        docs = {f"t{i}": tokens(t) for i, t in enumerate(trig)}
        docs.update({f"d{i}": tokens(d) for i, d in enumerate(dont)})
        if len(docs) < 2:
            continue
        v = tfidf(docs)
        for i, t in enumerate(trig):
            for j, d in enumerate(dont):
                score = cosine(v[f"t{i}"], v[f"d{j}"])
                internal.append({"intent": spec["name"], "score": score, "trigger": t, "dont": d})
    internal.sort(key=lambda r: -r["score"])
    # Keep the best row per intent, then the strongest few overall. A fixed
    # threshold alone made this section silently empty at 0.55 while the top row
    # sat at 0.52 -- an empty section reads as "checked, nothing there", which is
    # the one thing it must never say when it has not looked hard enough.
    seen_intents: set[str] = set()
    ranked = []
    for row in internal:
        if row["intent"] in seen_intents:
            continue
        seen_intents.add(row["intent"])
        ranked.append(row)
    internal = ranked[:6]

    return {
        "names": names,
        "family": family,
        "unknown_neighbours": unknown_neighbours,
        "asymmetric": asymmetric,
        "fallback_links": fallback_links,
        "self_listed": self_listed,
        "pairs": pairs,
        "tier_a": tier_a,
        "tier_b": tier_b,
        "half": half,
        "internal": internal,
        "ch_pairs": ch_pairs,
        "_specs": specs,
    }


def clip(text: str, width: int = 150) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def render(r: dict, top: int) -> str:
    fam_counts = Counter(r["family"].values())
    named_both = sum(1 for p in r["pairs"] if p["named_both"])

    out = [
        "# Intent specification review",
        "",
        "Generated by `spec_review.py`. Do not edit by hand — re-run it.",
        "",
        f"{len(r['names'])} intents, {len(fam_counts)} families. "
        f"{len(r['pairs'])} intent pairs claim overlapping ground; "
        f"{named_both} of those name each other explicitly.",
        "",
        "**This tool is not a judge.** Shared wording is evidence that two specs are",
        "talking about the same thing, never proof that they conflict. Every row is a",
        "question for you. A zero in any section is not a pass.",
        "",
        "Work top to bottom: Section 1 is provably wrong and changes what the rest says.",
        "",
        "---",
        "",
        "## 1. Mechanical — no judgement needed",
        "",
    ]

    if r["unknown_neighbours"]:
        out += [
            f"### {len(r['unknown_neighbours'])} neighbour name(s) not in the taxonomy",
            "",
            "Stage 3 samples hard negatives from this field. A name that does not exist",
            "produces nothing, silently — the intent simply gets no hard negatives.",
            "",
            "| Intent | Names a neighbour that does not exist |",
            "|---|---|",
        ]
        out += [f"| `{a}` | `{b}` |" for a, b in r["unknown_neighbours"]]
        out += [""]
    else:
        out += ["- ✅ Every `neighbor_intents` name exists in the taxonomy.", ""]

    if r["self_listed"]:
        out += (
            [
                f"### {len(r['self_listed'])} spec(s) list themselves as their own neighbour",
                "",
            ]
            + [f"- `{n}`" for n in r["self_listed"]]
            + [""]
        )

    if r["asymmetric"]:
        same = [p for p in r["asymmetric"] if r["family"][p[0]] == r["family"][p[1]]]
        out += [
            f"### {len(r['asymmetric'])} one-directional neighbour link(s)",
            "",
            "`A` treats `B` as a boundary risk but `B` does not return the favour, so hard",
            "negatives are generated in one direction only. Usually the fix is to add the",
            "back-link; occasionally it is to delete the forward one because the two are not",
            "really neighbours. Decide which, per row.",
            "",
            f"{r['fallback_links']} further link(s) point at the Fallback intent and are NOT",
            "listed: Fallback is a neighbour of everything by definition and cannot name 59",
            "intents back. Reporting those buries the real ones.",
            "",
            f"**{len(same)} of these are within a single family** — read those first. Two",
            "siblings in one family that disagree about whether they are neighbours is the",
            "shape that produces a confident wrong label rather than a Fallback.",
            "",
            "| Names | Does not name back | Same family |",
            "|---|---|:-:|",
        ]
        out += [
            f"| `{a}` | `{b}` | {'**yes**' if r['family'][a] == r['family'][b] else 'no'} |"
            for a, b in sorted(
                r["asymmetric"], key=lambda p: (r["family"][p[0]] != r["family"][p[1]], p[0])
            )
        ]
        out += [""]
    else:
        out += ["- ✅ Every neighbour link is reciprocated.", ""]

    out += [
        "---",
        "",
        "## 2. Overlapping claims — the actual review",
        "",
        "Both specs claim similar ground. What differs between the tiers is whether either",
        "spec does anything about it.",
        "",
        "An overlap with a stated boundary is a hard classification problem. An overlap",
        "nobody noticed is an *unlabelable* one: two intents competing for the same region",
        "of embedding space with no rule to split them. That shows up downstream as",
        "unstable confidence on command intents, which is what inflates False-Accept Rate.",
        "",
        "For each row, one of three things is true. Write down which:",
        "",
        "1. They genuinely do not overlap — the shared words are incidental. Nothing to do.",
        "2. They overlap and the boundary is obvious — add one line to the `do_not_trigger`",
        "   of each, **naming the other intent**. Common case, and cheap.",
        "3. They overlap and you cannot state the boundary in one sentence — that is a",
        "   taxonomy problem, not a spec problem. Escalate it; do not paper over it.",
        "",
        "### 2a. Nothing on either side — highest priority",
        "",
        "Neither spec names the other, and neither so much as mentions its subject in a",
        "rule field. Nothing anywhere separates these two.",
        "",
    ]
    if r["tier_a"]:
        out += ["| # | Intent A | Intent B | Overlap | Same family |", "|---:|---|---|---:|:-:|"]
        for i, p in enumerate(r["tier_a"], 1):
            out.append(
                f"| {i} | `{p['a']}` | `{p['b']}` | {p['score']:.2f} | "
                f"{'**yes**' if p['same_family'] else 'no'} |"
            )
        out += [""]
    else:
        out += ["- ✅ None.", ""]

    out += [
        "### 2b. Subject mentioned, sibling never named",
        "",
        "The topic appears in a rule field, but the sibling intent is never named. A human",
        'reads "TV or phone volume requests" and knows which intent is meant; the generator',
        "gets one spec at a time and cannot resolve it to a sibling it has never been shown.",
        "",
        "Cheapest section to clear: for most of these the boundary already exists in prose",
        "and only needs the intent name added beside it.",
        "",
    ]
    if r["tier_b"]:
        out += ["| # | Intent A | Intent B | Overlap | Same family |", "|---:|---|---|---:|:-:|"]
        for i, p in enumerate(r["tier_b"], 1):
            out.append(
                f"| {i} | `{p['a']}` | `{p['b']}` | {p['score']:.2f} | "
                f"{'**yes**' if p['same_family'] else 'no'} |"
            )
        out += [""]
    else:
        out += ["- ✅ None.", ""]

    out += [
        "### 2c. Named by one side only",
        "",
        "The boundary exists but is not mutual. The silent side has nothing to work from.",
        "",
    ]
    if r["half"]:
        out += ["| Intent A | Intent B | Overlap | Which side is silent |", "|---|---|---:|---|"]
        for p in r["half"]:
            silent = p["b"] if not p["b_named"] else p["a"]
            seen_by = p["a"] if p["a_named"] else p["b"]
            where = ", ".join(p["a_named"] or p["b_named"])
            out.append(
                f"| `{p['a']}` | `{p['b']}` | {p['score']:.2f} | "
                f"`{silent}` is silent; `{seen_by}` names it in _{where}_ |"
            )
        out += [""]
    else:
        out += ["- ✅ Every named overlap is named on both sides.", ""]

    weak = [
        p for p in r["ch_pairs"] if p.get("missing") or not (p.get("cmd_nb") and p.get("help_nb"))
    ]
    out += [
        "---",
        "",
        "## 3. Command ↔ Help boundary",
        "",
        "The product's costliest error class, and the reason False-Accept Rate is the",
        "priority metric: an information question misread as a command physically changes",
        "a hearing aid. `generator_config.yaml` already names every Cmd intent that has a",
        "Help counterpart, so this needs no inference — only checking that both specs know",
        "about each other.",
        "",
        "Precedence rule 4 in the prompt decides these by the action required rather than",
        "the opening words, but the rule is generic. The spec is where the *specific*",
        "boundary lives, and the generator sees one spec at a time.",
        "",
        "**The pass condition is mutual `neighbor_intents` membership, not prose.** A"
        " boundary written into `do_not_trigger` documents the distinction; only"
        ' `neighbor_intents` reaches the prompt as "most likely confusions" and feeds'
        " hard-negative sampling. Restating a neighbour in prose and deleting it from the"
        " list is a downgrade this section used to score as a pass.",
        "",
        f"**{len(weak)} of {len(r['ch_pairs'])} pairs are not mutual neighbours.**",
        "",
    ]
    if weak:
        out += [
            "| Command | Help | Cmd → nb | Help → nb | stated in prose only |",
            "|---|---|:-:|:-:|---|",
        ]
        for p in weak:
            if p.get("missing"):
                out.append(f"| `{p['cmd']}` | `{p['help']}` | — | — | *not in taxonomy* |")
                continue
            prose = []
            if not p["cmd_nb"] and p["cmd_prose"]:
                prose.append(f"`{p['cmd']}` in _{', '.join(p['cmd_prose'])}_")
            if not p["help_nb"] and p["help_prose"]:
                prose.append(f"`{p['help']}` in _{', '.join(p['help_prose'])}_")
            out.append(
                f"| `{p['cmd']}` | `{p['help']}` | "
                f"{'yes' if p['cmd_nb'] else '**no**'} | "
                f"{'yes' if p['help_nb'] else '**no**'} | "
                f"{'; '.join(prose) or '—'} |"
            )
        out += [
            "",
            "> A Help intent that never names its Command counterpart is the one most worth",
            "> fixing. Several Command intents map to a single Help intent (four volume",
            "> commands to `Help_Volume`, eight activity commands to `Help_Activity`), so one",
            "> silent Help spec leaves the whole group without a stated boundary.",
            "",
        ]
    else:
        out += [
            "- ✅ Every Command/Help pair lists its counterpart in `neighbor_intents`, "
            "on both sides.",
            "",
        ]

    out += [
        "---",
        "",
        "## 4. Specs arguing with themselves",
        "",
        "A `trigger_conditions` bullet and a `do_not_trigger` bullet in the **same spec**",
        "that say close to the same thing. `Help_Volume` shipped exactly this: its first",
        "bullet sent a phrasing to Help and its second called that phrasing a command.",
        "It survived review because nobody reads one spec's two lists against each other.",
        "",
    ]
    out += [
        f"The closest such pair in each of the {len(r['internal'])} most affected specs, "
        f"strongest first. A score at or above {INTERNAL_THRESHOLD:.2f} is worth reading "
        "carefully; below that the two lists are merely sharing a sentence frame, which "
        "they do by design. Nothing here is automatically a defect.",
        "",
    ]
    for row in r["internal"]:
        mark = "  ← read this one" if row["score"] >= INTERNAL_THRESHOLD else ""
        out += [
            f"**`{row['intent']}`** — similarity {row['score']:.2f}{mark}",
            "",
            f"- TRIGGER: {clip(row['trigger'])}",
            f"- DO NOT: {clip(row['dont'])}",
            "",
        ]

    reviewed_internal = []
    stale_internal = []
    for row in r["internal"]:
        intent = row["intent"]
        if intent not in REVIEWED_INTERNAL:
            continue
        spec = next((x for x in r["_specs"] if x["name"] == intent), None)
        if spec is not None and REVIEWED_INTERNAL_HASHES.get(intent) == internal_review_key(spec):
            reviewed_internal.append(row)
        else:
            stale_internal.append(row)
    out += ["---", "", "## 5. Review decisions", ""]
    if reviewed_internal:
        out += [
            "The flagged similarities below were reviewed by hand and accepted as valid "
            "boundaries, not contradictions. No spec change is required for them.",
            "",
        ]
        for row in reviewed_internal:
            out.append(f"- ✅ **`{row['intent']}`** — {REVIEWED_INTERNAL[row['intent']]}")
        out += [""]
    if stale_internal:
        out += [
            "**Stale approvals — re-review required.** The approved `trigger_conditions` "
            "or `do_not_trigger` text has changed since sign-off, so the decision below no "
            "longer describes what is in the spec.",
            "",
        ]
        for row in stale_internal:
            out.append(
                f"- ❌ **`{row['intent']}`** — approval hash no longer matches "
                "the current spec text."
            )
        out += [""]
    out += [
        "---",
        "",
        "## 6. Sign-off",
        "",
        "Stage 0 is the gate on Stage 1, and `intent_specs.yaml` still carries",
        "`REQUIRES HUMAN REVIEW`. Cheap pilots against unreviewed specs are fine. A paid",
        "full run is not: the specs are the source of truth for every downstream label,",
        "so a spec defect is not a data-quality problem that later stages can filter out.",
        "",
        "Per family, so the work can be put down and picked up:",
        "",
        "| Family | Intents | Reviewed |",
        "|---|---:|:-:|",
    ]
    for fam, count in sorted(fam_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        out.append(f"| {fam} | {count} | ☐ |")
    out += [
        "",
        "When every box is ticked, drop the `REQUIRES HUMAN REVIEW` note from",
        "`intent_specs.yaml`'s `meta` block and record who signed off, in the same commit.",
        "",
    ]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--specs", type=Path, default=SPECS)
    ap.add_argument("--top", type=int, default=25, help="rows per ranked section")
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args(argv)

    result = analyse(load_specs(args.specs), args.top)
    text = render(result, args.top)
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
        print(f"wrote {args.markdown}")
    else:
        print(text)
    blocking = len(result["unknown_neighbours"]) + len(result["self_listed"])
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
