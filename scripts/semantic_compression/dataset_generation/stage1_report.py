"""Score generated output automatically, so prompt tuning is not eyeballing.

Reading 7,680 utterances by hand is neither practical nor reliable -- by the
third intent a reviewer is skimming, and skimming is exactly how a subtle
distribution problem survives. This computes the numbers that actually decide
whether the generator is doing its job, so a tuning loop becomes: change one
thing, re-run two intents, compare a table.

The metrics, in rough order of how much they matter:

1. **Diversity gain vs seed.** The whole premise of this project is that the
   legacy data is permutation-heavy and the generated data should be
   semantically wider. If generated diversity is not clearly above the seed's,
   the prompt is not working, and no other number matters.
2. **Boundary leakage.** For each generated utterance, which intent's centroid
   is it actually nearest? A positive that sits closer to a neighbour is a
   future false accept, and this is the cheapest available proxy for what the
   trained model will do.
3. **Near-duplicate rate.** Internal redundancy the avoid-list failed to stop.
4. **Type mix.** A run that is 90% ExplicitCommand has not produced the
   implicit and indirect phrasings the architecture asks for, however diverse
   it looks lexically.
5. **Vocabulary novelty.** Too low means the model is rewording the seeds. Very
   high can mean it is drifting off-product -- worth a human look.

   READ THIS BEFORE ACTING ON A LOW FIGURE. Novelty falls as the generated
   distribution moves TOWARD real usage, by construction, and that is not a
   defect. Measured on Cmd.VolumeIncrease by utterance length:

       <= 4 words   50 rows    44 unique tokens    novelty 36.4%
       5-7 words    72 rows    91 unique tokens    novelty 61.5%
       8-12 words   54 rows   100 unique tokens    novelty 66.0%

   A four-word volume command can only be built from a handful of words --
   louder, up, turn, volume, raise, more, left, right -- and every one of them
   is already in the seeds. Since 57% of the real seeds are four words or fewer,
   asking the generator for the commonest real shape necessarily drives this
   number down. Four successive runs went 0.805 -> 0.773 -> 0.730 -> 0.711 while
   the short share rose, and the data got BETTER over that span.

   Use diversity gain (section 1) to tell the two cases apart. It measures how
   different the generated utterances are FROM EACH OTHER. If novelty falls
   while diversity gain holds or rises, the corpus has shifted to shorter
   utterances that are still mutually distinct -- which is the goal. If both
   fall together, that is genuine narrowing and worth acting on. Across those
   same four runs diversity gain went +0.060 -> +0.077 -> +0.082 -> +0.085.

Lexical metrics need nothing installed. Embedding metrics (1 and 2, in their
stronger form) activate when sentence-transformers and the configured model are
available; without them the report falls back to token-overlap versions of the
same ideas and says so.

Usage::

    python stage1_report.py                    # all intents with output
    python stage1_report.py --only Cmd.VolumeIncrease
    python stage1_report.py --markdown report.md
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any, Sequence

from seed_loader import (
    GeneratorConfig,
    SeedCorpusError,
    dedupe_key,
    load_config,
    load_seed_corpus,
)

DEFAULT_CONFIG = Path(__file__).with_name("generator_config.yaml")


# ---------------------------------------------------------------------------
# Lexical metrics -- always available
# ---------------------------------------------------------------------------


def _tokens(text: str) -> frozenset[str]:
    return frozenset(dedupe_key(text).split())


def _mean_pairwise_distance(texts: Sequence[str], *, cap: int = 200) -> float:
    """Mean token-set Jaccard distance. 1.0 = every pair shares no words."""
    sets = [_tokens(t) for t in texts[:cap]]
    if len(sets) < 2:
        return 0.0
    total = pairs = 0.0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = len(sets[i] | sets[j])
            total += 1.0 - (len(sets[i] & sets[j]) / union if union else 0.0)
            pairs += 1
    return total / pairs if pairs else 0.0


def _near_duplicate_rate(texts: Sequence[str], threshold: float = 0.75) -> float:
    """Share of utterances that overlap another by more than ``threshold``."""
    sets = [_tokens(t) for t in texts]
    flagged = set()
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = len(sets[i] | sets[j])
            if union and len(sets[i] & sets[j]) / union >= threshold:
                flagged.add(i)
                flagged.add(j)
    return len(flagged) / len(sets) if sets else 0.0


def _vocabulary_novelty(generated: Sequence[str], seeds: Sequence[str]) -> float:
    """Share of generated tokens that never appear in the seed corpus."""
    seed_vocab: set[str] = set()
    for s in seeds:
        seed_vocab |= _tokens(s)
    gen_vocab: set[str] = set()
    for g in generated:
        gen_vocab |= _tokens(g)
    if not gen_vocab:
        return 0.0
    return len(gen_vocab - seed_vocab) / len(gen_vocab)


def _length_spread(texts: Sequence[str]) -> tuple[float, float]:
    lengths = [len(t.split()) for t in texts]
    if not lengths:
        return 0.0, 0.0
    mean = sum(lengths) / len(lengths)
    var = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    return mean, math.sqrt(var)


# ---------------------------------------------------------------------------
# Embedding metrics -- optional
# ---------------------------------------------------------------------------


def _load_embedder(config: GeneratorConfig) -> Any:
    """Return a sentence-transformers model, or None if unavailable.

    Deliberately non-fatal: the lexical metrics carry the report on their own,
    and a missing 400MB download should not block a tuning loop.
    """
    name = (config.raw.get("deduplication") or {}).get("embedding_model")
    if not name:
        return None
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(name)
    except Exception:  # noqa: BLE001 -- absent package or failed download
        return None


def _boundary_leakage(
    embedder: Any,
    generated: dict[str, list[str]],
    taxonomy_seeds: dict[str, list[str]],
    *,
    near_similarity: float = 0.65,
) -> dict[str, dict[str, Any]]:
    """For each intent, the share of its utterances nearest ANOTHER intent.

    Centroids are built from the SEED data, not the generated data, so this
    asks the question that matters: does the new material still land in the
    region the real product data occupies?

    ``taxonomy_seeds`` must be the WHOLE taxonomy, not just the intents that
    happen to have generated output. An earlier version keyed it off the
    generated set, which meant a run scored with ``--only`` built exactly one
    centroid -- and with no rival to lose to, every utterance was nearest its
    own intent and the metric reported a flawless 0.0% by construction. The same
    silence would have covered a full run that stopped early: the fewer intents
    completed, the better the safety metric would have looked.

    Every seed is used. The previous ``[:120]`` cap was a head slice of a
    permutation-heavy export, so it hit hardest exactly where it mattered most:
    `Default Fallback Intent` has 613 seeds, and its centroid was being placed
    by the first 120 lines in file order while 493 were ignored -- for the class
    that decides False-Accept Rate. Encoding the full corpus is a few thousand
    short strings, once per run.
    """
    import numpy as np

    labels = [i for i in taxonomy_seeds if taxonomy_seeds[i]]
    centroids = []
    for intent in labels:
        vecs = embedder.encode(taxonomy_seeds[intent], normalize_embeddings=True)
        centroids.append(np.asarray(vecs).mean(axis=0))
    matrix = np.vstack(centroids)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    # Splitting the misses by how far away they landed is what makes this
    # number readable. A raw leakage percentage conflates two findings with
    # opposite meanings:
    #
    #   FAR  -- the utterance drifted to a different region of the space. Real,
    #           and worth acting on: something about batteries filed under a
    #           volume intent is a defect.
    #   NEAR -- it landed on a neighbour whose centroid is nearly on top of this
    #           one. Measured here, Cmd.VolumeMute and Cmd.VolumeUnmute sit at
    #           0.902 despite being opposites, so which of them "wins" an
    #           utterance is close to a coin toss and says nothing about the
    #           data.
    #
    # Note the cut is centroid distance, NOT the `families` map. Families are
    # generation scaffolding for hard-negative sampling; they do not track
    # topic. Cmd.VolumeIncrease and Help_Volume are in different families and
    # sit 0.822 apart -- squarely NEAR. Reading family membership as distance
    # would send a reviewer chasing noise.
    index = {name: k for k, name in enumerate(labels)}
    cross = matrix @ matrix.T

    out: dict[str, dict[str, Any]] = {}
    for intent, texts in generated.items():
        if intent not in index or not texts:
            continue
        own = index[intent]
        vecs = np.asarray(embedder.encode(texts, normalize_embeddings=True))
        nearest = (vecs @ matrix.T).argmax(axis=1)
        near: list[str] = []
        far: list[str] = []
        for k in nearest:
            if k == own:
                continue
            (near if cross[own, k] >= near_similarity else far).append(labels[k])
        wn = collections.Counter(near).most_common(1)
        wf = collections.Counter(far).most_common(1)
        out[intent] = {
            "near_rate": len(near) / len(texts),
            "far_rate": len(far) / len(texts),
            "worst_near": wn[0][0] if wn else "-",
            "worst_far": wf[0][0] if wf else "-",
        }
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping unparseable lines rather than failing.

    These files are append-only and written during long runs, so the last line
    can be a partial write left by a crash or an interrupt. Refusing to report
    at all because of one truncated line would deny the user the diagnosis at
    exactly the moment they need it most.
    """
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"  warning: skipped unparseable line {number} in {path.name}")
    return rows


def _budget(config: GeneratorConfig, intent: str) -> int:
    """The configured target for an intent.

    Duplicated from ``generator.py`` rather than imported: that module pulls in
    the provider SDKs, and this report is deliberately runnable with nothing
    installed.
    """
    budgets = (config.raw.get("generation") or {}).get("utterances_per_intent") or {}
    return int((budgets.get("overrides") or {}).get(intent, budgets.get("default", 120)))


def _quota_profile(config: GeneratorConfig, intent: str) -> dict[str, Any]:
    """Resolve an intent's quota profile, longest matching prefix wins.

    Duplicated from ``generator.py`` on purpose: importing it would drag in the
    provider SDKs, and this report is meant to run with nothing installed.
    """
    quotas = (config.raw.get("generation") or {}).get("quotas") or {}
    profiles = quotas.get("profiles") or {}
    chosen, best = None, -1
    for pattern, name in (quotas.get("assign") or {}).items():
        if (intent == pattern or intent.startswith(pattern)) and len(pattern) > best:
            chosen, best = name, len(pattern)
    return dict(profiles.get(chosen) or {})


def _quota_compliance(
    config: GeneratorConfig, intent: str, rows: list[dict[str, Any]]
) -> list[tuple[str, str, int, str]]:
    """Compare what the prompt demanded against what came back.

    This exists because the alternative was recomputing the same handful of
    counts by hand after every run, which is where a reviewer quietly starts
    reading the wrong number. Quotas are stated per batch as fractions, so the
    expectation scales to however many rows the intent actually produced.
    """
    profile = _quota_profile(config, intent)
    total = len(rows)
    if not total:
        return []

    def want(fraction: float) -> int:
        return round(float(fraction) * total)

    out: list[tuple[str, str, int, str]] = []
    if profile.get("min_short"):
        got = sum(1 for r in rows if len(r["utterance"].split()) <= 4)
        target = want(profile["min_short"])
        out.append(("<= 4 words", f"min {target}", got, "ok" if got >= target else "UNDER"))
    for type_name, bounds in (profile.get("types") or {}).items():
        low, high = bounds
        got = sum(1 for r in rows if r.get("type") == type_name)
        lo = want(low)
        if high is None:
            out.append((type_name, f"min {lo}", got, "ok" if got >= lo else "UNDER"))
        else:
            hi = want(high)
            verdict = "ok" if lo <= got <= hi else ("UNDER" if got < lo else "OVER")
            out.append((type_name, f"{lo}-{hi}", got, verdict))
    if profile.get("asr_simulated"):
        lo, hi = (want(v) for v in profile["asr_simulated"])
        got = sum(1 for r in rows if r.get("source") == "ASR-Simulated")
        verdict = "ok" if lo <= got <= hi else ("UNDER" if got < lo else "OVER")
        out.append(("ASR-Simulated", f"{lo}-{hi}", got, verdict))

    mix = (config.raw.get("generation") or {}).get("difficulty_mix") or {}
    for name, fraction in mix.items():
        got = sum(1 for r in rows if r.get("difficulty") == name)
        target = want(fraction)
        drift = abs(got - target) / max(target, 1)
        out.append((f"difficulty {name}", f"~{target}", got, "ok" if drift <= 0.20 else "OFF"))
    return out


def load_generated(config: GeneratorConfig) -> dict[str, list[dict[str, Any]]]:
    directory = config.checkpoint_dir / "stage1"
    if not directory.is_dir():
        raise SeedCorpusError(f"No Stage 1 output at {directory} -- run generator.py first")
    data: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(directory.glob("*.jsonl")):
        rows = [r for r in _read_jsonl(path) if "utterance" in r and "intent" in r]
        if rows:
            data[rows[0]["intent"]] = rows
    return data


def build_report(config: GeneratorConfig, only: list[str] | None = None) -> str:
    generated = load_generated(config)
    if only:
        generated = {k: v for k, v in generated.items() if k in only}
    if not generated:
        raise SeedCorpusError("No generated output matched")

    corpus = load_seed_corpus(config)
    gen_texts = {k: [r["utterance"] for r in v] for k, v in generated.items()}
    seeds = {k: corpus.intents.get(k, []) for k in generated}

    embedder = _load_embedder(config)
    # The full taxonomy, not `seeds` -- `seeds` is keyed by the generated
    # intents and is right for the lexical metrics below, but as a centroid
    # source it would leave the metric nothing to compare against.
    near_cut = float((config.raw.get("reporting") or {}).get("near_family_similarity", 0.65))
    leakage = (
        _boundary_leakage(embedder, gen_texts, corpus.intents, near_similarity=near_cut)
        if embedder
        else {}
    )

    lines = [
        "# Stage 1 Quality Report",
        "",
        f"{len(generated)} intents, {sum(len(v) for v in gen_texts.values())} utterances.",
        "",
        "## 0. Coverage vs budget",
        "",
        "An intent can end a run short of its budget without anything looking",
        "wrong: batches that keep failing validation are abandoned, and API",
        "errors do the same. Nothing else in this report compares output against",
        "what was asked for, so a truncated intent is otherwise invisible -- and",
        "truncation lands hardest where the budget is largest, which is Fallback,",
        "the class that decides FAR.",
        "",
        "| Intent | have | want | coverage |",
        "|---|---:|---:|---:|",
    ]
    short: list[str] = []
    for intent in sorted(gen_texts):
        have = len(gen_texts[intent])
        want = _budget(config, intent)
        share = have / want if want else 1.0
        if share < 0.95:
            short.append(intent)
        flag = "  ⚠" if share < 0.95 else ""
        lines.append(f"| `{intent}` | {have} | {want} | {share:.0%}{flag} |")
    if short:
        lines += [
            "",
            f"**{len(short)} intent(s) finished short of budget: "
            f"{', '.join(f'`{s}`' for s in short)}.** Rerunning `generator.py`",
            "resumes from the store and will attempt the shortfall again.",
        ]

    lines += ["", "## 0b. Quota compliance", ""]
    any_quota = False
    for intent in sorted(generated):
        checks = _quota_compliance(config, intent, generated[intent])
        if not checks:
            continue
        any_quota = True
        lines += [
            f"**`{intent}`** — the prompt states these as counts per batch; the",
            f"targets below are scaled to the {len(generated[intent])} rows produced.",
            "",
            "| constraint | asked | got | |",
            "|---|---|---:|---|",
        ]
        for name, asked, got, verdict in checks:
            mark = "" if verdict == "ok" else f"  **{verdict}**"
            lines.append(f"| {name} | {asked} | {got} |{mark} |")
        lines.append("")
    if not any_quota:
        lines += [
            "_No quota profile applies to these intents. `Help*` and Fallback are",
            "deliberately unconstrained until their distributions have been",
            "measured rather than guessed._",
            "",
        ]

    lines += [
        "## 1. Diversity vs seed  *(the headline metric)*",
        "",
        "Mean pairwise token distance. The premise of this project is that the",
        "legacy seeds are permutation-heavy, so `gain` should be clearly positive.",
        "A gain near zero means the generator is rewording rather than extending.",
        "",
        "| Intent | n | seed | generated | gain |",
        "|---|---:|---:|---:|---:|",
    ]

    gains = []
    for intent in sorted(gen_texts):
        s = _mean_pairwise_distance(seeds[intent]) if seeds[intent] else 0.0
        g = _mean_pairwise_distance(gen_texts[intent])
        gain = g - s
        gains.append(gain)
        flag = "" if gain > 0.03 else "  ⚠"
        lines.append(
            f"| `{intent}` | {len(gen_texts[intent])} | {s:.3f} | {g:.3f} | {gain:+.3f}{flag} |"
        )
    if gains:
        lines += ["", f"**Mean gain across intents: {sum(gains)/len(gains):+.3f}**"]

    lines += [
        "",
        "## 1b. Saturation — is the budget right?",
        "",
        "Two signals, both taken in generation order.",
        "",
        "`new vocab per quarter` is how many previously-unseen words each",
        "quarter contributed. A generator with semantic room left keeps",
        "introducing vocabulary; one that has run dry recycles it.",
        "",
        "`last-quarter redundancy` is the share of the final quarter that",
        "restates something from the first three.",
        "",
        "Together these answer the budget question empirically rather than by",
        "guesswork. Note that mean pairwise distance does NOT work here -- on",
        "short utterances it sits near 1.0 and barely moves, so it reads",
        "saturated whatever the truth is.",
        "",
        "| Intent | n | new vocab per quarter | last-quarter redundancy | read |",
        "|---|---:|---|---:|---|",
    ]
    for intent in sorted(gen_texts):
        texts = gen_texts[intent]
        if len(texts) < 8:
            lines.append(f"| `{intent}` | {len(texts)} | — | — | too few to judge |")
            continue

        # New vocabulary contributed by each quarter. A generator still finding
        # semantic room keeps introducing words; one that has run dry recycles.
        cuts = [max(1, int(len(texts) * f)) for f in (0.25, 0.50, 0.75, 1.0)]
        seen: set[str] = set()
        fresh: list[int] = []
        previous = 0
        for cut in cuts:
            for text in texts[previous:cut]:
                seen |= _tokens(text)
            fresh.append(len(seen))
            previous = cut
        added = [fresh[0]] + [fresh[i] - fresh[i - 1] for i in range(1, 4)]

        # How much of the final quarter merely restates earlier material.
        tail = texts[cuts[2] :]
        head_sets = [_tokens(t) for t in texts[: cuts[2]]]
        echoed = 0
        for text in tail:
            ts = _tokens(text)
            for hs in head_sets:
                union = len(ts | hs)
                if union and len(ts & hs) / union >= 0.5:
                    echoed += 1
                    break
        redundancy = echoed / len(tail) if tail else 0.0

        growth = added[3] / added[0] if added[0] else 0.0
        if redundancy >= 0.5 or growth < 0.25:
            read = "saturated — budget is enough"
        elif redundancy >= 0.3 or growth < 0.5:
            read = "levelling off"
        else:
            read = "**still climbing — raise budget**"

        lines.append(
            f"| `{intent}` | {len(texts)} | {' → '.join(str(a) for a in added)} "
            f"| {redundancy:.0%} | {read} |"
        )

    lines += [
        "",
        "## 2. Internal redundancy",
        "",
        "Share of utterances that overlap another in the same intent by ≥75% of",
        "their tokens. This is what the avoid-list is supposed to prevent.",
        "",
        "| Intent | near-dupes | vocab novelty | mean words | sd |",
        "|---|---:|---:|---:|---:|",
    ]
    for intent in sorted(gen_texts):
        dup = _near_duplicate_rate(gen_texts[intent])
        nov = _vocabulary_novelty(gen_texts[intent], seeds[intent])
        mean, sd = _length_spread(gen_texts[intent])
        flag = "  ⚠" if dup > 0.15 else ""
        lines.append(f"| `{intent}` | {dup:.1%}{flag} | {nov:.1%} | {mean:.1f} | {sd:.1f} |")

    lines += ["", "## 3. Type and difficulty mix", ""]
    types: collections.Counter = collections.Counter()
    diffs: collections.Counter = collections.Counter()
    sources: collections.Counter = collections.Counter()
    for rows in generated.values():
        types.update(r["type"] for r in rows)
        diffs.update(r["difficulty"] for r in rows)
        sources.update(r["source"] for r in rows)
    total = sum(types.values()) or 1

    lines += ["| Type | count | share |", "|---|---:|---:|"]
    for name, count in types.most_common():
        lines.append(f"| {name} | {count} | {count/total:.1%} |")

    implicit = types.get("ImplicitCommand", 0) + types.get("ObservationPlusCommand", 0)
    lines += [
        "",
        f"**Indirect share (Implicit + ObservationPlusCommand): {implicit/total:.1%}**",
        "",
        "This is the number to watch. A run dominated by ExplicitCommand has not",
        "produced the indirect phrasings the architecture asks for, however",
        "diverse it looks lexically.",
        "",
        "| Difficulty | share |   | Source | share |",
        "|---|---:|---|---|---:|",
    ]
    dtotal = sum(diffs.values()) or 1
    stotal = sum(sources.values()) or 1
    rows_d = list(diffs.most_common())
    rows_s = list(sources.most_common())
    for i in range(max(len(rows_d), len(rows_s))):
        left = f"{rows_d[i][0]} | {rows_d[i][1]/dtotal:.1%}" if i < len(rows_d) else " | "
        right = f"{rows_s[i][0]} | {rows_s[i][1]/stotal:.1%}" if i < len(rows_s) else " | "
        lines.append(f"| {left} |   | {right} |")

    lines += ["", "## 4. Boundary leakage", ""]
    if leakage:
        lines += [
            "For each generated utterance, which intent's SEED centroid is it",
            "actually nearest? The misses are split by how far away they landed,",
            "because the two halves mean opposite things.",
            "",
            "**FAR** — nearest centroid sits in a different region of the space",
            f"(similarity below {near_cut:.2f}). Review these; do NOT assume they",
            "are defects. Checked by hand on the first real run, all three FAR rows",
            "were correctly labelled — they were simply the batch's hardest",
            "utterances, sitting closest to a neighbouring region because one",
            "clause of a compound pulled that way. That makes them the best",
            "candidates for `dev_hard.csv`, not deletion candidates. What a FAR",
            "figure genuinely catches is gross drift: an utterance about batteries",
            "filed under a volume intent would land near 0.2 with a wide margin.",
            "",
            "**NEAR** — nearest centroid is a neighbour sitting almost on top of",
            "this one. Measured on this corpus, `Cmd.VolumeMute` and",
            "`Cmd.VolumeUnmute` are 0.902 apart despite being opposites, and",
            "`Cmd.VolumeIncrease` vs `Cmd.VolumeDecrease` 0.876, while an",
            "unrelated intent (`Help_Battery`) sits at 0.38. A sentence embedding",
            "captures what an utterance is ABOUT and barely captures direction or",
            "speech act, so among such neighbours the winner is close to a coin",
            "toss. A NEAR figure is not a data defect and chasing it wastes review",
            "time.",
            "",
            "The cut is centroid distance, not the `families` map: families are",
            "generation scaffolding and do not track topic. `Cmd.VolumeIncrease`",
            "and `Help_Volume` are in different families yet sit 0.822 apart.",
            "",
            "What a high NEAR figure DOES say is that the boundary is not",
            "separable by topic alone — which is precisely the case Stage 3's hard",
            "negatives exist to teach, and `command_help_pairs` exists to name.",
            "",
            "This metric answers *did the generator drift off-product?* It cannot",
            "answer *is this row labelled correctly?* — the resolution is below",
            "the granularity of this taxonomy, and it never could.",
            "",
            "| Intent | FAR (review) | into | NEAR (noise) | into |",
            "|---|---:|---|---:|---|",
        ]
        # Deliberately no warning threshold. FAR is a review list, not a defect
        # count: measured on the first real run it was 1.7%, and every one of
        # those rows was correct. A flag here would train the reader to ignore
        # the column, which is worse than no column at all.
        for intent, r in sorted(leakage.items(), key=lambda x: -x[1]["far_rate"]):
            lines.append(
                f"| `{intent}` | {r['far_rate']:.1%} | `{r['worst_far']}` "
                f"| {r['near_rate']:.1%} | `{r['worst_near']}` |"
            )
    else:
        lines += [
            "_Skipped: sentence-transformers or the configured embedding model is",
            "not available. The lexical metrics above still stand; install the",
            "package to enable this section._",
        ]

    rejects = config.checkpoint_dir / "rejections.jsonl"
    lines += ["", "## 5. Rejections", ""]
    rejected_rows = _read_jsonl(rejects)
    if rejected_rows:
        reasons: collections.Counter = collections.Counter()
        for entry in rejected_rows:
            for reason in entry.get("reasons", []):
                reasons[str(reason).split(":")[-1].strip()[:60]] += 1
        lines += [f"{len(rejected_rows)} rejected batches. Most common causes:", ""]
        lines += [f"- {r} — ×{c}" for r, c in reasons.most_common(8)]
    else:
        lines.append("None logged.")

    lines += [
        "",
        "---",
        "",
        "## What this cannot tell you",
        "",
        "No metric here catches an **invented capability**. An utterance like",
        '"set a sleep timer on my hearing aids" scores as diverse, on-topic and',
        "well-formed; it is only wrong because the product has no sleep timer.",
        "That still needs a person — but on a sample of 30-50 utterances, chosen",
        "from the Hard slice and the highest-leakage intents above, not on all of",
        "them.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score Stage 1 output.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--markdown", type=Path, default=None, help="Write to a file.")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        report = build_report(config, args.only)
    except SeedCorpusError as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.markdown:
        args.markdown.write_text(report, encoding="utf-8")
        print(f"Wrote {args.markdown}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
