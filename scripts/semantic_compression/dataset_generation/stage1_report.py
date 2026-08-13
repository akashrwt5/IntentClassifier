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
    seeds: dict[str, list[str]],
) -> dict[str, tuple[float, str]]:
    """For each intent, the share of its utterances nearest ANOTHER intent.

    Centroids are built from the SEED data, not the generated data, so this
    asks the question that matters: does the new material still land in the
    region the real product data occupies?
    """
    import numpy as np

    labels = [i for i in seeds if seeds[i]]
    centroids = []
    for intent in labels:
        vecs = embedder.encode(seeds[intent][:120], normalize_embeddings=True)
        centroids.append(np.asarray(vecs).mean(axis=0))
    matrix = np.vstack(centroids)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    out: dict[str, tuple[float, str]] = {}
    for intent, texts in generated.items():
        if intent not in labels or not texts:
            continue
        vecs = np.asarray(embedder.encode(texts, normalize_embeddings=True))
        nearest = (vecs @ matrix.T).argmax(axis=1)
        wrong = [labels[k] for k in nearest if labels[k] != intent]
        rate = len(wrong) / len(texts)
        worst = collections.Counter(wrong).most_common(1)
        out[intent] = (rate, worst[0][0] if worst else "-")
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
    leakage = _boundary_leakage(embedder, gen_texts, seeds) if embedder else {}

    lines = [
        "# Stage 1 Quality Report",
        "",
        f"{len(generated)} intents, {sum(len(v) for v in gen_texts.values())} utterances.",
        "",
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
            "Share of generated utterances whose nearest SEED centroid belongs to",
            "a different intent. These are the samples most likely to become false",
            "accepts, and this is the cheapest proxy available before training.",
            "",
            "| Intent | leaked | most often into |",
            "|---|---:|---|",
        ]
        for intent, (rate, worst) in sorted(leakage.items(), key=lambda x: -x[1][0]):
            flag = "  ⚠" if rate > 0.20 else ""
            lines.append(f"| `{intent}` | {rate:.1%}{flag} | `{worst}` |")
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
