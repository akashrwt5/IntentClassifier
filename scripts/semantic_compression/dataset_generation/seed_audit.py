"""Audit the legacy Dialogflow seed corpus before any generation happens.

This is the evidence base for the ``taxonomy:`` rules in
``generator_config.yaml``. It answers, from the data rather than from memory:

* Which encodings and invisible characters are actually present?
* How many utterances survive normalisation and de-duplication?
* Which intents share utterances with other intents -- i.e. which labels the
  classifier is being asked to separate on identical text?

Cross-intent collisions are the important output. A pair at 100% overlap is not
a hard problem for the model, it is an unlabelable one, and no amount of
synthetic generation downstream can repair it.

Usage::

    python seed_audit.py                      # writes seed_audit_report.md
    python seed_audit.py --config other.yaml
    python seed_audit.py --stdout             # print instead of writing
    python seed_audit.py --include-phrases    # local review only -- see below

The default report contains NO raw utterance text. The seed corpus is real
production ASR capture including customer PII, and this repository is public,
so the report is safe to commit only in its default form.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import unicodedata
from pathlib import Path

from seed_loader import (
    GeneratorConfig,
    SeedCorpusError,
    dedupe_key,
    decode_seed_file,
    filter_transcription_noise,
    load_config,
    load_seed_corpus,
    normalize_utterance,
)

DEFAULT_CONFIG = Path(__file__).with_name("generator_config.yaml")

# Characters that are legitimate in an utterance; anything else non-ASCII is
# worth surfacing because it usually indicates a broken export.
_EXPECTED_NON_ASCII = {"'", '"', "-"}


def _raw_scan(config: GeneratorConfig) -> tuple[dict, dict, collections.Counter]:
    """Scan every .txt in the seed dir BEFORE taxonomy rules are applied."""
    raw_lines: dict[str, list[str]] = {}
    encodings: dict[str, str] = {}
    odd_chars: collections.Counter = collections.Counter()

    for path in sorted(config.seed_dir.glob("*.txt")):
        encoding, text = decode_seed_file(path)
        encodings[path.stem] = encoding
        for char in text:
            if char in "\r\n\t " or char.isascii():
                continue
            if char in _EXPECTED_NON_ASCII:
                continue
            odd_chars[f"U+{ord(char):04X} {unicodedata.name(char, '<unnamed>')}"] += 1
        raw_lines[path.stem] = [
            normalize_utterance(line) for line in text.splitlines() if line.strip()
        ]

    return raw_lines, encodings, odd_chars


def _collisions(raw_lines: dict[str, list[str]]) -> list[tuple[str, str, int, float]]:
    """Find utterances that appear under more than one intent file.

    Returns ``(intent_a, intent_b, shared, pct_of_smaller)`` sorted by how
    completely the smaller intent is swallowed by the larger one -- 100% means
    the smaller intent has no distinguishing utterance at all.
    """
    index: dict[str, set[str]] = collections.defaultdict(set)
    sizes: dict[str, int] = {}
    for intent, lines in raw_lines.items():
        keys = {dedupe_key(line) for line in lines if dedupe_key(line)}
        sizes[intent] = len(keys)
        for key in keys:
            index[key].add(intent)

    pairs: collections.Counter = collections.Counter()
    for owners in index.values():
        if len(owners) < 2:
            continue
        ordered = sorted(owners)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                pairs[(left, right)] += 1

    out: list[tuple[str, str, int, float]] = []
    for (left, right), shared in pairs.items():
        smaller = min(sizes[left], sizes[right]) or 1
        out.append((left, right, shared, 100.0 * shared / smaller))
    out.sort(key=lambda row: (-row[3], -row[2]))
    return out


def _runtime_labels(config: GeneratorConfig) -> set[str] | None:
    """Load the label set the shipping app actually dispatches, if available.

    Returns ``None`` when no runtime label file is configured or present, so
    the audit still works standalone.
    """
    relative = config.raw.get("paths", {}).get("runtime_labels")
    if not relative:
        return None
    path = (config.config_path.parent / relative).resolve()
    if not path.is_file():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("map"), dict):
        return {str(v) for v in data["map"].values()}
    if isinstance(data, dict) and isinstance(data.get("labels"), list):
        return {str(x) for x in data["labels"]}
    if isinstance(data, list):
        return {str(x) for x in data}
    if isinstance(data, dict):
        return {str(k) for k in data}
    return None


def build_report(config: GeneratorConfig, *, include_phrases: bool = False) -> str:
    raw_lines, encodings, odd_chars = _raw_scan(config)
    corpus = load_seed_corpus(config)
    collisions = _collisions(raw_lines)

    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "# Seed Corpus Audit",
        "",
        f"_Generated {now} by `seed_audit.py` from `{config.seed_dir.name}/`._",
        "",
        "## 1. Headline numbers",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Seed files on disk | {len(raw_lines)} |",
        f"| Excluded as entity lists | {len(corpus.excluded)} |",
        f"| Merged into another intent | {len(corpus.merged)} |",
        f"| Dropped from taxonomy | {len(corpus.dropped)} |",
        f"| **Resolved intents** | **{len(corpus)}** |",
        f"| Raw non-empty lines | {sum(len(v) for v in raw_lines.values())} |",
        f"| Unique utterances after normalisation | {sum(len(v) for v in corpus.intents.values())} |",
        "",
        "## 2. Encodings",
        "",
    ]

    enc_counts = collections.Counter(encodings.values())
    lines += ["| Encoding | Files |", "|---|---|"]
    lines += [f"| `{enc}` | {count} |" for enc, count in enc_counts.most_common()]
    lines += [
        "",
        "> Reading these with `encoding='utf-8'` raises `UnicodeDecodeError`. Any",
        "> caller that catches and skips on failure loses the intent silently.",
        "",
        "## 3. Unexpected characters",
        "",
    ]

    if odd_chars:
        lines += ["| Codepoint | Occurrences |", "|---|---|"]
        lines += [f"| {name} | {count} |" for name, count in odd_chars.most_common(20)]
        lines += [
            "",
            "> `U+00A0 NO-BREAK SPACE` is used as a word separator in parts of the",
            "> export. Tokenising on `str.split()` without NFKC normalisation",
            "> collapses a whole utterance into a single token.",
        ]
    else:
        lines.append("None found.")

    lines += ["", "## 4. Cross-intent collisions", ""]
    lines += [
        "Utterances appearing under more than one intent. `% of smaller` is the",
        "share of the smaller intent's unique utterances that the larger one also",
        "claims; at 100% the smaller intent has no distinguishing evidence.",
        "",
        "| Intent A | Intent B | Shared | % of smaller |",
        "|---|---|---:|---:|",
    ]
    shown = [row for row in collisions if row[2] >= 2][:25]
    for left, right, shared, pct in shown:
        lines.append(f"| `{left}` | `{right}` | {shared} | {pct:.0f}% |")
    if not shown:
        lines.append("| _none_ | | | |")

    runtime = _runtime_labels(config)
    if runtime is not None:
        derived = set(corpus.intents)
        missing = sorted(runtime - derived)
        extra = sorted(derived - runtime)
        lines += ["", "## 4b. Taxonomy vs the deployed label space", ""]
        if not missing and not extra:
            lines += [
                f"✅ The {len(derived)} derived intents match the runtime label set exactly.",
            ]
        else:
            lines += [
                f"⚠️ **MISMATCH.** Derived {len(derived)} intents; runtime dispatches "
                f"{len(runtime)}. They overlap on {len(derived & runtime)}.",
                "",
                "The seed folder is not the same thing as the shipping label space, and",
                "the two have drifted. Training on the derived set alone would ship a",
                "model whose outputs the app cannot dispatch, and would drop intents it",
                "currently serves. Reconcile this BEFORE Stage 1 generation.",
                "",
                "| Direction | Intent |",
                "|---|---|",
            ]
            lines += [f"| runtime only — no seeds, no spec | `{name}` |" for name in missing]
            lines += [f"| derived only — runtime cannot dispatch | `{name}` |" for name in extra]

    lines += ["", "## 5. Applied taxonomy rules", ""]
    if corpus.excluded:
        lines += ["**Excluded (entity value lists):**", ""]
        lines += [f"- `{name}` — {reason}" for name, reason in sorted(corpus.excluded.items())]
        lines.append("")
    if corpus.merged:
        lines += ["**Merged:**", ""]
        lines += [f"- `{src}` → `{dst}`" for src, dst in sorted(corpus.merged.items())]
        lines.append("")
    if corpus.dropped:
        lines += ["**Dropped:**", ""]
        for name, reason in sorted(corpus.dropped.items()):
            lines.append(f"- `{name}` — {' '.join(reason.split())}")
        lines.append("")

    lines += ["## 6. Per-intent counts (resolved taxonomy)", ""]
    lines += ["| Intent | Family | Raw lines | Unique |", "|---|---|---:|---:|"]
    family_of = config.family_of
    for intent in corpus.intent_names:
        raw = corpus.raw_counts.get(intent, 0)
        merged_in = sum(
            corpus.raw_counts.get(src, 0) for src, dst in corpus.merged.items() if dst == intent
        )
        lines.append(
            f"| `{intent}` | {family_of.get(intent, '—')} | "
            f"{raw + merged_in} | {len(corpus.intents[intent])} |"
        )

    guard_cfg = config.sampling.get("noise_guard") or {}
    guard_on = bool(guard_cfg.get("enabled", False))
    flagged: list[tuple[str, str]] = []
    for intent in corpus.intent_names:
        _, dropped = filter_transcription_noise(
            corpus.intents[intent],
            max_hapax_tokens=int(guard_cfg.get("max_hapax_tokens", 2)),
            min_utterances_to_apply=int(guard_cfg.get("min_utterances_to_apply", 30)),
        )
        flagged.extend((intent, phrase) for phrase in dropped)

    if flagged:
        state = "ACTIVE — these are withheld" if guard_on else "REPORT-ONLY — nothing is withheld"
        per_intent = collections.Counter(intent for intent, _ in flagged)
        lines += [
            "",
            "## 7. Suspected transcription noise (review candidates)",
            "",
            f"Noise guard is **{state}** (`seed_sampling.noise_guard.enabled`).",
            "",
            "Phrases carrying several words unique to their intent. Max-diversity",
            "sampling is attracted to outliers, so these are among the likeliest",
            "lines to reach the LLM. The heuristic is *not* reliable enough to",
            "filter on: it penalises lexical novelty, which is the very signal this",
            "project wants. Treat the counts as a human review queue, not a verdict.",
            "",
            f"{len(flagged)} phrases flagged across {len(per_intent)} intents.",
            "",
            "| Intent | Flagged |",
            "|---|---:|",
        ]
        lines += [f"| `{intent}` | {count} |" for intent, count in per_intent.most_common()]

        if include_phrases:
            lines += [
                "",
                "### Flagged phrase text",
                "",
                "> ⚠ Contains raw production transcripts. Do NOT commit this section.",
                "",
                "| Intent | Flagged phrase |",
                "|---|---|",
            ]
            lines += [f"| `{intent}` | {phrase} |" for intent, phrase in flagged[:40]]
            if len(flagged) > 40:
                lines.append(f"| _… {len(flagged) - 40} more_ | |")
        else:
            lines += [
                "",
                "Phrase text is omitted by default. These are raw production ASR",
                "transcripts and this repository is public; the flagged lines are",
                "disproportionately the ones carrying names, addresses and other",
                "personal detail, precisely because the heuristic selects for rare",
                "words. Run `seed_audit.py --include-phrases` locally to read them,",
                "and do not commit that output.",
            ]

    thin = [i for i in corpus.intent_names if len(corpus.intents[i]) < 15]
    if thin:
        lines += [
            "",
            "## 8. Thin intents",
            "",
            "Fewer than 15 unique seed utterances. The bootstrapper has little",
            "evidence to reverse-engineer boundaries from, so review these specs by",
            "hand before Stage 1 generation.",
            "",
        ]
        lines += [f"- `{i}` ({len(corpus.intents[i])})" for i in thin]

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the Dialogflow seed corpus.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stdout", action="store_true", help="Print instead of writing the file.")
    parser.add_argument(
        "--include-phrases",
        action="store_true",
        help=(
            "Include the raw text of noise-flagged utterances. These are real "
            "production transcripts -- for local review only, never commit."
        ),
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        report = build_report(config, include_phrases=args.include_phrases)
    except SeedCorpusError as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.stdout:
        print(report)
        return 0

    destination = config.output_file("seed_audit_report")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
