"""Shared seed-corpus loading for the NLU Super Dataset pipeline.

The legacy Dialogflow export is messy in ways that silently corrupt downstream
generation, so every consumer (``seed_audit.py``, ``bootstrap_specs.py``, and
later ``generator.py``) loads the corpus through this module rather than
re-implementing ``open()``:

* Files are UTF-16 (61 of 63), not UTF-8. A naive ``encoding="utf-8"`` read
  raises and, if the caller swallows it, drops the intent entirely.
* Word separators are frequently U+00A0 NO-BREAK SPACE, not U+0020. Splitting
  on whitespace without normalising yields one giant "token" per utterance.
* Three files are entity value lists, not intents.
* Several intents are duplicates, stale renames, or rollup parents of others.

Taxonomy resolution is config-driven (``generator_config.yaml``) so the rules
stay auditable instead of buried in code.
"""

from __future__ import annotations

import csv
import json
import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

# Encodings tried in order. utf-8-sig first so BOM-prefixed UTF-8 does not fall
# through to a lossy latin-1 read; latin-1 last because it never raises and
# would mask a genuine decode problem if tried earlier.
_ENCODING_CANDIDATES: tuple[str, ...] = (
    "utf-8-sig",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "cp1252",
    "latin-1",
)

_SMART_CHARS = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "…": "...",
}

_PUNCT_RE = re.compile(r"[^\w\s']", re.UNICODE)
_WS_RE = re.compile(r"\s+")


class SeedCorpusError(RuntimeError):
    """Raised when the seed corpus or its configuration is unusable."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratorConfig:
    """Parsed ``generator_config.yaml`` with paths resolved to absolutes."""

    raw: dict[str, Any]
    config_path: Path

    def _resolve(self, key: str) -> Path:
        return (self.config_path.parent / self.raw["paths"][key]).resolve()

    @property
    def seed_dir(self) -> Path:
        return self._resolve("seed_dir")

    @property
    def output_dir(self) -> Path:
        return self._resolve("output_dir")

    @property
    def checkpoint_dir(self) -> Path:
        return self._resolve("checkpoint_dir")

    def output_file(self, key: str) -> Path:
        return self.output_dir / self.raw["paths"][key]

    @property
    def llm(self) -> dict[str, Any]:
        return self.raw["llm"]

    @property
    def sampling(self) -> dict[str, Any]:
        return self.raw["seed_sampling"]

    @property
    def taxonomy(self) -> dict[str, Any]:
        return self.raw["taxonomy"]

    @property
    def random_seed(self) -> int:
        return int(self.raw["reproducibility"]["random_seed"])

    @property
    def fallback_intent(self) -> str:
        return str(self.taxonomy["fallback_intent"])

    @property
    def family_of(self) -> dict[str, str]:
        """Invert the family -> [intents] map into intent -> family."""
        mapping: dict[str, str] = {}
        for family, intents in self.raw.get("families", {}).items():
            for intent in intents:
                if intent in mapping:
                    raise SeedCorpusError(
                        f"Intent {intent!r} is listed in two families: "
                        f"{mapping[intent]!r} and {family!r}"
                    )
                mapping[intent] = family
        return mapping

    @property
    def command_help_pairs(self) -> dict[str, str]:
        return dict(self.raw.get("command_help_pairs", {}))


def load_config(config_path: str | Path) -> GeneratorConfig:
    """Load and minimally validate ``generator_config.yaml``."""
    path = Path(config_path).resolve()
    if not path.is_file():
        raise SeedCorpusError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise SeedCorpusError(f"Config did not parse to a mapping: {path}")
    for required in ("paths", "llm", "seed_sampling", "taxonomy", "reproducibility"):
        if required not in raw:
            raise SeedCorpusError(f"Config missing required section: {required!r}")

    return GeneratorConfig(raw=raw, config_path=path)


# ---------------------------------------------------------------------------
# Decoding and normalisation
# ---------------------------------------------------------------------------


def decode_seed_file(path: Path) -> tuple[str, str]:
    """Decode a seed file, returning ``(encoding_used, text)``.

    A decode that "succeeds" but leaves NUL bytes in the result means we picked
    a single-byte codec for what is really UTF-16, so that result is rejected
    and the next candidate is tried.
    """
    raw = path.read_bytes()
    for encoding in _ENCODING_CANDIDATES:
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
        if "\x00" in text:
            continue
        return encoding, text
    raise SeedCorpusError(f"Could not decode {path.name} with any known encoding")


def normalize_utterance(text: str) -> str:
    """Normalise a raw seed line into a clean, display-ready utterance.

    NFKC folds the NO-BREAK SPACEs that riddle the Dialogflow export into plain
    spaces; smart quotes are ASCII-folded so ``"it's"`` and ``"it’s"`` do not
    survive as two distinct utterances.
    """
    text = unicodedata.normalize("NFKC", text)
    for source, target in _SMART_CHARS.items():
        text = text.replace(source, target)
    text = "".join(ch for ch in text if ch == "\t" or not unicodedata.category(ch).startswith("C"))
    return _WS_RE.sub(" ", text).strip()


def dedupe_key(text: str) -> str:
    """Aggressive key for duplicate detection: casefolded, punctuation-free."""
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", normalize_utterance(text).lower())).strip()


def read_seed_file(path: Path) -> tuple[str, list[str]]:
    """Return ``(encoding_used, normalised_non_empty_lines)`` for one seed file."""
    encoding, text = decode_seed_file(path)
    lines = [normalize_utterance(line) for line in text.splitlines()]
    return encoding, [line for line in lines if line]


# ---------------------------------------------------------------------------
# Corpus assembly
# ---------------------------------------------------------------------------


@dataclass
class SeedCorpus:
    """The resolved taxonomy plus its de-duplicated seed utterances."""

    intents: dict[str, list[str]] = field(default_factory=dict)
    encodings: dict[str, str] = field(default_factory=dict)
    excluded: dict[str, str] = field(default_factory=dict)
    merged: dict[str, str] = field(default_factory=dict)
    dropped: dict[str, str] = field(default_factory=dict)
    raw_counts: dict[str, int] = field(default_factory=dict)

    @property
    def intent_names(self) -> list[str]:
        return sorted(self.intents)

    def __len__(self) -> int:
        return len(self.intents)


def read_dialogflow_usersays(path: Path, *, collapse_templates: bool = True) -> list[str]:
    """Read a Dialogflow ``*_usersays_*.json`` export.

    These files carry more than the flat ``.txt`` export does: each utterance is
    segmented, and segments tagged with a ``meta`` entity are slot fillers. That
    structure exposes the permutation problem directly -- ``Cmd.MemoryChange``
    ships 630 entries that are only 58 carrier templates crossed with 41 memory
    names.

    With ``collapse_templates`` (the default) one representative per template is
    returned instead of every permutation. Feeding 18 permutations of the same
    two templates to the spec writer would show it almost nothing; 18 distinct
    templates show it the intent's actual shape. The slot vocabulary is supplied
    separately, via ``taxonomy.slot_vocabularies``.
    """
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise SeedCorpusError(f"{path.name} is not a Dialogflow usersays array")

    by_template: dict[str, str] = {}
    plain: list[str] = []

    for entry in entries:
        segments = entry.get("data") or []
        literal = normalize_utterance("".join(s.get("text", "") for s in segments))
        if not literal:
            continue
        plain.append(literal)
        template = normalize_utterance(
            "".join("{slot}" if s.get("meta") else s.get("text", "") for s in segments)
        )
        by_template.setdefault(template, literal)

    return list(by_template.values()) if collapse_templates else plain


def _load_additional_sources(config: GeneratorConfig) -> dict[str, list[str]]:
    """Pull seed utterances for intents that have no ``.txt`` in the export.

    The flat ``.txt`` export is an incomplete snapshot of the taxonomy: three
    shipping intents have no file in it at all. Their evidence comes from other
    parts of the SAME Dialogflow export (``*_usersays_*.json``) and from the
    curated PVA phrase workbook -- never from previously generated data, which
    would make each pass a copy of the last.
    """
    sources = config.taxonomy.get("additional_seed_sources") or []
    collected: dict[str, list[str]] = {}

    for source in sources:
        path = (config.config_path.parent / source["path"]).resolve()
        if not path.is_file():
            raise SeedCorpusError(f"additional_seed_sources path not found: {path}")

        kind = str(source.get("format", "csv")).lower()

        if kind == "dialogflow_usersays":
            intent = str(source["intent"])
            collected.setdefault(intent, []).extend(
                read_dialogflow_usersays(
                    path, collapse_templates=bool(source.get("collapse_templates", True))
                )
            )
            continue

        if kind != "csv":
            raise SeedCorpusError(f"Unknown additional_seed_sources format: {kind!r}")

        wanted = set(source.get("intents") or [])
        text_col = source.get("text_column", "text")
        intent_col = source.get("intent_column", "intent")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                intent = (row.get(intent_col) or "").strip()
                if wanted and intent not in wanted:
                    continue
                utterance = normalize_utterance(row.get(text_col) or "")
                if utterance:
                    collected.setdefault(intent, []).append(utterance)

    return collected


def load_seed_corpus(config: GeneratorConfig) -> SeedCorpus:
    """Read the seed directory and apply every taxonomy rule from the config."""
    seed_dir = config.seed_dir
    if not seed_dir.is_dir():
        raise SeedCorpusError(f"Seed directory not found: {seed_dir}")

    taxonomy = config.taxonomy
    excluded = {name: "entity value list, not an intent" for name in taxonomy["exclude_files"]}
    merge_rules: dict[str, dict[str, Any]] = taxonomy.get("merge_intents") or {}
    drop_rules: dict[str, dict[str, Any]] = taxonomy.get("drop_intents") or {}

    corpus = SeedCorpus()
    staged: dict[str, list[str]] = {}

    for path in sorted(seed_dir.glob("*.txt")):
        intent = path.stem
        encoding, lines = read_seed_file(path)
        corpus.encodings[intent] = encoding
        corpus.raw_counts[intent] = len(lines)

        if intent in excluded:
            corpus.excluded[intent] = excluded[intent]
            continue
        if intent in drop_rules:
            corpus.dropped[intent] = str(drop_rules[intent].get("reason", "")).strip()
            continue

        target = intent
        if intent in merge_rules:
            target = str(merge_rules[intent]["into"])
            corpus.merged[intent] = target

        staged.setdefault(target, []).extend(lines)

    for intent, lines in _load_additional_sources(config).items():
        if intent in drop_rules or intent in excluded:
            continue
        staged.setdefault(intent, []).extend(lines)
        corpus.raw_counts[intent] = corpus.raw_counts.get(intent, 0) + len(lines)
        corpus.encodings.setdefault(intent, "csv")

    for merge_source, merge_target in corpus.merged.items():
        if merge_target not in staged:
            raise SeedCorpusError(
                f"merge_intents maps {merge_source!r} into {merge_target!r}, "
                "which has no seed file"
            )

    for intent, lines in staged.items():
        corpus.intents[intent] = _dedupe_preserving_order(lines)

    _validate_against_families(config, corpus)
    return corpus


def _dedupe_preserving_order(lines: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        key = dedupe_key(line)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def _validate_against_families(config: GeneratorConfig, corpus: SeedCorpus) -> None:
    """Fail loudly if the family map and the resolved taxonomy disagree.

    An intent with no family would silently lose hard-negative scoping in Stage
    3; a family entry with no intent means the config is stale.
    """
    family_of = config.family_of
    resolved = set(corpus.intents)

    missing_family = sorted(resolved - set(family_of))
    if missing_family:
        raise SeedCorpusError(
            "Intents present in the seed corpus but absent from `families` in "
            f"generator_config.yaml: {missing_family}"
        )

    phantom = sorted(set(family_of) - resolved)
    if phantom:
        raise SeedCorpusError(
            "Intents listed under `families` that do not exist in the resolved "
            f"taxonomy (stale config?): {phantom}"
        )


# ---------------------------------------------------------------------------
# Seed sampling
# ---------------------------------------------------------------------------


def _tokens(text: str) -> frozenset[str]:
    return frozenset(dedupe_key(text).split())


def filter_transcription_noise(
    phrases: Sequence[str],
    *,
    max_hapax_tokens: int = 2,
    min_utterances_to_apply: int = 30,
) -> tuple[list[str], list[str]]:
    """Split phrases into ``(kept, dropped)``, dropping likely ASR garbage.

    The Dialogflow logs contain genuine mis-transcriptions filed under command
    intents -- one ``Cmd.VolumeIncrease`` seed is garbled speech carrying two
    unrelated personal names. This matters specifically because we sample by
    maximum diversity: farthest-point selection is *attracted* to outliers, so
    the noisiest line in a file is among the likeliest to be shown to the LLM.

    A specification must not be reverse-engineered from a transcription error,
    so we drop phrases carrying more than ``max_hapax_tokens`` words that occur
    exactly once across the whole intent. The guard only applies to intents with
    enough utterances for that signal to mean anything; below that threshold
    every word looks rare.

    Note this filters *evidence shown to the spec writer*, not training data.
    """
    if len(phrases) < min_utterances_to_apply:
        return list(phrases), []

    frequency: dict[str, int] = {}
    for phrase in phrases:
        for token in _tokens(phrase):
            frequency[token] = frequency.get(token, 0) + 1

    kept: list[str] = []
    dropped: list[str] = []
    for phrase in phrases:
        hapax = sum(1 for token in _tokens(phrase) if frequency.get(token, 0) <= 1)
        (dropped if hapax > max_hapax_tokens else kept).append(phrase)

    # Never let the guard gut an intent; if it would, treat it as unreliable.
    if len(kept) < min_utterances_to_apply // 2:
        return list(phrases), []
    return kept, dropped


def _jaccard_distance(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 0.0
    union = len(left | right)
    return 1.0 - (len(left & right) / union) if union else 0.0


def select_diverse_seeds(
    phrases: Sequence[str],
    k: int,
    *,
    strategy: str = "max_diversity",
    rng: Any = None,
    min_tokens: int = 1,
    max_tokens: int = 25,
) -> list[str]:
    """Pick ``k`` seed phrases to show the LLM.

    ``max_diversity`` runs a greedy farthest-point traversal over token-set
    Jaccard distance: seed the selection with the phrase of median length (a
    representative rather than an outlier), then repeatedly add whichever
    candidate is farthest from everything chosen so far.

    This matters because the export is permutation-heavy. ``head`` -- taking
    the first k lines -- returns near-identical siblings and hands the LLM the
    least informative evidence in the file.
    """
    pool = [p for p in phrases if min_tokens <= len(p.split()) <= max_tokens]
    if not pool:
        pool = list(phrases)
    if len(pool) <= k:
        return list(pool)

    if strategy == "head":
        return list(pool[:k])
    if strategy == "random":
        if rng is None:
            raise SeedCorpusError("strategy='random' requires an rng")
        return list(rng.sample(pool, k))
    if strategy != "max_diversity":
        raise SeedCorpusError(f"Unknown seed_sampling.strategy: {strategy!r}")

    # Ties are the norm, not the exception. In a chaotic intent like Fallback,
    # 80% of phrase pairs share zero tokens, so almost every candidate sits at
    # distance 1.0 simultaneously. Plain `max()` returns the first such index,
    # which silently degrades the whole traversal into file order -- and the
    # export is alphabetically sorted, so "diverse sampling" would quietly hand
    # back the A's. Ties are therefore broken with the seeded RNG: still
    # reproducible, but actually spread across the file.
    if rng is None:
        rng = random.Random(0)

    token_sets: list[frozenset[str]] = [_tokens(p) for p in pool]
    lengths = sorted(range(len(pool)), key=lambda i: len(pool[i].split()))
    start = lengths[len(lengths) // 2]

    chosen = [start]
    best_distance = [_jaccard_distance(token_sets[start], ts) for ts in token_sets]
    best_distance[start] = -1.0

    while len(chosen) < k:
        highest = max(best_distance)
        if highest < 0:
            break
        tied = [i for i, d in enumerate(best_distance) if d >= highest - 1e-9]
        nxt = tied[0] if len(tied) == 1 else rng.choice(tied)
        chosen.append(nxt)
        best_distance[nxt] = -1.0
        for i, ts in enumerate(token_sets):
            if best_distance[i] < 0:
                continue
            distance = _jaccard_distance(token_sets[nxt], ts)
            if distance < best_distance[i]:
                best_distance[i] = distance

    return [pool[i] for i in sorted(chosen)]


def sample_seeds_for_intent(
    config: GeneratorConfig,
    phrases: Sequence[str],
    *,
    rng: Any = None,
) -> tuple[list[str], list[str]]:
    """Apply the configured noise guard and sampling strategy to one intent.

    Returns ``(selected_seeds, dropped_as_noise)`` so callers can report what
    the guard removed rather than dropping it silently.
    """
    sampling = config.sampling
    guard = sampling.get("noise_guard") or {}
    if rng is None:
        rng = random.Random(config.random_seed)

    dropped: list[str] = []
    pool = list(phrases)
    if guard.get("enabled", False):
        pool, dropped = filter_transcription_noise(
            pool,
            max_hapax_tokens=int(guard.get("max_hapax_tokens", 2)),
            min_utterances_to_apply=int(guard.get("min_utterances_to_apply", 30)),
        )

    selected = select_diverse_seeds(
        pool,
        int(sampling["max_seeds_per_intent"]),
        strategy=str(sampling["strategy"]),
        rng=rng,
        min_tokens=int(sampling["min_tokens"]),
        max_tokens=int(sampling["max_tokens"]),
    )
    return selected, dropped
