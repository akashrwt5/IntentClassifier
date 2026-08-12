"""Phase 1 -- reverse-engineer formal Intent Specifications from seed data.

Authoring the Section 2 specification for 57 intents by hand costs days, so this
script reads the legacy Dialogflow seeds and has an LLM infer each spec under a
strict Pydantic schema. The specs -- not the generator's inference -- are the
source of truth for labelling in every later phase.

What this does beyond a plain "read files, call the LLM" loop:

* **Config-driven taxonomy.** Entity lists, merges and drops come from
  ``generator_config.yaml``, so the rules are auditable and match
  ``seed_audit.py``'s evidence.
* **Diverse seed sampling.** The export is permutation-heavy; the first ten
  lines of a file are near-identical siblings. We show the LLM a maximally
  diverse subset instead (see ``seed_loader.select_diverse_seeds``).
* **Constrained neighbours.** Neighbour intents are validated against the real
  taxonomy and rejected if hallucinated, because Stage 3 samples hard negatives
  from this field -- a phantom name silently produces nothing.
* **Deterministic families.** ``intent_family`` is injected from config rather
  than inferred, per the blueprint's "specs are the source of truth" rule.
* **Checkpoint + resume.** Each spec is persisted the moment it validates, so an
  API failure 50 intents in does not discard 50 intents of work.

Usage::

    python bootstrap_specs.py --dry-run      # no API calls, verify the pipeline
    python bootstrap_specs.py --limit 5      # small paid test batch
    python bootstrap_specs.py                # full run (resumes automatically)
    python bootstrap_specs.py --force        # ignore checkpoints, regenerate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from seed_loader import (
    GeneratorConfig,
    SeedCorpus,
    SeedCorpusError,
    load_config,
    load_seed_corpus,
    sample_seeds_for_intent,
)

DEFAULT_CONFIG = Path(__file__).with_name("generator_config.yaml")

MIN_NEIGHBOURS = 2
MAX_NEIGHBOURS = 6


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class IntentSpecification(BaseModel):
    """The Section 2 specification template, as returned by the LLM.

    ``intent_family`` is deliberately absent: families are assigned from
    ``generator_config.yaml`` after the call, never inferred.
    """

    name: str = Field(description="The exact intent name, verbatim, e.g. Cmd.VolumeIncrease")
    business_description: str = Field(
        description="One or two sentences describing what the user wants to accomplish."
    )
    trigger_conditions: list[str] = Field(
        description=(
            "Conditions under which this intent SHOULD fire. Describe user goals, "
            "not surface wordings. 3-6 entries."
        )
    )
    do_not_trigger: list[str] = Field(
        description=(
            "Conditions under which this intent must NOT fire, including pure "
            "observations, information-only questions, and adjacent-product "
            "requests (TV/phone volume, streaming). 3-6 entries."
        )
    )
    boundary_cases: list[str] = Field(
        description=(
            "Rules for resolving ambiguity against Fallback and neighbours. "
            "Ambiguity is always resolved toward Fallback, never by inventing a "
            "capability. 2-5 entries."
        )
    )
    neighbor_intents: list[str] = Field(
        description=(
            "Exact intent names from the provided allowed list that are most "
            "likely to be confused with this one. 2-6 entries."
        )
    )
    positive_example: str = Field(
        description="A natural utterance that clearly triggers this intent."
    )
    hard_negative_example: str = Field(
        description=(
            "A near-miss utterance that a naive classifier would assign here but "
            "which must NOT trigger it -- typically an observation, an "
            "information-only question, or a neighbour intent's command."
        )
    )


SYSTEM_PROMPT = """\
You are a Principal Conversational AI Architect with 15+ years building \
production intent classifiers for on-device, low-resource models.

You are reverse-engineering a formal Intent Specification from legacy \
Dialogflow seed utterances for a hearing-aid voice assistant. This spec becomes \
the sole source of truth for labelling a synthetic training set, so precision \
matters more than coverage.

Hard rules:
1. NEVER invent a product capability. If the seed utterances do not evidence a \
behaviour, it does not exist. An under-specified spec is recoverable; a \
hallucinated capability poisons every downstream sample.
2. Describe user GOALS in trigger_conditions, not surface wordings. \
"User asks for the hearing aids to be louder" is useful. "User says 'turn it \
up'" is not -- that is a template, and templates teach an embedding model \
nothing.
3. do_not_trigger must always cover: pure observations that request no change; \
information-only questions ("how do I ...?"); and requests aimed at an adjacent \
product surface (TV, phone, streaming source) rather than the hearing aid.
4. Ambiguity resolves toward Fallback. Never resolve it by widening the intent.
5. neighbor_intents must be chosen ONLY from the allowed list given to you, \
copied verbatim. These drive hard-negative sampling; a name that is not in the \
list produces nothing.
6. hard_negative_example must be genuinely difficult -- lexically close to the \
positive but semantically outside the intent. A random unrelated sentence is \
useless here.

Context that should shape your judgement: a false ACCEPT (an information \
question misread as a command, physically changing a hearing aid's volume) is \
far costlier for these users than a false reject. Draw the boundaries \
conservatively.
"""

HUMAN_PROMPT = """\
Intent name: {intent_name}
Intent family: {intent_family}

Seed utterances (a diverse sample of {seed_shown} drawn from {seed_total} unique \
utterances):
{seed_phrases}

Sibling intents in the same family (most likely confusions):
{siblings}

Paired Help/Command counterpart, if any (the Command-vs-Question boundary):
{counterpart}

ALLOWED neighbor_intents values -- copy verbatim, choose {min_n}-{max_n}:
{allowed_intents}
{correction}
Produce the formal Intent Specification.
"""

FALLBACK_NOTE = """\

NOTE: this is the catch-all Fallback intent. Its trigger_conditions describe \
out-of-scope input (greetings, small talk, weather, smart-home, unrelated \
device requests, and pure observations that request no change). Its \
do_not_trigger must state that any actionable in-scope command belongs to its \
specific intent instead.
"""


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _safe_name(intent: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", intent)


class Checkpoints:
    """Per-intent JSON checkpoints so a mid-run failure costs one intent."""

    def __init__(self, root: Path, fingerprint: str) -> None:
        self.dir = root / "specs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = root / "state.json"
        self.fingerprint = fingerprint
        self.state: dict[str, Any] = {"fingerprint": fingerprint, "attempts": {}}
        if self.state_path.is_file():
            try:
                loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
                if loaded.get("fingerprint") == fingerprint:
                    self.state = loaded
                else:
                    print("Config changed since last run — checkpoints will be ignored.")
                    self.state = {"fingerprint": fingerprint, "attempts": {}}
            except (json.JSONDecodeError, OSError):
                pass

    def load(self, intent: str) -> dict[str, Any] | None:
        path = self.dir / f"{_safe_name(intent)}.json"
        if self.state.get("fingerprint") != self.fingerprint or not path.is_file():
            return None
        try:
            return dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            return None

    def save(self, intent: str, spec: dict[str, Any], attempts: int) -> None:
        path = self.dir / f"{_safe_name(intent)}.json"
        path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
        self.state.setdefault("attempts", {})[intent] = attempts
        self.state_path.write_text(
            json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_spec(
    spec: IntentSpecification,
    intent: str,
    allowed: set[str],
) -> list[str]:
    """Return human-readable errors; an empty list means the spec is acceptable.

    Neighbour validation is the load-bearing check. Roadmap Phase 1.5 promises
    "verify referenced Neighbor Intents actually exist", and doing it here --
    while the LLM call is still in flight and correctable -- is far cheaper than
    discovering phantom names during Stage 3.
    """
    errors: list[str] = []

    if spec.name != intent:
        errors.append(f"`name` must be exactly {intent!r}, got {spec.name!r}.")
    if len(spec.business_description.split()) < 4:
        errors.append("`business_description` is too short to be useful.")

    for field_name, minimum in (
        ("trigger_conditions", 2),
        ("do_not_trigger", 2),
        ("boundary_cases", 1),
    ):
        values = [v.strip() for v in getattr(spec, field_name) if v.strip()]
        if len(values) < minimum:
            errors.append(f"`{field_name}` needs at least {minimum} non-empty entries.")

    neighbours = [n.strip() for n in spec.neighbor_intents if n.strip()]
    unknown = [n for n in neighbours if n not in allowed]
    if unknown:
        errors.append(
            f"These neighbor_intents are not in the taxonomy and must be replaced "
            f"with names from the allowed list: {unknown}."
        )
    if intent in neighbours:
        errors.append("`neighbor_intents` must not contain the intent itself.")
    valid = [n for n in neighbours if n in allowed and n != intent]
    if len(valid) < MIN_NEIGHBOURS:
        errors.append(f"`neighbor_intents` needs at least {MIN_NEIGHBOURS} valid entries.")

    if not spec.positive_example.strip():
        errors.append("`positive_example` is empty.")
    if not spec.hard_negative_example.strip():
        errors.append("`hard_negative_example` is empty.")
    if spec.positive_example.strip().lower() == spec.hard_negative_example.strip().lower():
        errors.append("`positive_example` and `hard_negative_example` are identical.")

    return errors


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _build_chain(config: GeneratorConfig) -> Any:
    """Import LangChain lazily so ``--dry-run`` works without the SDK installed."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    llm_cfg = config.llm
    llm = ChatOpenAI(
        model=llm_cfg["model"],
        temperature=llm_cfg["temperature"],
        timeout=llm_cfg.get("request_timeout_seconds", 90),
    )
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
    return prompt | llm.with_structured_output(IntentSpecification)


def _payload(
    config: GeneratorConfig,
    corpus: SeedCorpus,
    intent: str,
    seeds: list[str],
) -> dict[str, Any]:
    family_of = config.family_of
    family = family_of.get(intent, "Unassigned")
    siblings = [i for i in corpus.intent_names if family_of.get(i) == family and i != intent]
    counterpart = config.command_help_pairs.get(intent)
    if counterpart is None:
        counterpart = next(
            (cmd for cmd, help_i in config.command_help_pairs.items() if help_i == intent),
            None,
        )

    body = "\n".join(f"- {phrase}" for phrase in seeds)
    if intent == config.fallback_intent:
        body += FALLBACK_NOTE

    return {
        "intent_name": intent,
        "intent_family": family,
        "seed_phrases": body,
        "seed_shown": len(seeds),
        "seed_total": len(corpus.intents[intent]),
        "siblings": ", ".join(siblings) if siblings else "(none — singleton family)",
        "counterpart": counterpart or "(none)",
        "allowed_intents": ", ".join(corpus.intent_names),
        "min_n": MIN_NEIGHBOURS,
        "max_n": MAX_NEIGHBOURS,
        "correction": "",
    }


def generate_one(
    chain: Any,
    payload: dict[str, Any],
    intent: str,
    allowed: set[str],
    config: GeneratorConfig,
) -> tuple[IntentSpecification | None, int, list[str]]:
    """Invoke the LLM, validating and re-prompting on failure.

    Implements the blueprint's Validation Failure Policy at spec level: reject
    the sample, regenerate only that sample, repeat to the retry limit, and log
    what was rejected.
    """
    max_retries = int(config.llm.get("max_retries", 3))
    backoff = float(config.llm.get("retry_backoff_seconds", 2.0))
    last_errors: list[str] = []

    for attempt in range(1, max_retries + 1):
        try:
            spec: IntentSpecification = chain.invoke(payload)
        except Exception as exc:  # noqa: BLE001 -- surface any SDK/transport error
            last_errors = [f"LLM call failed: {exc}"]
            if attempt < max_retries:
                time.sleep(backoff * attempt)
            continue

        errors = validate_spec(spec, intent, allowed)
        if not errors:
            return spec, attempt, []

        last_errors = errors
        payload = dict(payload)
        payload["correction"] = (
            "\nYour previous attempt was REJECTED for these reasons. Fix every "
            "one of them:\n" + "\n".join(f"- {e}" for e in errors) + "\n"
        )
        if attempt < max_retries:
            time.sleep(backoff * attempt)

    return None, max_retries, last_errors


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _origin_counts(specs: list[dict[str, Any]]) -> dict[str, int]:
    """Tally where each spec actually came from.

    Provenance is not decoration here. A reviewer needs to know whether a given
    spec was inferred by a model or written by a person, because that changes
    how hard they should look at it.
    """
    counts: dict[str, int] = {}
    for spec in specs:
        origin = str(spec.get("_provenance", {}).get("model", "unknown"))
        counts[origin] = counts.get(origin, 0) + 1
    return dict(sorted(counts.items()))


def export_yaml(config: GeneratorConfig, specs: list[dict[str, Any]]) -> Path:
    destination = config.output_file("intent_specs")
    document = {
        "meta": {
            "generated_by": "bootstrap_specs.py",
            "intent_count": len(specs),
            "sources": _origin_counts(specs),
            "note": (
                "Derived from legacy Dialogflow seeds. REQUIRES HUMAN REVIEW "
                "before Stage 1 generation; these specs are the source of truth "
                "for every downstream label. See `sources` for provenance."
            ),
        },
        "intents": specs,
    }
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle, sort_keys=False, default_flow_style=False, width=88)
    return destination


def export_summary(config: GeneratorConfig, specs: list[dict[str, Any]]) -> Path:
    destination = config.output_file("specs_summary")
    lines = [
        "# Intent Specifications Summary",
        "",
        f"{len(specs)} intents. Provenance: "
        + ", ".join(f"{n} × {k}" for k, n in _origin_counts(specs).items())
        + ".",
        "",
        "> **Review these before Stage 1.** They are the source of truth for every",
        "> downstream label, so an error here is multiplied by the per-intent",
        "> generation budget.",
        "",
        "| Intent | Family | Business description | Positive | Hard negative |",
        "|---|---|---|---|---|",
    ]
    for spec in sorted(specs, key=lambda s: (s.get("intent_family", ""), s["name"])):
        description = " ".join(str(spec["business_description"]).split())
        lines.append(
            f"| `{spec['name']}` | {spec.get('intent_family', '—')} | {description} | "
            f"{spec.get('positive_example', '')} | {spec.get('hard_negative_example', '')} |"
        )

    lines += ["", "## Neighbour graph", "", "| Intent | Neighbours |", "|---|---|"]
    for spec in sorted(specs, key=lambda s: s["name"]):
        lines.append(f"| `{spec['name']}` | {', '.join(spec.get('neighbor_intents', []))} |")

    lines.append("")
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def load_hand_authored(
    config: GeneratorConfig,
    allowed: set[str],
    family_of: dict[str, str],
) -> list[dict[str, Any]]:
    """Load specs written by hand, validating them exactly like generated ones.

    Some intents must never reach the LLM -- currently ``Default Fallback
    Intent``, whose seed utterances are raw production transcripts carrying real
    customer PII. Hand-authoring is not a lower standard: these specs run
    through the same ``validate_spec`` checks, so a stale neighbour name here
    fails just as loudly as a hallucinated one from the model.
    """
    names = list(config.taxonomy.get("hand_authored_intents") or [])
    if not names:
        return []

    path = config.output_file("hand_authored_specs")
    if not path.is_file():
        raise SeedCorpusError(
            f"taxonomy.hand_authored_intents lists {names} but {path.name} does not exist"
        )

    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    by_name = {str(entry.get("name")): entry for entry in document.get("intents", [])}

    missing = [name for name in names if name not in by_name]
    if missing:
        raise SeedCorpusError(f"{path.name} is missing hand-authored specs for: {missing}")

    records: list[dict[str, Any]] = []
    for name in names:
        if name not in allowed:
            raise SeedCorpusError(
                f"{path.name} defines {name!r}, which is not in the resolved taxonomy"
            )
        spec = IntentSpecification.model_validate(by_name[name])
        errors = validate_spec(spec, name, allowed)
        if errors:
            raise SeedCorpusError(
                f"Hand-authored spec for {name!r} is invalid:\n"
                + "\n".join(f"  · {e}" for e in errors)
            )
        record = spec.model_dump()
        record["intent_family"] = family_of.get(name, "Unassigned")
        origin = str(by_name[name].get("authored_by", "authored-directly"))
        record["_provenance"] = {"model": origin, "attempts": 0, "source": path.name}
        records.append(record)

    return records


def _fingerprint(config: GeneratorConfig, intents: list[str]) -> str:
    payload = json.dumps(
        {
            "model": config.llm["model"],
            "temperature": config.llm["temperature"],
            "sampling": config.sampling,
            "seed": config.random_seed,
            "intents": intents,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap Intent Specifications from seeds.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the taxonomy and sample seeds without calling the LLM.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N intents.")
    parser.add_argument("--force", action="store_true", help="Ignore existing checkpoints.")
    parser.add_argument("--only", nargs="*", default=None, help="Process only these intents.")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        corpus = load_seed_corpus(config)
    except SeedCorpusError as exc:
        print(f"ERROR: {exc}")
        return 1

    allowed = set(corpus.intents)
    family_of = config.family_of

    hand_authored_names = set(config.taxonomy.get("hand_authored_intents") or [])
    try:
        hand_specs = load_hand_authored(config, allowed, family_of)
    except SeedCorpusError as exc:
        print(f"ERROR: {exc}")
        return 1

    intents = [i for i in corpus.intent_names if i not in hand_authored_names]
    if args.only:
        unknown = [i for i in args.only if i not in corpus.intents]
        if unknown:
            print(f"ERROR: unknown intents: {unknown}")
            return 1
        intents = [i for i in args.only if i not in hand_authored_names]
    if args.limit:
        intents = intents[: args.limit]

    print(
        f"Resolved taxonomy: {len(corpus)} intents "
        f"({len(corpus.excluded)} excluded, {len(corpus.merged)} merged, "
        f"{len(corpus.dropped)} dropped). "
        f"{len(hand_specs)} hand-authored, {len(intents)} to generate."
    )
    for name in sorted(hand_authored_names):
        print(f"  hand-authored, never sent to the LLM: {name}")

    rng = random.Random(config.random_seed)

    if args.dry_run:
        total_noise = 0
        for intent in intents:
            seeds, noise = sample_seeds_for_intent(config, corpus.intents[intent], rng=rng)
            total_noise += len(noise)
            print(
                f"  {intent:42s} family={family_of.get(intent, '—'):18s} "
                f"unique={len(corpus.intents[intent]):4d} sampled={len(seeds):3d} "
                f"noise_dropped={len(noise):3d}"
            )
        print(
            f"\nDry run complete — no LLM calls made, no files written. "
            f"Noise guard withheld {total_noise} seed phrases. "
            f"{len(hand_specs)} hand-authored spec(s) loaded and validated."
        )
        return 0

    # Every intent may already be authored outside the LLM path. In that case
    # this tool still has work to do -- validate and export -- so it must not
    # demand an API key it will never use.
    if intents and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='...'")
        return 1

    checkpoints = Checkpoints(config.checkpoint_dir, _fingerprint(config, corpus.intent_names))
    chain = _build_chain(config) if intents else None

    specs: list[dict[str, Any]] = list(hand_specs)
    failures: list[tuple[str, list[str]]] = []

    for position, intent in enumerate(intents, start=1):
        if not args.force:
            cached = checkpoints.load(intent)
            if cached is not None:
                print(f"[{position}/{len(intents)}] {intent} — resumed from checkpoint")
                specs.append(cached)
                continue

        seeds, _noise = sample_seeds_for_intent(config, corpus.intents[intent], rng=rng)
        payload = _payload(config, corpus, intent, seeds)
        spec, attempts, errors = generate_one(chain, payload, intent, allowed, config)

        if spec is None:
            print(f"[{position}/{len(intents)}] {intent} — FAILED after {attempts} attempts")
            for error in errors:
                print(f"      · {error}")
            failures.append((intent, errors))
            continue

        record = spec.model_dump()
        record["intent_family"] = family_of.get(intent, "Unassigned")
        record["neighbor_intents"] = [
            n for n in record["neighbor_intents"] if n in allowed and n != intent
        ]
        record["_provenance"] = {
            "model": config.llm["model"],
            "attempts": attempts,
            "seeds_shown": len(seeds),
            "seeds_available": len(corpus.intents[intent]),
        }
        checkpoints.save(intent, record, attempts)
        specs.append(record)
        suffix = f" (after {attempts} attempts)" if attempts > 1 else ""
        print(f"[{position}/{len(intents)}] {intent} — ok{suffix}")

    if not specs:
        print("No specifications generated.")
        return 1

    yaml_path = export_yaml(config, specs)
    summary_path = export_summary(config, specs)

    print(f"\nWrote {yaml_path}")
    print(f"Wrote {summary_path}")
    print(f"Succeeded: {len(specs)}   Failed: {len(failures)}")
    if failures:
        print("Failed intents (re-run to retry; successes are checkpointed):")
        for intent, _ in failures:
            print(f"  - {intent}")
        return 1

    print("\nPhase 1 bootstrap complete. Review specs_summary.md before Stage 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
