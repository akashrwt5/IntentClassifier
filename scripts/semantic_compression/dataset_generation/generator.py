"""Phase 2/3 -- the Super Dataset generator.

Stage 1 (positives) is implemented here. Stages 2 and 3 plug into the same
engine and are the remaining Phase 3 work.

Three things drive the design, all of them consequences of the blueprint rather
than of convenience:

* **Batching, not one big call.** Asking for an intent's whole budget in one
  request produces visibly degrading output: the model exhausts its genuinely
  distinct phrasings early and pads the tail with the word-substitution
  variants Section 1 exists to eliminate. Batches are small, and every batch is
  shown what has already been accepted so it cannot repeat it.

* **Specs are the prompt.** Step 0 says load the specification first and treat
  it as the sole source of truth. The spec's own ``do_not_trigger`` and
  ``boundary_cases`` are pasted into each request verbatim, so the model is
  bounded by the reviewed artefact rather than by its own inference.

* **Reject at generation time.** A label that contradicts the taxonomy is far
  cheaper to catch now, where a retry fixes it, than in the Section 8 confusion
  matrix after training.

Usage::

    python generator.py --dry-run          # plan the run, no API calls
    python generator.py --only Cmd.VolumeIncrease
    python generator.py --limit 3          # small paid smoke test
    python generator.py                    # full Stage 1 (resumes)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

from llm_client import (
    build_structured_llm,
    invoke_with_validation,
    required_api_key,
)
from schemas import LONG_FORM_TYPES, GeneratedBatch, UtteranceType
from seed_loader import (
    GeneratorConfig,
    SeedCorpusError,
    dedupe_key,
    decode_seed_file,
    load_config,
    load_seed_corpus,
    normalize_utterance,
)

DEFAULT_CONFIG = Path(__file__).with_name("generator_config.yaml")


SYSTEM_PROMPT = """\
# Role
You are a Principal Conversational AI Architect, Senior Machine Learning \
Engineer and Dataset Engineer with more than 15 years building production \
Conversational AI deployed to millions of users. You specialise in datasets for \
small (~10MB) sentence-embedding models that run entirely on-device, for a \
HEARING AID voice assistant.

Your objective is not paraphrasing. It is to engineer a semantic dataset that \
maximises intent-classification accuracy, semantic diversity, robustness and \
generalisation, while minimising intent overlap, dataset bias, label leakage \
and ambiguous samples. Think like an ML engineer optimising a production model, \
not like a chatbot producing text.

# Philosophy
Prioritise semantic diversity over syntactic diversity; real user behaviour \
over textbook grammar; production robustness over dataset size.

Word-substitution variants are near-worthless here. "Increase volume" / "Raise \
volume" / "Turn volume up" occupy almost the same point in embedding space and \
teach the model nothing. Utterances like "Can you make it easier to hear?", \
"I'm struggling to hear people" and "Could you speak louder?" introduce \
genuinely different semantic representations. Produce the second kind.

# Type (use these EXACT values, and these definitions)
- ExplicitCommand: direct request for action. NOTE: a grammatical question that \
requests an action ("Can you turn it up?") is a COMMAND, not a Question.
- ImplicitCommand: indirect request expressing a personal need that implies an \
action ("I'm struggling to hear people").
- Observation: factual statement describing a state WITHOUT requesting a \
change. Reserved for the Fallback intent.
- ObservationPlusCommand: an observation followed by a request.
- Question: a request for INFORMATION ONLY.
- Negation: an instruction cancelling an action.
- Conversation: casual speech.
- Fallback: out-of-scope or unsupported.

# Difficulty
- Easy: direct commands, high lexical similarity to seed data.
- Medium: indirect requests, polite phrasing, mild ASR variation.
- Hard: compound utterances, heavy ASR corruption, long conversational \
phrasing, boundary cases.

# Precedence rules -- apply consistently
1. Compound: if an utterance contains BOTH an observation and an actionable \
command, the command determines the intent. Pure observations are Fallback.
2. Conflicting commands: the LAST explicit command wins.
3. Negations: cancelling with no replacement is Fallback; with a replacement, \
classify by the replacement.
4. Anything still unresolved defers to the spec's Boundary Cases, then Fallback.

# Hard constraints -- violating any of these makes the batch worthless
- NEVER invent a product capability that the specification does not state.
- NEVER change the business meaning of the intent.
- NEVER produce a near-duplicate of an utterance you were told to avoid, or of \
another utterance in this same batch.
- NEVER exceed about 20 words, EXCEPT for ObservationPlusCommand, Conversation \
and Fallback, which may be longer.
- NEVER emit unnatural or unrealistic speech. ASR corruption must be plausible \
mis-hearing, not random noise.
- Every utterance you return must belong to the requested intent and satisfy \
its Trigger Conditions.

Context that should shape every judgement: for these users a FALSE ACCEPT -- an \
information question misread as a command, physically changing a hearing aid -- \
is far costlier than a false reject. Keep the intent's boundaries tight.
"""

HUMAN_PROMPT = """\
# Intent Specification (the sole source of truth)
Intent name: {intent}
Intent family: {family}

Business description:
{business_description}

Trigger conditions:
{trigger_conditions}

Do NOT trigger:
{do_not_trigger}

Boundary / uncertain cases:
{boundary_cases}

Neighbour intents (most likely confusions):
{neighbor_intents}

Reference positive: {positive_example}
Reference hard negative (must NOT be generated as a positive): \
{hard_negative_example}
{slot_block}{seed_block}
# Task
Generate exactly {batch_size} NEW positive utterances for `{intent}`.

Every utterance must have `intent` set to exactly `{intent}`.
Aim for roughly this difficulty mix: {difficulty_mix}.

Cover a spread of speaking styles, not a spread of wordings:
polite/indirect requests; short abrupt commands; elderly phrasing; partial
sentences and filled pauses; compound observation-then-command; and plausible
ASR mis-transcription (mark those `ASR-Simulated`, everything else
`LLM-Generated`).

{avoid_block}{correction}
Return the batch."""


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _bullets(values: Iterable[str]) -> str:
    items = [str(v).strip() for v in values if str(v).strip()]
    return "\n".join(f"- {v}" for v in items) if items else "- (none stated)"


def _slot_block(config: GeneratorConfig, intent: str) -> str:
    """Inline the intent's slot vocabulary, if it has one.

    ``Cmd.MemoryChange`` is the reason this exists: 94% of its real utterances
    name one of the memory values, so a generator that never sees that list
    produces data covering a fraction of the slot space the product uses.
    """
    vocab_files = (config.taxonomy.get("slot_vocabularies") or {}).get(intent) or []
    values: list[str] = []
    for stem in vocab_files:
        path = config.seed_dir / f"{stem}.txt"
        if not path.is_file():
            continue
        _, text = decode_seed_file(path)
        values += [normalize_utterance(line) for line in text.splitlines() if line.strip()]

    if not values:
        return ""
    unique = sorted({v for v in values if v})
    return (
        "\n# Slot vocabulary for this intent\n"
        "Distribute your utterances across these values; do not favour the "
        "first few.\n" + ", ".join(unique) + "\n"
    )


def _seed_block(seeds: list[str]) -> str:
    if not seeds:
        return ""
    return (
        "\n# Real seed utterances for this intent (style reference only)\n"
        "Do NOT copy or lightly reword these. They show how real users speak; "
        "your job is to extend that space, not restate it.\n"
        + "\n".join(f"- {s}" for s in seeds)
        + "\n"
    )


def _avoid_block(accepted: list[str], limit: int) -> str:
    if not accepted:
        return ""
    shown = accepted[-limit:]
    return (
        "# Already generated -- do NOT repeat or lightly reword any of these\n"
        + "\n".join(f"- {a}" for a in shown)
        + "\n\n"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_batch(
    batch: GeneratedBatch,
    *,
    intent: str,
    fallback_intent: str,
    accepted_keys: set[str],
    max_words: int,
    expected_size: int,
) -> list[str]:
    """Return rejection reasons for a generated batch; empty means accept."""
    errors: list[str] = []

    if not batch.utterances:
        return ["batch is empty"]
    if len(batch.utterances) < max(1, expected_size // 2):
        errors.append(
            f"returned {len(batch.utterances)} utterances; {expected_size} were requested"
        )

    seen: set[str] = set()
    for item in batch.utterances:
        text = item.utterance

        if item.intent != intent:
            errors.append(f"{text!r}: intent is {item.intent!r}, must be {intent!r}")

        # Section 6's Command-vs-Observation policy, enforced at generation
        # time. A pure Observation belongs to Fallback by definition, so one
        # labelled with an actionable intent is a contradiction -- and it is
        # exactly the sample that would later teach a false accept.
        if item.type is UtteranceType.OBSERVATION and intent != fallback_intent:
            errors.append(
                f"{text!r}: type Observation is only valid for {fallback_intent!r}; "
                "if it implies an action use ImplicitCommand or ObservationPlusCommand"
            )

        if item.type not in LONG_FORM_TYPES and len(text.split()) > max_words:
            errors.append(
                f"{text!r}: {len(text.split())} words exceeds the {max_words}-word limit "
                f"for type {item.type.value}"
            )

        key = dedupe_key(text)
        if key in seen:
            errors.append(f"{text!r}: duplicated within this batch")
        elif key in accepted_keys:
            errors.append(f"{text!r}: duplicates an utterance already accepted")
        seen.add(key)

    return errors


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _budget(config: GeneratorConfig, intent: str) -> int:
    gen = config.raw.get("generation", {}).get("utterances_per_intent", {})
    return int((gen.get("overrides") or {}).get(intent, gen.get("default", 120)))


def _difficulty_mix(config: GeneratorConfig) -> str:
    mix = config.raw.get("generation", {}).get("difficulty_mix") or {}
    if not mix:
        return "40% Easy, 40% Medium, 20% Hard"
    return ", ".join(f"{int(v * 100)}% {k}" for k, v in mix.items())


def _load_specs(config: GeneratorConfig) -> dict[str, dict[str, Any]]:
    import yaml

    path = config.output_file("intent_specs")
    if not path.is_file():
        raise SeedCorpusError(f"{path.name} not found -- run bootstrap_specs.py first")
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {s["name"]: s for s in document.get("intents", [])}


class Stage1Store:
    """Append-only per-intent store, so a crash costs one batch, not one run."""

    def __init__(self, root: Path) -> None:
        self.dir = root / "stage1"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rejects = root / "rejections.jsonl"

    def _path(self, intent: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in intent)
        return self.dir / f"{safe}.jsonl"

    def load(self, intent: str) -> list[dict[str, Any]]:
        path = self._path(intent)
        if not path.is_file():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def append(self, intent: str, rows: list[dict[str, Any]]) -> None:
        with self._path(intent).open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def log_rejections(self, rejections: list[Any]) -> None:
        if not rejections:
            return
        with self.rejects.open("a", encoding="utf-8") as handle:
            for r in rejections:
                handle.write(
                    json.dumps(
                        {"intent": r.intent, "attempt": r.attempt, "reasons": r.reasons},
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the Super Dataset (Stage 1).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Plan the run, make no calls.")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Ignore existing Stage 1 output.")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        corpus = load_seed_corpus(config)
        specs = _load_specs(config)
    except SeedCorpusError as exc:
        print(f"ERROR: {exc}")
        return 1

    intents = [i for i in corpus.intent_names if i in specs]
    missing = [i for i in corpus.intent_names if i not in specs]
    if missing:
        print(f"ERROR: no specification for: {missing}")
        return 1
    if args.only:
        unknown = [i for i in args.only if i not in specs]
        if unknown:
            print(f"ERROR: unknown intents: {unknown}")
            return 1
        intents = list(args.only)
    if args.limit:
        intents = intents[: args.limit]

    gen_cfg = config.raw.get("generation", {})
    batch_size = int(gen_cfg.get("batch_size", 25))
    avoid_max = int(gen_cfg.get("avoid_list_max", 60))
    max_words = int(gen_cfg.get("max_utterance_words", 20))
    seed_ref = int(gen_cfg.get("seed_reference_count", 8))

    store = Stage1Store(config.checkpoint_dir)
    plan = []
    for intent in intents:
        have = 0 if args.force else len(store.load(intent))
        want = _budget(config, intent)
        plan.append((intent, have, want, max(0, -(-(want - have) // batch_size))))

    total_calls = sum(p[3] for p in plan)
    print(f"Stage 1 plan — {len(plan)} intents, {total_calls} LLM calls at batch={batch_size}")
    for intent, have, want, calls in plan:
        if calls or args.dry_run:
            print(f"  {intent:34s} have={have:4d} want={want:4d} calls={calls:3d}")
    print(f"  target utterances: {sum(p[2] for p in plan)}")

    if args.dry_run:
        print("\nDry run — no API calls made, nothing written.")
        return 0

    key = required_api_key(config)
    if not os.environ.get(key):
        print(f"ERROR: {key} is not set (llm.provider={config.llm.get('provider')})")
        return 1

    from langchain_core.prompts import ChatPromptTemplate

    chain = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    ) | build_structured_llm(config, GeneratedBatch, stage="generation")

    mix = _difficulty_mix(config)
    failures: list[str] = []

    for position, (intent, _, want, _) in enumerate(plan, start=1):
        spec = specs[intent]
        rows = [] if args.force else store.load(intent)
        accepted = [r["utterance"] for r in rows]
        keys = {dedupe_key(a) for a in accepted}
        seeds = corpus.intents.get(intent, [])[:seed_ref]

        while len(accepted) < want:
            size = min(batch_size, want - len(accepted))
            payload = {
                "intent": intent,
                "family": spec.get("intent_family", "Unassigned"),
                "business_description": spec["business_description"],
                "trigger_conditions": _bullets(spec["trigger_conditions"]),
                "do_not_trigger": _bullets(spec["do_not_trigger"]),
                "boundary_cases": _bullets(spec["boundary_cases"]),
                "neighbor_intents": ", ".join(spec["neighbor_intents"]),
                "positive_example": spec.get("positive_example", ""),
                "hard_negative_example": spec.get("hard_negative_example", ""),
                "slot_block": _slot_block(config, intent),
                "seed_block": _seed_block(seeds),
                "avoid_block": _avoid_block(accepted, avoid_max),
                "batch_size": size,
                "difficulty_mix": mix,
            }

            outcome = invoke_with_validation(
                chain,
                payload,
                validate=lambda b: validate_batch(
                    b,
                    intent=intent,
                    fallback_intent=config.fallback_intent,
                    accepted_keys=keys,
                    max_words=max_words,
                    expected_size=size,
                ),
                config=config,
                label=intent,
            )
            store.log_rejections(outcome.rejections)

            if not outcome.ok:
                print(f"[{position}/{len(plan)}] {intent} — batch FAILED, moving on")
                for reason in outcome.rejections[-1].reasons[:3]:
                    print(f"      · {reason}")
                failures.append(intent)
                break

            new_rows = []
            for item in outcome.value.utterances:
                key_ = dedupe_key(item.utterance)
                if key_ in keys:
                    continue
                keys.add(key_)
                accepted.append(item.utterance)
                new_rows.append(
                    {
                        "utterance": item.utterance,
                        "intent": item.intent,
                        "intent_family": spec.get("intent_family", "Unassigned"),
                        "type": item.type.value,
                        "difficulty": item.difficulty.value,
                        "source": item.source.value,
                    }
                )
            store.append(intent, new_rows)
            print(
                f"[{position}/{len(plan)}] {intent} — {len(accepted)}/{want} "
                f"(+{len(new_rows)}, attempts={outcome.attempts})"
            )

    print(f"\nStage 1 complete. Failures: {len(set(failures))}")
    print(f"Output: {store.dir}")
    if store.rejects.is_file():
        print(f"Rejection log: {store.rejects}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
