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
import re
from pathlib import Path
from typing import Any, Iterable

from llm_client import (
    Rejection,
    build_structured_llm,
    invoke_with_validation,
    required_api_key,
    resolve_api_key,
)
from schemas import LONG_FORM_TYPES, GeneratedBatch, UtteranceType, batch_model_for
from seed_loader import (
    GeneratorConfig,
    SeedCorpusError,
    dedupe_key,
    decode_seed_file,
    load_config,
    load_seed_corpus,
    normalize_utterance,
    sample_seeds_for_intent,
)

DEFAULT_CONFIG = Path(__file__).with_name("generator_config.yaml")

#: Surface markers of a need or a request. An ImplicitCommand has to imply an
#: action; an utterance carrying none of these is describing a state, not
#: asking for anything.
#:
#: This list is BRITTLE and will stay brittle. It tries to detect meaning by
#: matching strings, and every widening of the generator's phrasing finds
#: another hole in it. Three have been found so far, each on a perfectly
#: ordinary request the check then threw away:
#:
#:     "I'm not getting enough volume in my right ear"   -- no `enough` phrase
#:     "Right side needs volume"                         -- `need` did not match `needs`
#:     "I'd like the right aid to stop making any sound" -- `would` does not match `I'd`
#:
#: Patch the holes as they appear, but do not expect the next patch to be the
#: last one. What makes that acceptable is the cost: since validation became
#: per-row, a false positive drops a single utterance and the next batch makes
#: up the shortfall. It is a rounding error, not a failed run. It stops being
#: acceptable if the drop rate ever climbs, which the report's rejection
#: section is there to show.
#:
#: Widen it with care. An attempt to cover the mute intents by adding
#: `quiet\w*`, `silen\w*` and `off` was reverted before it shipped: those words
#: describe a STATE as readily as a request, so the list would have started
#: passing "everything sounds quiet today" and "my aid is off" -- the pure
#: observations this check exists to reject, and the exact rows that teach a
#: false accept. Only verbs survive here. `mute` and `silence` are actions;
#: `quiet` and `silent` are conditions.
_REQUEST_SIGNAL = re.compile(
    r"(\b(need\w*|want\w*|could|can|cannot|can'?t|would|please|help|make|turn|raise|"
    r"give|boost|increase|louder|higher|more|up|wish|let|unable|barely|hardly|"
    r"mute|silence|"
    r"struggl\w*|trying|difficult\w*)\b|hard time|trouble\s+\w+ing|"
    r"not getting (enough|much)|make out|keep up|catch what|"
    r"\b(i'?d|we'?d|would|i'?ll) like\b)",
    re.IGNORECASE,
)

#: Openers that cancel a previous instruction.
_NEGATION_OPENER = re.compile(r"^\s*(don'?t|do not|no,|not that|stop|cancel)\b", re.IGNORECASE)


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
- Easy: short, single-clause and direct. No observation clause, no hedging.
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
{composition_block}
Cover a spread of speaking styles, not a spread of wordings:
polite/indirect requests; short abrupt commands; elderly phrasing; partial
sentences and filled pauses; and plausible ASR mis-transcription (mark those
`ASR-Simulated`, everything else `LLM-Generated`).

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


def _evidence_block(
    config: GeneratorConfig,
    phrases: list[str],
    intent: str,
    k: int,
) -> str:
    """The block of real-user evidence shown to the model for one intent.

    Normally a maximum-diversity sample of that intent's own seeds. For intents
    listed under ``generation.privacy.no_seed_block_intents`` it is a
    hand-written description of how the intent sounds instead, and no seed text
    leaves the machine.

    The distinction this exists to fix: hand-authoring a SPECIFICATION keeps
    that spec off a third-party API, which is what
    ``taxonomy.hand_authored_intents`` achieves. It does nothing about the
    GENERATION call, which is a separate request carrying seed utterances of its
    own. ``Default Fallback Intent`` is 613 raw production ASR transcripts, so
    the two paths have to be handled separately.

    The substitution lives in ONE function on purpose. ``render_prompt.py``
    renders the identical prompt for manual evaluation, and a privacy rule
    applied to only one of the two call sites leaks the first time somebody uses
    the other one -- which is exactly how the seed sampler came to be applied on
    one path and not the other.
    """
    privacy = config.raw.get("generation", {}).get("privacy") or {}
    if intent in set(privacy.get("no_seed_block_intents") or []):
        replacement = str((privacy.get("seed_block_replacement") or {}).get(intent, ""))
        return f"\n{replacement.strip()}\n" if replacement.strip() else ""
    selected, _noise = sample_seeds_for_intent(config, phrases, k=k)
    return _seed_block(selected)


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
    cancellation_capability: bool = False,
) -> tuple[list[str], dict[int, list[str]]]:
    """Validate a generated batch, returning ``(batch_errors, row_errors)``.

    Section 7's Validation Failure Policy is explicit that a failure should
    "regenerate only the failed sample (not the full batch)", so the two kinds
    of failure are kept apart rather than flattened into one list:

    * **batch_errors** are structural -- an empty response, or one short enough
      to suggest the call itself went wrong. Nothing in it is trustworthy, so
      the whole batch is retried.
    * **row_errors** are faults in individual utterances, keyed by position.
      The driver drops those rows and keeps the rest; the shortfall is picked up
      by the next batch, which is what "regenerate only the failed sample"
      amounts to in a batched pipeline.

    Flattening the two was expensive in practice: a single duplicate failed all
    25 utterances in its batch, and three such failures abandoned the intent's
    entire remaining budget -- worst exactly where the budget is largest.
    """
    batch_errors: list[str] = []
    row_errors: dict[int, list[str]] = {}

    def fault(index: int, message: str) -> None:
        row_errors.setdefault(index, []).append(message)

    if not batch.utterances:
        return ["batch is empty"], row_errors
    if len(batch.utterances) < max(1, expected_size // 2):
        batch_errors.append(
            f"returned {len(batch.utterances)} utterances; {expected_size} were requested"
        )

    seen: set[str] = set()
    for index, item in enumerate(batch.utterances):
        text = item.utterance

        if item.intent != intent:
            fault(index, f"{text!r}: intent is {item.intent!r}, must be {intent!r}")

        # Section 6's Command-vs-Observation policy, enforced at generation
        # time. A pure Observation belongs to Fallback by definition, so one
        # labelled with an actionable intent is a contradiction -- and it is
        # exactly the sample that would later teach a false accept.
        if item.type is UtteranceType.OBSERVATION and intent != fallback_intent:
            fault(
                index,
                f"{text!r}: type Observation is only valid for {fallback_intent!r}; "
                "if it implies an action use ImplicitCommand or ObservationPlusCommand",
            )

        # The Observation rule above is bypassed whenever a model files a pure
        # observation under ImplicitCommand instead -- which happens in
        # practice. Left in, such a row teaches that "everything sounds faint"
        # IS a command to raise the volume, i.e. exactly the false accept this
        # project is built to avoid.
        if item.type is UtteranceType.IMPLICIT_COMMAND and not _REQUEST_SIGNAL.search(text):
            fault(
                index,
                f"{text!r}: labelled ImplicitCommand but states no need and requests "
                f"nothing. A pure observation belongs to {fallback_intent!r}; "
                "express a need or relabel.",
            )

        # Section 6 precedence rule 3, stated the way the blueprint states it:
        # a cancellation offering no replacement belongs to the FALLBACK INTENT.
        # It is not a positive for this intent wearing a different `type`, so
        # relabelling it Negation and leaving it filed under the very intent it
        # cancels is the wrong remedy -- which is what the previous version of
        # this check asked for.
        #
        # Skipped for intents whose capability IS cancellation, where the opener
        # carries no negation at all: "stop streaming" invokes
        # Cmd.StreamingStop rather than negating anything. 40% of that intent's
        # shipping seeds and 31% of Cmd.EdgeModeDeactivate's open this way.
        if (
            not cancellation_capability
            and item.type is not UtteranceType.NEGATION
            and _NEGATION_OPENER.match(text)
        ):
            fault(
                index,
                f"{text!r}: opens by cancelling an action but is typed "
                f"{item.type.value}. Cancelling with no replacement belongs to "
                f"{fallback_intent!r}, not here; if it does name a replacement, "
                "classify by that replacement and type it Negation.",
            )

        if item.type not in LONG_FORM_TYPES and len(text.split()) > max_words:
            fault(
                index,
                f"{text!r}: {len(text.split())} words exceeds the {max_words}-word limit "
                f"for type {item.type.value}",
            )

        key = dedupe_key(text)
        if key in seen:
            fault(index, f"{text!r}: duplicated within this batch")
        elif key in accepted_keys:
            fault(index, f"{text!r}: duplicates an utterance already accepted")
        seen.add(key)

    return batch_errors, row_errors


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _budget(config: GeneratorConfig, intent: str) -> int:
    gen = config.raw.get("generation", {}).get("utterances_per_intent", {})
    return int((gen.get("overrides") or {}).get(intent, gen.get("default", 120)))


def _cancellation_capability(config: GeneratorConfig, intent: str) -> bool:
    """Whether cancelling IS this intent's capability rather than a negation.

    Listed in ``generation.validation.cancellation_capability_intents`` so the
    exemption stays auditable next to the rule it exempts, rather than being a
    set literal buried in a regex check.
    """
    names = (config.raw.get("generation", {}).get("validation") or {}).get(
        "cancellation_capability_intents"
    ) or []
    return intent in set(names)


def _difficulty_mix(config: GeneratorConfig) -> str:
    mix = config.raw.get("generation", {}).get("difficulty_mix") or {}
    if not mix:
        return "40% Easy, 40% Medium, 20% Hard"
    return ", ".join(f"{int(v * 100)}% {k}" for k, v in mix.items())


def _quota_profile(config: GeneratorConfig, intent: str) -> dict[str, Any]:
    """The quota profile that applies to ``intent``, longest prefix wins.

    Command, Help and Fallback intents have fundamentally different natural
    shapes -- a command is short and imperative, a Help utterance is a question,
    Fallback is deliberately shapeless -- so one set of numbers cannot serve all
    three. Only ``command`` currently carries numbers; the others are empty by
    design until they have been measured rather than guessed.
    """
    quotas = config.raw.get("generation", {}).get("quotas") or {}
    profiles = quotas.get("profiles") or {}
    chosen, best = None, -1
    for pattern, name in (quotas.get("assign") or {}).items():
        if (intent == pattern or intent.startswith(pattern)) and len(pattern) > best:
            chosen, best = name, len(pattern)
    return dict(profiles.get(chosen) or {})


def _composition_block(config: GeneratorConfig, intent: str, size: int) -> str:
    """Render the batch's composition requirements as COUNTS.

    A list of proportions is read and approximated away: the run that asked for
    "roughly 35/40/25" returned 21/40/39, and the style list that named "short
    abrupt commands" produced one utterance of four words or fewer out of 180.
    Counts are checkable by the model before it answers, and by the report
    afterwards.
    """

    def n(fraction: float) -> int:
        """Count for a fraction of this batch. Zero means "state no rule"."""
        return max(0, round(float(fraction) * size))

    def apportion(shares: dict[str, float]) -> dict[str, int]:
        """Split ``size`` across shares so the parts sum to EXACTLY ``size``.

        Rounding each share independently does not: at a batch of 18 the
        difficulty mix rounds to 6/7/4, and telling a model to produce
        "6 Easy, 7 Medium, 4 Hard" out of 18 is an instruction it cannot
        satisfy. Largest-remainder apportionment gives every part its floor and
        hands the leftovers to the largest fractions, which always totals
        ``size``.

        Only batch sizes 5, 10, 20 and 25 arise from the budgets as they stand,
        and none of those are affected -- but the failure is silent and would
        surface as an unexplained drop in compliance the first time somebody
        set a budget that is not a multiple of five.
        """
        total = sum(shares.values()) or 1.0
        exact = {k: size * v / total for k, v in shares.items()}
        counts = {k: int(v) for k, v in exact.items()}
        for key in sorted(exact, key=lambda k: exact[k] - counts[k], reverse=True):
            if sum(counts.values()) >= size:
                break
            counts[key] += 1
        return counts

    rules: list[str] = []
    profile = _quota_profile(config, intent)

    def span(bounds: Any, noun: str) -> str | None:
        """One rule line, or None when the batch is too small to constrain."""
        low, high = bounds
        lo = n(low)
        if high is None:
            return f"- at least {lo} {noun}." if lo else None
        hi = max(n(high), lo)
        if not hi:
            return None
        if lo == hi:
            return f"- exactly {lo} {noun}."
        return f"- between {lo} and {hi} {noun}."

    if n(profile.get("min_short", 0)):
        rules.append(
            f"- at least {n(profile['min_short'])} must be FOUR WORDS OR FEWER. "
            'Real users say "Louder" and "Turn it up" far more often than '
            "they say anything longer; this is the commonest shape, not a "
            "garnish."
        )
    # Type minima must leave room for one another. If they cannot, the batch is
    # too small to carry every constraint and the weakest are dropped rather
    # than issuing a set of rules with no satisfying assignment.
    budget_left = size - (n(profile.get("min_short", 0)) if False else 0)
    type_rules: list[str] = []
    claimed = 0
    for type_name, bounds in (profile.get("types") or {}).items():
        line = span(bounds, f"must have type `{type_name}`")
        if line is None:
            continue
        claimed += n(bounds[0])
        if claimed > size:
            break
        type_rules.append(line)
    rules += type_rules

    if profile.get("asr_simulated"):
        line = span(profile["asr_simulated"], "must be source `ASR-Simulated`")
        if line:
            rules.append(line + " The rest are `LLM-Generated`.")

    mix = config.raw.get("generation", {}).get("difficulty_mix") or {}
    if mix:
        counts = apportion(mix)
        rules.append(
            "- difficulty: exactly " + ", ".join(f"{counts[k]} {k}" for k in mix if counts[k]) + "."
        )

    if not rules:
        return ""
    return (
        "\nComposition of this batch -- these are COUNTS, not suggestions. "
        f"Of the {size} utterances:\n" + "\n".join(rules) + "\nCount them before you answer.\n"
    )


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

    def reset(self, intent: str) -> None:
        """Clear an intent's output AND its rejection history for ``--force``.

        ``append`` opens in ``"a"`` mode, so without this a forced rerun leaves
        the previous run's rows in place and every downstream report scores the
        union of two runs as if it were one.

        The rejection log needs the same treatment, and for a subtler reason:
        it is shared across intents and is never truncated, so after a rerun it
        mixes this run's drops with the previous run's. That misled a review
        once already -- three stale entries were read as current failures. Only
        this intent's entries are removed, so a targeted rerun does not destroy
        the record for intents it did not touch.
        """
        path = self._path(intent)
        if path.is_file():
            path.write_text("", encoding="utf-8")
        if self.rejects.is_file():
            kept = [
                line
                for line in self.rejects.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line).get("intent") != intent
            ]
            self.rejects.write_text("".join(line + "\n" for line in kept), encoding="utf-8")

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
    parser.add_argument(
        "--batches",
        type=int,
        default=0,
        help=(
            "Stop each intent after N batches instead of running to budget. "
            "A paid smoke test: one batch is a few cents and is the only way to "
            "measure what a full run will actually produce, because the manual "
            "harness cannot exercise the API model or its structured output."
        ),
    )
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
    max_stalled = int(gen_cfg.get("max_consecutive_failed_batches", 3))

    store = Stage1Store(config.checkpoint_dir)
    plan = []
    for intent in intents:
        have = 0 if args.force else len(store.load(intent))
        want = _budget(config, intent)
        plan.append((intent, have, want, max(0, -(-(want - have) // batch_size))))

    if args.batches:
        plan = [(i, h, w, min(c, args.batches)) for i, h, w, c in plan]
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
    if not resolve_api_key(config):
        secrets_file = (config.raw.get("paths") or {}).get("secrets_file", "(unset)")
        print(f"ERROR: no API key for llm.provider={config.llm.get('provider')!r}.")
        print(f"  Either export {key}, or put it in {secrets_file} as:")
        print(f'      {key}: "..."')
        return 1

    from langchain_core.prompts import ChatPromptTemplate

    # The template is fixed; the schema is not. Each intent gets a batch model
    # whose `intent` field is constrained to that one name, so the chain is
    # assembled per intent rather than once up front.
    template = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )

    failures: list[str] = []

    for position, (intent, _, want, _) in enumerate(plan, start=1):
        spec = specs[intent]
        if args.force:
            store.reset(intent)
        rows = [] if args.force else store.load(intent)
        accepted = [r["utterance"] for r in rows]
        keys = {dedupe_key(a) for a in accepted}

        # The evidence block is the only view of real user speech the model
        # gets, so it must not be the head of the file: the export is
        # permutation-heavy and its first lines are near-identical siblings, the
        # least informative evidence available (Architecture Section 2, "Seed
        # Evidence Selection"). Measured across the export, head-slicing hands
        # over a sample at mean pairwise distance 0.710 where a diverse pick
        # reaches 0.882, and on the worst intents the gap is far wider.
        evidence = _evidence_block(config, corpus.intents.get(intent, []), intent, seed_ref)
        cancels = _cancellation_capability(config, intent)
        chain = template | build_structured_llm(config, batch_model_for(intent), stage="generation")
        stalled = 0
        batches_done = 0

        while len(accepted) < want:
            if args.batches and batches_done >= args.batches:
                print(
                    f"[{position}/{len(plan)}] {intent} — stopping after "
                    f"{batches_done} batch(es) at {len(accepted)}/{want} (--batches)"
                )
                break
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
                "seed_block": evidence,
                "avoid_block": _avoid_block(accepted, avoid_max),
                "batch_size": size,
                "composition_block": _composition_block(config, intent, size),
            }

            # Only structural failures drive the retry loop. Per-utterance
            # faults are collected here and applied below by dropping those rows
            # (Section 7: regenerate the failed sample, not the batch).
            row_faults: dict[int, list[str]] = {}

            def _validate(candidate: GeneratedBatch) -> list[str]:
                batch_errors, row_errors = validate_batch(
                    candidate,
                    intent=intent,
                    fallback_intent=config.fallback_intent,
                    accepted_keys=keys,
                    max_words=max_words,
                    expected_size=size,
                    cancellation_capability=cancels,
                )
                row_faults.clear()
                row_faults.update(row_errors)
                return batch_errors

            outcome = invoke_with_validation(
                chain,
                payload,
                validate=_validate,
                config=config,
                label=intent,
            )
            batches_done += 1
            store.log_rejections(outcome.rejections)

            if not outcome.ok:
                stalled += 1
                print(
                    f"[{position}/{len(plan)}] {intent} — batch unusable "
                    f"({stalled}/{max_stalled})"
                )
                for reason in outcome.rejections[-1].reasons[:3]:
                    print(f"      · {reason}")
                if stalled >= max_stalled:
                    print(f"      giving up on {intent} at {len(accepted)}/{want}")
                    failures.append(intent)
                    break
                continue

            new_rows = []
            dropped: list[str] = []
            surplus = 0
            for index, item in enumerate(outcome.value.utterances):
                # Models do not always return exactly the count they were asked
                # for. Nothing above rejects an over-long batch -- validation
                # only objects when too FEW come back -- so without this the
                # store quietly overshoots its budget. Measured with a stub that
                # returned three extra per call: 123 rows against a budget of
                # 120, and a coverage report reading 102%.
                if len(accepted) >= want:
                    surplus += 1
                    continue
                if index in row_faults:
                    dropped += row_faults[index]
                    continue
                key_ = dedupe_key(item.utterance)
                if key_ in keys:
                    continue
                keys.add(key_)
                accepted.append(item.utterance)
                new_rows.append(
                    {
                        "utterance": item.utterance,
                        # The requested name, not `item.intent`: the field is now
                        # an Enum, and str() of a string Enum does not give the
                        # value on Python 3.11+.
                        "intent": intent,
                        "intent_family": spec.get("intent_family", "Unassigned"),
                        "type": item.type.value,
                        "difficulty": item.difficulty.value,
                        "source": item.source.value,
                    }
                )
            store.append(intent, new_rows)
            if dropped:
                store.log_rejections([Rejection(intent, outcome.attempts, dropped)])

            # A batch yielding nothing usable is not progress. The while
            # condition alone cannot see that, so it would loop forever.
            stalled = 0 if new_rows else stalled + 1
            print(
                f"[{position}/{len(plan)}] {intent} — {len(accepted)}/{want} "
                f"(+{len(new_rows)}, dropped={len(dropped)}"
                + (f", surplus={surplus}" if surplus else "")
                + f", attempts={outcome.attempts})"
            )
            if not new_rows and stalled >= max_stalled:
                print(f"      giving up on {intent} at {len(accepted)}/{want}")
                failures.append(intent)
                break

    print(f"\nStage 1 complete. Failures: {len(set(failures))}")
    print(f"Output: {store.dir}")
    if store.rejects.is_file():
        print(f"Rejection log: {store.rejects}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
