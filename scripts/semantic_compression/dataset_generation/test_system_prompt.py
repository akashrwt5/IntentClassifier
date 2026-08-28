#!/usr/bin/env python3
"""Pin the load-bearing rules in SYSTEM_PROMPT. No API, no deps, no network.

WHY THIS FILE EXISTS
--------------------
Two prompt rules were written, reviewed, committed and measured -- and were never
sent to the model. They went into ``prompt.txt``, which ``render_prompt.py``
*writes* and nothing reads. The generator never had them.

Nothing caught it because a missing prompt rule does not crash, does not warn,
and does not show up as a failure. It shows up as a corpus that is slightly wrong
in a way that looks like model behaviour. Two conclusions were drawn from the
silence and both were wrong: an anti-length instruction was recorded as "measured
and did nothing" when it was never sent, and a boundary improvement was credited
to a rule that was not live.

So these assertions are deliberately about *content*, not structure. A test that
only checked "SYSTEM_PROMPT is a non-empty string" would have passed throughout.

Parsed with ``ast`` rather than imported, so this runs with nothing installed --
which matters, because a guard that needs the generator's full dependency stack
is a guard that gets skipped in exactly the environment where it is needed.

    python3 -m pytest test_system_prompt.py -q
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
GENERATOR = HERE / "generator.py"
RENDERED = HERE / "prompt.txt"


def system_prompt() -> str:
    """SYSTEM_PROMPT's literal value, without importing the module."""
    tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "SYSTEM_PROMPT":
            return ast.literal_eval(node.value)
    raise AssertionError("generator.py has no SYSTEM_PROMPT assignment")


PROMPT = system_prompt()


# --- the two rules that were lost ---------------------------------------


def test_help_versus_command_precedence_rule_is_live():
    """Decide by the action required, not the opening words.

    boundary_lint.py exists to measure obedience to this rule and its docstring
    asserts the rule is in force. Without the rule in the prompt, that linter is
    grading the model against an instruction it never received.
    """
    assert "Help versus Command" in PROMPT
    assert re.search(r"by the ACTION the assistant must perform", PROMPT)
    assert "not by the opening words" in PROMPT


def test_difficulty_is_not_tied_to_length():
    """Hard must not mean long.

    The classifier's measured weak spot is short utterances (79% at <=4 words
    against 89% at 8-12). A prompt that lets Hard drift into Long spends the
    generation budget on the part the model already handles.
    """
    assert "Difficulty is NOT length" in PROMPT
    assert "Never reach for extra words" in PROMPT


def test_hard_definition_does_not_reintroduce_length():
    """The Hard bullet itself must not name length as a property of Hard."""
    hard = re.search(r"^- Hard: (.+?)(?=\n- |\n\n)", PROMPT, flags=re.M | re.S)
    assert hard, "no Hard bullet in the Difficulty block"
    body = hard.group(1).lower()
    for banned in ("long conversational", "longer", "multi-clause length"):
        assert banned not in body, f"Hard bullet reintroduces length via {banned!r}"


def test_easy_is_defined_structurally_not_by_seed_similarity():
    """The retired wording produced verbatim seed reproductions.

    Section 7 of the architecture doc repudiates it by name. It still sits in
    prompt.txt, which is exactly why this assertion is on the live prompt.
    """
    assert "high lexical similarity" not in PROMPT
    assert "single-clause" in PROMPT


# --- structure that other tools depend on --------------------------------


def test_precedence_rules_are_contiguously_numbered():
    """A renumbering slip is how a rule goes missing without looking missing."""
    block = re.search(r"# Precedence rules.*?\n\n", PROMPT, flags=re.S)
    assert block, "no precedence rules block"
    numbers = [int(n) for n in re.findall(r"^(\d+)\. ", block.group(0), flags=re.M)]
    assert numbers == list(range(1, len(numbers) + 1)), numbers
    assert len(numbers) >= 5


def test_the_eight_type_values_are_all_defined():
    """schemas.UtteranceType is an enum; the prompt must define every member."""
    for value in (
        "ExplicitCommand",
        "ImplicitCommand",
        "Observation",
        "ObservationPlusCommand",
        "Question",
        "Negation",
        "Conversation",
        "Fallback",
    ):
        assert f"- {value}:" in PROMPT, f"Type {value} is not defined in the prompt"


def test_false_accept_framing_survives():
    """The product's costliest error class must stay in front of the model."""
    assert "FALSE ACCEPT" in PROMPT


# --- the artefact that caused all this -----------------------------------


def test_rendered_prompt_is_not_a_second_source_of_truth():
    """``prompt.txt`` must not disagree with the live prompt.

    It is build output: render_prompt.py reads gen.SYSTEM_PROMPT and writes it.
    If it exists, its system section must match. Hand-editing it is what lost
    two rules for a week, so this fails loudly rather than warning.

    The fix when this fails is never to edit prompt.txt. Change SYSTEM_PROMPT and
    re-render, or delete the file -- it has no consumer.
    """
    if not RENDERED.is_file():
        return  # deleting it is a valid resolution, and the better one
    text = RENDERED.read_text(encoding="utf-8")
    for probe, label in (
        ("Help versus Command", "Help-vs-Command precedence rule"),
        ("Difficulty is NOT length", "anti-length instruction"),
    ):
        assert (probe in text) == (probe in PROMPT), (
            f"prompt.txt and SYSTEM_PROMPT disagree about the {label}. "
            "prompt.txt is build output -- regenerate it, do not edit it."
        )
    assert "high lexical similarity" not in text, (
        "prompt.txt carries the retired Easy definition. It is stale: "
        "regenerate with `python render_prompt.py --intent <X> --manual > prompt.txt`."
    )
