"""Strict metadata schemas for generated utterances (Architecture Section 7).

The ``Type`` enum is the load-bearing part. Section 3 Step 8 fixes eight values
and gives each a definition; leaving the field as a free string would let the
model invent neighbours like "Command" or "ImplicitRequest" that look right in
review and then silently fragment every per-type metric in the Section 8
confusion matrix. Making it an ``Enum`` moves that failure from evaluation time
to generation time, where a retry can still fix it.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field, field_validator

# Utterances that are nothing but punctuation or filler survive a naive
# non-empty check but carry no signal.
_MEANINGFUL = re.compile(r"[A-Za-z0-9]")


class UtteranceType(str, Enum):
    """Section 3, Step 8. Definitions are quoted in the generator prompt."""

    EXPLICIT_COMMAND = "ExplicitCommand"
    IMPLICIT_COMMAND = "ImplicitCommand"
    OBSERVATION = "Observation"
    OBSERVATION_PLUS_COMMAND = "ObservationPlusCommand"
    QUESTION = "Question"
    NEGATION = "Negation"
    CONVERSATION = "Conversation"
    FALLBACK = "Fallback"


class Difficulty(str, Enum):
    """Section 7. Definitions are quoted in the generator prompt."""

    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class Source(str, Enum):
    """Section 7."""

    LLM_GENERATED = "LLM-Generated"
    ASR_SIMULATED = "ASR-Simulated"
    HUMAN_SEED = "Human-Seed"


#: Types that legitimately exceed the normal voice-command length ceiling,
#: because the blueprint asks for them to be multi-clause.
LONG_FORM_TYPES = frozenset(
    {
        UtteranceType.OBSERVATION_PLUS_COMMAND,
        UtteranceType.CONVERSATION,
        UtteranceType.FALLBACK,
    }
)


class GeneratedUtterance(BaseModel):
    """One row of the rich dataset."""

    utterance: str = Field(description="The user utterance, verbatim. No quotes, no numbering.")
    intent: str = Field(description="The exact intent name this utterance belongs to.")
    type: UtteranceType = Field(description="One of the eight allowed Type values.")
    difficulty: Difficulty = Field(description="Easy, Medium or Hard.")
    source: Source = Field(
        description="LLM-Generated normally; ASR-Simulated when deliberately corrupted."
    )

    @field_validator("utterance")
    @classmethod
    def _meaningful(cls, value: str) -> str:
        text = " ".join(value.split())
        if not _MEANINGFUL.search(text):
            raise ValueError("utterance has no alphanumeric content")
        return text


class GeneratedBatch(BaseModel):
    """What the model returns for one request."""

    utterances: list[GeneratedUtterance] = Field(
        description="The generated utterances for this batch."
    )
