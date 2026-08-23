#!/usr/bin/env python3
"""The ImplicitCommand guard, alone, so that it can be tested.

WHY IT LIVES IN ITS OWN FILE
----------------------------
This is thirty lines of regex that decides whether a generated row really asks
for something. It went unmeasured for months, and when it was finally measured
it was flagging fourteen rows to catch six -- deleting eight genuine requests to
do it, and emptying one intent's ImplicitCommand quota in every run ever
recorded.

Part of why nobody measured it is that it could not be reached without importing
generator.py, which pulls in pydantic and langchain through llm_client. A guard
whose test needs the whole generation stack installed is a guard that does not
get tested. It needs one regex, so it now lives where one regex is all you need.

The measurement lives in test_request_signal.py.
"""

from __future__ import annotations

import re

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
    # Verbs the product's own commands are built from. Their absence is why
    # Cmd.MemoryChange's ImplicitCommand quota came back 0 in every run ever
    # recorded: its natural verb is `switch`, and the guard had never heard of it.
    r"switch|put|tell|show|set|change|start|stop|send|read|play|find|last|"
    r"struggl\w*|trying|difficult\w*)\b|hard time|trouble\s+\w+ing|"
    r"not getting (enough|much)|make out|keep up|catch what|"
    r"\b(i'?d|we'?d|would|i'?ll) like\b)",
    re.IGNORECASE,
)


def _requests_something(text: str) -> bool:
    """Does this utterance ask for anything? Apostrophes normalised first.

    The pattern above already carried a fix for "I'd like ..." -- written with a
    STRAIGHT apostrophe. The model emits curly ones: 205 of 1,040 generated rows
    contain U+2019 against 29 with U+0027. So the fix never fired on real output,
    and "I'd like the right hearing aid to stop making any sound" was deleted as
    an utterance that "requests nothing".

    Measured on test_request_signal.py's fixture, normalising the apostrophe and
    adding the missing verbs moves the guard's flag precision from 43% to 75%
    while catching exactly the same observations. It does not make the guard
    good -- 5 of 11 observations still get through, in every variant tried --
    which is why the caller now warns rather than deletes.
    """
    return bool(_REQUEST_SIGNAL.search(text.replace("\u2019", "'").replace("\u02bc", "'")))
