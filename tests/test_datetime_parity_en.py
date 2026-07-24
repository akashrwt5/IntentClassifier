"""
English datetime golden-corpus regression test.

English historically had NO datetime parity fixture (it was the hardcoded
special case). This test locks the baseline captured in
`nlu_datetime_parity_en_golden.json` so any change to the datetime parser — in
particular the ongoing eviction of English tokens into the Language Pack — must
reproduce the exact same (iso, time_explicit, explicit_day) for every case.

Runs the extractor two ways and requires identical output:
  1. built-in English defaults (no pack)
  2. pack-fed tokens (weekdays + word-numbers injected from packs/en)
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

_BASE = Path(__file__).parent.parent
sys.path.insert(0, str(_BASE / "scripts"))

from nlu.entities import EntityExtractor  # noqa: E402

_GOLDEN = json.loads(
    (_BASE / "tests" / "datetime_parity" / "nlu_datetime_parity_en_golden.json").read_text()
)
_NOW = datetime.fromisoformat(_GOLDEN["now_iso"])
_CASES = _GOLDEN["cases"]


def _pack_tokens():
    """Weekdays + word-numbers from the en pack datetime grammar (mirrors what
    the engine injects), so this test also covers the pack-fed path."""
    grammar = json.loads(
        (_BASE / "packs" / "en" / "datetime" / "grammar.json").read_text()
    )
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekdays = [(grammar["weekdays"].get(n) or [n.lower()])[0].lower() for n in order]
    word_nums = {str(k): int(v) for k, v in grammar["word_numbers"].items()}
    return weekdays, word_nums


_DEFAULT = EntityExtractor()
_wd, _wn = _pack_tokens()
_PACKED = EntityExtractor(weekdays=_wd, word_nums=_wn)


@pytest.mark.parametrize("case", _CASES, ids=[c["utterance"] for c in _CASES])
@pytest.mark.parametrize("extractor,label", [(_DEFAULT, "default"), (_PACKED, "packed")])
def test_en_datetime_golden(extractor, label, case):
    iso, _span, _conf, te, ed = extractor.extract_datetime(case["utterance"], now=_NOW)
    assert [iso, te, ed] == [case["iso"], case["time_explicit"], case["explicit_day"]], (
        f"[{label}] {case['utterance']!r}: "
        f"got ({iso}, {te}, {ed}) expected ({case['iso']}, {case['time_explicit']}, {case['explicit_day']})"
    )
