#!/usr/bin/env python3
"""Shared surface-form normalisation for the English TF-IDF intent model.

WHY THIS EXISTS
---------------
skl2onnx's ONNX TfidfVectorizer tokenizer does NOT replicate Python's `\\w`
word-boundary semantics around the apostrophe. sklearn's default token pattern
(`\\b\\w\\w+\\b`) drops the "'s" in "what's up" and yields tokens ["what", "up"]
plus the bigram "what up"; the exported ONNX graph tokenises the same string
differently and loses that disambiguating bigram. The result is a train/inference
mismatch: `pipeline.pkl` predicts one intent and the shipped `model.onnx`
predicts another for ANY apostrophe input (observed: "what's up" -> OOS in the
pkl, -> Cmd.VolumeIncrease in ONNX).

FIX
---
Fold the apostrophe out of existence BEFORE the text reaches the vectorizer, in
BOTH training and inference, so the ONNX tokenizer and sklearn see an identical,
apostrophe-free surface form. Known contractions are expanded to full words
("what's" -> "what is") — which also tokenise into real, learnable words — and
any residual apostrophes (possessives, "o'clock") are removed.

CONTRACT
--------
`normalize_text()` MUST be applied at every entry point that feeds the English
model:
  * training            (nlu_training/train.py, before fit + ONNX export)
  * Python inference     (nlu_engine.classifier, the TF-IDF/ONNX path)
  * on-device / Swift    (port this exact logic for iOS/Android parity)

The transform is idempotent. It is intentionally scoped to the TF-IDF path; the
keyword stage matches raw text. This mirrors the accent-folding rationale in
`multilingual/text_norm.py`; the two should be consolidated (see decisions.md).
"""

import re
from functools import lru_cache

# Explicit contraction expansions. Order here does not matter: the alternation is
# built longest-first by `_contraction_re` and anchored on word boundaries, so a
# key that prefixes another cannot shadow it.
#
# ENGLISH FALLBACK ONLY. Contractions are language-specific — fr "j'ai"/"n'est",
# da "det's" — and expanding them needs that language's own table, so a pack
# supplies its own via `normalize_text(text, contractions=...)`. The `_DEFAULT_`
# prefix is the neutrality guard's convention for an overridable DATA table
# (see scripts/ci/check_language_neutral.py check 2): without it this would be
# English match vocabulary embedded in the engine, which is what made negation
# suppression a silent no-op for three languages before A4.
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]

_DEFAULT_CONTRACTIONS = json.loads((BASE_DIR / "language_packs" / "en" / "contractions.json").read_text(encoding="utf-8"))

_APOSTROPHES = ("’", "ʼ", "`")
_SPACE_RE = re.compile(r"\s+")


@lru_cache(maxsize=8)
def _contraction_re(keys: tuple[str, ...]) -> "re.Pattern":
    """Compiled alternation for one contraction table, built once per table.

    Cached on the key tuple rather than module-level so a pack's table costs the
    same as the default one. Longest-first so a key that prefixes another cannot
    shadow it.
    """
    ordered = sorted(keys, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(k) for k in ordered) + r")\b")


def normalize_text(text: str, contractions: dict | None = None) -> str:
    """lowercase -> unify apostrophes -> expand contractions -> drop residual
    apostrophes -> collapse whitespace.

    `contractions` comes from the pack/lexicon for the language being processed;
    None uses the English fallback table, so existing callers behave as before.

    Examples:
        "what's up"        -> "what is up"
        "don't mute it"    -> "do not mute it"
        "mom's reminder"   -> "moms reminder"
        "turn up the volume" -> "turn up the volume"   (unchanged)
    """
    table = contractions if contractions is not None else _DEFAULT_CONTRACTIONS
    t = str(text).lower().strip()
    for ap in _APOSTROPHES:
        t = t.replace(ap, "'")
    if table:
        t = _contraction_re(tuple(table)).sub(lambda m: table[m.group(1)], t)
    t = t.replace("'", "")            # residual possessives / o'clock -> oclock
    return _SPACE_RE.sub(" ", t).strip()


if __name__ == "__main__":
    for s in ["what's up", "What's up", "don't mute it", "mom's reminder",
              "turn up the volume", "how's it going", "can't hear you"]:
        print(f"{s!r:24} -> {normalize_text(s)!r}")
