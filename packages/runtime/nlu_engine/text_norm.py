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

# Plural -> singular folding for the TF-IDF featurizer. GENERATED, not
# hand-maintained: `nlu_training.train` derives it from the training corpus
# (suffix rule -> vocabulary guard -> ratio guard -> exception list) and writes
# it here, the same status as `vocab` and `idf`. A hand-written table would only
# ever cover the words someone remembered to type into it.
#
# WHY. The featurizer is a stock TfidfVectorizer with no stemmer, so `program`
# and `programs` are two unrelated vocabulary slots. When one form dominates a
# label and the other dominates a different label, PLURALITY ITSELF becomes the
# intent predictor. Measured on en: `program` 623 rows under Cmd.MemoryChange vs
# 23 under Help_ChangingMemories, while `programs` runs 6 vs 44 — so "change
# programs" classified as help and "change program" as a command.
#
# Like `contractions`, a pack supplies its own via `normalize_text(lemmas=...)`
# and an absent file simply means no folding.
_LEMMAS_PATH = BASE_DIR / "language_packs" / "en" / "lemmas.json"
_DEFAULT_LEMMAS = (json.loads(_LEMMAS_PATH.read_text(encoding="utf-8"))
                   if _LEMMAS_PATH.exists() else {})

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


@lru_cache(maxsize=8)
def _lemma_re(keys: tuple[str, ...]) -> "re.Pattern":
    """Compiled alternation for one lemma table, built once per table.

    Same construction as `_contraction_re` — longest-first so a key that
    prefixes another cannot shadow it, cached on the key tuple so a pack's
    table costs the same as the default one.
    """
    ordered = sorted(keys, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(k) for k in ordered) + r")\b")


def normalize_text(text: str, contractions: dict | None = None,
                   lemmas: dict | None = None) -> str:
    """lowercase -> unify apostrophes -> expand contractions -> drop residual
    apostrophes -> fold plurals -> collapse whitespace.

    `contractions` and `lemmas` come from the pack/lexicon for the language
    being processed; None uses the English fallback table, so existing callers
    behave as before. An empty `lemmas` table disables folding entirely.

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
    # Plural folding runs AFTER contraction expansion and apostrophe removal so
    # it sees clean word forms, and before whitespace collapse. The table maps
    # only to forms already in the corpus, so the transform stays idempotent.
    lem = lemmas if lemmas is not None else _DEFAULT_LEMMAS
    if lem:
        t = _lemma_re(tuple(lem)).sub(lambda m: lem[m.group(1)], t)
    return _SPACE_RE.sub(" ", t).strip()


if __name__ == "__main__":
    for s in ["what's up", "What's up", "don't mute it", "mom's reminder",
              "turn up the volume", "how's it going", "can't hear you"]:
        print(f"{s!r:24} -> {normalize_text(s)!r}")
