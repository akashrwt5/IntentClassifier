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
pkl, -> device.volume.increase in ONNX).

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
_DEFAULT_CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "ain't": "is not",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "wouldn't": "would not", "couldn't": "could not",
    "shouldn't": "should not", "mustn't": "must not", "needn't": "need not",
    "what's": "what is", "that's": "that is", "it's": "it is",
    "there's": "there is", "here's": "here is", "where's": "where is",
    "how's": "how is", "who's": "who is", "she's": "she is", "he's": "he is",
    "let's": "let us", "i'm": "i am",
    "you're": "you are", "we're": "we are", "they're": "they are",
    "i've": "i have", "you've": "you have", "we've": "we have",
    "they've": "they have", "would've": "would have", "could've": "could have",
    "should've": "should have",
    "i'll": "i will", "you'll": "you will", "we'll": "we will",
    "they'll": "they will", "it'll": "it will", "that'll": "that will",
    "i'd": "i would", "you'd": "you would", "we'd": "we would",
    "they'd": "they would",
}

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
