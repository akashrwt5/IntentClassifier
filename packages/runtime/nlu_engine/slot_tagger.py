"""
Slot Tagger runtime helpers for NLUEngine.

This module is the SINGLE SOURCE OF TRUTH for how BIO slot-tagger features are built.
The trainer (``scripts/bio_tagger_prototype.py``), the Python runtime and the exported
mobile weights all read :data:`FEATURE_SPEC` from here, so a rule can only be changed
in one place. ``nlu_export.export_slot_weights`` embeds the spec in the weights JSON,
which is what lets the Swift and Kotlin ports hold no English knowledge of their own.

Provides:
- FEATURE_SPEC: the data behind preprocessing, tokenisation and every lexical flag.
- robust_preprocess: normalizes speech-to-text slips, contractions, fillers, and slang.
- extract_token_features: multi-scale character n-gram and context features.
- extract_title_span: the one BIO decoding rule, shared by training and inference.
- extract_slot_title: runs the trained BIO model to extract open reminder titles.
"""

import logging
import re
from typing import Any, Optional

logger = logging.getLogger("nlu.slot_tagger")

# Replacements are stored in the "$1" back-reference style so the exported JSON can be
# consumed by Kotlin and Swift unchanged; Python needs r"\1", hence _py_replacement().
FEATURE_SPEC: dict[str, Any] = {
    "version": 2,
    # Applied in order, before tokenisation, at BOTH training and inference time.
    "preprocess": [
        # 1. ASR / speech-to-text phonetic error repair
        [r"\bset\s+a\s+remind\s+her\b", "set a reminder"],
        [r"\bremind\s+me\s+two\b", "remind me to"],
        [r"\bat\s+for\s+(am|pm)\b", "at 4 $1"],
        [r"\bto\s+by\b", "to buy"],
        # 2. Slang & colloquial carrier normalization
        [r"\bgimme\s+a\s+ping\s+to\b", "remind me to"],
        [r"\bdrop\s+a\s+reminder\s+to\b", "remind me to"],
        [r"\bhit\s+me\s+up\s+to\b", "remind me to"],
        [r"\bjot\s+down\s+a\s+reminder\s+to\b", "remind me to"],
        # 3. Conversational fillers
        [r"^\s*(?:um|uh|uhhh|you know|well basically|hey so yeah|like)\b[,.]?\s*", ""],
        [r"\b(?:um|uh|uhhh)\b", ""],
        # 4. Contraction expansion
        [r"\bdon't\b|\bdont\b", "do not"],
        [r"\bcan't\b|\bcant\b", "cannot"],
        [r"\bi've\s+gotta\b|\bive\s+gotta\b", "i have to"],
        [r"\bi'd\s+like\b|\bid\s+like\b", "i would like"],
    ],
    "tokenizer": r"\b\w+(?:'\w+)?\b|[^\w\s]",
    "affix_sizes": [2, 3, 4],
    "neighbour_suffix": 3,
    "rel_pos_decimals": 1,
    # `after_trigger` is true when one of these appears within `trigger_window` tokens back.
    "trigger_words": ["to", "about", "for", "that", "remind", "reminder", "alarm"],
    "trigger_window": 6,
    "prepositions": ["to", "for", "about", "that", "at", "by", "on"],
    "time_words": ["am", "pm", "tomorrow", "today", "tonight", "morning", "night", "at", "in"],
    "labels": {"begin": "B-TITLE", "inside": "I-TITLE", "outside": "O"},
}

_WHITESPACE = re.compile(r"\s+")
_BACKREF = re.compile(r"\$(\d)")


def _py_replacement(replacement: str) -> str:
    """Converts a "$1" style back-reference to the r"\\1" form Python's re.sub wants."""
    return _BACKREF.sub(r"\\\1", replacement)


_PREPROCESS = [
    (re.compile(pattern, re.I), _py_replacement(replacement))
    for pattern, replacement in FEATURE_SPEC["preprocess"]
]
_TOKENIZER = re.compile(FEATURE_SPEC["tokenizer"])
_TRIGGERS = frozenset(FEATURE_SPEC["trigger_words"])
_PREPOSITIONS = frozenset(FEATURE_SPEC["prepositions"])
_TIME_WORDS = frozenset(FEATURE_SPEC["time_words"])
_BEGIN = FEATURE_SPEC["labels"]["begin"]
_INSIDE = FEATURE_SPEC["labels"]["inside"]


def robust_preprocess(text: str) -> str:
    """Normalizes speech-to-text errors, contractions, slang and fillers, in spec order."""
    for pattern, replacement in _PREPROCESS:
        text = pattern.sub(replacement, text)
    return _WHITESPACE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Splits into words and single punctuation marks, keeping one apostrophe inside a word."""
    return _TOKENIZER.findall(text)


def prepare(text: str) -> list[str]:
    """Preprocesses then tokenises. Training and inference MUST both go through this."""
    return tokenize(robust_preprocess(text))


def _trigger_distance(tokens: list[str], i: int) -> int:
    """Distance back to the nearest trigger word before i, or 99 when there is none."""
    best = 99
    for idx, token in enumerate(tokens):
        if token.lower() not in _TRIGGERS:
            continue
        distance = i - idx
        if 0 < distance < best:
            best = distance
    return best


def extract_token_features(tokens: list[str], i: int) -> dict:
    """Extract rich character n-gram, positional, and syntactic features for token at i."""
    word = tokens[i]
    w_low = word.lower()
    decimals = FEATURE_SPEC["rel_pos_decimals"]
    neighbour = FEATURE_SPEC["neighbour_suffix"]

    features: dict[str, Any] = {
        "bias": 1.0,
        "word.lower()": w_low,
        "len": len(w_low),
        "isdigit": word.isdigit(),
        "after_trigger": _trigger_distance(tokens, i) < FEATURE_SPEC["trigger_window"],
        "rel_pos": round(i / max(len(tokens), 1), decimals),
    }
    # Multi-scale character n-grams (prefix & suffix) for strong typo tolerance.
    for size in FEATURE_SPEC["affix_sizes"]:
        features[f"p{size}"] = w_low[:size]
        features[f"s{size}"] = w_low[-size:]

    if i > 0:
        prev = tokens[i - 1].lower()
        features["-1:w"] = prev
        features[f"-1:s{neighbour}"] = prev[-neighbour:]
        features["-1:is_prep"] = prev in _PREPOSITIONS
    else:
        features["BOS"] = True

    if i < len(tokens) - 1:
        nxt = tokens[i + 1].lower()
        features["+1:w"] = nxt
        features[f"+1:s{neighbour}"] = nxt[-neighbour:]
        features["+1:is_time"] = nxt in _TIME_WORDS
    else:
        features["EOS"] = True

    return features


def extract_title_span(tokens: list[str], tags: list[str]) -> Optional[str]:
    """
    The one BIO decoding rule: return the LAST complete B-/I- run, or None.

    Taking the last run is what the training-time annotator produced, so training and
    inference agree. An I- tag with no B- in front of it does not open a span.
    """
    found: Optional[str] = None
    buffer: list[str] = []
    for token, tag in zip(tokens, tags):
        if tag == _BEGIN:
            if buffer:
                found = " ".join(buffer)
            buffer = [token]
        elif tag == _INSIDE and buffer:
            buffer.append(token)
        else:
            if buffer:
                found = " ".join(buffer)
                buffer = []
    if buffer:
        found = " ".join(buffer)
    return found


def extract_slot_title(slot_model: Any, text: str) -> Optional[str]:
    """Run the BIO Slot Tagger on text and extract the title."""
    if slot_model is None or not isinstance(slot_model, dict):
        return None

    vec = slot_model.get("vectorizer")
    clf = slot_model.get("classifier")
    if vec is None or clf is None:
        return None

    try:
        tokens = prepare(text)
        if not tokens:
            return None

        features = [extract_token_features(tokens, i) for i in range(len(tokens))]
        tags = list(clf.predict(vec.transform(features)))
        return extract_title_span(tokens, tags)
    except Exception as e:  # noqa: BLE001 - a tagger failure must never fail the turn
        logger.warning("nlu.slot_tagger.inference_error err=%s", e)
        return None
