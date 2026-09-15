"""
Slot Tagger runtime helpers for NLUEngine.

Provides:
- robust_preprocess: normalizes speech-to-text slips, contractions, fillers, and slang.
- extract_token_features: multi-scale character n-gram and context features for BIO token classification.
- extract_slot_title: runs the trained BIO model to extract open reminder titles.
"""

import logging
import re
from typing import Optional, Any

logger = logging.getLogger("nlu.slot_tagger")


def robust_preprocess(text: str) -> str:
    """Preprocess text to normalize speech-to-text errors, contractions, slang, and fillers."""
    # 1. ASR / Speech-to-text phonetic error repair
    text = re.sub(r"\bset\s+a\s+remind\s+her\b", "set a reminder", text, flags=re.I)
    text = re.sub(r"\bremind\s+me\s+two\b", "remind me to", text, flags=re.I)
    text = re.sub(r"\bat\s+for\s+(am|pm)\b", r"at 4 \1", text, flags=re.I)
    text = re.sub(r"\bto\s+by\b", "to buy", text, flags=re.I)

    # 2. Slang & colloquial carrier normalization
    text = re.sub(r"\bgimme\s+a\s+ping\s+to\b", "remind me to", text, flags=re.I)
    text = re.sub(r"\bdrop\s+a\s+reminder\s+to\b", "remind me to", text, flags=re.I)
    text = re.sub(r"\bhit\s+me\s+up\s+to\b", "remind me to", text, flags=re.I)
    text = re.sub(r"\bjot\s+down\s+a\s+reminder\s+to\b", "remind me to", text, flags=re.I)

    # 3. Conversational fillers at sentence start
    text = re.sub(
        r"^\s*(?:um|uh|uhhh|you know|well basically|hey so yeah|like)\b[,.]?\s*",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\b(?:um|uh|uhhh)\b", "", text, flags=re.I)

    # 4. Contractions expansion
    text = re.sub(r"\bdon't\b|\bdont\b", "do not", text, flags=re.I)
    text = re.sub(r"\bcan't\b|\bcant\b", "cannot", text, flags=re.I)
    text = re.sub(r"\bi've\s+gotta\b|\bive\s+gotta\b", "i have to", text, flags=re.I)
    text = re.sub(r"\bi'd\s+like\b|\bid\s+like\b", "i would like", text, flags=re.I)

    # Clean double spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Tokenize text into words and punctuation."""
    return re.findall(r"\b\w+(?:'\w+)?\b|[^\w\s]", text)


def extract_token_features(tokens: list[str], i: int) -> dict:
    """Extract rich character n-gram, positional, and syntactic features for token at i."""
    word = tokens[i]
    w_low = word.lower()

    # Find position relative to nearest trigger word ('to', 'for', 'about', 'that', 'remind')
    trigger_dist = 99
    for idx, t in enumerate(tokens):
        if t.lower() in ("to", "about", "for", "that", "remind", "reminder", "alarm"):
            dist = i - idx
            if 0 < dist < abs(trigger_dist):
                trigger_dist = dist

    features = {
        "bias": 1.0,
        "word.lower()": w_low,
        "len": len(w_low),
        # Multi-scale character n-grams (prefix & suffix) for strong typo tolerance
        "p2": w_low[:2],
        "p3": w_low[:3],
        "p4": w_low[:4] if len(w_low) >= 4 else w_low,
        "s2": w_low[-2:],
        "s3": w_low[-3:],
        "s4": w_low[-4:] if len(w_low) >= 4 else w_low,
        "isupper": word.isupper(),
        "istitle": word.istitle(),
        "isdigit": word.isdigit(),
        "after_trigger": trigger_dist < 6,
        "rel_pos": round(i / max(len(tokens), 1), 1),
    }

    # Context window: previous token
    if i > 0:
        prev = tokens[i - 1].lower()
        features.update(
            {
                "-1:w": prev,
                "-1:s3": prev[-3:],
                "-1:is_prep": prev in ("to", "for", "about", "that", "at", "by", "on"),
            }
        )
    else:
        features["BOS"] = True

    # Context window: next token
    if i < len(tokens) - 1:
        nxt = tokens[i + 1].lower()
        features.update(
            {
                "+1:w": nxt,
                "+1:s3": nxt[-3:],
                "+1:is_time": nxt
                in ("am", "pm", "tomorrow", "today", "tonight", "morning", "night", "at", "in"),
            }
        )
    else:
        features["EOS"] = True

    return features


def extract_slot_title(slot_model: Any, text: str) -> Optional[str]:
    """Run the BIO Slot Tagger on text and extract the title."""
    if slot_model is None or not isinstance(slot_model, dict):
        return None

    vec = slot_model.get("vectorizer")
    clf = slot_model.get("classifier")
    if vec is None or clf is None:
        return None

    try:
        clean = robust_preprocess(text)
        tokens = tokenize(clean)
        if not tokens:
            return None

        features = [extract_token_features(tokens, i) for i in range(len(tokens))]
        X_vec = vec.transform(features)
        tags = clf.predict(X_vec)

        title_tokens = []
        is_capturing = False
        for token, tag in zip(tokens, tags):
            if tag == "B-TITLE":
                is_capturing = True
                title_tokens.append(token)
            elif tag == "I-TITLE" and is_capturing:
                title_tokens.append(token)
            elif tag != "I-TITLE":
                is_capturing = False

        if title_tokens:
            return " ".join(title_tokens)
    except Exception as e:
        logger.warning("nlu.slot_tagger.inference_error err=%s", e)

    return None
