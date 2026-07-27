#!/usr/bin/env python3
"""
Shared text normalisation for the multilingual intent models.

WHY THIS EXISTS
---------------
skl2onnx's ONNX TfidfVectorizer tokenizer does NOT replicate Python's Unicode
`\\w` word-boundary semantics. Words containing accented letters (æ ø å ä ö ü
é è à ç …) get tokenised differently at ONNX inference than during sklearn
training, so those tokens miss the learned vocabulary and accuracy collapses on
non-English languages (Danish lost ~10 points).

FIX
---
Fold every input to lowercase ASCII *before* it ever reaches the vectorizer, in
BOTH training and inference. Once the surface form is plain ASCII the ONNX
tokenizer and sklearn agree, so the exported model matches the in-memory model.

CONTRACT
--------
`normalize_text()` MUST be applied at every entry point that feeds these models:
  * training            (train_multilingual.py load_clean)
  * Python inference     (test_multilingual_models.py, predict.py, …)
  * on-device / Swift    (port this exact logic — see _FOLD + NFKD below)

The transform is idempotent: normalising already-normalised text is a no-op.
"""

import unicodedata

# Letters that do NOT decompose under Unicode NFKD (they are atomic codepoints,
# not base-letter + combining-mark), so they need an explicit map. Lowercase
# only — normalize_text() lowercases first. Keep this list in sync with the
# Swift port.
_FOLD = {
    "ø": "o",    # Danish/Norwegian
    "æ": "ae",   # Danish/Norwegian
    "œ": "oe",   # French ligature
    "ß": "ss",   # German sharp s
    "ð": "d",    # eth
    "þ": "th",   # thorn
    "đ": "d",
    "ł": "l",    # Polish
}


def normalize_text(text: str) -> str:
    """lowercase → explicit fold → NFKD strip-combining-marks → ASCII surface form.

    Examples:
        'Højre Øre'              -> 'hojre ore'
        'Fußgänger'              -> 'fussganger'
        "j'écoute la traduction" -> "j'ecoute la traduction"
    """
    text = str(text).lower().strip()
    # Atomic letters NFKD can't split (ø, æ, ß, …).
    text = "".join(_FOLD.get(ch, ch) for ch in text)
    # Decompose the rest (é→e+◌́, ä→a+◌̈, å→a+◌̊, ç→c+◌̧, ü, à, î, ñ, …) and drop marks.
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


if __name__ == "__main__":
    for s in ["Højre Øre", "Fußgänger", "j'écoute la traduction",
              "hvorfor skærer mit højre øre", "Wie heißt das"]:
        print(f"{s!r:45} -> {normalize_text(s)!r}")
