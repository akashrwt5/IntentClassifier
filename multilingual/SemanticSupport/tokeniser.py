"""
MultilingualTokeniser — fixed-length WordPiece tokeniser for the
paraphrase-multilingual-MiniLM-L12-v2 ONNX encoder.

Design invariants (required for ANE / Core ML downstream compilation):
  * Output shapes are ALWAYS (1, SEQ_LEN) regardless of input length.
  * Sequences shorter than SEQ_LEN are right-padded with zero token-ids
    and zero attention-mask values.
  * Sequences longer than SEQ_LEN are hard-truncated after [CLS] and the
    final token slot is reserved for [SEP].
  * No dynamic resizing, no variable-length lists emitted after shape lock.

The multilingual MiniLM vocabulary has ~105k entries and is case-preserving
(do_lower_case=False), unlike the English MiniLM-L6-v2 which lowercases.
"""

import re
import unicodedata
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

SEQ_LEN = 64    # locked into the exported ONNX graph — do not change at runtime

_VOCAB_CACHE: Optional[dict] = None


def _load_vocab(vocab_path: Path) -> dict:
    global _VOCAB_CACHE
    if _VOCAB_CACHE is None:
        with open(vocab_path, encoding="utf-8") as f:
            _VOCAB_CACHE = {line.strip(): i for i, line in enumerate(f)}
    return _VOCAB_CACHE


def _clean_text(text: str) -> str:
    """Apply Unicode NFC normalisation and collapse whitespace.

    Mirrors HuggingFace BertTokenizer's BasicTokenizer: control chars and
    zero-width joiners are removed; whitespace is collapsed to single spaces.
    We intentionally do NOT lowercase here — the multilingual MiniLM vocab is
    case-preserving (do_lower_case=False in tokenizer_config.json).
    """
    out = []
    for ch in unicodedata.normalize("NFC", text):
        cat = unicodedata.category(ch)
        if cat == "Cc" or cat == "Cf":
            continue
        if unicodedata.category(ch) == "Zs" or ch in (" ", "\t", "\n", "\r"):
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out).strip()


def _tokenize_chinese_chars(text: str) -> str:
    """Add whitespace around CJK characters so they become individual tokens."""
    out = []
    for ch in text:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF or
                0x3400 <= cp <= 0x4DBF or
                0x20000 <= cp <= 0x2A6DF or
                0x2A700 <= cp <= 0x2B73F or
                0x2B740 <= cp <= 0x2B81F or
                0x2B820 <= cp <= 0x2CEAF or
                0xF900 <= cp <= 0xFAFF or
                0x2F800 <= cp <= 0x2FA1F):
            out.extend([" ", ch, " "])
        else:
            out.append(ch)
    return "".join(out)


def _wordpiece(word: str, vocab: dict, unk: str = "[UNK]") -> list:
    """Greedy longest-match WordPiece segmentation for a single word."""
    if len(word) > 100:
        return [unk]
    chars = list(word)
    is_bad = False
    start = 0
    sub_tokens = []
    while start < len(chars):
        end = len(chars)
        cur = None
        while start < end:
            substr = "".join(chars[start:end])
            if start > 0:
                substr = "##" + substr
            if substr in vocab:
                cur = substr
                break
            end -= 1
        if cur is None:
            is_bad = True
            break
        sub_tokens.append(cur)
        start = end
    return [unk] if is_bad else sub_tokens


def tokenize(text: str, vocab: dict) -> list:
    """Full tokenization pipeline: clean → split → WordPiece."""
    text = _tokenize_chinese_chars(_clean_text(text))
    # Split on whitespace and punctuation, keeping punctuation as separate tokens.
    # re.findall preserves punctuation as individual tokens, matching HuggingFace
    # BasicTokenizer behaviour (do_tokenize_chinese_chars=True, strip_accents=None).
    word_tokens = re.findall(r"\w+|[^\w\s]", text)
    wp_tokens = []
    for word in word_tokens:
        wp_tokens.extend(_wordpiece(word, vocab))
    return wp_tokens


class MultilingualTokeniser:
    """Deterministic, fixed-length tokeniser for ANE-compatible static-shape inference.

    All outputs have shape (1, SEQ_LEN) — the batch dimension and sequence
    dimension are both constants baked into the ONNX graph.
    """

    def __init__(self, vocab_path: Path):
        self.vocab = _load_vocab(vocab_path)
        self._unk_id = self.vocab.get("[UNK]", 0)
        self._pad_id = self.vocab.get("[PAD]", 0)
        self._cls_id = self.vocab.get("[CLS]", 101)
        self._sep_id = self.vocab.get("[SEP]", 102)

    def encode(self, text: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (input_ids, attention_mask, token_type_ids), each shape (1, SEQ_LEN).

        The output shapes are invariant — every call returns exactly (1, SEQ_LEN)
        arrays regardless of text length. This is required for the ONNX graph
        which has these dimensions baked in as static values for ANE compilation.
        """
        tokens = tokenize(text, self.vocab)

        # Reserve 2 slots for [CLS] and [SEP]; truncate body to fit.
        max_body = SEQ_LEN - 2
        tokens = tokens[:max_body]

        ids = [self._cls_id] + [self.vocab.get(t, self._unk_id) for t in tokens] + [self._sep_id]
        real_len = len(ids)  # ≤ SEQ_LEN

        # Right-pad to exact SEQ_LEN with pad token (id=0, mask=0).
        input_ids      = np.full(SEQ_LEN, self._pad_id, dtype=np.int64)
        attention_mask = np.zeros(SEQ_LEN, dtype=np.int64)
        token_type_ids = np.zeros(SEQ_LEN, dtype=np.int64)

        input_ids[:real_len]      = ids
        attention_mask[:real_len] = 1

        return (
            input_ids[np.newaxis, :],       # (1, SEQ_LEN)
            attention_mask[np.newaxis, :],  # (1, SEQ_LEN)
            token_type_ids[np.newaxis, :],  # (1, SEQ_LEN)
        )

    def unk_rate(self, text: str) -> float:
        """Fraction of WordPiece tokens that resolved to [UNK]. Useful for diagnostics."""
        tokens = tokenize(text, self.vocab)
        if not tokens:
            return 0.0
        unk_count = sum(1 for t in tokens if self.vocab.get(t, self._unk_id) == self._unk_id)
        return unk_count / len(tokens)
