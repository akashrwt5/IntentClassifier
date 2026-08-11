"""
Shared helpers: tokenizer, vocab, IO, leak guard.

The tokenizer here is THE definition used everywhere (train, eval, export,
runtime). If it changes, the vocab and every artifact must be rebuilt.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# Punctuation is discarded. "volume up" and "volume up?" are the SAME input.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    t = unicodedata.normalize("NFKD", str(text)).replace("’", "'")
    return _TOKEN_RE.findall(t.lower())


def token_key(text: str) -> str:
    """Canonical identity of an utterance, as the model sees it."""
    return " ".join(tokenize(text))


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_rows(path: Path, text_col=None, label_col=None) -> list[tuple[str, str]]:
    rows = read_csv(path)
    if not rows:
        return []
    cols = list(rows[0].keys())
    tc = text_col or next(c for c in cols if c.lower() in ("text", "utterance", "phrase"))
    lc = label_col or next(
        (c for c in cols if c.lower() in ("intent", "label", "expected", "expected_intent")),
        None,
    )
    return [(r[tc], r[lc] if lc else "") for r in rows]


# ------------------------------------------------------------------ vocab
#
# Two tokenizer modes.
#
#   word    — one id per whole word. Anything unseen becomes [UNK], which
#             destroys all information about it. Experiments v1-v4 showed this
#             creates a hard Pareto frontier: the model cannot tell
#             "quieter" (real command, unseen word) from "asdfghjkl" (junk),
#             because after tokenization both are literally the same token.
#
#   subword — WordPiece-style greedy longest match over learned pieces, with
#             every single character in the vocab as a floor. NO INPUT IS EVER
#             [UNK]: "quieter" -> "quiet" + "##er" keeps its meaning, while
#             "asdfghjkl" shatters into many rare character pieces. That
#             difference is the signal the word-level model never had.


def build_vocab(texts, min_freq: int = 1) -> dict[str, int]:
    freq = Counter(tok for t in texts for tok in tokenize(t))
    vocab = {"[PAD]": config.PAD_ID, "[UNK]": config.UNK_ID}
    for tok, n in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])):
        if n >= min_freq:
            vocab.setdefault(tok, len(vocab))
    return vocab


def build_subword_vocab(texts, size: int = 3000, max_piece: int = 8) -> dict[str, int]:
    """Learn WordPiece-ish pieces from the corpus. Characters guarantee coverage."""
    word_freq = Counter(tok for t in texts for tok in tokenize(t))

    vocab = {"[PAD]": config.PAD_ID, "[UNK]": config.UNK_ID}
    # every character seen anywhere — this is what removes [UNK] entirely
    chars = {c for w in word_freq for c in w}
    for c in sorted(chars):
        vocab.setdefault(c, len(vocab))
        vocab.setdefault("##" + c, len(vocab))

    # candidate pieces scored by frequency * length (prefer long frequent ones)
    scores: Counter = Counter()
    for w, f in word_freq.items():
        n = len(w)
        for i in range(n):
            for j in range(i + 2, min(i + max_piece, n) + 1):
                piece = w[i:j] if i == 0 else "##" + w[i:j]
                scores[piece] += f * (j - i)

    for piece, _ in scores.most_common():
        if len(vocab) >= size:
            break
        vocab.setdefault(piece, len(vocab))
    return vocab


def _wordpiece(word: str, vocab: dict) -> list[str]:
    """Greedy longest match. Falls back to characters, so never fails."""
    out, start = [], 0
    while start < len(word):
        end = len(word)
        cur = None
        while start < end:
            piece = word[start:end] if start == 0 else "##" + word[start:end]
            if piece in vocab:
                cur = piece
                break
            end -= 1
        if cur is None:  # unseen character
            out.append("[UNK]")
            start += 1
        else:
            out.append(cur)
            start = end
    return out


def tokenize_subword(text: str, vocab: dict) -> list[str]:
    return [p for w in tokenize(text) for p in _wordpiece(w, vocab)]


def encode(text: str, vocab: dict[str, int], max_len: int = config.MAX_LEN, mode="word"):
    if mode == "subword":
        pieces = tokenize_subword(text, vocab)
        ids = [vocab.get(p, config.UNK_ID) for p in pieces][:max_len]
    else:
        ids = [vocab.get(t, config.UNK_ID) for t in tokenize(text)][:max_len]
    n = len(ids)
    ids += [config.PAD_ID] * (max_len - n)
    return ids, n


def save_vocab(vocab: dict, path: Path, mode: str = "word") -> None:
    path.write_text(json.dumps({"mode": mode, "vocab": vocab}, indent=2), encoding="utf-8")


def load_vocab(path: Path) -> tuple[dict, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "vocab" in raw and "mode" in raw:
        return raw["vocab"], raw["mode"]
    return raw, "word"  # legacy files (v1-v4) are bare word vocabs


# ------------------------------------------------------------------ guards


def assert_no_leak(train_texts, eval_texts, name: str) -> int:
    """Fail loudly if any eval row is present in training, tokenizer-view."""
    tr = {token_key(t) for t in train_texts}
    ev = {token_key(t) for t in eval_texts}
    overlap = tr & ev
    if overlap:
        sample = list(overlap)[:5]
        raise SystemExit(
            f"LEAK: {len(overlap)} rows of {name} appear in training data.\n"
            f"  examples: {sample}\n"
            f"  Fix the data before training — otherwise the score is meaningless."
        )
    return 0


def class_weights(labels, label_list) -> dict[str, float]:
    """w[c] = n_total / (n_classes * n[c])  — standard inverse-frequency."""
    counts = Counter(labels)
    n_total, n_classes = len(labels), len(label_list)
    return {c: n_total / (n_classes * counts[c]) for c in label_list if counts[c]}
