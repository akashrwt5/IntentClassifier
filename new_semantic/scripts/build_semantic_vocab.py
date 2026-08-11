#!/usr/bin/env python3
"""
Give the student real word meanings instead of a corpus-only lookup table.

THE PROBLEM THIS FIXES
----------------------
The shipped student learns `nn.Embedding(1982, 64)` from scratch on 24k
in-domain sentences. The E5 teacher only supplied soft CLASS distributions, so
none of its knowledge about English words ever reached the student. Result:

    increase / raise / boost / amplify / crank up the volume  -> VolumeIncrease
    elevate  / heighten                the volume             -> fallback, 0.5131

Both unknown words score IDENTICALLY because after tokenisation they are the
same input: `[UNK] the volume`. The model has no way to know "elevate" is a
synonym of "raise" — it never saw the word and carries no external semantics.

WHAT THIS SCRIPT DOES
---------------------
1. Vocabulary = every corpus word (never dropped) + the most common general
   English words taken from the teacher's own tokenizer.
2. Each word is embedded ALONE by the teacher, giving a 384-d vector that
   already places `elevate` next to `raise`.
3. PCA 384 -> EMBED_DIM (64), then written as an init matrix for
   `nn.Embedding`.

So a word absent from training data still arrives with a vector in roughly the
right place, instead of collapsing to `[UNK]`.

Output: models/en/embed_init_<tag>.npz  {vocab, matrix, meta}

Requires: torch, sentence-transformers, scikit-learn (training-time only).

Usage:
    python scripts/build_semantic_vocab.py                     # 8000 words
    python scripts/build_semantic_vocab.py --size 12000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.common import load_rows, tokenize  # noqa: E402

WORD_RE = re.compile(r"^[a-z][a-z'-]*[a-z]$|^[a-z]$")


def general_words(limit: int, teacher: str) -> list[str]:
    """Common English words, taken from the teacher's tokenizer.

    BERT-family vocabularies are ordered roughly by corpus frequency, so the
    early whole-word entries are the common words. Sub-word pieces ('##ing')
    and symbols are dropped — this student is word-level.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(teacher)
    out = []
    for piece in tok.get_vocab():
        if piece.startswith("##") or piece.startswith("["):
            continue
        w = piece.lower()
        if WORD_RE.match(w) and len(w) > 1:
            out.append(w)
    # get_vocab() is a dict; recover frequency order via the id
    ids = tok.get_vocab()
    out.sort(key=lambda w: ids.get(w, 10**9))
    return out[:limit]


def domain_words(
    corpus_words: list[str], limit: int, teacher: str, enc, batch: int
) -> list[str]:
    """Expand the vocabulary with words that are NEAR THE DOMAIN, not words that
    are common on Wikipedia.

    `general_words` fills the budget in teacher-tokenizer id order, which for a
    BERT-family model is roughly corpus frequency. Measured on the shipped
    8,000-token vocabulary, that produced:

        present : aaron, abdul, abbey, abraham, output
        ABSENT  : elevate, heighten, diminish, dampen, magnify, lessen

    75% of entries never appeared in a training row, and on `locked` the rows
    containing one scored 0.7000 against 0.9196 for rows that did not — a
    22-point gap that clears the interval. The mechanism is direct: with frozen
    embeddings, an entry the transformer never saw in training contributes a
    fixed vector nothing ever taught it to interpret.

        "turn up the volume"  -> Cmd.VolumeIncrease    0.89
        "turn up the output"  -> Cmd.ActivityCalories  0.96

    So select by SEMANTIC PROXIMITY to the corpus instead: score each candidate
    by its highest cosine similarity to any word the training data actually
    uses, and keep the closest. 'elevate' scores high against 'increase' and
    earns its slot; 'aaron' does not.
    """
    import numpy as _np

    pool = [w for w in general_words(limit * 8, teacher) if w not in set(corpus_words)]
    if not pool or limit <= 0:
        return []
    print(f"  scoring {len(pool)} candidates against {len(corpus_words)} corpus words ...")
    Ec = enc.encode(corpus_words, batch_size=batch, normalize_embeddings=True)
    Ep = enc.encode(pool, batch_size=batch, normalize_embeddings=True,
                    show_progress_bar=True)
    best = (_np.asarray(Ep) @ _np.asarray(Ec).T).max(axis=1)
    order = _np.argsort(-best)[:limit]
    kept = [pool[i] for i in order]
    print(f"  kept {len(kept)}: closest {kept[:8]} ... furthest {kept[-4:]}")
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--size",
        type=int,
        default=8000,
        help="total vocabulary target (corpus words are always kept)",
    )
    ap.add_argument(
        "--expand",
        choices=("frequency", "domain"),
        default="frequency",
        help="how to spend the budget beyond the training corpus. 'frequency' "
        "(default, reproduces every existing artifact) takes the teacher "
        "tokenizer's most frequent whole words — which is Wikipedia frequency, "
        "so it yields names and misses domain synonyms. 'domain' keeps the "
        "candidates most similar to words the corpus actually uses. Run "
        "scripts/vocab_health.py to see what the current choice produced.",
    )
    ap.add_argument("--dim", type=int, default=config.EMBED_DIM)
    ap.add_argument("--tag", default="sem")
    ap.add_argument(
        "--teacher",
        default=config.TEACHER,
        help="the encoder whose semantics get baked in. MUST be passed to change "
        "it — `--tag` only names the output file. That gap already cost one "
        "experiment: `--tag bge` produced a file called embed_init_bge.npz that "
        "contained e5-small embeddings, and the 'bge' run came out bit-identical "
        "to the e5 run because nothing had actually changed.",
    )
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    teacher = args.teacher
    print(f"teacher          : {teacher}")
    # The tag is only a filename. Refuse a tag that names a different model —
    # checked BEFORE any import or download so it fails in a second, not after
    # fetching 130 MB of the wrong encoder.
    for hint, must in (("bge", "bge"), ("e5", "e5"), ("minilm", "MiniLM")):
        if hint in args.tag.lower() and must.lower() not in teacher.lower():
            raise SystemExit(
                f"ABORT: --tag {args.tag!r} says {hint!r} but --teacher is "
                f"{teacher!r}. The artifact would be named after a model it does "
                f"not contain. Pass --teacher, or rename the tag."
            )

    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    corpus_words = sorted({w for t, _ in load_rows(config.TRAIN_CSV) for w in tokenize(t)})
    print(f"corpus words     : {len(corpus_words)}")

    extra_budget = max(0, args.size - len(corpus_words) - 2)
    enc = SentenceTransformer(teacher)

    if args.expand == "domain":
        print(f"expansion        : domain (budget {extra_budget})")
        pool = domain_words(corpus_words, extra_budget, teacher, enc, args.batch)
    else:
        pool = [w for w in general_words(extra_budget * 3, teacher) if w not in set(corpus_words)][
            :extra_budget
        ]
        print(f"general words    : {len(pool)}  (from {teacher} tokenizer, frequency order)")

    words = corpus_words + pool
    vocab = {"[PAD]": config.PAD_ID, "[UNK]": config.UNK_ID}
    for w in words:
        vocab.setdefault(w, len(vocab))
    print(f"final vocabulary : {len(vocab)}")

    print(f"embedding {len(words)} words with {teacher} ...")
    E = enc.encode(words, batch_size=args.batch, normalize_embeddings=True, show_progress_bar=True)

    print(f"PCA {E.shape[1]} -> {args.dim}")
    pca = PCA(n_components=args.dim, random_state=config.SEED)
    Z = pca.fit_transform(E).astype(np.float32)
    # scale to the magnitude a freshly-initialised nn.Embedding would have, so
    # the initial forward pass is not dominated by these vectors
    Z *= (1.0 / max(np.abs(Z).std(), 1e-6)) * 0.02

    M = np.zeros((len(vocab), args.dim), dtype=np.float32)
    for w, z in zip(words, Z):
        M[vocab[w]] = z
    # [PAD] stays zero; [UNK] gets the mean so it is not an outlier
    M[config.UNK_ID] = Z.mean(0)

    out = config.MODELS / f"embed_init_{args.tag}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        matrix=M,
        vocab=json.dumps(vocab),
        meta=json.dumps(
            {
                "teacher": teacher,
                "dim": args.dim,
                "corpus_words": len(corpus_words),
                "general_words": len(pool),
                "expansion": args.expand,
                "vocab_size": len(vocab),
                "explained_variance": float(pca.explained_variance_ratio_.sum()),
            }
        ),
    )
    print(f"\nPCA keeps {pca.explained_variance_ratio_.sum() * 100:.1f}% of variance")
    print(f"embedding table  : {M.nbytes / 1e6:.2f} MB fp32")
    print(f"wrote {out}")

    # sanity: are synonyms actually neighbours now?
    print("\nnearest neighbours in the projected space (sanity check):")
    idx = {w: i for i, w in enumerate(words)}
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    for probe in ("louder", "raise", "quieter", "remind"):
        if probe not in idx:
            continue
        sims = Zn @ Zn[idx[probe]]
        near = [words[j] for j in np.argsort(-sims)[1:6]]
        print(f"  {probe:<10} -> {near}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
