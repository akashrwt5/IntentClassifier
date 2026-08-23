"""Which architectures fit a given INT8 size budget.

The question is not "how much can we shave off bge" — it is "what shape of model
fits in 2-5 MB", and those turn out to be different questions.

BERT-style parameter count:

    token embedding     V * H
    position embedding  P * H          <- P can be cut to max_len (64), not 512
    type + norms        ~4H
    per layer           12H^2 + 13H    <- 4H^2 attention, 8H^2 FFN (I = 4H)
    classifier head     H * n_intents

INT8 is roughly one byte per parameter, so params in millions ~= MB.

The term that decides everything is 12H^2 per layer. It is quadratic in H, so
halving the hidden size quarters the cost of depth. That is why bge-small
(H=384) cannot be pruned into this budget while keeping useful depth: its
embedding alone eats most of the budget and each layer costs 1.77 MB. The
existing 0.236 MB student implies H around 26 — it was distilled small, not
pruned small, and those are not interchangeable.

    python scripts/size_budget.py --min-mb 2 --max-mb 5
    python scripts/size_budget.py --measure-vocab bge-small-en-v1.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

N_INTENTS = 57


def params(
    hidden: int, layers: int, vocab: int, max_pos: int = 64, n_intents: int = N_INTENTS
) -> dict:
    emb = vocab * hidden
    pos = max_pos * hidden
    misc = 4 * hidden
    per_layer = 12 * hidden * hidden + 13 * hidden
    head = hidden * n_intents + n_intents
    total = emb + pos + misc + per_layer * layers + head
    return dict(
        embedding=emb,
        layers=per_layer * layers,
        other=pos + misc + head,
        total=total,
        mb=total / 1e6,
        embedding_share=emb / total,
    )


def measure_vocab(encoder: str) -> int:
    """How many distinct wordpiece tokens the corpus actually produces."""
    import pandas as pd
    from transformers import AutoTokenizer
    from encoders import discover_local_encoders
    from pipeline import DATA

    local = discover_local_encoders()
    if encoder not in local:
        raise SystemExit(f"'{encoder}' not found: {list(local)}")
    tok = AutoTokenizer.from_pretrained(str(local[encoder]), local_files_only=True)
    texts = []
    for f in (
        "train_augmented.csv",
        "validation.csv",
        "test.csv",
        "stt_test.csv",
        "hard_negative_test.csv",
        "ood_test.csv",
        "minimal_pair_test.csv",
        "negation_test.csv",
        "contextual_test.csv",
    ):
        p = DATA / f
        if p.exists():
            texts += pd.read_csv(p)["text"].astype(str).tolist()
    used = set()
    for i in range(0, len(texts), 512):
        for ids in tok(texts[i : i + 512])["input_ids"]:
            used.update(ids)
    print(
        f"{encoder}: corpus uses {len(used)} of {tok.vocab_size} tokens "
        f"({100*len(used)/tok.vocab_size:.1f}%) across {len(texts)} rows"
    )
    return len(used)


CANDIDATES = [
    # (hidden, layers, note)
    (384, 1, "bge pruned to 1 layer"),
    (384, 2, "bge pruned to 2 layers"),
    (384, 4, "bge pruned to 4 layers"),
    (256, 2, "distilled student"),
    (256, 3, "distilled student"),
    (256, 4, "distilled student"),
    (192, 4, "distilled student"),
    (192, 6, "distilled student"),
    (160, 6, "distilled student"),
    (128, 4, "distilled student"),
    (128, 6, "distilled student"),
    (128, 8, "distilled student"),
    (96, 6, "distilled student"),
    (64, 4, "distilled student"),
    (32, 3, "roughly the existing 0.236 MB student"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--max-pos", type=int, default=64)
    ap.add_argument("--min-mb", type=float, default=2.0)
    ap.add_argument("--max-mb", type=float, default=5.0)
    ap.add_argument("--measure-vocab", default=None)
    args = ap.parse_args()

    vocab = measure_vocab(args.measure_vocab) if args.measure_vocab else args.vocab

    print(
        f"\nvocab={vocab}  max_pos={args.max_pos}  intents={N_INTENTS}  "
        f"budget={args.min_mb}-{args.max_mb} MB (INT8)\n"
    )
    print(
        f"{'H':>5} {'L':>3} {'INT8 MB':>9} {'emb MB':>8} {'layer MB':>9} "
        f"{'emb%':>6} {'fits':>6}  note"
    )
    print("-" * 78)
    for h, l, note in CANDIDATES:
        p = params(h, l, vocab, args.max_pos)
        fits = args.min_mb <= p["mb"] <= args.max_mb
        print(
            f"{h:>5} {l:>3} {p['mb']:>9.2f} {p['embedding']/1e6:>8.2f} "
            f"{p['layers']/1e6:>9.2f} {100*p['embedding_share']:>5.0f}% "
            f"{'YES' if fits else '-':>6}  {note}"
        )

    print("\nvocabulary sensitivity at a few shapes (INT8 MB):")
    print(f"{'H/L':>8} " + "".join(f"{v:>9}" for v in (4000, 6000, 8000, 12000)))
    for h, l in ((256, 3), (192, 4), (128, 6)):
        row = "".join(
            f"{params(h, l, v, args.max_pos)['mb']:>9.2f}" for v in (4000, 6000, 8000, 12000)
        )
        print(f"{h:>4}/{l:<3} {row}")

    print(
        "\nCutting position embeddings from 512 to 64 saves "
        f"{448 * 256 / 1e6:.2f} MB at H=256 and costs nothing — the runtime "
        "already truncates at 64 tokens."
    )


if __name__ == "__main__":
    main()
