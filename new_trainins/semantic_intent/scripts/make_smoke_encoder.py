"""Build a tiny, randomly-initialized BERT locally.

This is NOT a model candidate. It exists so the transformer training + ONNX
export + parity path can be exercised end to end with no network access, which
means that when the real E5/MiniLM/BGE weights are dropped into
models/encoders/, the only thing that changes is the weights.

Its tokenizer is trained on the actual corpus, so the vocabulary and the
WordPiece behaviour are representative even though the weights are noise.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "encoders" / "smoke-tinybert"


def main(
    vocab_size: int = 4096, hidden: int = 128, layers: int = 2, heads: int = 2, inter: int = 256
) -> None:
    from tokenizers import BertWordPieceTokenizer
    from transformers import BertConfig, BertModel, BertTokenizerFast

    OUT.mkdir(parents=True, exist_ok=True)
    corpus = ROOT / "data" / "raw" / "en.csv"
    txt = OUT / "_corpus.txt"
    txt.write_text("\n".join(pd.read_csv(corpus)["text"].astype(str)))

    tk = BertWordPieceTokenizer(lowercase=True)
    tk.train(
        [str(txt)],
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )
    tk.save_model(str(OUT))  # vocab.txt — this is what Android reads
    tk.save(str(OUT / "tokenizer.json"))
    txt.unlink()

    # transformers>=5 builds BertTokenizerFast from tokenizer.json, not vocab.txt
    fast = BertTokenizerFast(
        tokenizer_file=str(OUT / "tokenizer.json"),
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )
    fast.save_pretrained(OUT)

    cfg = BertConfig(
        vocab_size=fast.vocab_size,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=inter,
        max_position_embeddings=128,
    )
    BertModel(cfg).save_pretrained(OUT)
    n = sum(p.numel() for p in BertModel(cfg).parameters())
    print(
        f"smoke encoder at {OUT}: vocab={fast.vocab_size} hidden={hidden} "
        f"layers={layers} params={n/1e6:.2f}M"
    )


if __name__ == "__main__":
    main()
