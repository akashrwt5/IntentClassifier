"""Make the model smaller without changing what it does for this app.

35 MB against the 0.236 MB the previous student needed. Two thirds of that is
not the language model at all:

    bge-small-en-v1.5   33.4M params
      token embedding   11.7M   (30522 x 384)  <- 35% of the model
      12 layers          21.3M
      pooler etc          0.4M

The embedding table carries a vocabulary built for the whole internet. This app
sees hearing-aid commands in English, and the corpus touches a small fraction of
those rows. Every untouched row is weight that ships to the phone and is never
read.

Two reductions, deliberately separate because they carry different risk:

  VOCABULARY PRUNING is close to free. Keep only the tokens the corpus actually
  produces, plus the special tokens. In-domain text tokenizes identically, so
  in-domain behaviour is unchanged by construction. What it costs is graceful
  handling of words never seen — including novel ASR errors at runtime, which
  is exactly the input that is already hardest. The STT suite is where that
  cost shows up, so check it rather than assuming.

  LAYER PRUNING is not free. Dropping transformer layers removes capacity, and
  the layers that resolve word order are the ones corrective negation depends
  on. Always re-run finetune_encoder.py afterwards and re-measure — a pruned
  encoder that was never fine-tuned back is a different, worse model.

    python scripts/shrink_encoder.py --encoder bge-small-en-v1.5 --prune-vocab
    python scripts/shrink_encoder.py --encoder bge-small-en-v1.5 \\
           --prune-vocab --keep-layers 6

Neither gets near 0.236 MB. Reaching that needs a small model distilled from
this one, which is a phase the original plan does not contain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from encoders import ROOT, discover_local_encoders  # noqa: E402
from pipeline import DATA  # noqa: E402


def corpus_texts() -> list[str]:
    """Everything the model will ever be asked about, in training or evaluation.
    Missing a file here silently removes tokens the suites depend on."""
    files = [
        "train_augmented.csv",
        "train.csv",
        "validation.csv",
        "test.csv",
        "stt_test.csv",
        "hard_negative_test.csv",
        "ood_test.csv",
        "minimal_pair_test.csv",
        "negation_test.csv",
        "contextual_test.csv",
    ]
    out = []
    for f in files:
        p = DATA / f
        if p.exists():
            out += pd.read_csv(p)["text"].astype(str).tolist()
    return out


def main() -> None:
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="bge-small-en-v1.5")
    ap.add_argument("--prune-vocab", action="store_true")
    ap.add_argument(
        "--keep-layers", type=int, default=None, help="keep only the first N transformer layers"
    )
    ap.add_argument("--suffix", default=None)
    args = ap.parse_args()

    local = discover_local_encoders()
    if args.encoder not in local:
        raise SystemExit(f"'{args.encoder}' not in models/encoders/: {list(local)}")
    src = local[args.encoder]
    tok = AutoTokenizer.from_pretrained(str(src), local_files_only=True)
    model = AutoModel.from_pretrained(str(src), local_files_only=True)
    before = sum(p.numel() for p in model.parameters())

    suffix = args.suffix or (
        ("-v" if args.prune_vocab else "") + (f"-L{args.keep_layers}" if args.keep_layers else "")
    )
    if not suffix:
        raise SystemExit("nothing to do — pass --prune-vocab and/or --keep-layers")
    dst = ROOT / "models" / "encoders" / f"{args.encoder}{suffix}"
    dst.mkdir(parents=True, exist_ok=True)

    report = dict(source=args.encoder, params_before=int(before))

    # --- vocabulary ------------------------------------------------------
    if args.prune_vocab:
        texts = corpus_texts()
        used = set()
        for i in range(0, len(texts), 512):
            for ids in tok(texts[i : i + 512])["input_ids"]:
                used.update(ids)
        specials = [
            tok.cls_token_id,
            tok.sep_token_id,
            tok.pad_token_id,
            tok.unk_token_id,
            tok.mask_token_id,
        ]
        used.update(x for x in specials if x is not None)
        keep = sorted(used)
        old2new = {o: n for n, o in enumerate(keep)}

        emb = model.get_input_embeddings().weight.data
        new_emb = torch.nn.Embedding(len(keep), emb.shape[1])
        new_emb.weight.data = emb[torch.tensor(keep)].clone()
        model.set_input_embeddings(new_emb)
        model.config.vocab_size = len(keep)

        inv = {v: k for k, v in tok.get_vocab().items()}
        tokens = [inv[o] for o in keep]
        (dst / "vocab.txt").write_text("\n".join(tokens) + "\n")

        # The tokenizer MUST be rebuilt, not copied. Saving the source
        # tokenizer alongside a pruned embedding produces a model that emits
        # token ids up to the ORIGINAL vocab size into an embedding table that
        # no longer has those rows — an IndexError on the first out-of-range
        # token, and silently wrong ids before that.
        from tokenizers import BertWordPieceTokenizer
        from transformers import BertTokenizerFast

        lower = bool(
            getattr(
                tok, "do_lower_case", getattr(tok, "init_kwargs", {}).get("do_lower_case", True)
            )
        )
        new_tok = BertWordPieceTokenizer(
            vocab={t: i for i, t in enumerate(tokens)}, lowercase=lower
        )
        new_tok.save(str(dst / "tokenizer.json"))
        BertTokenizerFast(
            tokenizer_file=str(dst / "tokenizer.json"),
            do_lower_case=lower,
            unk_token=tok.unk_token,
            sep_token=tok.sep_token,
            pad_token=tok.pad_token,
            cls_token=tok.cls_token,
            mask_token=tok.mask_token,
        ).save_pretrained(dst)

        (dst / "vocab_id_map.json").write_text(
            json.dumps(
                {
                    "note": "old token id -> new token id, for anything that stored "
                    "ids from the source tokenizer",
                    "map": {str(k): v for k, v in old2new.items()},
                }
            )
        )
        report.update(
            vocab_before=int(emb.shape[0]),
            vocab_after=len(keep),
            vocab_coverage=round(len(keep) / emb.shape[0], 4),
        )
        print(
            f"vocabulary {emb.shape[0]} -> {len(keep)} " f"({100*len(keep)/emb.shape[0]:.1f}% kept)"
        )

    # --- layers ----------------------------------------------------------
    if args.keep_layers:
        n_before = model.config.num_hidden_layers
        if args.keep_layers >= n_before:
            raise SystemExit(f"model already has {n_before} layers")
        model.encoder.layer = torch.nn.ModuleList(list(model.encoder.layer)[: args.keep_layers])
        model.config.num_hidden_layers = args.keep_layers
        report.update(layers_before=n_before, layers_after=args.keep_layers)
        print(f"layers {n_before} -> {args.keep_layers}")

    model.save_pretrained(dst)
    if not args.prune_vocab:
        tok.save_pretrained(dst)

    # --- self-check -------------------------------------------------------
    # This exists because the first version of this script shipped a pruned
    # embedding next to the ORIGINAL tokenizer, and the smoke test — two short
    # sentences — did not catch it. Frequent tokens have low ids, so short
    # in-domain text encodes fine right up until it doesn't. The check now runs
    # the whole corpus, which is the only thing that would have caught it.
    if args.prune_vocab:
        from transformers import AutoTokenizer

        check_tok = AutoTokenizer.from_pretrained(str(dst), local_files_only=True)
        vs = model.config.vocab_size
        worst, mismatched = 0, 0
        for i in range(0, len(texts), 512):
            chunk = texts[i : i + 512]
            for old_ids, new_ids in zip(tok(chunk)["input_ids"], check_tok(chunk)["input_ids"]):
                worst = max(worst, max(new_ids))
                if [old2new.get(x, -1) for x in old_ids] != new_ids:
                    mismatched += 1
        if worst >= vs:
            raise SystemExit(
                f"FAILED: tokenizer emits id {worst} into an embedding of "
                f"{vs} rows. The saved tokenizer does not match the pruned "
                f"vocabulary — do not use this model."
            )
        if mismatched:
            raise SystemExit(
                f"FAILED: {mismatched} of {len(texts)} corpus rows tokenize "
                f"differently after pruning. In-domain tokenization must be "
                f"identical by construction; something dropped a needed token."
            )
        print(
            f"self-check: {len(texts)} corpus rows, max token id {worst} "
            f"< {vs}, 0 tokenization changes"
        )
        report.update(selfcheck_rows=len(texts), selfcheck_max_token_id=int(worst))

    after = sum(p.numel() for p in model.parameters())
    report.update(
        params_after=int(after),
        fp32_mb=round(after * 4 / 1e6, 2),
        int8_mb_estimate=round(after / 1e6, 2),
        reduction=round(1 - after / before, 4),
    )
    (dst / "shrink.json").write_text(json.dumps(report, indent=2))

    print(f"params {before/1e6:.1f}M -> {after/1e6:.1f}M " f"({100*(1-after/before):.1f}% smaller)")
    print(f"estimated INT8 size: {after/1e6:.1f} MB")
    print(f"saved -> {dst}")
    print(
        "\nMeasure before trusting this. The STT suite is where vocabulary "
        "pruning hurts, and layer pruning needs fine-tuning to recover:"
    )
    print(
        f"  python scripts/train_classifier.py --encoder {args.encoder}{suffix} "
        f"--classifier mlp --train train_augmented --out models/shrunk"
    )
    if args.keep_layers:
        print(
            f"  python scripts/finetune_encoder.py --encoder "
            f"{args.encoder}{suffix}   # required after layer pruning"
        )


if __name__ == "__main__":
    main()
