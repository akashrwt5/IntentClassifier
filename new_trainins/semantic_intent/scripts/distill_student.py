"""Distil a small student that fits a 2-5 MB budget.

Pruning bge cannot get there. Its hidden size is 384, so each transformer layer
costs 12*384^2 = 1.77M parameters and the pruned embedding costs another 1.7-3
MB. Inside 5 MB that leaves room for exactly one layer, and one layer of a
twelve-layer model is not a model — it is a fragment. `size_budget.py` shows the
whole table.

The term that matters is 12H^2 per layer: quadratic in width, linear in depth.
Narrowing the model buys depth cheaply. At H=192 a layer costs 0.44M instead of
1.77M, so six layers fit in the same space one layer of bge needed. And depth is
the part worth keeping here — corrective negation turns on word order, which is
resolved by attention across layers, so a wide-and-shallow student would give up
precisely the capability that is already weakest.

The student learns from the teacher's full probability distribution, not just
the correct label. The distribution is where the teacher's knowledge actually
lives: told "this is Cmd.VolumeIncrease" the student learns one bit, but told
"0.7 increase, 0.2 decrease, 0.05 Help_Volume" it also learns which intents are
close to which, which is the structure it has too little capacity to rediscover
on its own.

    python scripts/size_budget.py --measure-vocab bge-small-en-v1.5
    python scripts/distill_student.py --teacher models/final --hidden 192 --layers 6

Then treat the student like any other encoder — the whole downstream pipeline is
unchanged:

    python scripts/train_classifier.py --encoder student-h192-l6 --classifier mlp \\
           --train train_augmented --out models/final_student
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from encoders import ROOT  # noqa: E402
from finetune_encoder import batches, pick_device  # noqa: E402
from pipeline import DATA, IntentModel  # noqa: E402


def build_student(
    hidden: int, layers: int, vocab_size: int, max_pos: int, n_classes: int, device: str
):
    import torch.nn as nn
    from transformers import BertConfig, BertModel

    heads = next(h for h in (12, 8, 6, 5, 4, 3, 2, 1) if hidden % h == 0)
    cfg = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=4 * hidden,
        max_position_embeddings=max_pos,
    )
    enc = BertModel(cfg, add_pooling_layer=False).to(device)
    head = nn.Linear(hidden, n_classes).to(device)
    return enc, head, heads


def pooled(enc, ids, mask):
    h = enc(input_ids=ids, attention_mask=mask).last_hidden_state
    m = mask.unsqueeze(-1).float()
    emb = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
    return emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-9)


def main() -> None:
    from sklearn.metrics import f1_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="models/final")
    ap.add_argument("--train", default="train_augmented")
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--max-pos", type=int, default=64)
    ap.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="a student starts from random weights with no "
        "pretraining; 8 epochs is nowhere near enough, and "
        "the one-cycle schedule anneals to zero at the end "
        "so a short run looks converged when it is not",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument(
        "--temperature",
        type=float,
        default=3.0,
        help="softens the teacher distribution; higher exposes more "
        "of the teacher's ranking below the top class",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="weight on the teacher's distribution vs the hard label",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--tokenizer-from",
        default=None,
        help="encoder whose (pruned) tokenizer the student should "
        "use. Without this the student inherits the teacher's "
        "full vocabulary and the embedding alone blows the "
        "size budget.",
    )
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)

    teacher = IntentModel.load(ROOT / args.teacher)
    prefix = getattr(teacher.encoder, "prefix", "")

    # The student's embedding table is V x H, and V dominates a small student:
    # at H=192 the teacher's full 30522-token vocabulary costs 5.9 MB before a
    # single transformer layer exists. Pruning it to the ~4-8k tokens this
    # corpus actually produces is what makes the budget reachable, and it is
    # lossless for in-domain text by construction.
    if args.tokenizer_from:
        from transformers import AutoTokenizer
        from encoders import discover_local_encoders

        local = discover_local_encoders()
        if args.tokenizer_from not in local:
            raise SystemExit(f"'{args.tokenizer_from}' not in models/encoders/: " f"{list(local)}")
        tok = AutoTokenizer.from_pretrained(str(local[args.tokenizer_from]), local_files_only=True)
        print(
            f"student vocabulary from {args.tokenizer_from}: "
            f"{tok.vocab_size} tokens "
            f"(teacher has {teacher.encoder.tok.vocab_size})"
        )
    else:
        tok = teacher.encoder.tok
        print(
            f"WARNING: student inherits the teacher's full "
            f"{tok.vocab_size}-token vocabulary. At H={args.hidden} that is "
            f"{tok.vocab_size * args.hidden / 1e6:.1f} MB of embedding alone. "
            f"Run shrink_encoder.py --prune-vocab and pass --tokenizer-from."
        )

    train = pd.read_csv(DATA / f"{args.train}.csv")
    val = pd.read_csv(DATA / "validation.csv")
    labels = teacher.labels
    idx = {l: i for i, l in enumerate(labels)}
    Xtr = [prefix + t for t in train["text"].astype(str)]
    Xva = [prefix + t for t in val["text"].astype(str)]
    ytr = [idx[l] for l in train["intent"]]
    yva = [idx[l] for l in val["intent"]]

    print(f"device={device}  teacher={args.teacher}  " f"student H={args.hidden} L={args.layers}")
    # Teacher logits come from the teacher's OWN tokenizer — that is the model
    # whose knowledge is being copied. Only the student is retokenized.
    print("computing teacher logits once...")
    t_logits = torch.tensor(teacher.logits(Xtr) / teacher.temperature, dtype=torch.float32)

    enc, head, heads = build_student(
        args.hidden, args.layers, tok.vocab_size, args.max_pos, len(labels), device
    )
    n_params = sum(p.numel() for p in enc.parameters())
    print(
        f"student: {n_params/1e6:.2f}M params ({heads} heads) " f"-> ~{n_params/1e6:.2f} MB INT8\n"
    )

    params = list(enc.parameters()) + list(head.parameters())
    steps = args.epochs * int(np.ceil(len(Xtr) / args.batch_size))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=0.1, anneal_strategy="linear"
    )
    ce = torch.nn.CrossEntropyLoss()
    kl = torch.nn.KLDivLoss(reduction="batchmean")
    T = args.temperature

    best = dict(f1=-1.0, epoch=-1)
    history = []
    for ep in range(args.epochs):
        enc.train()
        head.train()
        total, n = 0.0, 0
        order = np.arange(len(Xtr))
        rng.shuffle(order)
        for i in range(0, len(order), args.batch_size):
            sel = order[i : i + args.batch_size]
            b = tok(
                [Xtr[j] for j in sel],
                padding=True,
                truncation=True,
                max_length=args.max_pos,
                return_tensors="pt",
            )
            ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            y = torch.tensor([ytr[j] for j in sel], device=device)
            tl = t_logits[sel].to(device)

            opt.zero_grad()
            sl = head(pooled(enc, ids, mask))
            # T^2 keeps the soft-target gradients on the same scale as the hard
            # ones; without it the distillation term shrinks as T rises and the
            # temperature knob quietly turns into a learning-rate knob.
            loss = args.alpha * kl(
                torch.log_softmax(sl / T, -1), torch.softmax(tl / T, -1)
            ) * T * T + (1 - args.alpha) * ce(sl, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(sel)
            n += len(sel)

        enc.eval()
        head.eval()
        preds = []
        with torch.no_grad():
            for j in range(0, len(Xva), 128):
                b = tok(
                    Xva[j : j + 128],
                    padding=True,
                    truncation=True,
                    max_length=args.max_pos,
                    return_tensors="pt",
                )
                preds.append(
                    head(pooled(enc, b["input_ids"].to(device), b["attention_mask"].to(device)))
                    .cpu()
                    .numpy()
                )
        f1 = f1_score(yva, np.vstack(preds).argmax(1), average="macro", zero_division=0)
        history.append(dict(epoch=ep, loss=total / n, val_macro_f1=float(f1)))
        print(f"epoch {ep}: loss={total/n:.4f}  val_macroF1={f1:.4f}")

        if f1 > best["f1"]:
            best = dict(f1=float(f1), epoch=ep)
            name = args.name or f"student-h{args.hidden}-l{args.layers}"
            dst = ROOT / "models" / "encoders" / name
            dst.mkdir(parents=True, exist_ok=True)
            enc.save_pretrained(dst)
            tok.save_pretrained(dst)

    name = args.name or f"student-h{args.hidden}-l{args.layers}"
    dst = ROOT / "models" / "encoders" / name
    (dst / "distill.json").write_text(
        json.dumps(
            dict(
                teacher=args.teacher,
                best_epoch=best["epoch"],
                val_macro_f1=best["f1"],
                params=int(n_params),
                int8_mb_estimate=round(n_params / 1e6, 2),
                args=vars(args),
                history=history,
            ),
            indent=2,
        )
    )

    print(f"\nkept epoch {best['epoch']} (val macro-F1 {best['f1']:.4f})")
    print(f"saved -> {dst}   ~{n_params/1e6:.2f} MB INT8\n")
    print(
        "The head trained here is discarded. Re-fit it through the normal "
        "path so calibration, the OOD scorer and every threshold are derived "
        "the same way they were for the teacher — otherwise you are comparing "
        "two pipelines, not two encoders:"
    )
    print(
        f"  python scripts/train_classifier.py --encoder {name} "
        f"--classifier mlp --train {args.train} --out models/final_student"
    )
    print("  python scripts/evaluate_onnx.py --model models/final_student ...")
    print(
        "\nThe number that decides whether this ships is accepted precision, "
        "not macro-F1. A student can lose a point of accuracy and still keep "
        "the gate's promise, and that trade is usually worth 30 MB."
    )


if __name__ == "__main__":
    main()
