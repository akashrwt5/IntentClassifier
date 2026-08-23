"""Fine-tune the encoder end to end — the real fix for corrective negation.

Why this and not more data: on intent pairs explicitly taught the corrective
frame the model scores 0.74; on a held-out control group of three families it
scores 0.48, which is chance. It learned the pairs, not the rule. There are
thousands of possible pairs across 57 intents, so teaching them one at a time
cannot close that gap.

The reason it cannot is structural. "not edge mode, i meant mask mode" and
"not mask mode, i meant edge mode" contain identical tokens; only the ORDER
differs, and word order is resolved inside the transformer's attention layers.
Those layers are frozen. A classifier head sitting on top of a fixed embedding
can only reweight what the encoder already separated — if the encoder maps both
sentences to nearly the same point, no head can pull them apart.

So the encoder has to move. This trains it jointly with the head.

    python scripts/finetune_encoder.py --encoder bge-small-en-v1.5 --epochs 3

On an M-series Mac this uses MPS and takes a few minutes. The output is a normal
encoder directory, so everything downstream is unchanged:

    python scripts/train_classifier.py --encoder bge-small-en-v1.5-ft ...

Guard rails, because fine-tuning can quietly destroy more than it fixes:
  * validation macro-F1 is checked every epoch and the best epoch is kept
  * the corrective control group is reported per epoch, so you can see whether
    the thing this exists for is actually moving
  * a low learning rate with warmup — the encoder is already good, the goal is
    to bend it, not retrain it
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402
from encoders import ROOT, discover_local_encoders  # noqa: E402
from pipeline import DATA  # noqa: E402


def pick_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Net:
    """Encoder + mean pooling + L2 norm + linear head, trained together."""

    def __init__(self, path: Path, n_classes: int, device: str, dropout: float):
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.encoder = AutoModel.from_pretrained(str(path), local_files_only=True)
        dim = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_classes))
        self.encoder.to(device)
        self.head.to(device)
        self.device = device

    def forward(self, input_ids, attention_mask):
        import torch
        h = self.encoder(input_ids=input_ids,
                         attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        emb = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        emb = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-9)
        return self.head(emb)

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.head.parameters())

    def train(self, flag=True):
        self.encoder.train(flag); self.head.train(flag)


def batches(texts, labels, tok, device, bs, max_len, shuffle, rng=None):
    import torch
    idx = np.arange(len(texts))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        sel = idx[i : i + bs]
        enc = tok([texts[j] for j in sel], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt")
        yield (enc["input_ids"].to(device), enc["attention_mask"].to(device),
               torch.tensor([labels[j] for j in sel], device=device))


def evaluate(net, texts, labels, tok, device, bs, max_len) -> np.ndarray:
    import torch
    net.train(False)
    out = []
    with torch.no_grad():
        for ids, mask, _ in batches(texts, labels, tok, device, bs, max_len, False):
            out.append(net.forward(ids, mask).cpu().numpy())
    net.train(True)
    return np.vstack(out)


def main() -> None:
    import torch
    from sklearn.metrics import f1_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="bge-small-en-v1.5")
    ap.add_argument("--train", default="train_augmented")
    ap.add_argument("--epochs", type=int, default=12,
                    help="Was 3, which was far too few and looked finished. "
                         "A 4-epoch run ended at val macro-F1 0.7439 while the "
                         "frozen-encoder pipeline reaches 0.90, with loss still "
                         "falling (2.87 -> 0.79) and the best epoch being the "
                         "LAST one. OneCycle anneals the learning rate to zero, "
                         "so a short schedule completes tidily and hides that "
                         "the model never got there — the same trap "
                         "distill_student.py documents.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="-ft")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)

    local = discover_local_encoders()
    if args.encoder not in local:
        raise SystemExit(f"encoder '{args.encoder}' not in models/encoders/: "
                         f"{list(local)}")
    src = local[args.encoder]

    train = pd.read_csv(DATA / f"{args.train}.csv")
    val = pd.read_csv(DATA / "validation.csv")
    labels = sorted(train["intent"].unique())
    idx = {l: i for i, l in enumerate(labels)}
    ytr = [idx[l] for l in train["intent"]]
    yva = [idx[l] for l in val["intent"]]

    net = Net(src, len(labels), device, args.dropout)
    prefix = "query: " if "e5" in args.encoder.lower() else ""
    Xtr = [prefix + t for t in train["text"].astype(str)]
    Xva = [prefix + t for t in val["text"].astype(str)]

    # The control group: corrective rows for families never taught the frame.
    hn = pd.read_csv(DATA / "hard_negative_test.csv")
    ctl = hn[hn["reason"] == "corrective_HELD_OUT_family"]
    Xctl = [prefix + t for t in ctl["text"].astype(str)]
    yctl = [idx.get(l, -1) for l in ctl["intent"]]

    # The lexical-association arms, tracked per epoch. This is the failure that
    # sent us here: on the frozen encoder, arm A scored 1.000 and arm B 0.389 on
    # the SAME request in the SAME frame, differing only in which describing
    # word was used. If fine-tuning is doing what it is supposed to do — moving
    # structure into the encoder rather than reweighting a fixed embedding —
    # the A-B gap is the number that closes. Watching only val macro-F1 would
    # hide it: F1 barely moved across every earlier attempt while this gap
    # stayed at 0.6.
    import structure_probe as sp
    arm_rows = sp.build_conflict()
    arms = {}
    for a in sp.ARMS:
        sub = [r for r in arm_rows if r["arm"] == a]
        arms[a] = ([prefix + r["text"] for r in sub],
                   [idx.get(r["intent"], -1) for r in sub])

    def arm_scores(net) -> dict:
        out = {}
        for a, (xs, ys) in arms.items():
            p = evaluate(net, xs, ys, net.tok, device, 64, args.max_len)
            out[a] = float((p.argmax(1) == np.array(ys)).mean())
        out["gap_up"] = out["A_up_symptom_word"] - out["B_up_request_word"]
        out["gap_down"] = out["C_down_symptom_word"] - out["D_down_request_word"]
        return out

    steps = args.epochs * int(np.ceil(len(Xtr) / args.batch_size))
    opt = torch.optim.AdamW([
        {"params": net.encoder.parameters(), "lr": args.lr},
        {"params": net.head.parameters(), "lr": args.head_lr},
    ], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr, args.head_lr], total_steps=steps,
        pct_start=args.warmup_frac, anneal_strategy="linear")
    lossf = torch.nn.CrossEntropyLoss()

    print(f"device={device}  {len(Xtr)} train rows  {len(labels)} classes  "
          f"{steps} steps")
    print(f"control group: {len(Xctl)} corrective rows from families never "
          f"taught the frame\n")

    best = dict(f1=-1.0, epoch=-1)
    history = []
    for ep in range(args.epochs):
        net.train(True)
        total, n = 0.0, 0
        for ids, mask, y in batches(Xtr, ytr, net.tok, device,
                                    args.batch_size, args.max_len, True, rng):
            opt.zero_grad()
            loss = lossf(net.forward(ids, mask), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            total += loss.detach().item() * len(y); n += len(y)

        pv = evaluate(net, Xva, yva, net.tok, device, 64, args.max_len)
        f1 = f1_score(yva, pv.argmax(1), average="macro", zero_division=0)
        pc = evaluate(net, Xctl, yctl, net.tok, device, 64, args.max_len)
        ctl_acc = float((pc.argmax(1) == np.array(yctl)).mean())
        a = arm_scores(net)
        history.append(dict(epoch=ep, loss=total / n, val_macro_f1=float(f1),
                            corrective_held_out=ctl_acc, **a))
        print(f"epoch {ep}: loss={total/n:.4f}  val_macroF1={f1:.4f}  "
              f"corrective_held_out={ctl_acc:.4f}")
        print(f"          lexical gap: up {a['gap_up']:+.3f} "
              f"(A {a['A_up_symptom_word']:.3f} / B {a['B_up_request_word']:.3f})   "
              f"down {a['gap_down']:+.3f} "
              f"(C {a['C_down_symptom_word']:.3f} / D {a['D_down_request_word']:.3f})")

        if f1 > best["f1"]:
            best = dict(f1=float(f1), epoch=ep, ctl=ctl_acc,
                        gap_up=a["gap_up"], gap_down=a["gap_down"])
            dst = ROOT / "models" / "encoders" / f"{args.encoder}{args.suffix}"
            dst.mkdir(parents=True, exist_ok=True)
            net.encoder.save_pretrained(dst)
            net.tok.save_pretrained(dst)

    dst = ROOT / "models" / "encoders" / f"{args.encoder}{args.suffix}"
    (dst / "finetune.json").write_text(json.dumps(
        dict(source=args.encoder, best_epoch=best["epoch"],
             val_macro_f1=best["f1"],
             corrective_held_out_at_best=best.get("ctl"),
             args=vars(args), history=history), indent=2))

    print(f"\nkept epoch {best['epoch']} (val macro-F1 {best['f1']:.4f}, "
          f"corrective held-out {best.get('ctl'):.4f}, "
          f"lexical gap up {best.get('gap_up', float('nan')):+.3f} / "
          f"down {best.get('gap_down', float('nan')):+.3f})")
    print("Frozen-encoder baseline for those gaps: up +0.611, down +0.611 "
          "before F16; up +0.333, down +0.389 after. If fine-tuning is working, "
          "this is where it shows.")
    print(f"saved -> {dst}")
    print("\nThe encoder alone is saved; the head trained here is discarded on "
          "purpose. Re-run train_classifier.py against the fine-tuned encoder so "
          "calibration, the OOD scorer and every threshold are fitted the same "
          "way they were for the frozen encoder — otherwise the comparison is "
          "between two different pipelines, not two encoders:")
    print(f"  python scripts/train_classifier.py --encoder "
          f"{args.encoder}{args.suffix} --classifier mlp "
          f"--train {args.train} --out models/final_ft")


if __name__ == "__main__":
    main()
