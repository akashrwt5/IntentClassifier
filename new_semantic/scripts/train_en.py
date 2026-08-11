#!/usr/bin/env python3
"""
Train the English tiny semantic student.

    teacher (E5, offline)  ──embeddings──┐
                                         ├─► student (0.75 MB, on-device)
    train.csv (labels) ──────────────────┘

Loss = CE_WEIGHT * weighted_CE(labels)  +  KD_WEIGHT * KL(student ‖ teacher, T)

Class weights are ON by default: train.csv is deliberately uncapped, so 11
intents have ~1,850 rows while 23 have under 50. Without weights the model
simply learns to ignore the small classes.

Requires (NOT repo runtime deps — training only):
    pip install torch sentence-transformers scikit-learn

Usage:
    python scripts/train_en.py
    python scripts/train_en.py --no-class-weights     # ablation
    python scripts/train_en.py --no-distill           # CE only, teacher-free
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.common import (  # noqa: E402
    assert_no_leak,
    build_subword_vocab,
    build_vocab,
    class_weights,
    encode,
    load_rows,
    save_vocab,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_student(
    vocab_size: int, n_classes: int, init_matrix=None, freeze=False, dim: int | None = None
):
    """`dim` must come from the artifact, not from config.

    build_semantic_vocab.py can emit any --dim; if this defaulted to
    config.EMBED_DIM the shapes would silently disagree and torch would raise
    deep inside __init__ with a bare size mismatch. Callers that load a
    checkpoint must pass the dim recorded in its train summary.
    """
    import torch
    from torch import nn

    d = dim if dim is not None else config.EMBED_DIM
    if init_matrix is not None and init_matrix.shape[1] != d:
        raise SystemExit(
            f"embedding init is {init_matrix.shape[1]}-d but the student is being "
            f"built {d}-d. Pass dim=<matrix width>, or rebuild the init with "
            f"--dim {d}."
        )
    if d % config.NHEAD:
        raise SystemExit(f"EMBED_DIM {d} must be divisible by NHEAD {config.NHEAD}")

    class TinyIntentClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d, padding_idx=config.PAD_ID)
            if init_matrix is not None:
                with torch.no_grad():
                    self.embedding.weight.copy_(torch.tensor(init_matrix))
                self.embedding.weight.requires_grad = not freeze
            layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=config.NHEAD,
                dim_feedforward=config.FF_DIM,
                dropout=config.DROPOUT,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.NUM_LAYERS)
            self.norm = nn.LayerNorm(d)
            self.classifier = nn.Linear(d, n_classes)

        def forward(self, ids, mask):
            x = self.embedding(ids)
            x = self.encoder(x, src_key_padding_mask=~mask)
            m = mask.unsqueeze(-1).float()
            pooled = (x * m).sum(1) / m.sum(1).clamp(min=1e-6)
            return self.classifier(self.norm(pooled))

    return TinyIntentClassifier()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--no-distill", action="store_true")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--tag", default="v1")
    ap.add_argument(
        "--tokenizer",
        choices=["word", "subword"],
        default="word",
        help="'subword' removes [UNK] entirely: unseen words split into known "
        "pieces ('quieter' -> quiet+##er) while junk shatters into many rare "
        "pieces. That is the signal the word-level model never had.",
    )
    ap.add_argument("--vocab-size", type=int, default=3000, help="subword mode only")
    ap.add_argument("--max-len", type=int, default=0, help="0 = auto (24 for word, 32 for subword)")
    ap.add_argument(
        "--init-embeddings",
        type=Path,
        default=None,
        help="npz from build_semantic_vocab.py. Replaces the corpus-only "
        "vocabulary with a larger one whose embeddings START in the teacher's "
        "semantic space, so a word never seen in training ('elevate') still "
        "lands near its synonyms instead of collapsing to [UNK].",
    )
    ap.add_argument(
        "--teacher",
        default=None,
        help="encoder for the KD soft targets. Default: whatever built the "
        "--init-embeddings file, else config.TEACHER. There are TWO teachers in "
        "this pipeline — the one that seeds the embedding table and the one that "
        "supplies soft targets — and a real teacher swap has to change BOTH. "
        "Leaving them mismatched silently produces a run that is not the "
        "experiment you think it is.",
    )
    ap.add_argument(
        "--freeze-embeddings",
        action="store_true",
        help="keep the teacher-derived vectors fixed. Prevents 24k in-domain "
        "sentences from overwriting the general semantics that are the whole "
        "point of loading them.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="run the SAME config on 2-3 seeds before trusting any ranking: "
        "differences between v1-v5 are 1-3 points and run-to-run noise has "
        "never been measured.",
    )

    # ---- v2 knobs: fix OOD rejection --------------------------------
    ap.add_argument(
        "--fallback-weight-floor",
        type=float,
        default=0.0,
        help="minimum class weight for Default Fallback Intent. It is only 4%% "
        "of train.csv now, so inverse-frequency DOWN-weights it to ~0.44 — "
        "backwards for the class we care most about. Try 3.0.",
    )
    ap.add_argument(
        "--fallback-oversample",
        type=int,
        default=1,
        help="repeat each real fallback row N times (no synthetic text)",
    )
    ap.add_argument(
        "--unk-aug",
        type=float,
        default=0.0,
        help="fraction of EXTRA synthetic fallback rows built by HEAVILY "
        "corrupting sentences (>=70%% of tokens -> [UNK]). Teaches "
        "'mostly-unknown input -> fallback'. Creates SYNTHETIC text.",
    )
    ap.add_argument(
        "--unk-robust",
        type=float,
        default=0.0,
        help="fraction of EXTRA in-scope rows with only 1-2 tokens corrupted, "
        "keeping their TRUE label. This is the counter-example --unk-aug needs: "
        "without it the model learns 'any UNK -> fallback' and rejects real "
        "commands that merely contain an unfamiliar word (stress test collapses). "
        "Use TOGETHER with --unk-aug, roughly 1:1.",
    )
    ap.add_argument("--train-csv", type=Path, default=None, help="override training CSV path")
    args = ap.parse_args()

    import torch
    from sklearn.model_selection import train_test_split
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    seed_all(args.seed)
    config.MODELS.mkdir(parents=True, exist_ok=True)
    config.REPORTS.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- data
    train_path = args.train_csv if args.train_csv else config.TRAIN_CSV
    rows = load_rows(train_path)
    texts = [t for t, _ in rows]
    labels = [l for _, l in rows]
    print(f"train CSV       : {train_path.name}")
    print(f"train rows      : {len(rows)}")

    # hard guard — never train on anything that is also in an eval set
    for path, name in (
        (config.LOCKED_TEST, "locked test"),
        (config.STRESS_TEST, "stress test"),
        (config.OOD_TEST, "OOD test"),
    ):
        if path.exists():
            assert_no_leak(texts, [t for t, _ in load_rows(path)], name)
    print("leak guard      : OK (locked / stress / OOD all disjoint)")

    label_list = sorted(set(labels))
    l2i = {l: i for i, l in enumerate(label_list)}
    print(f"intents         : {len(label_list)}")

    max_len = args.max_len or (32 if args.tokenizer == "subword" else config.MAX_LEN)

    # vocab is built BEFORE augmentation so synthetic rows cannot add tokens
    init_matrix = None
    embed_teacher = None
    if args.init_embeddings:
        blob = np.load(args.init_embeddings, allow_pickle=False)
        vocab = json.loads(str(blob["vocab"]))
        init_matrix = blob["matrix"]
        meta_e = json.loads(str(blob["meta"]))
        embed_teacher = meta_e.get("teacher")
        print(
            f"embeddings      : {embed_teacher}, {meta_e['corpus_words']} corpus "
            f"+ {meta_e['general_words']} general words"
        )
    elif args.tokenizer == "subword":
        vocab = build_subword_vocab(texts, size=args.vocab_size)
    else:
        vocab = build_vocab(texts)
    print(f"tokenizer       : {args.tokenizer}  (max_len {max_len})")
    print(f"vocab           : {len(vocab)} tokens")

    # KD teacher: explicit flag > whatever seeded the embeddings > config
    teacher = args.teacher or embed_teacher or config.TEACHER
    if embed_teacher and teacher != embed_teacher:
        print(
            f"! KD teacher {teacher} != embedding teacher {embed_teacher} — "
            f"that is two different semantic spaces in one model"
        )
    print(f"KD teacher      : {teacher}")

    n_orig = len(texts)
    fb_share_before = labels.count(config.FALLBACK_INTENT) / len(labels)

    # ---- oversample real fallback rows (no synthetic text) ----------
    if args.fallback_oversample > 1:
        fb_rows = [t for t, l in zip(texts, labels) if l == config.FALLBACK_INTENT]
        extra = fb_rows * (args.fallback_oversample - 1)
        texts = texts + extra
        labels = labels + [config.FALLBACK_INTENT] * len(extra)
        print(f"fallback x{args.fallback_oversample}    : +{len(extra)} rows (real, repeated)")

    # ---- synthetic UNK augmentation ---------------------------------
    # Distinct filler tokens, not one repeated string: a single repeated token
    # is a degenerate pattern the model can memorise instead of learning
    # "this word is unknown".
    rng = random.Random(args.seed)
    fillers = [f"zzq{i}unk" for i in range(400)]

    def corrupt(text: str, lo: float, hi: float) -> str | None:
        toks = text.split()
        if not toks:
            return None
        k = max(1, round(len(toks) * rng.uniform(lo, hi)))
        pos = set(rng.sample(range(len(toks)), min(k, len(toks))))
        return " ".join(rng.choice(fillers) if i in pos else w for i, w in enumerate(toks))

    in_scope = [(t, l) for t, l in zip(texts, labels) if l != config.FALLBACK_INTENT]

    n_synth = 0
    if args.unk_aug > 0:
        # HEAVY corruption -> fallback. 70-100% so it cannot be mistaken for a
        # real command carrying one unfamiliar word.
        made = []
        for _ in range(int(len(in_scope) * args.unk_aug)):
            src, _ = rng.choice(in_scope)
            c = corrupt(src, 0.7, 1.0)
            if c:
                made.append(c)
        texts += made
        labels += [config.FALLBACK_INTENT] * len(made)
        n_synth += len(made)
        print(f"unk-aug         : +{len(made)} synthetic FALLBACK rows (70-100% corrupt)")

    n_robust = 0
    if args.unk_robust > 0:
        # LIGHT corruption -> keep the TRUE label. Teaches that an in-scope
        # command with one unfamiliar word is still that command.
        made_t, made_l = [], []
        for _ in range(int(len(in_scope) * args.unk_robust)):
            src, lab = rng.choice(in_scope)
            toks = src.split()
            if len(toks) < 3:
                continue
            k = 1 if len(toks) < 6 else rng.choice([1, 2])
            pos = set(rng.sample(range(len(toks)), k))
            made_t.append(
                " ".join(rng.choice(fillers) if i in pos else w for i, w in enumerate(toks))
            )
            made_l.append(lab)
        texts += made_t
        labels += made_l
        n_robust = len(made_t)
        n_synth += n_robust
        print(f"unk-robust      : +{n_robust} synthetic IN-SCOPE rows (1-2 tokens corrupt)")

    fb_share_after = labels.count(config.FALLBACK_INTENT) / len(labels)
    if len(texts) != n_orig:
        print(f"fallback share  : {fb_share_before * 100:.1f}% -> {fb_share_after * 100:.1f}%")

    y = np.array([l2i[l] for l in labels])

    X = np.array([encode(t, vocab, max_len, args.tokenizer)[0] for t in texts], dtype=np.int64)
    M = X != config.PAD_ID

    Xtr, Xva, Mtr, Mva, ytr, yva, ttr, tva = train_test_split(
        X, M, y, texts, test_size=config.VAL_SIZE, random_state=args.seed, stratify=y
    )
    print(f"split           : train {len(ytr)} / val {len(yva)}")

    # ---------------------------------------------------------- teacher
    teacher_logits = None
    if not args.no_distill:
        from sentence_transformers import SentenceTransformer

        t0 = time.time()
        enc = SentenceTransformer(teacher)
        emb_tr = enc.encode(ttr, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
        # class prototypes in teacher space -> soft targets
        proto = np.stack([emb_tr[ytr == i].mean(0) for i in range(len(label_list))])
        proto /= np.linalg.norm(proto, axis=1, keepdims=True)
        teacher_logits = (emb_tr @ proto.T) * 10.0  # scaled cosine
        print(f"teacher embeds  : {time.time() - t0:.1f}s")

    # ---------------------------------------------------------- weights
    w = None
    if config.USE_CLASS_WEIGHTS and not args.no_class_weights:
        cw = class_weights(labels, label_list)
        if args.fallback_weight_floor > 0:
            old = cw[config.FALLBACK_INTENT]
            cw[config.FALLBACK_INTENT] = max(old, args.fallback_weight_floor)
            print(f"fallback weight : {old:.3f} -> {cw[config.FALLBACK_INTENT]:.3f} (floored)")
        w = torch.tensor([cw[l] for l in label_list], dtype=torch.float32)
        print(f"class weights   : ON  (min {w.min():.3f} / max {w.max():.3f})")
    else:
        print("class weights   : OFF")

    # ---------------------------------------------------------- train
    # dim comes from the artifact, never from config: build_semantic_vocab.py
    # can emit any --dim and config.EMBED_DIM would silently disagree.
    embed_dim = init_matrix.shape[1] if init_matrix is not None else config.EMBED_DIM
    if embed_dim != config.EMBED_DIM:
        print(
            f"embed dim       : {embed_dim} (from the init matrix; "
            f"config.EMBED_DIM is {config.EMBED_DIM})"
        )
    model = build_student(
        len(vocab), len(label_list), init_matrix, args.freeze_embeddings, embed_dim
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"student params  : {n_params:,}  (~{n_params * 4 / 1e6:.2f} MB fp32)")

    opt = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss(weight=w)

    tensors = [torch.tensor(Xtr), torch.tensor(Mtr), torch.tensor(ytr)]
    if teacher_logits is not None:
        tensors.append(torch.tensor(teacher_logits, dtype=torch.float32))
    loader = DataLoader(TensorDataset(*tensors), batch_size=config.BATCH_SIZE, shuffle=True)

    Xva_t, Mva_t, yva_t = torch.tensor(Xva), torch.tensor(Mva), torch.tensor(yva)

    best_acc, best_epoch, bad = 0.0, -1, 0
    ckpt = config.MODELS / f"student_{args.tag}.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            ids, mask, yb = batch[0], batch[1], batch[2]
            opt.zero_grad()
            logits = model(ids, mask)
            loss = ce(logits, yb)
            if teacher_logits is not None:
                tl = batch[3]
                T = config.TEMPERATURE
                kd = nn.functional.kl_div(
                    nn.functional.log_softmax(logits / T, dim=-1),
                    nn.functional.softmax(tl / T, dim=-1),
                    reduction="batchmean",
                ) * (T * T)
                loss = config.CE_WEIGHT * loss + config.KD_WEIGHT * kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())

        model.eval()
        with torch.no_grad():
            pred = model(Xva_t, Mva_t).argmax(-1).numpy()
        acc = float((pred == yva).mean())
        # macro recall — the metric that actually exposes ignored small classes
        macro = float(
            np.mean(
                [(pred[yva == i] == i).mean() for i in range(len(label_list)) if (yva == i).any()]
            )
        )
        history.append(
            {"epoch": epoch, "loss": total / len(loader), "val_acc": acc, "val_macro_recall": macro}
        )
        print(
            f"  ep{epoch:>3}  loss={total / len(loader):.4f}  val_acc={acc:.4f}  macro_recall={macro:.4f}"
        )

        if macro > best_acc:
            best_acc, best_epoch, bad = macro, epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= config.PATIENCE:
                print(f"  early stop (patience {config.PATIENCE})")
                break

    save_vocab(vocab, config.MODELS / f"vocab_{args.tag}.json", args.tokenizer)
    (config.MODELS / f"labels_{args.tag}.json").write_text(
        json.dumps(label_list, indent=2), encoding="utf-8"
    )

    summary = {
        "tag": args.tag,
        "seed": args.seed,
        "train_csv": str(config.TRAIN_CSV),
        "train_rows": len(rows),
        "intents": len(label_list),
        "vocab_size": len(vocab),
        "embed_dim": embed_dim,
        "init_embeddings": str(args.init_embeddings) if args.init_embeddings else None,
        "freeze_embeddings": args.freeze_embeddings,
        "tokenizer": args.tokenizer,
        "max_len": max_len,
        "student_params": n_params,
        "student_mb_fp32": round(n_params * 4 / 1e6, 3),
        "teacher": None if args.no_distill else teacher,
        "embedding_teacher": embed_teacher,
        "class_weights": bool(w is not None),
        "fallback_weight_floor": args.fallback_weight_floor or None,
        "fallback_oversample": args.fallback_oversample,
        "unk_aug_fraction": args.unk_aug or None,
        "unk_robust_fraction": args.unk_robust or None,
        "synthetic_fallback_rows": n_synth - n_robust,
        "synthetic_in_scope_rows": n_robust,
        "synthetic_rows": n_synth,
        "synthetic_text": n_synth > 0,
        "fallback_share_before": round(fb_share_before, 4),
        "fallback_share_after": round(fb_share_after, 4),
        "rows_after_augmentation": len(texts),
        "best_epoch": best_epoch,
        "best_val_macro_recall": round(best_acc, 4),
        "checkpoint": str(ckpt),
        "leak_guard": "passed",
        "history": history,
    }
    out = config.REPORTS / f"train_{args.tag}_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nbest epoch {best_epoch}  val macro-recall {best_acc:.4f}")
    print(f"wrote {ckpt}")
    print(f"wrote {out}")
    print("\nNext:  python scripts/evaluate.py --tag " + args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
