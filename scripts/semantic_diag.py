#!/usr/bin/env python3
"""
semantic_diag.py — diagnose antonym/paraphrase failures and compare
TF-IDF (Stage 2) vs MiniLM semantic head (Stage 3) on the same dataset.

Answers the question: "why does 'it's too quiet, make it louder' predict
volume.decrease?" and "does a semantic head actually fix it?"

Usage:
    python scripts/semantic_diag.py --data balanced_intents_final.xlsx
    python scripts/semantic_diag.py --data datasets/train.csv --skip-semantic

Reads (for the semantic stage):
    models/minilm-l6-v2.onnx
    models/minilm-vocab.txt

Writes nothing by default. --save-head writes models/semantic_head_diag.npz
so you can diff it against the shipped head without clobbering it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

# Lexical polarity probes. The bug is that a word can appear on BOTH sides:
# "loud" shows up in decrease phrases ("it's too loud, turn it down") far more
# often than in increase phrases, so bag-of-words learns loud -> decrease.
POLARITY_PROBES = [
    "loud",
    "louder",
    "quiet",
    "quieter",
    "soft",
    "softer",
    "up",
    "down",
    "low",
    "high",
    "raise",
    "lower",
]

# Hard paraphrases: phrasings deliberately outside the training vocabulary.
# Extend this list — it is the only honest measure of generalisation.
HARD_SET: List[Tuple[str, str]] = [
    ("it's too loudy here can you make it quiter", "device.volume.decrease"),
    ("it's too quiter here can you make it louder", "device.volume.increase"),
    ("this restaurant is deafening, tone it down a notch", "device.volume.decrease"),
    ("i'm struggling to follow the conversation, give me more", "device.volume.increase"),
    ("my ears are ringing from how strong this is", "device.volume.decrease"),
    ("everything is a whisper, i need more of it", "device.volume.increase"),
    ("crank em", "device.volume.increase"),
    ("dial it back please", "device.volume.decrease"),
    ("i need total silence for a moment", "device.volume.mute"),
    ("bring the sound back i want to hear again", "device.volume.unmute"),
    ("kill the audio completely", "device.volume.mute"),
    ("stop being silent, i want audio again", "device.volume.unmute"),
    ("i am going to a noisy cafe, adjust my setup for that", "device.memory.change"),
    ("flip me over to the outdoor configuration", "device.memory.change"),
    ("don't let me forget to take my pills at nine", "reminders.task.create"),
    ("i already took my medication, cross that off", "reminders.task.complete"),
    ("what have i got lined up for today", "help.reminder.show"),
    ("read out my pending to dos", "help.reminder.show"),
    ("pipe my music straight into my ears", "streaming.session.start"),
    ("cut the bluetooth feed to my aids", "streaming.session.stop"),
    ("i misplaced my handset, help me track it", "find.phone.locate"),
    ("where did i leave my mobile", "find.phone.locate"),
    ("make it less aggressive on my ears", "device.volume.decrease"),
    ("boost it a couple notches", "device.volume.increase"),
    ("i cannot make out a single word she is saying", "device.volume.increase"),
    ("this is blasting way beyond comfortable", "device.volume.decrease"),
]


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV, or a multi-sheet XLSX where each sheet is one intent."""
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        xl = pd.ExcelFile(path)
        frames = []
        for sheet in xl.sheet_names:
            d = xl.parse(sheet)
            d.columns = [str(c).strip().lower() for c in d.columns]
            if "intent" not in d.columns:
                d["intent"] = sheet
            frames.append(d[["text", "intent"]])
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path)
        df.columns = [str(c).strip().lower() for c in df.columns]

    df["text"] = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].drop_duplicates(subset=["text", "intent"])
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------
# Stage A — lexical polarity audit (no model needed)
# ----------------------------------------------------------------------
def polarity_audit(df: pd.DataFrame) -> None:
    print("\n=== 1. Lexical polarity audit ===")
    print("A word listed under an intent it does NOT semantically belong to is")
    print("a bag-of-words trap: the classifier will follow the count, not the sense.\n")
    lower = df["text"].str.lower()
    for word in POLARITY_PROBES:
        hit = df[lower.str.contains(rf"\b{word}\b", regex=True)]
        if hit.empty:
            continue
        counts = hit["intent"].value_counts()
        top = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in counts.head(4).items())
        purity = counts.iloc[0] / counts.sum()
        flag = "  <-- ambiguous" if purity < 0.85 else ""
        print(f"  {word:9s} {top}{flag}")

    # Contrastive coverage: how many phrases carry a state word AND an action word?
    up = r"\b(louder|increase|raise|boost|turn up|amplify)\b"
    down = r"\b(quieter|softer|decrease|reduce|lower|turn down)\b"
    both = lower.str.contains(up, regex=True) & lower.str.contains(down, regex=True)
    print(
        f"\n  contrastive phrases (state + opposite action): {both.sum()} / {len(df)} "
        f"({both.mean():.1%})"
    )
    if both.mean() < 0.05:
        print("  -> too low. Augment with 'it's too X, make it Y' templates.")


# ----------------------------------------------------------------------
# Stage B — TF-IDF baseline
# ----------------------------------------------------------------------
def fit_tfidf(df: pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=15.0)),
        ]
    )
    pipe.fit(df["text"], df["intent"])
    return pipe


# ----------------------------------------------------------------------
# Stage C — MiniLM embeddings + linear head
# ----------------------------------------------------------------------
class MiniLMEmbedder:
    """Mean-pooled, L2-normalised MiniLM sentence embeddings via ONNX Runtime."""

    def __init__(self, onnx_path: Path, vocab_path: Path, max_len: int = 128):
        import onnxruntime as ort
        from tokenizers import BertWordPieceTokenizer

        self.tok = BertWordPieceTokenizer(str(vocab_path), lowercase=True)
        self.tok.enable_truncation(max_len)
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def __call__(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = [self.tok.encode(t) for t in batch]
            width = max(len(e.ids) for e in enc)
            ids = np.zeros((len(batch), width), np.int64)
            mask = np.zeros((len(batch), width), np.int64)
            for j, e in enumerate(enc):
                ids[j, : len(e.ids)] = e.ids
                mask[j, : len(e.ids)] = 1
            hidden = self.sess.run(
                None,
                {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)},
            )[0]
            m = mask[..., None].astype(np.float32)
            vec = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9
            out.append(vec.astype(np.float32))
        return np.vstack(out)


def fit_semantic_head(embed, df: pd.DataFrame):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X = embed(df["text"].tolist())
    x_tr, x_te, y_tr, y_te = train_test_split(
        X, df["intent"], test_size=0.15, stratify=df["intent"], random_state=0
    )
    head = LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced")
    head.fit(x_tr, y_tr)
    print(f"  semantic head in-distribution accuracy: {head.score(x_te, y_te):.4f}")
    return head


# ----------------------------------------------------------------------
# Stage D — hard paraphrase comparison
# ----------------------------------------------------------------------
def compare(tfidf, head, embed, hard: List[Tuple[str, str]]) -> None:
    texts = [t for t, _ in hard]
    gold = np.array([g for _, g in hard])

    p_tf = tfidf.predict_proba(texts)
    classes = tfidf.classes_

    print("\n=== 3. Hard paraphrase set (out-of-vocabulary phrasings) ===")
    if head is None:
        pred = classes[p_tf.argmax(1)]
        print(f"  TF-IDF: {(pred == gold).sum()}/{len(gold)}")
        for t, g, p, c in zip(texts, gold, pred, p_tf.max(1)):
            mark = " " if p == g else "x"
            print(f"  {mark} {t[:46]:48s} {p:24s} {c:.2f}")
        return

    p_sem = head.predict_proba(embed(texts))
    if list(head.classes_) != list(classes):
        order = [list(head.classes_).index(c) for c in classes]
        p_sem = p_sem[:, order]

    pred_tf = classes[p_tf.argmax(1)]
    pred_sem = classes[p_sem.argmax(1)]
    print(f"  TF-IDF alone   : {(pred_tf == gold).sum()}/{len(gold)}")
    print(f"  Semantic alone : {(pred_sem == gold).sum()}/{len(gold)}")
    for thr in (0.5, 0.6, 0.7):
        hyb = np.where(p_tf.max(1) < thr, pred_sem, pred_tf)
        print(f"  Hybrid rescue@{thr}: {(hyb == gold).sum()}/{len(gold)}")
    for w in (0.3, 0.5, 0.7):
        blend = classes[(w * p_sem + (1 - w) * p_tf).argmax(1)]
        print(f"  Blend w_sem={w} : {(blend == gold).sum()}/{len(gold)}")

    ct, cs = p_tf.max(1), p_sem.max(1)
    print(
        f"\n  mean confidence when correct -> TF-IDF {ct[pred_tf == gold].mean():.2f}"
        f" | semantic {cs[pred_sem == gold].mean():.2f}"
    )
    print("  (semantic's value is calibrated confidence on unseen phrasings,")
    print("   which is what keeps a correct answer from being dropped by the gate)")

    print(f"\n  {'text':48s} {'gold':24s} {'tfidf':26s} semantic")
    for i, (t, g) in enumerate(zip(texts, gold)):
        m1 = " " if pred_tf[i] == g else "x"
        m2 = " " if pred_sem[i] == g else "x"
        print(
            f"  {t[:46]:48s} {g:24s} {m1}{pred_tf[i]:22s}{ct[i]:.2f} "
            f"{m2}{pred_sem[i]:22s}{cs[i]:.2f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--skip-semantic", action="store_true")
    ap.add_argument("--save-head", action="store_true")
    args = ap.parse_args()

    df = load_dataset(args.data)
    print(f"loaded {len(df)} phrases, {df['intent'].nunique()} intents")
    print(df["intent"].value_counts().to_string())

    polarity_audit(df)

    print("\n=== 2. Fitting models ===")
    tfidf = fit_tfidf(df)

    head = embed = None
    if not args.skip_semantic:
        onnx_path = MODEL_DIR / "minilm-l6-v2.onnx"
        vocab_path = MODEL_DIR / "minilm-vocab.txt"
        if not onnx_path.exists():
            print(f"  {onnx_path} missing — run scripts/download_minilm.py; skipping.")
        else:
            embed = MiniLMEmbedder(onnx_path, vocab_path)
            head = fit_semantic_head(embed, df)
            if args.save_head:
                out = MODEL_DIR / "semantic_head_diag.npz"
                np.savez(
                    out,
                    weights=head.coef_.astype(np.float32),
                    bias=head.intercept_.astype(np.float32),
                    labels=head.classes_,
                    embedder=np.array(["onnx"]),
                )
                print(f"  wrote {out} ({os.path.getsize(out) / 1024:.1f} KB)")

    compare(tfidf, head, embed, HARD_SET)


if __name__ == "__main__":
    main()
