#!/usr/bin/env python3
"""
Train the multilingual semantic classification head.

Reads multilingual/data/{fr,de,da}.csv (and optionally en.csv), generates
384-dim embeddings using the local INT8 static-shape ONNX encoder, and fits a
LogisticRegression head.  The "Default Fallback Intent" class is injected as an
explicit out-of-scope class so rejection is learned, not threshold-guessed —
the same discipline as the English head in scripts/train_semantic_head.py.

Pipeline:
  1. Load fr + de + da training phrases (and optionally en)
  2. Append a curated OOS class (multilingual/data/oos_multilingual.csv if present)
  3. Stratified 85/15 holdout split → train → per-class accuracy + macro-F1
  4. Retrain on 100% of data → save semantic_head_multilingual.npz

Output: multilingual/SemanticSupport/models/semantic_head_multilingual.npz

Run:
    python scripts/SemanticSupport/train_multilingual_semantic_head.py
    python scripts/SemanticSupport/train_multilingual_semantic_head.py --include-en
    python scripts/SemanticSupport/train_multilingual_semantic_head.py --max-per-intent 200
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR    = Path(__file__).parent.parent.parent
ML_DIR      = BASE_DIR / "multilingual"
DATA_DIR    = ML_DIR / "data"
MODEL_DIR   = ML_DIR / "SemanticSupport" / "models"
ONNX_PATH   = MODEL_DIR / "paraphrase-multilingual-minilm-l12-v2.int8.onnx"
VOCAB_PATH  = MODEL_DIR / "multilingual-vocab.txt"
HEAD_NPZ    = MODEL_DIR / "semantic_head_multilingual.npz"

FALLBACK_INTENT = "Default Fallback Intent"
SEQ_LEN         = 64


# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Train multilingual semantic head")
parser.add_argument("--include-en", action="store_true",
                    help="Also include multilingual/data/en.csv in training")
parser.add_argument("--max-per-intent", type=int, default=250,
                    help="Cap per-class rows to limit near-duplicate spam (default: 250)")
parser.add_argument("--C", type=float, default=3.0,
                    help="LogisticRegression regularisation strength (default: 3.0)")
args = parser.parse_args()


# ── Embedding ─────────────────────────────────────────────────────────────────

sys.path.insert(0, str(BASE_DIR / "multilingual" / "SemanticSupport"))
from tokeniser import MultilingualTokeniser, SEQ_LEN as _SEQ_LEN


def embed_batch(session, tokeniser: MultilingualTokeniser, texts: list) -> np.ndarray:
    """Embed a list of texts using the static-shape ONNX session.

    Each text is tokenised individually (shape (1, 64)) and then stacked so
    the batch runs as a single ONNX call. The static shape constraint means
    every call has the same (N, 64) dimension — no dynamic resizing.
    """
    batch_ids   = []
    batch_mask  = []
    batch_types = []
    for text in texts:
        ids, mask, types = tokeniser.encode(text)
        batch_ids.append(ids[0])    # (64,)
        batch_mask.append(mask[0])  # (64,)
        batch_types.append(types[0])

    input_ids      = np.stack(batch_ids,   axis=0).astype(np.int64)   # (N, 64)
    attention_mask = np.stack(batch_mask,  axis=0).astype(np.int64)   # (N, 64)
    token_type_ids = np.stack(batch_types, axis=0).astype(np.int64)   # (N, 64)

    outputs    = session.run(None, {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    })
    token_emb  = outputs[0]                                             # (N, 64, 384)
    expanded   = attention_mask[:, :, np.newaxis].astype(np.float32)   # (N, 64, 1)
    summed     = (token_emb * expanded).sum(axis=1)                    # (N, 384)
    counts     = expanded.sum(axis=1)                                   # (N, 1)
    vecs       = summed / np.where(counts == 0, 1.0, counts)

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype(np.float32)


def embed_all(texts: list, batch_size: int = 64) -> np.ndarray:
    import onnxruntime as ort
    if not ONNX_PATH.exists():
        print(f"ERROR: {ONNX_PATH} not found.\n"
              "Run `python scripts/SemanticSupport/download_models.py` first.")
        sys.exit(1)
    sess      = ort.InferenceSession(str(ONNX_PATH))
    tokeniser = MultilingualTokeniser(VOCAB_PATH)
    chunks    = []
    t0        = time.time()
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        chunks.append(embed_batch(sess, tokeniser, chunk))
        if (i // batch_size) % 20 == 0:
            done = min(i + batch_size, len(texts))
            print(f"  embedded {done}/{len(texts)}  ({time.time()-t0:.0f}s)")
    return np.vstack(chunks)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_language_csv(path: Path, lang_tag: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    # Accept both 'text' and 'utterance' as the phrase column.
    text_col = next(
        (c for c in df.columns if c in ("text", "utterance", "phrase", "sentence")),
        df.columns[0]
    )
    df = df.rename(columns={text_col: "text"})
    df["text"]   = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df[["text", "intent"]].dropna().drop_duplicates(subset=["text", "intent"])
    df = df[df["intent"] != FALLBACK_INTENT]   # exclude noisy OOS from source files
    print(f"  [{lang_tag}] {len(df)} phrases, "
          f"{df['intent'].nunique()} intents from {path.name}")
    return df


def main() -> None:
    print("=== Multilingual Semantic Head Training ===\n")

    if not VOCAB_PATH.exists():
        print(f"ERROR: {VOCAB_PATH} not found.\n"
              "Run `python scripts/SemanticSupport/download_models.py` first.")
        sys.exit(1)

    # ── Load per-language CSVs ─────────────────────────────────────────────
    languages = ["fr", "de", "da"]
    if args.include_en:
        languages.append("en")

    frames = []
    for lang in languages:
        csv_path = DATA_DIR / f"{lang}.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found — skipping {lang}")
            continue
        frames.append(load_language_csv(csv_path, lang))

    if not frames:
        print("ERROR: No language CSV files found.")
        sys.exit(1)

    data = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["text", "intent"])
    print(f"\nTotal in-scope phrases (before cap): {len(data)}")

    # ── Per-class cap (same as English head: removes near-duplicate spam) ──
    data = (
        data.groupby("intent")
            .tail(args.max_per_intent)
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
    )
    print(f"In-scope after capping at {args.max_per_intent} per intent: {len(data)}")
    print(f"Intents: {data['intent'].nunique()}")

    # ── OOS class: inject explicit out-of-scope data if available ─────────
    oos_path = DATA_DIR / "oos_multilingual.csv"
    oos_texts = []
    if oos_path.exists():
        oos_df = pd.read_csv(oos_path, encoding="utf-8-sig")
        oos_df.columns = [c.strip().lower() for c in oos_df.columns]
        text_col = next(
            (c for c in oos_df.columns if c in ("text", "utterance", "phrase")),
            oos_df.columns[0]
        )
        oos_texts = oos_df[text_col].astype(str).str.strip().dropna().tolist()
        print(f"Curated OOS phrases ('{FALLBACK_INTENT}'): {len(oos_texts)}")
    else:
        print(f"  [oos] {oos_path} not found — training WITHOUT a learned OOS class")
        print("  (consider creating multilingual/data/oos_multilingual.csv for better rejection)")

    texts   = data["text"].tolist() + oos_texts
    intents = data["intent"].tolist() + [FALLBACK_INTENT] * len(oos_texts)
    print(f"\nTraining set: {len(texts)} phrases, {len(set(intents))} classes\n")

    # ── Embed ──────────────────────────────────────────────────────────────
    print("Embedding all phrases via ONNX (same path runtime will use)...")
    X = embed_all(texts)
    y = np.array(intents)

    # ── Held-out evaluation ────────────────────────────────────────────────
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    clf = LogisticRegression(max_iter=2000, C=args.C, class_weight="balanced")
    clf.fit(X_tr, y_tr)

    y_pred   = clf.predict(X_te)
    acc      = accuracy_score(y_te, y_pred)
    macro_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)

    print(f"\n── Held-out evaluation (15% never seen during training) ──")
    print(f"  Overall accuracy : {acc:.3f}")
    print(f"  Macro-F1         : {macro_f1:.3f}")

    # Per-language macro-F1 breakdown (requires language-tagged test split).
    # We approximate by checking per-intent accuracy since the data is merged.
    print("\n  Per-class accuracy:")
    for intent in sorted(set(y_tr) | set(y_te)):
        mask = y_te == intent
        if mask.sum() > 0:
            acc_cls = (y_pred[mask] == intent).mean()
            print(f"    {intent[:40]:40s}  {acc_cls:6.1%}  ({mask.sum():3d} samples)")

    # ── Accuracy gates ─────────────────────────────────────────────────────
    # Target: macro-F1 fr ≥ 0.80, de ≥ 0.78, da ≥ 0.75.
    # Because the merged head does not separate per-language scores we report
    # the overall macro-F1 and warn if it falls below the strictest gate (0.75).
    # Per-language gates are validated in test_multilingual_semantic.py with a
    # language-stratified holdout.
    GATE_OVERALL = 0.75
    if macro_f1 < GATE_OVERALL:
        print(f"\n  ⚠  WARNING: macro-F1 {macro_f1:.3f} < gate {GATE_OVERALL:.2f}. "
              "Review training data before deploying.")
    else:
        print(f"\n  ✓  Macro-F1 {macro_f1:.3f} passes overall gate {GATE_OVERALL:.2f}.")

    # ── Final model on 100% of data ────────────────────────────────────────
    print("\nRetraining on full dataset...")
    clf_final = LogisticRegression(max_iter=2000, C=args.C, class_weight="balanced")
    clf_final.fit(X, y)

    np.savez_compressed(
        HEAD_NPZ,
        weights  = clf_final.coef_.astype(np.float32),         # (n_classes, 384)
        bias     = clf_final.intercept_.astype(np.float32),    # (n_classes,)
        labels   = np.array(clf_final.classes_, dtype=object),
        embedder = np.array(["multilingual-onnx"], dtype=object),
        seq_len  = np.array([SEQ_LEN]),
        model_id = np.array([
            "paraphrase-multilingual-MiniLM-L12-v2"
        ], dtype=object),
    )

    size_kb = HEAD_NPZ.stat().st_size / 1024
    print(f"\n✅ Head saved : {HEAD_NPZ}  ({size_kb:.0f} KB)")
    print(f"   Classes    : {len(clf_final.classes_)}")
    print(f"   Macro-F1   : {macro_f1:.3f}  (15% holdout)")


if __name__ == "__main__":
    main()
