#!/usr/bin/env python3
"""
Train the semantic classification head (SetFit-style, frozen encoder).

Replaces the 1-NN cosine search over a 4.6 MB index with a logistic
regression trained on MiniLM embeddings of all training phrases —
INCLUDING Default Fallback Intent phrases as an explicit out-of-scope
class, so rejection is learned instead of threshold-guessed.

Pipeline:
  1. Embed every phrase in data/intent_data_new[_2].csv with MiniLM (ONNX)
  2. Stratified 85/15 split → train head → report held-out metrics
  3. Retrain on 100% of data → save models/semantic_head.npz (~95 KB)
     and models/semantic_head.json (for iOS)

Run:
    python scripts/train_semantic_head.py              # default: version 2
    python scripts/train_semantic_head.py --version 1  # original files
    python scripts/train_semantic_head.py --version 2  # corrected _2 files
    python scripts/train_semantic_head.py -v 1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- Args ----------
_parser = argparse.ArgumentParser(description="Train MiniLM semantic head")
_parser.add_argument("--version", "-v", type=int, choices=[1, 2], default=2,
                     help="Data version: 1=original files, 2=corrected _2 files (default: 2)")
_args = _parser.parse_args()

BASE_DIR   = Path(__file__).parent.parent
if _args.version == 1:
    DATA_PATH    = BASE_DIR / "data" / "intent_data_new.csv"
    OOS_PATH     = BASE_DIR / "data" / "semantic_oos.csv"
    HOLDOUT_PATH = BASE_DIR / "data" / "semantic_holdout_100.csv"
else:
    DATA_PATH    = BASE_DIR / "data" / "intent_data_new_2.csv"
    OOS_PATH     = BASE_DIR / "data" / "semantic_oos_2.csv"
    HOLDOUT_PATH = BASE_DIR / "data" / "semantic_holdout_2.csv"

print(f"Data version: {_args.version}  |  {DATA_PATH.name}  |  oos: {OOS_PATH.name}  |  holdout: {HOLDOUT_PATH.name}")
MODEL_DIR  = BASE_DIR / "models"
ONNX_PATH  = MODEL_DIR / "minilm-l6-v2.onnx"
VOCAB_PATH = MODEL_DIR / "minilm-vocab.txt"
HEAD_NPZ   = MODEL_DIR / "semantic_head.npz"
HEAD_JSON  = MODEL_DIR / "semantic_head.json"

FALLBACK_INTENT = "Default Fallback Intent"

sys.path.insert(0, str(BASE_DIR / "scripts"))
from nlu.semantic import SemanticFallback  # for the WordPiece tokeniser


# ── Batched ONNX embedding ───────────────────────────────────────────────────

def load_vocab() -> dict:
    with open(VOCAB_PATH, encoding="utf-8") as f:
        return {line.strip(): i for i, line in enumerate(f)}


def embed_batch(session, vocab, texts, max_len=64):
    """Embed a batch of texts with dynamic padding to the batch max length."""
    all_ids = []
    for text in texts:
        tokens = ["[CLS]"] + SemanticFallback._wordpiece(text.lower(), vocab) + ["[SEP]"]
        tokens = tokens[:max_len]
        all_ids.append([vocab.get(t, vocab["[UNK]"]) for t in tokens])

    batch_len = max(len(ids) for ids in all_ids)
    n = len(all_ids)
    input_ids = np.zeros((n, batch_len), dtype=np.int64)
    mask      = np.zeros((n, batch_len), dtype=np.int64)
    for i, ids in enumerate(all_ids):
        input_ids[i, :len(ids)] = ids
        mask[i, :len(ids)] = 1

    outputs = session.run(None, {
        "input_ids":      input_ids,
        "attention_mask": mask,
        "token_type_ids": np.zeros((n, batch_len), dtype=np.int64),
    })
    token_emb = outputs[0]                                  # (n, seq, 384)
    expanded  = mask[:, :, np.newaxis].astype(np.float32)
    summed    = (token_emb * expanded).sum(axis=1)
    vecs      = summed / expanded.sum(axis=1)
    norms     = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms     = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype(np.float32)


def embed_all(texts, batch_size=64):
    import onnxruntime as ort
    session = ort.InferenceSession(str(ONNX_PATH))
    vocab   = load_vocab()
    chunks  = []
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        chunks.append(embed_batch(session, vocab, texts[i:i + batch_size]))
        if (i // batch_size) % 20 == 0:
            done = min(i + batch_size, len(texts))
            print(f"  embedded {done}/{len(texts)}  ({time.time()-t0:.0f}s)")
    return np.vstack(chunks)


# ── Training ─────────────────────────────────────────────────────────────────

def main():
    if not ONNX_PATH.exists():
        print(f"ERROR: {ONNX_PATH} not found. Run scripts/download_minilm.py first.")
        sys.exit(1)

    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    data.columns = [c.strip().lower() for c in data.columns]
    data["text"]   = data["text"].astype(str).str.lower().str.strip()
    data["intent"] = data["intent"].astype(str).str.strip()
    data = data.dropna().drop_duplicates(subset=["text", "intent"])

    # The fallback class in intent_data_new.csv is Dialogflow's failure log —
    # largely garbled ASR of IN-DOMAIN requests. Training on it would teach the
    # head to replicate old Dialogflow's failures, so it is EXCLUDED. Instead we
    # train an EXPLICIT out-of-scope class from a CURATED clean OOS set
    # (data/semantic_oos.csv): general-knowledge / other-domain queries that the
    # head should learn to reject rather than guess at via a probability floor.
    in_scope = data[data["intent"] != FALLBACK_INTENT]
    noisy_fb = data[data["intent"] == FALLBACK_INTENT]   # noisy eval only

    # ── Lever A: Cap per-class at 250 to reduce near-duplicate spam ──
    # MemoryChange 1821 → ~250; Help_Battery 18 → stays 18. This removes
    # 70% of carrier-phrase permutations ("can you change X" ×185 variants)
    # while preserving diversity on tiny tail classes.
    MAX_PER_INTENT = 250
    in_scope = (
        in_scope.groupby("intent")
            .tail(MAX_PER_INTENT)          # keep newest rows when over cap
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
    )
    print(f"In-scope after capping at {MAX_PER_INTENT} per intent: {len(in_scope)}")

    oos_texts = []
    if OOS_PATH.exists():
        oos = pd.read_csv(OOS_PATH, encoding="utf-8-sig")
        oos.columns = [c.strip().lower() for c in oos.columns]
        oos["text"] = oos["text"].astype(str).str.lower().str.strip()
        oos = oos.dropna(subset=["text"]).drop_duplicates(subset=["text"])
        # Leakage guard: never train on a holdout OOS phrase.
        if HOLDOUT_PATH.exists():
            hold = pd.read_csv(HOLDOUT_PATH, encoding="utf-8-sig")
            hold.columns = [c.strip().lower() for c in hold.columns]
            htext = next((c for c in hold.columns
                          if c.strip().lower() in ("text", "utterance", "query", "sentence", "phrase")),
                         hold.columns[0])
            hold_set = set(hold[htext].astype(str).str.lower().str.strip())
            before = len(oos)
            oos = oos[~oos["text"].isin(hold_set)]
            if before != len(oos):
                print(f"  [oos] dropped {before - len(oos)} phrase(s) overlapping the holdout")
        oos_texts = oos["text"].tolist()
        print(f"Curated OOS phrases (trained as '{FALLBACK_INTENT}'): {len(oos_texts)}")
    else:
        print(f"  [oos] {OOS_PATH} not found — training WITHOUT a learned OOS class")

    texts   = in_scope["text"].tolist() + oos_texts
    intents = in_scope["intent"].tolist() + [FALLBACK_INTENT] * len(oos_texts)
    print(f"Training phrases: {len(texts)}  Intents: {len(set(intents))}")
    print(f"Noisy fallback-log eval set: {len(noisy_fb)} phrases (NOT trained on)")

    print("\nEmbedding all phrases via ONNX (same path mobile will use)...")
    X    = embed_all(texts)
    X_fb = embed_all(noisy_fb["text"].tolist())
    y    = np.array(intents)

    # ── Held-out evaluation ──
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    # Lever B: Reduce C from 10.0 to 1.5 to prevent overfitting of tiny
    # tail classes in high-dimensional (384-dim) space.
    clf = LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced")
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"\n── Held-out evaluation (15% never seen during training) ──")
    print(f"  Overall accuracy: {acc:.3f}")

    # Lever C: Per-class metrics to expose tail-class regressions
    from sklearn.metrics import classification_report, f1_score
    print("\n  Per-class accuracy:")
    unique_intents = sorted(set(y_tr) | set(y_te))
    for intent in unique_intents:
        mask = y_te == intent
        if mask.sum() > 0:
            acc_per_class = (y_pred[mask] == intent).mean()
            count = mask.sum()
            print(f"    {intent[:35]:35s}  {acc_per_class:6.1%}  ({count:3d} test samples)")

    macro_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    print(f"\n  Macro-F1 (unweighted per-class): {macro_f1:.3f}")

    # ── Rejection threshold curve ──
    # In-scope held-out max-probs vs fallback-set max-probs: pick the
    # threshold that keeps in-scope rescues while rejecting out-of-scope.
    p_in = clf.predict_proba(X_te).max(axis=1)
    p_fb = clf.predict_proba(X_fb).max(axis=1)
    print(f"\n── Rejection curve (head max-probability threshold) ──")
    print(f"  {'thresh':>7} {'in-scope kept':>14} {'out-of-scope rejected':>22}")
    for th in (0.30, 0.40, 0.50, 0.60, 0.70):
        kept     = float((p_in >= th).mean())
        rejected = float((p_fb < th).mean())
        print(f"  {th:>7.2f} {kept:>13.1%} {rejected:>21.1%}")
    print("  (out-of-scope set is noisy — it contains garbled in-domain requests,")
    print("   so 100% rejection is neither achievable nor desirable)")

    # ── Final model on 100% of data (in-scope + curated OOS class) ──
    print("\nRetraining on full dataset (in-scope + OOS class)...")
    clf = LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced")
    clf.fit(X, y)

    np.savez_compressed(
        HEAD_NPZ,
        weights=clf.coef_.astype(np.float32),       # (n_classes, 384)
        bias=clf.intercept_.astype(np.float32),     # (n_classes,)
        labels=np.array(clf.classes_, dtype=object),
        embedder=np.array(["onnx"], dtype=object),  # which embed path built this
    )
    HEAD_JSON.write_text(json.dumps({
        "embedder": "onnx",
        "labels":   list(clf.classes_),
        "weights":  [[round(float(w), 6) for w in row] for row in clf.coef_],
        "bias":     [round(float(b), 6) for b in clf.intercept_],
    }))

    print(f"\n✅ Head saved: {HEAD_NPZ}  ({HEAD_NPZ.stat().st_size/1024:.0f} KB)")
    print(f"✅ iOS JSON:   {HEAD_JSON}  ({HEAD_JSON.stat().st_size/1024:.0f} KB)")
    print("\nThe legacy semantic_index.npz 1-NN path was removed in Sprint 3 "
          "(head-only rejection).")

    # Re-sign the bundle so the new semantic artifacts pass the startup
    # integrity check (the head files are tracked in the manifest).
    from nlu.manifest import generate_manifest
    generate_manifest(BASE_DIR)


if __name__ == "__main__":
    main()
