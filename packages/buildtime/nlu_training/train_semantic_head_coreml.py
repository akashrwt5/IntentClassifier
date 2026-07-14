#!/usr/bin/env python3
"""
Retrain the semantic head on FP16-CoreML embeddings (close the ONNX->CoreML gap).

WHY THIS EXISTS
---------------
train_semantic_head.py embeds every training phrase with the INT8 ONNX model
(embedder="onnx") and trains the logistic-regression head on those vectors.
But iOS does NOT run that ONNX model -- it runs the FP16 CoreML embedder
(MiniLMEmbedder.mlpackage). Those two embedders differ enough that
compare_coreml_quant.py measures only ~0.96 cosine between them and a ~4%
in-scope accuracy drop: the head sees a distribution at inference time that it
was never trained on.

We CANNOT fix this by converting the ONNX to CoreML (the shipped ONNX is
dynamically INT8-quantized; onnx2torch can't translate MatMulInteger /
DynamicQuantizeLinear -- see export_onnx_to_coreml.py). The robust fix is the
other direction: train the head on the embedder iOS actually runs. Then the
training and inference embedding spaces are identical and the gap is 0 by
construction.

This script is a CoreML-embedder twin of train_semantic_head.py. It produces:
    models/semantic_head_coreml.json   (iOS head, embedder="coreml")
    models/semantic_head_coreml.npz    (Python head)
It does NOT overwrite the ONNX-trained head. Compare the two with
compare_coreml_quant.py (point _load_head at the new JSON), and if the CoreML
head wins, copy it over semantic_head.json before bundling into the STT repo.

REQUIREMENTS
------------
  * macOS 13+ (CoreML predict() needs the macOS runtime)
  * models/MiniLMEmbedder.mlpackage present (run export_coreml.py first)
  * pip install coremltools scikit-learn pandas numpy

Run:
    python scripts/train_semantic_head_coreml.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR     = Path(__file__).resolve().parents[3]
DATA_PATH    = BASE_DIR / "datasets" / "01_source_base_training_data.csv"
OOS_PATH     = BASE_DIR / "datasets" / "semantic_oos.csv"
HOLDOUT_PATH = BASE_DIR / "datasets" / "semantic_holdout_100.csv"
MODEL_DIR    = BASE_DIR / "models"
COREML_PATH  = MODEL_DIR / "MiniLMEmbedder.mlpackage"
VOCAB_PATH   = MODEL_DIR / "minilm-vocab.txt"
HEAD_NPZ     = MODEL_DIR / "semantic_head_coreml.npz"
HEAD_JSON    = MODEL_DIR / "semantic_head_coreml.json"

FALLBACK_INTENT = "sys.oos.fallback"
MAX_LEN = 64

# The shipped CoreML encoder is exported at a FIXED (1, SEQ_LEN) shape
# (export_coreml.py --seq-len, default 32) so it stays resident on the ANE.
# We right-pad each phrase to SEQ_LEN (mask=0 on pads) to match it exactly.
SEQ_LEN = 32

sys.path.insert(0, str(BASE_DIR / "scripts"))
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[3] / 'packages' / 'runtime'))
from nlu_engine.semantic import SemanticFallback  # for the WordPiece tokeniser


# -- CoreML embedding ----------------------------------------------------------

def load_vocab() -> dict:
    with open(VOCAB_PATH, encoding="utf-8") as f:
        return {line.strip(): i for i, line in enumerate(f)}


def _tokenise(vocab, text):
    tokens = ["[CLS]"] + SemanticFallback._wordpiece(text.lower(), vocab) + ["[SEP]"]
    tokens = tokens[:MAX_LEN]
    ids = [vocab.get(t, vocab["[UNK]"]) for t in tokens]
    return ids, len(ids)


def _embed_one(model, vocab, text, seq_len=SEQ_LEN):
    """Embed a single text via CoreML -- mirrors SemanticEmbedder.swift exactly.

    The shipped CoreML encoder uses a FIXED (1, seq_len) shape (ANE-resident),
    so we right-pad each phrase to seq_len with attention_mask=0 on the pads and
    mean-pool only the real tokens. (Padding to seq_len is also valid against a
    legacy dynamic-shape model.)
    """
    ids, n = _tokenise(vocab, text)
    n = min(n, seq_len)
    ids_arr  = np.zeros((1, seq_len), dtype=np.int32); ids_arr[0, :n]  = ids[:n]
    mask_arr = np.zeros((1, seq_len), dtype=np.int32); mask_arr[0, :n] = 1
    tids_arr = np.zeros((1, seq_len), dtype=np.int32)
    out = model.predict({
        "input_ids":      ids_arr,
        "attention_mask": mask_arr,
        "token_type_ids": tids_arr,
    })
    key = next(iter(out))
    token_emb = np.array(out[key])[0]                  # [seq_len, 384]
    vec = token_emb[:n].mean(axis=0)                   # real tokens only
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)


def embed_all(model, vocab, texts, model_name="the embedder"):
    vecs = []
    bad  = []   # (index, text) for any phrase whose embedding is non-finite
    t0 = time.time()
    for i, text in enumerate(texts):
        v = _embed_one(model, vocab, text)
        if not np.isfinite(v).all():
            bad.append((i, text))
        vecs.append(v)
        if i % 200 == 0:
            print(f"  embedded {i}/{len(texts)}  ({time.time()-t0:.0f}s)")
    if bad:
        # Fail loud and specific: sklearn would otherwise abort later with an
        # opaque "Input X contains NaN" and no pointer to the cause. A non-finite
        # embedding means the CoreML model itself emits NaN/Inf — almost always a
        # broken export (e.g. an FP16-overflowing custom LayerNorm) or a bad
        # palettization, not a data problem.
        preview = "\n".join(f'    [{i}] "{t[:60]}"' for i, t in bad[:10])
        more = f"\n    ... and {len(bad) - 10} more" if len(bad) > 10 else ""
        sys.exit(
            f"\nERROR: {len(bad)}/{len(texts)} phrases produced a non-finite "
            f"embedding from {model_name}.\n"
            f"The CoreML encoder is emitting NaN/Inf, so the head cannot be "
            f"trained on it.\nFirst offenders:\n{preview}{more}\n\n"
            f"Diagnose which model is broken (FP16 base vs palettized int8):\n"
            f"    python scripts/diagnose_embedder_nan.py\n"
            f"Then regenerate it:\n"
            f"    python scripts/export_coreml.py --quantize\n"
        )
    return np.vstack(vecs)


# -- Training ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(COREML_PATH),
                    help="CoreML embedder to train the head against. Point this at "
                         "the package you will SHIP -- e.g. "
                         "models/MiniLMEmbedder_int8.mlpackage for the palettized "
                         f"model (default: {COREML_PATH.name}, the FP16 model).")
    args = ap.parse_args()

    try:
        import coremltools as ct
    except ImportError:
        sys.exit("ERROR: coremltools not installed (macOS only). pip install coremltools")

    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"ERROR: {model_path} not found. Run scripts/export_coreml.py first.")

    print(f"Loading CoreML embedder (cpuOnly): {model_path.name}")
    model = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    vocab = load_vocab()

    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    data.columns = [c.strip().lower() for c in data.columns]
    data["text"]   = data["text"].astype(str).str.lower().str.strip()
    data["intent"] = data["intent"].astype(str).str.strip()
    data = data.dropna().drop_duplicates(subset=["text", "intent"])

    # Same policy as train_semantic_head.py: the noisy Dialogflow fallback log is
    # excluded from training (eval only); a curated clean OOS set is trained as
    # an explicit Default Fallback Intent class so rejection is learned.
    in_scope = data[data["intent"] != FALLBACK_INTENT]
    noisy_fb = data[data["intent"] == FALLBACK_INTENT]

    oos_texts = []
    if OOS_PATH.exists():
        oos = pd.read_csv(OOS_PATH, encoding="utf-8-sig")
        oos.columns = [c.strip().lower() for c in oos.columns]
        oos["text"] = oos["text"].astype(str).str.lower().str.strip()
        oos = oos.dropna(subset=["text"]).drop_duplicates(subset=["text"])
        if HOLDOUT_PATH.exists():
            hold = pd.read_csv(HOLDOUT_PATH, encoding="utf-8-sig")
            hold.columns = [c.strip().lower() for c in hold.columns]
            htext = next((c for c in hold.columns
                          if c in ("text", "utterance", "query", "sentence", "phrase")),
                         hold.columns[0])
            hold_set = set(hold[htext].astype(str).str.lower().str.strip())
            before = len(oos)
            oos = oos[~oos["text"].isin(hold_set)]
            if before != len(oos):
                print(f"  [oos] dropped {before - len(oos)} phrase(s) overlapping the holdout")
        oos_texts = oos["text"].tolist()
        print(f"Curated OOS phrases (trained as '{FALLBACK_INTENT}'): {len(oos_texts)}")
    else:
        print(f"  [oos] {OOS_PATH} not found -- training WITHOUT a learned OOS class")

    texts   = in_scope["text"].tolist() + oos_texts
    intents = in_scope["intent"].tolist() + [FALLBACK_INTENT] * len(oos_texts)
    print(f"Training phrases: {len(texts)}  Intents: {len(set(intents))}")
    print(f"Noisy fallback-log eval set: {len(noisy_fb)} phrases (NOT trained on)")

    print("\nEmbedding all phrases via CoreML (the SAME path iOS will use)...")
    X    = embed_all(model, vocab, texts, model_path.name)
    X_fb = embed_all(model, vocab, noisy_fb["text"].tolist(), model_path.name)
    y    = np.array(intents)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    clf = LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced")
    clf.fit(X_tr, y_tr)

    acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"\n-- Held-out evaluation (15% never seen during training) --")
    print(f"  In-scope accuracy: {acc:.3f}")

    p_in = clf.predict_proba(X_te).max(axis=1)
    p_fb = clf.predict_proba(X_fb).max(axis=1)
    print(f"\n-- Rejection curve (head max-probability threshold) --")
    print(f"  {'thresh':>7} {'in-scope kept':>14} {'out-of-scope rejected':>22}")
    for th in (0.30, 0.40, 0.50, 0.60, 0.70):
        kept     = float((p_in >= th).mean())
        rejected = float((p_fb < th).mean())
        print(f"  {th:>7.2f} {kept:>13.1%} {rejected:>21.1%}")

    print("\nRetraining on full dataset (in-scope + OOS class)...")
    clf = LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced")
    clf.fit(X, y)

    np.savez_compressed(
        HEAD_NPZ,
        weights=clf.coef_.astype(np.float32),
        bias=clf.intercept_.astype(np.float32),
        labels=np.array(clf.classes_, dtype=object),
        embedder=np.array(["coreml"], dtype=object),
    )
    HEAD_JSON.write_text(json.dumps({
        "embedder": "coreml",
        "labels":   list(clf.classes_),
        "weights":  [[round(float(w), 6) for w in row] for row in clf.coef_],
        "bias":     [round(float(b), 6) for b in clf.intercept_],
    }))

    print(f"\nHead saved: {HEAD_NPZ}  ({HEAD_NPZ.stat().st_size/1024:.0f} KB)")
    print(f"iOS JSON:   {HEAD_JSON}  ({HEAD_JSON.stat().st_size/1024:.0f} KB)")
    print("\nNext steps:")
    print("  1. Verify the new head closes the gap. In compare_coreml_quant.py")
    print("     temporarily point _load_head() at semantic_head_coreml.json, then:")
    print("       python scripts/compare_coreml_quant.py")
    print("     FP16 in-scope accuracy should now match (or beat) the ONNX column.")
    print("  2. If it wins, promote it:")
    print("       cp models/semantic_head_coreml.json models/semantic_head.json")
    print("     and copy semantic_head.json into STT/STT/STT/Resources/.")
    print("  NOTE: do NOT re-run train_semantic_head.py afterwards -- it would")
    print("        overwrite semantic_head.json with the ONNX-trained head again.")


if __name__ == "__main__":
    main()
