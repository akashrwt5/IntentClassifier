#!/usr/bin/env python3
"""
Diagnose NaN / Inf in a CoreML MiniLM embedder before it reaches the head.

WHY THIS EXISTS
---------------
train_semantic_head_coreml.py embeds every training phrase through the CoreML
encoder and feeds the vectors to LogisticRegression. If any embedding contains
NaN, sklearn aborts with a cryptic "Input X contains NaN" deep in its
validation stack — with no hint about WHICH model or WHICH phrase produced it.

This script isolates that. It loads each embedder, runs it exactly the way the
trainer does (same tokeniser, same fixed-length padding, CPU_ONLY), and reports
NaN/Inf counts and the value range of the RAW model output — before mean-pool.
That tells you immediately whether the corruption is in:

  * the FP16 base export        (models/MiniLMEmbedder.mlpackage)
  * the palettized int8 export  (models/MiniLMEmbedder_int8.mlpackage)
  * both, or neither

Interpreting the verdict
------------------------
  * FP16 clean, int8 NaN  -> palettization is the culprit. Re-export with
    8-bit (`nbits=8`) palettization, or ship the FP16 model.
  * FP16 NaN (and int8)   -> the base PyTorch->CoreML export is broken
    (e.g. a custom LayerNorm that overflows FP16, or a bad torch/coremltools
    combo). Re-run `python scripts/export_coreml.py` with native nn.LayerNorm.
  * both clean            -> the .mlpackage files on disk are fine; the NaN you
    saw came from a stale model that has since been regenerated, or from a
    single pathological phrase. Re-run with --scan to find it.

Usage
-----
    # test both default embedders on a set of probe phrases
    python scripts/diagnose_embedder_nan.py

    # test one specific package
    python scripts/diagnose_embedder_nan.py --model models/MiniLMEmbedder_int8.mlpackage

    # if probes are clean, scan the FULL training set for an offending phrase
    python scripts/diagnose_embedder_nan.py --scan
"""

import argparse
import sys
from pathlib import Path

import numpy as np

BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
VOCAB_PATH = MODELS_DIR / "minilm-vocab.txt"
DATA_PATH  = BASE_DIR / "datasets" / "01_source_base_training_data.csv"

# Mirror the trainer exactly (train_semantic_head_coreml.py).
MAX_LEN = 64
SEQ_LEN = 32

DEFAULT_MODELS = [
    MODELS_DIR / "MiniLMEmbedder.mlpackage",
    MODELS_DIR / "MiniLMEmbedder_int8.mlpackage",
]

# A spread of short/long, simple/odd phrases. If a model is structurally broken
# (FP16 overflow, palettization corruption) it shows NaN on essentially any input.
PROBE_TEXTS = [
    "turn up the volume",
    "what's the weather like today",
    "remind me to call chris tomorrow at noon",
    "make it louder please i can barely hear anything over here",
    "x",                       # degenerate single char
    "aerobics activity start", # near a real intent label
]

sys.path.insert(0, str(BASE_DIR / "scripts"))


def load_vocab() -> dict:
    if not VOCAB_PATH.exists():
        sys.exit(f"ERROR: {VOCAB_PATH} not found — run scripts/download_minilm.py")
    with open(VOCAB_PATH, encoding="utf-8") as f:
        return {line.strip(): i for i, line in enumerate(f)}


def _tokenise(vocab, text):
    from nlu.semantic import SemanticFallback
    tokens = ["[CLS]"] + SemanticFallback._wordpiece(text.lower(), vocab) + ["[SEP]"]
    tokens = tokens[:MAX_LEN]
    ids = [vocab.get(t, vocab["[UNK]"]) for t in tokens]
    return ids, len(ids)


def _raw_output(model, vocab, text, seq_len=SEQ_LEN):
    """Return the RAW last_hidden_state [seq_len, 384] and the real-token count n.

    Padded to seq_len with mask=0, identical to train_semantic_head_coreml.py.
    No mean-pool / normalise here — we want to see corruption before it is
    averaged away or turned into 0/0.
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
    token_emb = np.array(out[next(iter(out))])[0]   # [seq_len, 384]
    return token_emb, n


def _describe(arr):
    """One-line health summary of an array."""
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    finite = arr[np.isfinite(arr)]
    if finite.size:
        rng = f"[{finite.min():.2f}, {finite.max():.2f}]"
    else:
        rng = "[no finite values]"
    return n_nan, n_inf, rng


def test_model(ct, path: Path, vocab, scan: bool) -> bool:
    """Return True if the model is healthy on all tested phrases."""
    print(f"\n{'='*70}")
    print(f"  {path.name}")
    print(f"{'='*70}")
    if not path.exists():
        print(f"  SKIP — not found at {path}")
        return True
    try:
        model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
    except Exception as e:
        print(f"  LOAD FAILED — {e}")
        return False

    healthy = True
    for text in PROBE_TEXTS:
        token_emb, n = _raw_output(model, vocab, text)
        real = token_emb[:n]                       # the rows that actually matter
        n_nan, n_inf, rng = _describe(real)
        flag = "  OK " if (n_nan == 0 and n_inf == 0) else " BAD "
        if n_nan or n_inf:
            healthy = False
        print(f"  [{flag}] nan={n_nan:>5} inf={n_inf:>4} range={rng:<22} \"{text[:42]}\"")

    if healthy and scan:
        healthy = _scan_training_set(model, vocab)
    elif not healthy:
        print("\n  -> This model emits non-finite values on ordinary input. It is")
        print("     structurally broken; regenerate it (see the header of this file).")
    else:
        print("\n  -> Healthy on all probe phrases. Re-run with --scan to check every")
        print("     training phrase if the trainer still reports NaN.")
    return healthy


def _scan_training_set(model, vocab) -> bool:
    """Embed every training phrase; report the first few that go non-finite."""
    import pandas as pd
    if not DATA_PATH.exists():
        print(f"\n  --scan: {DATA_PATH} not found, skipping full scan.")
        return True
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    texts = df["text"].astype(str).str.lower().str.strip().tolist()
    print(f"\n  --scan: embedding {len(texts)} training phrases ...")
    bad = []
    for i, text in enumerate(texts):
        token_emb, n = _raw_output(model, vocab, text)
        if not np.isfinite(token_emb[:n]).all():
            bad.append(text)
            if len(bad) <= 10:
                print(f"    NON-FINITE: \"{text[:60]}\"")
        if i % 1000 == 0 and i:
            print(f"    ... {i}/{len(texts)} scanned, {len(bad)} bad so far")
    if bad:
        print(f"\n  -> {len(bad)} phrase(s) produce non-finite embeddings.")
        return False
    print("\n  -> All training phrases embed cleanly. The model is healthy.")
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=None,
                    help="Embedder .mlpackage to test (repeatable). "
                         "Default: both FP16 and int8 packages in models/.")
    ap.add_argument("--scan", action="store_true",
                    help="If probes are clean, embed the FULL training set to find "
                         "any single phrase that produces a non-finite vector.")
    args = ap.parse_args()

    try:
        import coremltools as ct
    except ImportError:
        sys.exit("ERROR: coremltools not installed (macOS only). pip install coremltools")

    models = [Path(m) for m in args.model] if args.model else DEFAULT_MODELS
    vocab = load_vocab()

    print("CoreML embedder NaN/Inf diagnostic")
    print("(raw last_hidden_state over real tokens, CPU_ONLY, padded to "
          f"{SEQ_LEN} — the exact trainer path)")

    results = {p.name: test_model(ct, p, vocab, args.scan) for p in models}

    print(f"\n{'='*70}")
    print("  VERDICT")
    print(f"{'='*70}")
    for name, ok in results.items():
        print(f"  {'HEALTHY' if ok else 'BROKEN ':>8}  {name}")
    if not all(results.values()):
        print("\n  Fix the BROKEN model(s) before training the head. Guidance is in")
        print("  the docstring at the top of this script.")
        sys.exit(1)
    print("\n  All tested embedders are healthy.")


if __name__ == "__main__":
    main()
