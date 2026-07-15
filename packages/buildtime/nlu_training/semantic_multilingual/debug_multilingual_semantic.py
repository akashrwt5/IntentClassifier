#!/usr/bin/env python3
"""
CLI debug tool for the multilingual semantic fallback stage.

Prints:
  - Fixed-length tokenization arrays (input_ids, attention_mask, token positions)
  - UNK rate for the given text
  - Top-5 intent predictions with softmax confidence scores
  - Final rescue decision at the configured threshold

Usage:
    python scripts/SemanticSupport/debug_multilingual_semantic.py --text "..." --language fr
    python scripts/SemanticSupport/debug_multilingual_semantic.py --text "Wann soll ich Medikamente nehmen" --language de
    python scripts/SemanticSupport/debug_multilingual_semantic.py --text "hvad er klokken" --language da --threshold 0.4
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BASE_DIR / "scripts"))
sys.path.insert(0, str(BASE_DIR))

parser = argparse.ArgumentParser(description="Debug multilingual semantic fallback")
parser.add_argument("--text",      required=True, help="Input utterance to debug")
parser.add_argument("--language",  required=True, choices=["fr", "de", "da"],
                    help="Language code")
parser.add_argument("--threshold", type=float, default=0.55,
                    help="Confidence threshold for rescue decision (default: 0.55)")
parser.add_argument("--top-n",     type=int, default=5,
                    help="Number of top intents to display (default: 5)")
args = parser.parse_args()

MODEL_DIR  = BASE_DIR / "multilingual" / "SemanticSupport" / "models"
VOCAB_PATH = MODEL_DIR / "multilingual-vocab.txt"
HEAD_PATH  = MODEL_DIR / "semantic_head_multilingual.npz"
ONNX_PATH  = MODEL_DIR / "paraphrase-multilingual-minilm-l12-v2.int8.onnx"

for name, path in [("Vocab", VOCAB_PATH), ("Head", HEAD_PATH), ("ONNX", ONNX_PATH)]:
    if not path.exists():
        print(f"ERROR: {name} not found: {path}")
        print("Run download_models.py and train_multilingual_semantic_head.py first.")
        sys.exit(1)

import numpy as np
from multilingual.SemanticSupport.tokeniser import MultilingualTokeniser, SEQ_LEN
from multilingual.SemanticSupport.semantic  import MultilingualSemanticFallback, FALLBACK_INTENT

print(f"\n{'='*60}")
print(f"Text     : {args.text!r}")
print(f"Language : {args.language}")
print(f"Threshold: {args.threshold}")
print(f"{'='*60}")

# ── Tokenisation ──────────────────────────────────────────────────────────────

tok = MultilingualTokeniser(VOCAB_PATH)
from multilingual.SemanticSupport.tokeniser import tokenize

wp_tokens  = tokenize(args.text, tok.vocab)
input_ids, attention_mask, token_type_ids = tok.encode(args.text)

real_len = int(attention_mask[0].sum())
unk_count = sum(1 for t in wp_tokens if tok.vocab.get(t, tok._unk_id) == tok._unk_id)
unk_rate  = unk_count / len(wp_tokens) if wp_tokens else 0.0

# Reverse vocab for display
rev_vocab = {v: k for k, v in tok.vocab.items()}

print(f"\n── Tokenisation (SEQ_LEN={SEQ_LEN}) ──")
print(f"  WordPiece tokens  : {wp_tokens}")
print(f"  Real token count  : {real_len} (including [CLS] and [SEP])")
print(f"  UNK rate          : {unk_rate:.2%}  ({unk_count}/{len(wp_tokens)} tokens)")

print(f"\n  Positions [0..{real_len-1}] (real tokens):")
for i in range(real_len):
    tok_str = rev_vocab.get(int(input_ids[0, i]), "[?]")
    print(f"    [{i:3d}]  id={input_ids[0,i]:6d}  token={tok_str!r}")

if real_len < SEQ_LEN:
    pad_count = SEQ_LEN - real_len
    print(f"  Positions [{real_len}..{SEQ_LEN-1}]: {pad_count} × [PAD] (id=0, mask=0)")

print(f"\n  input_ids      shape: {input_ids.shape}  dtype: {input_ids.dtype}")
print(f"  attention_mask shape: {attention_mask.shape}")
print(f"  token_type_ids shape: {token_type_ids.shape}")

# ── Embedding + head scores ───────────────────────────────────────────────────

print(f"\n── Classifier scores ──")

# Load head directly for access to all class probabilities.
data    = np.load(HEAD_PATH, allow_pickle=True)
weights = data["weights"].astype(np.float32)   # (n_classes, 384)
bias    = data["bias"].astype(np.float32)       # (n_classes,)
labels  = data["labels"]

# Embed using the ONNX session directly (same path as runtime).
import onnxruntime as ort
sess = ort.InferenceSession(str(ONNX_PATH))
outputs = sess.run(None, {
    "input_ids":      input_ids,
    "attention_mask": attention_mask,
    "token_type_ids": token_type_ids,
})
token_emb = outputs[0][0]                                    # (64, 384)
mask_f    = attention_mask[0].astype(np.float32)             # (64,)
expanded  = mask_f[:, np.newaxis]                            # (64, 1)
summed    = (token_emb * expanded).sum(axis=0)               # (384,)
vec       = summed / mask_f.sum()                            # mean pool
norm      = np.linalg.norm(vec)
if norm > 0:
    vec = vec / norm
vec = vec.astype(np.float32)

print(f"  Embedding dim   : {vec.shape[0]}")
print(f"  Embedding norm  : {np.linalg.norm(vec):.4f}  (should be ~1.0)")

logits = weights @ vec + bias
logits -= logits.max()
probs  = np.exp(logits)
probs /= probs.sum()

top_n   = min(args.top_n, len(labels))
top_idx = np.argsort(probs)[::-1][:top_n]

print(f"\n  Top-{top_n} intents:")
for rank, idx in enumerate(top_idx, 1):
    marker = " ← RESCUE" if (
        str(labels[idx]) != FALLBACK_INTENT and float(probs[idx]) >= args.threshold
        and rank == 1
    ) else ""
    print(f"    {rank}. {str(labels[idx])[:50]:50s}  {probs[idx]:.4f}{marker}")

# ── Final decision ────────────────────────────────────────────────────────────

best_intent = str(labels[top_idx[0]])
best_conf   = float(probs[top_idx[0]])

print(f"\n── Rescue decision (threshold={args.threshold}) ──")
if best_intent == FALLBACK_INTENT:
    print(f"  FALLBACK  — head predicted the OOS class (conf={best_conf:.4f})")
elif best_conf >= args.threshold:
    print(f"  RESCUE    — {best_intent!r} at conf={best_conf:.4f} ≥ {args.threshold}")
else:
    print(f"  NO RESCUE — {best_intent!r} at conf={best_conf:.4f} < {args.threshold}")
print()
