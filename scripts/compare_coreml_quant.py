#!/usr/bin/env python3
"""
Data-driven FP16-vs-INT8 accuracy comparison for the MiniLM CoreML embedder.

Run this AFTER `python scripts/export_coreml.py --quantize`, which produces both
    models/MiniLMEmbedder.mlpackage        (FP16, ~45 MB)
    models/MiniLMEmbedder_int8.mlpackage   (INT8, ~22 MB)

Why this exists
───────────────
INT8 quantization is a SIZE optimization, not a free lunch. It perturbs the
384-dim embeddings slightly, and the semantic head was trained on FP16/FP32
embeddings. Borderline rescues near the 0.55 threshold can flip. This harness
measures the REAL impact on the actual holdout + out-of-scope data instead of
guessing, so the FP16-vs-INT8 ship decision is evidence-based.

What it reports
───────────────
  • In-scope accuracy on semantic_holdout_100.csv (FP16 vs INT8)
  • Out-of-scope rejection rate on semantic_oos.csv (FP16 vs INT8)
  • Mean / min cosine similarity between FP16 and INT8 embeddings
  • Number of decision flips (label changed OR rescue/reject decision changed)

A reasonable ship bar: accuracy delta < 1% AND OOS rejection not worse.

macOS only (CoreML prediction requires the macOS runtime).

Usage:
    pip install coremltools numpy
    python scripts/compare_coreml_quant.py
    python scripts/compare_coreml_quant.py --threshold 0.55
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data"

FALLBACK_INTENT = "Default Fallback Intent"
DEFAULT_THRESHOLD = 0.55

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN = "\033[96m"; BOLD = "\033[1m"; RESET = "\033[0m"


# ── Tokeniser: reuse the exact WordPiece logic the runtime uses ───────────────

def _load_tokeniser():
    """Return a tokenise(text) -> (ids, mask, tids) int32 function.

    Reuses SemanticFallback._wordpiece / vocab loading so the comparison runs
    through the identical tokenisation as production — no algorithm drift.
    """
    sys.path.insert(0, str(BASE_DIR))
    from scripts.nlu.semantic import SemanticFallback

    vocab_path = MODELS_DIR / "minilm-vocab.txt"
    if not vocab_path.exists():
        sys.exit(f"ERROR: {vocab_path} not found — run scripts/download_minilm.py")
    with open(vocab_path, encoding="utf-8") as f:
        vocab = {line.strip(): i for i, line in enumerate(f)}

    def tokenise(text, max_len=64):
        tokens = ["[CLS]"] + SemanticFallback._wordpiece(text.lower(), vocab) + ["[SEP]"]
        tokens = tokens[:max_len]
        ids = [vocab.get(t, vocab["[UNK]"]) for t in tokens]
        n = len(ids)
        return (np.array([ids], dtype=np.int32),
                np.ones((1, n), dtype=np.int32),
                np.zeros((1, n), dtype=np.int32))

    return tokenise


# ── Semantic head (the shipping artifact) ─────────────────────────────────────

def _load_head():
    """Load (weights, bias, labels) from semantic_head.json, falling back to .npz."""
    js = MODELS_DIR / "semantic_head.json"
    npz = MODELS_DIR / "semantic_head.npz"
    if js.exists():
        d = json.loads(js.read_text())
        return (np.array(d["weights"], dtype=np.float32),
                np.array(d["bias"], dtype=np.float32),
                [str(x) for x in d["labels"]])
    if npz.exists():
        d = np.load(npz, allow_pickle=True)
        return (d["weights"].astype(np.float32),
                d["bias"].astype(np.float32),
                [str(x) for x in d["labels"]])
    sys.exit("ERROR: neither semantic_head.json nor semantic_head.npz found.")


def _classify(vec, head, threshold):
    """Mirror SemanticFallback.classify + the engine's threshold/reject gate.

    Returns (decision, conf) where decision is the intent label, or "GENAI"
    when the head rejects (fallback class) or confidence is below threshold.
    """
    weights, bias, labels = head
    logits = weights @ vec + bias
    logits -= logits.max()
    probs = np.exp(logits); probs /= probs.sum()
    top = int(np.argmax(probs))
    label, conf = labels[top], float(probs[top])
    if label == FALLBACK_INTENT or conf < threshold:
        return "GENAI", conf
    return label, conf


# ── Embedding via a CoreML model ──────────────────────────────────────────────

def _embed(model, tokenise, text):
    """Mean-pool + L2-normalise a CoreML MiniLM prediction → 384-dim float32."""
    ids, mask, tids = tokenise(text)
    out = model.predict({"input_ids": ids, "attention_mask": mask, "token_type_ids": tids})
    key = next(iter(out))
    token_emb = np.array(out[key])[0]          # [seq, 384]
    m = mask[0][:, np.newaxis].astype(np.float32)
    vec = (token_emb * m).sum(axis=0) / m.sum()
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)


# ── Dataset loading ───────────────────────────────────────────────────────────

def _load_csv(path, label_col=None):
    """Read (utterance, expected) rows. expected is None for OOS files."""
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            text = (r.get("utterance") or "").strip()
            if not text:
                continue
            expected = r.get(label_col, "").strip() if label_col else None
            rows.append((text, expected))
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"Rescue threshold (default {DEFAULT_THRESHOLD})")
    args = ap.parse_args()

    try:
        import coremltools as ct
    except ImportError:
        sys.exit("ERROR: coremltools not installed (macOS only). pip install coremltools")

    fp16_path = MODELS_DIR / "MiniLMEmbedder.mlpackage"
    int8_path = MODELS_DIR / "MiniLMEmbedder_int8.mlpackage"
    if not fp16_path.exists():
        sys.exit(f"ERROR: {fp16_path} not found — run export_coreml.py first.")
    if not int8_path.exists():
        sys.exit(f"ERROR: {int8_path} not found — run export_coreml.py --quantize first.")

    print("=" * 64)
    print(f"  CoreML MiniLM — FP16 vs INT8 accuracy comparison")
    print(f"  threshold = {args.threshold}")
    print("=" * 64)

    tokenise = _load_tokeniser()
    head     = _load_head()
    print(f"  Head: {len(head[2])} classes")

    print("  Loading FP16 model ...")
    fp16 = ct.models.MLModel(str(fp16_path))
    print("  Loading INT8 model ...")
    int8 = ct.models.MLModel(str(int8_path))

    fp16_mb = sum(f.stat().st_size for f in fp16_path.rglob("*") if f.is_file()) / 1e6
    int8_mb = sum(f.stat().st_size for f in int8_path.rglob("*") if f.is_file()) / 1e6
    print(f"  Size: FP16 {fp16_mb:.1f} MB  vs  INT8 {int8_mb:.1f} MB "
          f"({100*(1-int8_mb/fp16_mb):.0f}% smaller)\n")

    holdout = _load_csv(DATA_DIR / "semantic_holdout_100.csv", label_col="expected_intent")
    oos     = _load_csv(DATA_DIR / "semantic_oos.csv")

    cosines, flips = [], []
    fp16_hold_ok = int8_hold_ok = 0
    fp16_oos_rej = int8_oos_rej = 0

    # In-scope holdout
    print(f"{BOLD}{CYAN}In-scope holdout ({len(holdout)} cases){RESET}")
    for text, expected in holdout:
        v16 = _embed(fp16, tokenise, text)
        v8  = _embed(int8, tokenise, text)
        cosines.append(float(np.dot(v16, v8)))
        d16, c16 = _classify(v16, head, args.threshold)
        d8,  c8  = _classify(v8,  head, args.threshold)
        if d16 == expected: fp16_hold_ok += 1
        if d8  == expected: int8_hold_ok += 1
        if d16 != d8:
            flips.append((text, d16, c16, d8, c8))

    # Out-of-scope (correct answer is rejection → GENAI)
    print(f"{BOLD}{CYAN}Out-of-scope ({len(oos)} cases){RESET}")
    for text, _ in oos:
        v16 = _embed(fp16, tokenise, text)
        v8  = _embed(int8, tokenise, text)
        cosines.append(float(np.dot(v16, v8)))
        d16, c16 = _classify(v16, head, args.threshold)
        d8,  c8  = _classify(v8,  head, args.threshold)
        if d16 == "GENAI": fp16_oos_rej += 1
        if d8  == "GENAI": int8_oos_rej += 1
        if d16 != d8:
            flips.append((text, d16, c16, d8, c8))

    # ── Report ────────────────────────────────────────────────────────────
    nh, no = len(holdout), len(oos)
    a16 = 100 * fp16_hold_ok / nh; a8 = 100 * int8_hold_ok / nh
    r16 = 100 * fp16_oos_rej / no; r8 = 100 * int8_oos_rej / no

    def delta(x, y):
        d = y - x
        col = GREEN if d >= -0.05 else (YELLOW if d >= -1.0 else RED)
        return f"{col}{d:+.1f}{RESET}"

    print(f"\n{BOLD}{'─'*64}{RESET}")
    print(f"{BOLD}  Results{RESET}")
    print(f"{BOLD}{'─'*64}{RESET}")
    print(f"  {'Metric':<32}{'FP16':>10}{'INT8':>10}{'Δ':>12}")
    print(f"  {'in-scope accuracy (%)':<32}{a16:>10.1f}{a8:>10.1f}{delta(a16,a8):>21}")
    print(f"  {'OOS rejection (%)':<32}{r16:>10.1f}{r8:>10.1f}{delta(r16,r8):>21}")
    print(f"\n  embedding cosine FP16↔INT8: "
          f"mean={np.mean(cosines):.4f}  min={np.min(cosines):.4f}")
    print(f"  decision flips: {len(flips)} / {nh+no}")

    if flips:
        print(f"\n{YELLOW}  Flipped decisions (first 15):{RESET}")
        for text, d16, c16, d8, c8 in flips[:15]:
            print(f"    \"{text[:40]:<40}\"  FP16={d16}({c16:.2f})  →  INT8={d8}({c8:.2f})")

    # ── Verdict ──────────────────────────────────────────────────────────
    acc_drop = a16 - a8
    oos_drop = r16 - r8
    print(f"\n{BOLD}{'='*64}{RESET}")
    if acc_drop <= 1.0 and oos_drop <= 1.0:
        print(f"{BOLD}{GREEN}  VERDICT: INT8 acceptable "
              f"(accuracy Δ {-acc_drop:+.1f}%, OOS Δ {-oos_drop:+.1f}%).{RESET}")
        print(f"  Ship INT8 to halve the bundle (~{fp16_mb-int8_mb:.0f} MB saved).")
    else:
        print(f"{BOLD}{RED}  VERDICT: keep FP16 "
              f"(accuracy Δ {-acc_drop:+.1f}%, OOS Δ {-oos_drop:+.1f}%).{RESET}")
        print(f"  INT8 regression exceeds the 1% bar; the size win isn't worth it.")
    print(f"{BOLD}{'='*64}{RESET}\n")


if __name__ == "__main__":
    main()
