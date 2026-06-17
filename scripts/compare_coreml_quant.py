#!/usr/bin/env python3
"""
Three-way comparison: ONNX (reference) vs FP16 CoreML vs Palettized CoreML.

The CRITICAL baseline is the ONNX embedder -- that is what the Python NLU
pipeline uses, what the semantic head was trained against, and what the
semantic_holdout_100.csv labels were generated with. Comparing only FP16
CoreML vs palettized CoreML can mask drift that both variants share:
if CoreML FP16 is already degraded vs ONNX, both CoreML variants look fine
relative to each other while both silently underperform the Python pipeline.

What this script reports
------------------------
  Column 1 - ONNX (onnxruntime, INT8): the Python production reference
  Column 2 - FP16 CoreML:              what the iOS app runs today
  Column 3 - Palettized CoreML:        the size-optimised candidate

For each column:
  * In-scope accuracy on semantic_holdout_100.csv
  * Out-of-scope rejection rate on semantic_oos.csv

Plus per-pair embedding cosine similarity and decision flip counts so you
can see exactly where FP16 CoreML drifts from ONNX (conversion error) and
where palettization additionally drifts from FP16 (compression error).

macOS only (CoreML prediction requires the macOS runtime).

Usage:
    pip install coremltools onnxruntime numpy
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


# -- Tokeniser -----------------------------------------------------------------

def _load_tokeniser():
    """Shared WordPiece tokeniser -- same logic for all three embedders."""
    sys.path.insert(0, str(BASE_DIR))
    from scripts.nlu.semantic import SemanticFallback

    vocab_path = MODELS_DIR / "minilm-vocab.txt"
    if not vocab_path.exists():
        sys.exit(f"ERROR: {vocab_path} not found -- run scripts/download_minilm.py")
    with open(vocab_path, encoding="utf-8") as f:
        vocab = {line.strip(): i for i, line in enumerate(f)}

    def tokenise(text, max_len=64):
        tokens = ["[CLS]"] + SemanticFallback._wordpiece(text.lower(), vocab) + ["[SEP]"]
        tokens = tokens[:max_len]
        ids = [vocab.get(t, vocab["[UNK]"]) for t in tokens]
        n = len(ids)
        return ids, n

    return tokenise, vocab


# -- Embedders -----------------------------------------------------------------

def _embed_onnx(session, tokenise_fn, text):
    """Embed via the ONNX model -- matches SemanticFallback._embed_onnx exactly."""
    ids, n = tokenise_fn(text)
    ids_arr  = np.array([ids], dtype=np.int64)
    mask_arr = np.ones((1, n), dtype=np.int64)
    tids_arr = np.zeros((1, n), dtype=np.int64)
    outputs = session.run(None, {
        "input_ids":      ids_arr,
        "attention_mask": mask_arr,
        "token_type_ids": tids_arr,
    })
    token_emb = outputs[0][0]                           # [seq, 384]
    mask_col  = mask_arr[0][:, np.newaxis].astype(np.float32)
    vec = (token_emb * mask_col).sum(axis=0) / mask_col.sum()
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)


def _embed_coreml(model, tokenise_fn, text):
    """Embed via a CoreML model (FP16 or palettized)."""
    ids, n = tokenise_fn(text)
    ids_arr  = np.array([ids], dtype=np.int32)
    mask_arr = np.ones((1, n), dtype=np.int32)
    tids_arr = np.zeros((1, n), dtype=np.int32)
    out = model.predict({
        "input_ids":      ids_arr,
        "attention_mask": mask_arr,
        "token_type_ids": tids_arr,
    })
    key = next(iter(out))
    token_emb = np.array(out[key])[0]                  # [seq, 384]
    mask_col  = mask_arr[0][:, np.newaxis].astype(np.float32)
    vec = (token_emb * mask_col).sum(axis=0) / mask_col.sum()
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)


# -- Head classifier -----------------------------------------------------------

def _load_head(head_path=None):
    js  = Path(head_path) if head_path else MODELS_DIR / "semantic_head.json"
    npz = MODELS_DIR / "semantic_head.npz"
    if js.exists():
        d = json.loads(js.read_text())
        return (np.array(d["weights"], dtype=np.float32),
                np.array(d["bias"],    dtype=np.float32),
                [str(x) for x in d["labels"]])
    if head_path:
        sys.exit(f"ERROR: head file {js} not found.")
    if npz.exists():
        d = np.load(npz, allow_pickle=True)
        return (d["weights"].astype(np.float32),
                d["bias"].astype(np.float32),
                [str(x) for x in d["labels"]])
    sys.exit("ERROR: neither semantic_head.json nor semantic_head.npz found.")


def _classify(vec, head, threshold):
    weights, bias, labels = head
    logits = weights @ vec + bias
    logits -= logits.max()
    probs = np.exp(logits); probs /= probs.sum()
    top = int(np.argmax(probs))
    label, conf = labels[top], float(probs[top])
    if label == FALLBACK_INTENT or conf < threshold:
        return "GENAI", conf
    return label, conf


# -- Dataset loading -----------------------------------------------------------

def _load_csv(path, label_col=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            text = (r.get("utterance") or r.get("text") or "").strip()
            if not text:
                continue
            expected = r.get(label_col, "").strip() if label_col else None
            rows.append((text, expected))
    return rows


# -- Main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"Rescue threshold (default {DEFAULT_THRESHOLD})")
    ap.add_argument("--head", default=None,
                    help="Path to a semantic head JSON to evaluate (default: "
                         "models/semantic_head.json). Use this to test a "
                         "CoreML-retrained head, e.g. "
                         "--head models/semantic_head_coreml.json")
    args = ap.parse_args()

    try:
        import coremltools as ct
    except ImportError:
        sys.exit("ERROR: coremltools not installed (macOS only). pip install coremltools")

    onnx_path = MODELS_DIR / "minilm-l6-v2.onnx"
    fp16_path = MODELS_DIR / "MiniLMEmbedder.mlpackage"
    pal_path  = MODELS_DIR / "MiniLMEmbedder_int8.mlpackage"

    if not fp16_path.exists():
        sys.exit(f"ERROR: {fp16_path} not found -- run export_coreml.py first.")
    if not pal_path.exists():
        sys.exit(f"ERROR: {pal_path} not found -- run export_coreml.py --quantize first.")

    print("=" * 70)
    print(f"  Three-way comparison: ONNX | FP16 CoreML | Palettized CoreML")
    print(f"  threshold = {args.threshold}")
    print("=" * 70)

    tokenise_fn, _ = _load_tokeniser()
    head = _load_head(args.head)
    head_name = args.head if args.head else "semantic_head.json"
    print(f"  Head: {len(head[2])} classes  ({head_name})")

    # ONNX session (the Python production reference)
    onnx_session = None
    if onnx_path.exists():
        try:
            import onnxruntime as ort
            onnx_session = ort.InferenceSession(str(onnx_path))
            print(f"  ONNX model: {onnx_path.name} (reference)")
        except ImportError:
            print(f"  WARN: onnxruntime not installed -- skipping ONNX column.")
            print(f"        pip install onnxruntime")
    else:
        print(f"  WARN: {onnx_path} not found -- skipping ONNX column.")
        print(f"        Run: python scripts/download_minilm.py")

    # CoreML models -- CPU only to avoid Metal MLIR crash on Mac during prediction
    print("  Loading FP16 CoreML (cpuOnly) ...")
    fp16 = ct.models.MLModel(str(fp16_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    print("  Loading palettized CoreML (cpuOnly) ...")
    pal  = ct.models.MLModel(str(pal_path),  compute_units=ct.ComputeUnit.CPU_ONLY)

    fp16_mb = sum(f.stat().st_size for f in fp16_path.rglob("*") if f.is_file()) / 1e6
    pal_mb  = sum(f.stat().st_size for f in pal_path.rglob("*")  if f.is_file()) / 1e6
    print(f"  Size: FP16 {fp16_mb:.1f} MB  ->  palettized {pal_mb:.1f} MB "
          f"({100*(1-pal_mb/fp16_mb):.0f}% smaller)\n")

    holdout = _load_csv(DATA_DIR / "semantic_holdout_100.csv", label_col="expected_intent")
    oos     = _load_csv(DATA_DIR / "semantic_oos.csv")

    # Accumulators
    ok   = {"onnx": 0, "fp16": 0, "pal": 0}
    rej  = {"onnx": 0, "fp16": 0, "pal": 0}
    # Pairwise cosine lists: onnx<->fp16, fp16<->pal
    cos_of  = []   # onnx vs fp16
    cos_fp  = []   # fp16 vs pal

    onnx_fp16_flips = []
    fp16_pal_flips  = []

    def embed_all(text):
        vo = _embed_onnx(onnx_session, tokenise_fn, text) if onnx_session else None
        vf = _embed_coreml(fp16, tokenise_fn, text)
        vp = _embed_coreml(pal,  tokenise_fn, text)
        return vo, vf, vp

    print(f"{BOLD}{CYAN}In-scope holdout ({len(holdout)} cases){RESET}")
    for text, expected in holdout:
        vo, vf, vp = embed_all(text)
        do, co = _classify(vo, head, args.threshold) if vo is not None else ("SKIP", 0)
        df, cf = _classify(vf, head, args.threshold)
        dp, cp = _classify(vp, head, args.threshold)

        if do != "SKIP" and do == expected: ok["onnx"] += 1
        if df == expected: ok["fp16"] += 1
        if dp == expected: ok["pal"]  += 1

        if vo is not None:
            cos_of.append(float(np.dot(vo, vf)))
            if do != df:
                onnx_fp16_flips.append((text, do, co, df, cf))
        cos_fp.append(float(np.dot(vf, vp)))
        if df != dp:
            fp16_pal_flips.append((text, df, cf, dp, cp))

    print(f"{BOLD}{CYAN}Out-of-scope ({len(oos)} cases){RESET}")
    for text, _ in oos:
        vo, vf, vp = embed_all(text)
        do, co = _classify(vo, head, args.threshold) if vo is not None else ("SKIP", 0)
        df, cf = _classify(vf, head, args.threshold)
        dp, cp = _classify(vp, head, args.threshold)

        if do != "SKIP" and do == "GENAI": rej["onnx"] += 1
        if df == "GENAI": rej["fp16"] += 1
        if dp == "GENAI": rej["pal"]  += 1

        if vo is not None:
            cos_of.append(float(np.dot(vo, vf)))
            if do != df:
                onnx_fp16_flips.append((text, do, co, df, cf))
        cos_fp.append(float(np.dot(vf, vp)))
        if df != dp:
            fp16_pal_flips.append((text, df, cf, dp, cp))

    # -- Report ------------------------------------------------------------
    nh, no = len(holdout), len(oos)

    def pct(n, d): return 100 * n / d if d else float("nan")
    def col_delta(a, b):
        d = b - a
        c = GREEN if d >= -0.05 else (YELLOW if d >= -1.0 else RED)
        return f"{c}{d:+.1f}{RESET}"

    ao = pct(ok["onnx"], nh); af = pct(ok["fp16"], nh); ap_ = pct(ok["pal"], nh)
    ro = pct(rej["onnx"], no); rf = pct(rej["fp16"], no); rp = pct(rej["pal"], no)

    print(f"\n{BOLD}{'-'*70}{RESET}")
    print(f"{BOLD}  Results{RESET}")
    print(f"{BOLD}{'-'*70}{RESET}")
    onnx_col = f"{'ONNX':>10}" if onnx_session else f"{'(no ONNX)':>10}"
    print(f"  {'Metric':<30}{onnx_col}{'FP16':>10}{'Palette':>10}  "
          f"{'O->F Δ':>8}  {'F->P Δ':>8}")
    print(f"  {'in-scope accuracy (%)':<30}{ao:>10.1f}{af:>10.1f}{ap_:>10.1f}  "
          f"{col_delta(ao,af):>17}  {col_delta(af,ap_):>17}")
    print(f"  {'OOS rejection (%)':<30}{ro:>10.1f}{rf:>10.1f}{rp:>10.1f}  "
          f"{col_delta(ro,rf):>17}  {col_delta(rf,rp):>17}")

    print()
    if cos_of:
        print(f"  ONNX<->FP16 embedding cosine:    "
              f"mean={np.mean(cos_of):.4f}  min={np.min(cos_of):.4f}")
    print(f"  FP16<Palettized cosine:        "
          f"mean={np.mean(cos_fp):.4f}  min={np.min(cos_fp):.4f}")

    if cos_of:
        print(f"\n  ONNX->FP16 decision flips:   {len(onnx_fp16_flips)} / {nh+no}")
    print(f"  FP16->Palettized flips:      {len(fp16_pal_flips)} / {nh+no}")

    if onnx_fp16_flips:
        print(f"\n{YELLOW}  ONNX->FP16 flips (first 10) -- CoreML conversion drift:{RESET}")
        for text, do, co, df, cf in onnx_fp16_flips[:10]:
            print(f"    \"{text[:38]:<38}\"  ONNX={do}({co:.2f})  FP16={df}({cf:.2f})")

    if fp16_pal_flips:
        print(f"\n{YELLOW}  FP16->Palettized flips (first 10) -- compression drift:{RESET}")
        for text, df, cf, dp, cp in fp16_pal_flips[:10]:
            print(f"    \"{text[:38]:<38}\"  FP16={df}({cf:.2f})  Pal={dp}({cp:.2f})")

    # -- Verdicts ---------------------------------------------------------
    print(f"\n{BOLD}{'='*70}{RESET}")

    if cos_of:
        conv_acc_drop = ao - af
        conv_oos_drop = ro - rf
        if conv_acc_drop > 2.0 or conv_oos_drop > 2.0:
            print(f"{BOLD}{RED}  ⚠️  CoreML FP16 has significant drift from ONNX "
                  f"(in-scope Δ {-conv_acc_drop:+.1f}%, OOS Δ {-conv_oos_drop:+.1f}%).{RESET}")
            print(f"  Investigate tokenisation or mean-pool differences before shipping.")
        else:
            print(f"{BOLD}{GREEN}  CoreML FP16 conversion drift: acceptable "
                  f"(in-scope Δ {-conv_acc_drop:+.1f}%, OOS Δ {-conv_oos_drop:+.1f}%).{RESET}")

    pal_acc_drop = af - ap_
    pal_oos_drop = rf - rp
    if pal_acc_drop <= 1.0 and pal_oos_drop <= 1.0:
        print(f"{BOLD}{GREEN}  Palettization drift: acceptable "
              f"(in-scope Δ {-pal_acc_drop:+.1f}%, OOS Δ {-pal_oos_drop:+.1f}%).{RESET}")
        print(f"  Safe to ship palettized model (~{fp16_mb-pal_mb:.0f} MB smaller).")
    else:
        print(f"{BOLD}{RED}  Palettization drift: too large "
              f"(in-scope Δ {-pal_acc_drop:+.1f}%, OOS Δ {-pal_oos_drop:+.1f}%).{RESET}")
        print(f"  Keep FP16 CoreML -- the size win isn't worth the regression.")
    print(f"{BOLD}{'='*70}{RESET}\n")


if __name__ == "__main__":
    main()
