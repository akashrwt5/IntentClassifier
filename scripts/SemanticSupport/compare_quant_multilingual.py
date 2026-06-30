#!/usr/bin/env python3
"""
Compare FP32 vs INT8 multilingual MiniLM models across three dimensions:

  1. Embedding drift    — cosine similarity between FP32 and INT8 embeddings
                          per language; shows how much quantization shifts the
                          vector space the head was trained on.
  2. Head accuracy      — classify() output compared between FP32 and INT8
                          embeddings on held-out phrases; shows real-world
                          intent prediction divergence.
  3. Latency            — wall-clock time per embedding call (single-sentence,
                          batch-of-32) to quantify the server-side speedup.

Decision guidance:
  cosine mean ≥ 0.999  → negligible drift, INT8 is safe to use
  cosine mean ≥ 0.995  → very small drift, INT8 is acceptable
  cosine mean < 0.990  → noticeable drift, measure head accuracy before deploying
  accuracy delta > 2%  → retrain head on INT8 embeddings before deploying

Run:
    python scripts/SemanticSupport/compare_quant_multilingual.py
    python scripts/SemanticSupport/compare_quant_multilingual.py --n-samples 200
    python scripts/SemanticSupport/compare_quant_multilingual.py --languages fr de
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR  = Path(__file__).parent.parent.parent
MODEL_DIR = BASE_DIR / "multilingual" / "SemanticSupport" / "models"
DATA_DIR  = BASE_DIR / "multilingual" / "data"

sys.path.insert(0, str(BASE_DIR))

FP32_ONNX = MODEL_DIR / "paraphrase-multilingual-minilm-l12-v2.onnx"
INT8_ONNX = MODEL_DIR / "paraphrase-multilingual-minilm-l12-v2.int8.onnx"
VOCAB_PATH = MODEL_DIR / "multilingual-vocab.txt"
HEAD_PATH  = MODEL_DIR / "semantic_head_multilingual.npz"

SEQ_LEN = 64

parser = argparse.ArgumentParser(description="Compare FP32 vs INT8 multilingual embeddings")
parser.add_argument("--n-samples",  type=int, default=100,
                    help="Phrases per language to sample for comparison (default: 100)")
parser.add_argument("--languages",  nargs="+", default=["fr", "de", "da"],
                    choices=["fr", "de", "da", "en"],
                    help="Languages to include (default: fr de da)")
parser.add_argument("--latency-reps", type=int, default=50,
                    help="Repetitions for latency benchmark (default: 50)")
parser.add_argument("--batch-size",   type=int, default=32,
                    help="Batch size for throughput latency test (default: 32)")
args = parser.parse_args()


# ── Embedding utilities ───────────────────────────────────────────────────────

def _make_session(onnx_path: Path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), opts)


def _embed_batch(sess, tokeniser, texts: list) -> np.ndarray:
    """Return L2-normalised (N, 384) float32 embeddings."""
    batch_ids, batch_mask, batch_types = [], [], []
    for text in texts:
        ids, mask, types = tokeniser.encode(text)
        batch_ids.append(ids[0])
        batch_mask.append(mask[0])
        batch_types.append(types[0])

    input_ids      = np.stack(batch_ids,   axis=0).astype(np.int64)
    attention_mask = np.stack(batch_mask,  axis=0).astype(np.int64)
    token_type_ids = np.stack(batch_types, axis=0).astype(np.int64)

    out      = sess.run(None, {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    })
    token_emb = out[0]                                                   # (N, 64, 384)
    expanded  = attention_mask[:, :, np.newaxis].astype(np.float32)     # (N, 64, 1)
    summed    = (token_emb * expanded).sum(axis=1)                      # (N, 384)
    counts    = np.where(expanded.sum(axis=1) == 0, 1.0,
                         expanded.sum(axis=1))                           # (N, 1)
    vecs      = summed / counts
    norms     = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms     = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype(np.float32)


def _classify_batch(vecs: np.ndarray, weights, bias, labels) -> list:
    logits = vecs @ weights.T + bias[np.newaxis, :]
    logits -= logits.max(axis=1, keepdims=True)
    probs   = np.exp(logits)
    probs  /= probs.sum(axis=1, keepdims=True)
    top_idx = np.argmax(probs, axis=1)
    return [(str(labels[i]), float(probs[r, i])) for r, i in enumerate(top_idx)]


# ── Load data ─────────────────────────────────────────────────────────────────

def load_samples(languages: list, n_per_lang: int) -> list:
    """Return up to n_per_lang (text, intent, lang) tuples per language."""
    import pandas as pd
    rows = []
    for lang in languages:
        csv = DATA_DIR / f"{lang}.csv"
        if not csv.exists():
            print(f"  [WARN] {csv} not found — skipping {lang}")
            continue
        df = pd.read_csv(csv, encoding="utf-8-sig")
        df.columns = [c.strip().lower() for c in df.columns]
        text_col = next(
            (c for c in df.columns if c in ("text", "utterance", "phrase", "sentence")),
            df.columns[0]
        )
        df = df.rename(columns={text_col: "text"})
        df = df[df["intent"] != "Default Fallback Intent"].dropna(subset=["text", "intent"])
        sample = df.sample(min(n_per_lang, len(df)), random_state=42)
        for _, row in sample.iterrows():
            rows.append((str(row["text"]), str(row["intent"]), lang))
    return rows


# ── Latency benchmark ─────────────────────────────────────────────────────────

def benchmark_latency(sess, tokeniser, probe_text: str, reps: int,
                      batch_texts: list) -> dict:
    # Single-sentence latency (warm up first)
    ids, mask, types = tokeniser.encode(probe_text)
    for _ in range(5):  # warm-up
        sess.run(None, {"input_ids": ids, "attention_mask": mask,
                        "token_type_ids": types})
    t0 = time.perf_counter()
    for _ in range(reps):
        sess.run(None, {"input_ids": ids, "attention_mask": mask,
                        "token_type_ids": types})
    single_ms = (time.perf_counter() - t0) / reps * 1000

    # Batch throughput
    _embed_batch(sess, tokeniser, batch_texts[:1])  # warm-up
    t0 = time.perf_counter()
    for _ in range(max(1, reps // 5)):
        _embed_batch(sess, tokeniser, batch_texts)
    batch_ms = (time.perf_counter() - t0) / max(1, reps // 5) * 1000

    return {"single_ms": single_ms, "batch_ms": batch_ms, "batch_size": len(batch_texts)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 64)
    print("  FP32 vs INT8 Multilingual MiniLM Comparison")
    print("=" * 64)

    # Prerequisite check
    for label, path in [("FP32 ONNX", FP32_ONNX), ("INT8 ONNX", INT8_ONNX),
                         ("Vocab",    VOCAB_PATH)]:
        if not path.exists():
            print(f"\nERROR: {label} not found: {path}")
            if path == FP32_ONNX:
                print("  Run: python scripts/SemanticSupport/download_models.py")
            elif path == INT8_ONNX:
                print("  Run: python scripts/SemanticSupport/quantize_multilingual.py")
            sys.exit(1)

    head_available = HEAD_PATH.exists()
    if not head_available:
        print("\n  NOTE: semantic_head_multilingual.npz not found — "
              "head accuracy comparison will be skipped.")
        print("  Run train_multilingual_semantic_head.py to enable it.\n")

    from multilingual.SemanticSupport.tokeniser import MultilingualTokeniser
    tokeniser = MultilingualTokeniser(VOCAB_PATH)

    print("\nLoading ONNX sessions ...")
    fp32_sess = _make_session(FP32_ONNX)
    int8_sess = _make_session(INT8_ONNX)
    print(f"  FP32: {FP32_ONNX.name}  ({FP32_ONNX.stat().st_size / 1e6:.1f} MB)")
    print(f"  INT8: {INT8_ONNX.name}  ({INT8_ONNX.stat().st_size / 1e6:.1f} MB)")
    print(f"  Size reduction: {(1 - INT8_ONNX.stat().st_size / FP32_ONNX.stat().st_size) * 100:.0f}%")

    # ── 1. Embedding drift ────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print("  1. Embedding drift (cosine similarity FP32 vs INT8)")
    print(f"{'─'*64}")
    print(f"  Sampling {args.n_samples} phrases per language: {args.languages}")

    samples = load_samples(args.languages, args.n_samples)
    if not samples:
        print("  ERROR: No data loaded. Check DATA_DIR paths.")
        sys.exit(1)

    texts    = [s[0] for s in samples]
    intents  = [s[1] for s in samples]
    langs    = [s[2] for s in samples]

    print(f"  Total phrases: {len(texts)}")
    print("  Embedding with FP32 ...")
    fp32_vecs = _embed_batch(fp32_sess, tokeniser, texts)
    print("  Embedding with INT8 ...")
    int8_vecs = _embed_batch(int8_sess, tokeniser, texts)

    # Cosine similarity (both are L2-normalised so dot product = cosine)
    cosines = (fp32_vecs * int8_vecs).sum(axis=1)

    print(f"\n  Overall cosine statistics ({len(cosines)} phrases):")
    print(f"    Mean   : {cosines.mean():.5f}")
    print(f"    Std    : {cosines.std():.5f}")
    print(f"    Min    : {cosines.min():.5f}")
    print(f"    p5     : {np.percentile(cosines, 5):.5f}")
    print(f"    p95    : {np.percentile(cosines, 95):.5f}")
    print(f"    Max    : {cosines.max():.5f}")

    print(f"\n  Per-language breakdown:")
    lang_cosines = {}
    for lang in args.languages:
        mask = np.array([l == lang for l in langs])
        if mask.sum() == 0:
            continue
        lc = cosines[mask]
        lang_cosines[lang] = lc.mean()
        print(f"    [{lang}] mean={lc.mean():.5f}  std={lc.std():.5f}  "
              f"min={lc.min():.5f}  n={mask.sum()}")

    # Interpret drift
    overall_mean = cosines.mean()
    print(f"\n  Interpretation:")
    if overall_mean >= 0.999:
        print(f"    ✅ mean cosine {overall_mean:.5f} — negligible drift. INT8 is safe.")
    elif overall_mean >= 0.995:
        print(f"    ✅ mean cosine {overall_mean:.5f} — very small drift. INT8 acceptable.")
    elif overall_mean >= 0.990:
        print(f"    ⚠  mean cosine {overall_mean:.5f} — moderate drift. "
              "Verify head accuracy before deploying.")
    else:
        print(f"    ❌ mean cosine {overall_mean:.5f} — significant drift. "
              "Consider retraining head on INT8 embeddings.")

    # ── 2. Head accuracy comparison ───────────────────────────────────────
    if head_available:
        print(f"\n{'─'*64}")
        print("  2. Head accuracy — intent prediction divergence")
        print(f"{'─'*64}")

        data   = np.load(HEAD_PATH, allow_pickle=True)
        W      = data["weights"].astype(np.float32)
        b      = data["bias"].astype(np.float32)
        labels = data["labels"]

        fp32_preds = _classify_batch(fp32_vecs, W, b, labels)
        int8_preds = _classify_batch(int8_vecs, W, b, labels)

        fp32_intents  = [p[0] for p in fp32_preds]
        int8_intents  = [p[0] for p in int8_preds]
        agree_overall = np.mean([a == b_ for a, b_ in zip(fp32_intents, int8_intents)])

        # Accuracy of each model against ground-truth labels
        fp32_acc = np.mean([a == t for a, t in zip(fp32_intents, intents)])
        int8_acc = np.mean([a == t for a, t in zip(int8_intents, intents)])

        print(f"\n  FP32 accuracy vs ground truth : {fp32_acc:.3f}")
        print(f"  INT8 accuracy vs ground truth : {int8_acc:.3f}")
        print(f"  Accuracy delta (FP32 - INT8)  : {fp32_acc - int8_acc:+.3f}")
        print(f"  FP32 / INT8 prediction agreement: {agree_overall:.3f} "
              f"({int(agree_overall * len(texts))}/{len(texts)} match)")

        # Per-language accuracy breakdown
        print(f"\n  Per-language accuracy (FP32 / INT8):")
        for lang in args.languages:
            mask = [i for i, l in enumerate(langs) if l == lang]
            if not mask:
                continue
            fp32_l = np.mean([fp32_intents[i] == intents[i] for i in mask])
            int8_l = np.mean([int8_intents[i] == intents[i] for i in mask])
            agree_l = np.mean([fp32_intents[i] == int8_intents[i] for i in mask])
            delta_marker = "⚠" if abs(fp32_l - int8_l) > 0.02 else "✅"
            print(f"    [{lang}]  FP32={fp32_l:.3f}  INT8={int8_l:.3f}  "
                  f"delta={fp32_l - int8_l:+.3f}  agree={agree_l:.3f}  {delta_marker}")

        print(f"\n  Interpretation:")
        delta = fp32_acc - int8_acc
        if abs(delta) <= 0.005:
            print(f"    ✅ delta {delta:+.3f} — accuracy impact negligible.")
        elif abs(delta) <= 0.02:
            print(f"    ✅ delta {delta:+.3f} — small accuracy impact, INT8 acceptable.")
        else:
            print(f"    ❌ delta {delta:+.3f} — noticeable accuracy drop. "
                  "Retrain head on INT8 embeddings:\n"
                  "       python scripts/SemanticSupport/train_multilingual_semantic_head.py "
                  "--use-int8")

        # Disagreement deep-dive: show cases where FP32 and INT8 disagree
        disagreements = [
            (texts[i], intents[i], fp32_intents[i], int8_intents[i], langs[i])
            for i in range(len(texts))
            if fp32_intents[i] != int8_intents[i]
        ]
        if disagreements:
            print(f"\n  Cases where FP32 and INT8 disagree "
                  f"({len(disagreements)}/{len(texts)}):")
            for text, true_intent, fp32_i, int8_i, lang in disagreements[:10]:
                mark = "✗" if fp32_i == true_intent else ("✓" if int8_i == true_intent else "?")
                print(f"    [{lang}] {mark} \"{text[:55]:<55}\"")
                print(f"           FP32={fp32_i!r}  INT8={int8_i!r}  true={true_intent!r}")
            if len(disagreements) > 10:
                print(f"    ... and {len(disagreements) - 10} more")

    # ── 3. Latency benchmark ──────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print("  3. Latency benchmark")
    print(f"{'─'*64}")

    probe   = texts[0]
    batched = texts[:args.batch_size]

    print(f"  Benchmarking over {args.latency_reps} reps "
          f"(batch size={args.batch_size}, single probe={probe[:40]!r}) ...")

    fp32_lat = benchmark_latency(fp32_sess, tokeniser, probe, args.latency_reps, batched)
    int8_lat = benchmark_latency(int8_sess, tokeniser, probe, args.latency_reps, batched)

    speedup_single = fp32_lat["single_ms"] / int8_lat["single_ms"]
    speedup_batch  = fp32_lat["batch_ms"]  / int8_lat["batch_ms"]

    print(f"\n  Single-sentence inference:")
    print(f"    FP32 : {fp32_lat['single_ms']:.2f} ms")
    print(f"    INT8 : {int8_lat['single_ms']:.2f} ms")
    print(f"    Speedup: {speedup_single:.2f}x")

    print(f"\n  Batch inference (N={args.batch_size}):")
    print(f"    FP32 : {fp32_lat['batch_ms']:.1f} ms  "
          f"({fp32_lat['batch_ms'] / args.batch_size:.2f} ms/sentence)")
    print(f"    INT8 : {int8_lat['batch_ms']:.1f} ms  "
          f"({int8_lat['batch_ms'] / args.batch_size:.2f} ms/sentence)")
    print(f"    Speedup: {speedup_batch:.2f}x")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("  SUMMARY")
    print(f"{'='*64}")
    print(f"  File size  : FP32={FP32_ONNX.stat().st_size/1e6:.1f} MB → "
          f"INT8={INT8_ONNX.stat().st_size/1e6:.1f} MB  "
          f"({(1-INT8_ONNX.stat().st_size/FP32_ONNX.stat().st_size)*100:.0f}% smaller)")
    print(f"  Emb drift  : mean cosine={overall_mean:.5f}")
    if head_available:
        print(f"  Accuracy   : FP32={fp32_acc:.3f}  INT8={int8_acc:.3f}  "
              f"delta={fp32_acc-int8_acc:+.3f}")
    print(f"  Speedup    : {speedup_single:.2f}x single  /  {speedup_batch:.2f}x batch")

    recommendation = (
        "✅ Use INT8 for server inference — accuracy loss within acceptable bounds."
        if (overall_mean >= 0.995 and (not head_available or abs(fp32_acc - int8_acc) <= 0.02))
        else "⚠  Review accuracy delta before switching to INT8 in production."
    )
    print(f"\n  Recommendation: {recommendation}")
    print()


if __name__ == "__main__":
    main()
