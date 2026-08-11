#!/usr/bin/env python3
"""
Install a trained student as the engine's Stage 3.

Copies the four artifacts into `models/semantic_student/<lang>/`, which is where
`NLUEngine._load_semantic` looks BEFORE falling back to the 23 MB MiniLM stage.

    student.onnx   the graph, self-contained (no .data sidecar)
    vocab.json     the exact vocabulary the model was trained with
    labels.json    logit column order
    meta.json      max_len + threshold + temperature + provenance

Then verifies the installed copy answers identically to the source, so a wrong
vocab or label order fails here rather than in production.

Stage 3 also has to be ENABLED. It is currently off:

    language_packs/en/nlu_schema.json -> "semantic_rescue_enabled": false

This script does not flip that — turning the stage on is a behaviour change that
should be a deliberate, separate commit, and it should follow a cascade
evaluation, not precede it.

Usage:
    python scripts/install_student.py --tag subw_vol5_s1 --threshold 0.40
    python scripts/install_student.py --tag subw_vol5_s1 --threshold 0.40 --temperature 0.68 --lang en
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.common import load_rows, load_vocab  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEST_ROOT = REPO / "models" / "semantic_student"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument(
        "--temperature", type=float, default=0.68,
        help="offline-fitted softmax temperature written to meta.json (default: 0.68)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    onnx = config.MODELS / f"student_{args.tag}.onnx"
    vocab_p = config.MODELS / f"vocab_{args.tag}.json"
    labels_p = config.MODELS / f"labels_{args.tag}.json"
    train_p = config.REPORTS / f"train_{args.tag}_summary.json"

    if args.temperature <= 0:
        raise SystemExit("--temperature must be > 0")

    if not onnx.exists():
        raise SystemExit(
            f"{onnx} not found. Export first:\n"
            f"  python scripts/export_onnx.py --tag {args.tag} "
            f"--threshold {args.threshold} --skip-int8"
        )
    sidecar = onnx.with_name(onnx.name + ".data")
    if sidecar.exists():
        raise SystemExit(
            f"ABORT: {sidecar.name} exists — the graph's weights are in an "
            f"external file. Re-export with the current export_onnx.py, which "
            f"folds them back in."
        )

    meta_src = json.loads(train_p.read_text(encoding="utf-8")) if train_p.exists() else {}
    tokenizer = meta_src.get("tokenizer", "word")
    if tokenizer not in {"word", "subword"}:
        raise SystemExit(f"ABORT: unsupported tokenizer mode: {tokenizer!r}")

    dest = DEST_ROOT / args.lang
    print(f"source : {onnx.name}  ({onnx.stat().st_size / 1e6:.3f} MB)")
    print(f"dest   : {dest}")
    if args.dry_run:
        print("\n(dry run — nothing copied)")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx, dest / "student.onnx")
    shutil.copy2(vocab_p, dest / "vocab.json")
    shutil.copy2(labels_p, dest / "labels.json")

    meta = {
        "tag": args.tag,
        "max_len": meta_src.get("max_len", config.MAX_LEN),
        "threshold": args.threshold,
        "tokenizer": tokenizer,
        "temperature": args.temperature,
        "vocab_size": meta_src.get("vocab_size"),
        "seed": meta_src.get("seed"),
        "teacher": meta_src.get("teacher"),
        "init_embeddings": meta_src.get("init_embeddings"),
        "freeze_embeddings": meta_src.get("freeze_embeddings"),
        "synthetic_rows": meta_src.get("synthetic_rows", 0),
        "synthetic_text": meta_src.get("synthetic_text", False),
        "source": str(onnx),
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---- verify the INSTALLED copy, through the engine's own class ----
    # Loaded from the file directly rather than via `nlu_engine.semantic`: the
    # package __init__ pulls in the whole engine (joblib, sklearn, a language
    # pack), none of which this check needs. Verifying the runtime class must
    # not require the runtime's entire dependency tree.
    import importlib.util

    _sem_path = REPO / "packages" / "runtime" / "nlu_engine" / "semantic.py"
    _spec = importlib.util.spec_from_file_location("_nlu_semantic", _sem_path)
    _sem = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_sem)
    StudentSemantic = _sem.StudentSemantic

    stage = StudentSemantic(dest, threshold=args.threshold)

    import onnxruntime as ort

    vocab, tokenizer_mode = load_vocab(vocab_p)
    labels = json.loads(labels_p.read_text(encoding="utf-8"))
    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    max_len = meta["max_len"]

    texts = []
    for p in (config.LOCKED_TEST, config.STRESS_TEST, config.OOD_TEST):
        if p.exists():
            texts += [t for t, _ in load_rows(p)]

    from scripts.common import encode

    mismatch = 0
    max_delta = 0.0
    for t in texts:
        ids, _ = encode(t, vocab, max_len, tokenizer_mode)
        X = np.array([ids], dtype=np.int64)
        M = X != config.PAD_ID
        ref = sess.run(None, {"input_ids": X, "attention_mask": M})[0][0]
        z = ref / meta["temperature"]
        z = z - z.max()
        e = np.exp(z)
        ref_p = e / e.sum()
        got_label, got_conf = stage.classify(t)
        if got_label != labels[int(ref_p.argmax())]:
            mismatch += 1
        max_delta = max(max_delta, abs(got_conf - float(ref_p.max())))

    print(f"\nverified on {len(texts)} utterances via nlu_engine.semantic.StudentSemantic")
    print(f"  label mismatches   {mismatch}")
    print(f"  max |conf delta|   {max_delta:.3e}")
    if mismatch:
        raise SystemExit(
            "ABORT: the installed copy disagrees with the source model. The "
            "engine's tokenizer or label order does not match training."
        )

    print(
        f"\ninstalled {stage.threshold} gate, {len(stage.labels)} intents, "
        f"{len(stage.vocab)} vocab"
    )
    print(f"\nStage 3 is still DISABLED. To turn it on:")
    print(f"  language_packs/{args.lang}/nlu_schema.json -> " f'"semantic_rescue_enabled": true')
    print("Do that AFTER running evaluate_cascade.py, not before.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
