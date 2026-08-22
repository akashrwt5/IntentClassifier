#!/usr/bin/env python3
"""One-time backfill: make the exported artifacts describe themselves.

The exports in output_models/ were written before the artifact contract
existed, so two facts about them live only in the sentence-transformers
directories they came from, or nowhere at all:

  * how the encoder was trained to pool (bge is CLS, the MiniLM students are
    mean) -- present in the source ``1_Pooling/config.json``, absent from the
    export;
  * a vocab.txt consistent with the pruned embedding matrix -- the exports
    carry the original 30,522-line file against a 10,000-row model, which
    Python ignores and a native wordpiece does not.

This script writes ``pooling.json`` beside each model and regenerates
``vocab.txt`` from the pruned ``tokenizer.json`` that already ships there.
It is idempotent and touches nothing else.

Deliberately NOT covered:

  * ``track1_pruned_l3`` -- its tokenizer.json was never pruned (30,522
    entries against a 10,000-row model), so no metadata makes it correct.
    Re-export or retire it; see build_pruned_l3.py.
  * ``track3_svd_l6`` -- superseded experimental track. Its pooling would be
    inferred from the base model rather than read from a source config, and
    this script does not write values it cannot source.

Run from scripts/semantic_compression/:  python3 backfill_artifact_metadata.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_models"

# Each export, and the sentence-transformers directory whose
# 1_Pooling/config.json is the authority on how it pools.
EXPORTS = {
    "final_distilled_onnx": "distilled_minilm_l3",
    "stage2_contrastive_onnx": "stage2_contrastive_minilm_l3",
    "stage2_contrastive_bge_small_onnx": "stage2_contrastive_bge_small_l3",
}


def source_pooling(st_dir: Path) -> str:
    cfg = st_dir / "1_Pooling" / "config.json"
    if not cfg.exists():
        raise SystemExit(f"cannot source pooling: {cfg} not found")
    mode = json.loads(cfg.read_text(encoding="utf-8"))["pooling_mode"]
    if mode not in ("cls", "mean"):
        raise SystemExit(f"{cfg} declares an unsupported pooling_mode: {mode!r}")
    return mode


def write_pooling(model_dir: Path, mode: str, source: str) -> None:
    payload = {
        "pooling_mode": mode,
        "_source": f"{source}/1_Pooling/config.json",
        "_note": (
            "Pooling is a property of how this encoder was trained, not a runtime "
            "choice. Reading the wrong token scores a working encoder near zero."
        ),
    }
    (model_dir / "pooling.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def rewrite_vocab_txt(model_dir: Path) -> tuple[int, int]:
    """Rewrite vocab.txt from the pruned tokenizer.json, in id order."""
    vocab = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))["model"][
        "vocab"
    ]
    ids = sorted(vocab.values())
    if ids != list(range(len(ids))):
        raise SystemExit(
            f"{model_dir}: tokenizer.json ids are not contiguous 0..N-1; "
            "cannot derive a wordpiece vocab.txt from it."
        )
    by_id = sorted(vocab.items(), key=lambda kv: kv[1])
    old = model_dir / "vocab.txt"
    before = len(old.read_text(encoding="utf-8").rstrip("\n").split("\n")) if old.exists() else 0
    old.write_text("\n".join(tok for tok, _ in by_id) + "\n", encoding="utf-8")
    return before, len(by_id)


def main() -> int:
    if not OUT.exists():
        raise SystemExit(f"{OUT} not found -- run this from scripts/semantic_compression/")

    for export, st_dir in EXPORTS.items():
        model_dir = OUT / export
        if not model_dir.exists():
            print(f"skip   {export}  (not present)")
            continue

        mode = source_pooling(HERE / st_dir)
        write_pooling(model_dir, mode, st_dir)
        before, after = rewrite_vocab_txt(model_dir)

        cfg_vocab = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))[
            "vocab_size"
        ]
        status = "OK" if after == cfg_vocab else f"STILL MISMATCHED (model has {cfg_vocab})"
        print(f"write  {export}")
        print(f"         pooling.json  -> {mode}  (from {st_dir})")
        print(f"         vocab.txt     -> {before} lines rewritten to {after}   {status}")

    print("\nNot backfilled, on purpose:")
    print("  track1_pruned_l3   tokenizer.json was never pruned (30,522 vs 10,000 rows).")
    print("                     No metadata fixes that. Re-export or retire.")
    print("  track3_svd_l6      superseded track; pooling would have to be inferred,")
    print("                     not sourced. This script does not write guessed values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
