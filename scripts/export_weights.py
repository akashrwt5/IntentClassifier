#!/usr/bin/env python3
"""
Export the Swift/CoreML weights from the SAME fitted pipeline that train.py
produced (models/intent_pipeline.pkl).

WHY: previously this script re-TRAINED its own TF-IDF+LR (different data, min_df,
even upper-cased labels), so the iOS weights and the Android ONNX were two
DIFFERENT models — a silent cross-platform drift. Deriving both from the one
pipeline guarantees they are the same model: same vocab, idf, coef, intercept,
labels, and TF-IDF config. Run `python scripts/train.py` first.

Usage:
    python scripts/export_weights.py [--pipeline models/intent_pipeline.pkl] [--out PATH]
"""
import argparse
import json
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).parent.parent
PIPELINE_PATH = BASE_DIR / "models" / "intent_pipeline.pkl"
DEFAULT_OUT = BASE_DIR / "models" / "intent_classifier_weights.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default=str(PIPELINE_PATH))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    pipe = joblib.load(args.pipeline)
    tfidf = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]

    # Preserve calibration/config metadata already present in a prior weights file.
    out = Path(args.out)
    prev = {}
    if out.exists():
        try:
            prev = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            prev = {}

    weights = {
        "labels": [str(c) for c in clf.classes_.tolist()],
        "vocab": {k: int(v) for k, v in tfidf.vocabulary_.items()},
        "idf": [round(float(x), 6) for x in tfidf.idf_.tolist()],
        "coef": [[round(float(x), 6) for x in row] for row in clf.coef_.tolist()],
        "intercept": [round(float(x), 6) for x in clf.intercept_.tolist()],
        # TF-IDF config so the Swift side / parity reference featurize identically.
        "ngram_range": list(tfidf.ngram_range),
        "sublinear_tf": bool(tfidf.sublinear_tf),
        "normalize": tfidf.norm or "l2",
        # calibration + app config (carried over, defaulted if absent).
        "temperature": float(prev.get("temperature", 1.0)),
        "conf_threshold": float(prev.get("conf_threshold", 0.70)),
        "conf_gap_threshold": float(prev.get("conf_gap_threshold", 0.20)),
        "genai_base_url": prev.get("genai_base_url", "https://genai.yourcompany.com/chat?query="),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(weights, f, separators=(",", ":"))

    print(f"✅ Exported {out} FROM {args.pipeline}")
    print(f"   labels={len(weights['labels'])}  vocab={len(weights['vocab'])}  "
          f"ngram={weights['ngram_range']}  sublinear={weights['sublinear_tf']}  "
          f"norm={weights['normalize']}")
    print(f"   first labels: {weights['labels'][:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
