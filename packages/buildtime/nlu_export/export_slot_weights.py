#!/usr/bin/env python3
"""
Export BIO Slot Tagger LogisticRegression weights to lightweight JSON
for iOS (Swift) and Android (Kotlin) native inference without ML library dependencies.

Usage:
    python -m nlu_export.export_slot_weights --lang en --out models/slot_tagger_weights.json
"""

import argparse
import json
from pathlib import Path
import joblib

REPO_ROOT = Path(__file__).resolve().parents[3]


def export_weights(lang: str, out_path: Path) -> Path:
    src = REPO_ROOT / "language_packs" / lang / "models" / "slot" / "slot_tagger.pkl"
    if not src.exists():
        src = REPO_ROOT / "dist" / "nlu_pack" / "models" / "slot" / lang / "slot_tagger.pkl"
    if not src.exists():
        src = REPO_ROOT / "models" / "slot" / lang / "slot_tagger.pkl"
    if not src.exists():
        src = REPO_ROOT / "models" / "bio_slot_tagger.pkl"

    if not src.exists():
        raise FileNotFoundError(f"slot_tagger.pkl not found for lang={lang}")

    data = joblib.load(src)
    vec = data["vectorizer"]
    clf = data["classifier"]

    vocab = vec.vocabulary_
    classes = [str(c) for c in clf.classes_]
    coef = clf.coef_
    intercept = [round(float(x), 4) for x in clf.intercept_]

    weights = {}
    for feat, idx in vocab.items():
        w_list = [round(float(coef[c_idx, idx]), 4) for c_idx in range(len(classes))]
        if any(abs(w) > 0.0001 for w in w_list):
            weights[feat] = w_list

    payload = {
        "version": "1.0.0",
        "lang": lang,
        "classes": classes,
        "intercept": intercept,
        "weights": weights,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Exported slot tagger weights to {out_path} ({len(weights)} active features)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export BIO slot tagger weights to JSON")
    parser.add_argument("--lang", default="en", help="Language code (default: en)")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "models" / "slot_tagger_weights.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()
    export_weights(args.lang, args.out)


if __name__ == "__main__":
    main()
