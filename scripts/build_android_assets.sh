#!/usr/bin/env bash
#
# Stage dist/bundle-en as the Android asset tree: dist/android-assets/nlu_pack/
#
# What is dropped, and why:
#   *.mlpackage, models/semantic_head/  iOS/CoreML; the semantic stage reports
#                                       disabled in cascade.json anyway
#   intent_classifier_weights*.json     the Kotlin TF-IDF port's weights. This
#                                       client runs ONNX. Shipping both puts
#                                       three temperatures in one APK and
#                                       guarantees someone pairs the wrong one
#   labels.pkl                          Python only
#   nlu_schema.json, nlu_entities.json  the compiler's INPUT shape. They are in
#                                       the pack for the reference engine; a
#                                       client reading them is the thing format
#                                       3.0 exists to stop
#   meta/                               report card and lineage, not read at run
#                                       time (keep it if support wants it)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-$REPO/dist/bundle-en}"
OUT="$REPO/dist/android-assets/nlu_pack"

[ -d "$SRC" ] || {
  echo "no bundle at $SRC — build it first:" >&2
  echo "  PYTHONPATH=packages/buildtime python -m nlu_compiler.content_bundle --lang en --out dist/bundle-en" >&2
  exit 1
}

rm -rf "$OUT"
mkdir -p "$OUT"

rsync -a \
  --exclude='*.mlpackage' \
  --exclude='*.mlmodelc' \
  --exclude='intent_classifier_weights*.json' \
  --exclude='labels.pkl' \
  --exclude='nlu_schema.json' \
  --exclude='nlu_entities.json' \
  --exclude='meta' \
  --exclude='semantic_head' \
  "$SRC/" "$OUT/"

printf '%s\n' "$OUT: $(find "$OUT" -type f | wc -l | tr -d ' ') files, $(du -sh "$OUT" | cut -f1)"
echo "now: scripts/sync_android_assets.sh [ANDROID_REPO]"
