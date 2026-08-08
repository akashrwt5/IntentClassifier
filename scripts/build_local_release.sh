#!/bin/bash
set -e

# Setup environment
if [ -d "test-venv" ]; then
  source test-venv/bin/activate
elif [ -d ".venv" ]; then
  source .venv/bin/activate
fi
export PYTHONPATH=packages/buildtime:packages/runtime
LANG_ARG="en"
VERSION="1.0.0"
CHANNEL="dev"

echo "=== Local Release Build Script ==="
echo "Building platform-specific artifacts for testing..."

echo "[1/4] Generating report card..."
mkdir -p dist
python -m nlu_training.evaluate --langs "$LANG_ARG" --out dist/report_card.json

# 1. Content Bundle
echo "[2/4] Compiling content bundle..."
mkdir -p dist/bundle-$LANG_ARG
python -m nlu_compiler.content_bundle \
  --lang "$LANG_ARG" \
  --out "dist/bundle-$LANG_ARG" \
  --version "$VERSION" \
  --channel "$CHANNEL" \
  --report dist/report_card.json \
  --report-gaps

# 2. Assemble Packs (will produce universal, ios, android slices)
echo "[3/4] Assembling sliced packs..."
ARGS=(--src "dist/bundle-$LANG_ARG" --version "$VERSION" --language "$LANG_ARG" --channel "$CHANNEL" --out dist --report dist/report_card.json)

# Add mock/local models if available to ensure we test the stripping logic
if [ -d "models/intent/$LANG_ARG/IntentClassifier.mlpackage" ]; then
  ARGS+=(--coreml "models/intent/$LANG_ARG/IntentClassifier.mlpackage")
fi
if [ -f "models/intent/$LANG_ARG/model.tflite" ]; then
  ARGS+=(--tflite "models/intent/$LANG_ARG/model.tflite")
fi
if [ -f "models/intent/$LANG_ARG/intent_classifier_weights.json" ]; then
  ARGS+=(--ios-weights "models/intent/$LANG_ARG/intent_classifier_weights.json")
fi

python scripts/ci/assemble_pack.py "${ARGS[@]}"

# 3. Verify
echo "[4/4] Verifying generated packs..."
python -m nlu_compiler.verify dist/pack-$LANG_ARG-v$VERSION-universal.nlu
python -m nlu_compiler.verify dist/pack-$LANG_ARG-v$VERSION-ios.nlu
python -m nlu_compiler.verify dist/pack-$LANG_ARG-v$VERSION-android.nlu

echo "ALL DONE! Sliced release packs are available in dist/"
