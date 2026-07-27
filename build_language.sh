#!/usr/bin/env bash
# Exit immediately if any command fails
set -e

# Default to "en" if no language is provided
LANG=${1:-en}
echo "🚀 Building language pack for: $LANG"

# Ensure Python can find our custom packages
export PYTHONPATH="packages/buildtime:packages/runtime"

echo "---------------------------------------------------"
echo "==> 1. Training the Model..."
.venv/bin/python -m nlu_training.train --lang "$LANG"

echo "---------------------------------------------------"
echo "==> 2. Calibrating Confidence Thresholds..."
.venv/bin/python -m nlu_training.fit_calibration --lang "$LANG" --write

echo "---------------------------------------------------"
echo "==> 3. Evaluating the Model..."
.venv/bin/python -m nlu_training.evaluate --langs "$LANG" --out evaluate_report.json

echo "---------------------------------------------------"
echo "==> 4. Exporting iOS Weights..."
.venv/bin/python packages/buildtime/nlu_export/export_ios_weights.py --lang "$LANG"

echo "---------------------------------------------------"
echo "==> 5. Exporting CoreML Package..."
.venv/bin/python packages/buildtime/nlu_export/export_coreml.py --lang "$LANG" --skip-minilm

echo "---------------------------------------------------"
echo "==> 6. Compiling the .nlu Bundle..."
.venv/bin/python -m nlu_compiler.content_bundle --lang "$LANG" --out "dist/bundle-$LANG" --report evaluate_report.json

echo "---------------------------------------------------"
echo "✅ Success! The .nlu bundle is ready at: dist/bundle-$LANG"
