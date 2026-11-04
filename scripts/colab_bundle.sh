#!/usr/bin/env bash
# Zip what the Colab notebook needs: the package and the dataset (no models, no venv).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
rm -f callgrader-colab.zip
zip -qr callgrader-colab.zip callgrader data results/.keep 2>/dev/null || zip -qr callgrader-colab.zip callgrader data
echo "callgrader-colab.zip: upload it to Colab with colab/train_and_serve.ipynb"
