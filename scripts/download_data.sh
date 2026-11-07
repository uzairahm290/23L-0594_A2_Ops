#!/usr/bin/env bash
# Pull the Kaggle dataset into data/raw. Needs ~/.kaggle/kaggle.json (Kaggle > Settings > API > Create token)
# or KAGGLE_USERNAME / KAGGLE_KEY in the environment.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
kaggle datasets download -d cashbowman/ai-generated-images-vs-real-images -p data/raw --unzip
find data/raw -maxdepth 2 -type d
