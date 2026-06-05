#!/usr/bin/env bash
set -euo pipefail

# Train all tabular models used by the current pipeline.
bash run_pipeline.sh train_rf train_knn train_xgb
