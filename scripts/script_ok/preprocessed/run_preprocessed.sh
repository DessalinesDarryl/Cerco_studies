#!/usr/bin/env bash
set -euo pipefail

# Run preprocessing and REM segmentation, then sync to requested architecture.
bash run_pipeline.sh preprocess segment_rem
bash scripts/script_ok/preprocessed/sync_rem_epochs.sh
