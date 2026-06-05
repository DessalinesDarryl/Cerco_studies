#!/usr/bin/env bash
set -euo pipefail

# Run pre-ML EEG steps and export feature tables.
bash run_pipeline.sh preprocess segment_rem features
bash scripts/script_ok/eeg/export_eeg_results.sh
