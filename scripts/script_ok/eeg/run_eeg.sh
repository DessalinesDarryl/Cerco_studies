#!/usr/bin/env bash
set -euo pipefail

# Run EEG-oriented steps and export artifacts to requested results folders.
bash run_pipeline.sh features train_rf plot_metrics
bash scripts/script_ok/eeg/export_eeg_results.sh
