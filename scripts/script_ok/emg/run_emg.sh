#!/usr/bin/env bash
set -euo pipefail

# Run EMG RSWA extraction and export outputs to requested results folders.
bash run_pipeline.sh rbd_emg
bash scripts/script_ok/emg/export_emg_results.sh
