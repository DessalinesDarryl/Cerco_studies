#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/eeg/csv results/eeg/boxplots

# Copy EEG feature tables produced before ML.
cp -f data/processed/features/*.csv results/eeg/csv/ 2>/dev/null || true

# Keep boxplots directory for optional exploratory plots only.

echo "EEG results exported to results/eeg"
