#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/eeg/csv results/eeg/boxplots

# Copy tabular metrics and stats.
find outputs/stats -maxdepth 1 -type f -name "*.csv" -exec cp -f {} results/eeg/csv/ \; 2>/dev/null || true

# Copy metric figures and boxplots if present.
find outputs/figures -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.svg" \) -exec cp -f {} results/eeg/boxplots/ \; 2>/dev/null || true

echo "EEG results exported to results/eeg"
