#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/emg/csv

cp -f data/processed/rbd/*.csv results/emg/csv/ 2>/dev/null || true

echo "EMG results exported to results/emg/csv"
