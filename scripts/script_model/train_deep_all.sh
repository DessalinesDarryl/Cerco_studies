#!/usr/bin/env bash
set -euo pipefail

# Train all deep models used by the current pipeline.
bash run_pipeline.sh train_cnn train_rnn
