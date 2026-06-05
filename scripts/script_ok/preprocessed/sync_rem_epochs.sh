#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="data/processed/rem_epochs"
DST_DIR="data/preprocessed/rem_epo"

mkdir -p "$DST_DIR"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Source directory not found: $SRC_DIR"
  exit 1
fi

cp -f "$SRC_DIR"/* "$DST_DIR"/ 2>/dev/null || true
echo "REM epochs synchronized to $DST_DIR"
