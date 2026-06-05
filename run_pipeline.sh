#!/usr/bin/env bash
# ==============================================================================
# run_pipeline.sh — Équivalent bash du Makefile (pour Git Bash / Windows)
# ==============================================================================
# Usage :
#   bash run_pipeline.sh <cible1> [cible2 ...]
#
# Exemples :
#   bash run_pipeline.sh preprocess segment_rem
#   bash run_pipeline.sh rbd_emg features dataset
#   bash run_pipeline.sh prepa_data
#
# Cibles disponibles :
#   preprocess, segment_rem, rbd_emg, features, dataset, prepa_data
# ==============================================================================

set -euo pipefail

PY="${PY:-python}"
NWORKERS="${NWORKERS:-10}"
LOG_DIR="data/logs"

mkdir -p "$LOG_DIR"

# --------------------------------------------------------------------------
# Fonctions = cibles du Makefile
# --------------------------------------------------------------------------

run_preprocess() {
    echo ">>> [preprocess]"
    $PY scripts/preprocess.py \
        --config configs/preproc/default.yaml \
        --n_workers "$NWORKERS" \
        2>&1 | tee "$LOG_DIR/logs_preprocess.txt"
}

run_segment_rem() {
    echo ">>> [segment_rem]"
    $PY scripts/segment_rem.py \
        --config configs/preproc/segment_rem.yaml \
        --n_workers "$NWORKERS" \
        2>&1 | tee "$LOG_DIR/logs_segment_rem.txt"
}

run_rbd_emg() {
    echo ">>> [rbd_emg]"
    $PY src/features/emg_rbd.py \
        --emg_channels Menton JAMBG JAMBD EMG1 EMG2 \
        2>&1 | tee "$LOG_DIR/logs_rbd_emg.txt"
}

run_features() {
    echo ">>> [features]"
    $PY scripts/extract_features.py \
        --config configs/preproc/features.yaml \
        --n_workers "$NWORKERS" \
        2>&1 | tee "$LOG_DIR/logs_extract_features.txt"
}

run_dataset() {
    echo ">>> [dataset]"
    $PY scripts/build_dataset.py \
        --eeg-features data/processed/features/features.csv \
        --emg-rbd data/processed/rbd/rbd_emg_events_and_summary_4s_per_channel.csv \
        --out-csv data/processed/features/dataset_final.csv \
        2>&1 | tee "$LOG_DIR/logs_dataset.txt"
}

run_prepa_data() {
    run_preprocess
    run_segment_rem
    run_rbd_emg
    run_features
    run_dataset
}

# --------------------------------------------------------------------------
# Dispatch des arguments
# --------------------------------------------------------------------------

if [[ $# -eq 0 ]]; then
    echo "Usage: bash run_pipeline.sh <cible1> [cible2 ...]"
    echo "Cibles : preprocess segment_rem rbd_emg features dataset prepa_data"
    exit 0
fi

for target in "$@"; do
    case "$target" in
        preprocess)   run_preprocess ;;
        segment_rem)  run_segment_rem ;;
        rbd_emg)      run_rbd_emg ;;
        features)     run_features ;;
        dataset)      run_dataset ;;
        prepa_data)   run_prepa_data ;;
        *)
            echo "[ERREUR] Cible inconnue : '$target'"
            exit 1
            ;;
    esac
done

echo ""
echo "=== Pipeline terminé avec succès ==="
