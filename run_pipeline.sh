#!/usr/bin/env bash
# ==============================================================================
# run_pipeline.sh — Équivalent bash du Makefile (pour Git Bash / Windows)
# ==============================================================================
# Usage :
#   bash run_pipeline.sh <cible1> [cible2 ...]
#
# Exemples :
#   bash run_pipeline.sh segment_rem rbd_emg features dataset
#   bash run_pipeline.sh step_rf
#   bash run_pipeline.sh segment_rem rbd_emg features dataset step_rf
#   bash run_pipeline.sh all_rf
#
# Cibles disponibles (identiques au Makefile) :
#   preprocess, segment_rem, rbd_emg, features, dataset, prepa_data
#   train_rf, train_knn, train_xgb, train_cnn, train_rnn
#   plot_metrics, xai_attr, xai_plots, stats, report
#   step_rf, step_knn, step_xgb, step_cnn, step_rnn
#   all_rf, all_knn, all_xgb, all_cnn, all_rnn
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

run_train_rf() {
    echo ">>> [train_rf]"
    $PY scripts/train_tabular_models.py \
        --config configs/eval/train_rf.yaml \
        2>&1 | tee "$LOG_DIR/logs_train_rf.txt"
}

run_train_knn() {
    echo ">>> [train_knn]"
    $PY scripts/train_tabular_models.py \
        --config configs/eval/train_knn.yaml \
        2>&1 | tee "$LOG_DIR/logs_train_knn.txt"
}

run_train_xgb() {
    echo ">>> [train_xgb]"
    $PY scripts/train_tabular_models.py \
        --config configs/eval/train_xgb.yaml \
        2>&1 | tee "$LOG_DIR/logs_train_xgb.txt"
}

run_train_cnn() {
    echo ">>> [train_cnn]"
    $PY scripts/train_deep_models.py \
        --config configs/eval/train_cnn.yaml \
        2>&1 | tee "$LOG_DIR/logs_train_cnn.txt"
}

run_train_rnn() {
    echo ">>> [train_rnn]"
    $PY scripts/train_deep_models.py \
        --config configs/eval/train_rnn.yaml \
        2>&1 | tee "$LOG_DIR/logs_train_rnn.txt"
}

run_plot_metrics() {
    echo ">>> [plot_metrics]"
    $PY scripts/plot_roc_cm.py \
        --config configs/eval/plots_metrics.yaml \
        2>&1 | tee "$LOG_DIR/logs_plot_metrics.txt"
}

run_xai_attr() {
    echo ">>> [xai_attr]"
    $PY scripts/xai_attributions.py \
        --config configs/xai/attributions.yaml \
        2>&1 | tee "$LOG_DIR/logs_xai_attr.txt"
}

run_xai_plots() {
    echo ">>> [xai_plots]"
    $PY scripts/xai_plots.py \
        --config configs/xai/plots.yaml \
        2>&1 | tee "$LOG_DIR/logs_xai_plots.txt"
}

run_stats() {
    echo ">>> [stats]"
    $PY scripts/group_stats.py \
        --config configs/eval/stats.yaml \
        2>&1 | tee "$LOG_DIR/logs_stats.txt"
}

run_report() {
    echo ">>> [report]"
    $PY scripts/build_reports.py \
        --xai-root data/outputs/xai \
        --stats-root data/outputs/stats \
        --fig-root data/outputs/figures \
        --out data/outputs/reports \
        2>&1 | tee "$LOG_DIR/logs_report.txt"
}

# --- Pipelines composés ---

run_step_rf()  { run_train_rf;  run_plot_metrics; run_xai_attr; run_xai_plots; run_stats; run_report; }
run_step_knn() { run_train_knn; run_plot_metrics; run_xai_attr; run_xai_plots; run_stats; run_report; }
run_step_xgb() { run_train_xgb; run_plot_metrics; run_xai_attr; run_xai_plots; run_stats; run_report; }
run_step_cnn() { run_train_cnn; run_plot_metrics; run_report; }
run_step_rnn() { run_train_rnn; run_plot_metrics; run_report; }

run_all_rf()  { run_prepa_data; run_step_rf; }
run_all_knn() { run_prepa_data; run_step_knn; }
run_all_xgb() { run_prepa_data; run_step_xgb; }
run_all_cnn() { run_prepa_data; run_step_cnn; }
run_all_rnn() { run_prepa_data; run_step_rnn; }

# --------------------------------------------------------------------------
# Dispatch des arguments
# --------------------------------------------------------------------------

if [[ $# -eq 0 ]]; then
    echo "Usage: bash run_pipeline.sh <cible1> [cible2 ...]"
    echo "Cibles : preprocess segment_rem rbd_emg features dataset prepa_data"
    echo "         train_rf train_knn train_xgb train_cnn train_rnn"
    echo "         plot_metrics xai_attr xai_plots stats report"
    echo "         step_rf step_knn step_xgb step_cnn step_rnn"
    echo "         all_rf all_knn all_xgb all_cnn all_rnn"
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
        train_rf)     run_train_rf ;;
        train_knn)    run_train_knn ;;
        train_xgb)    run_train_xgb ;;
        train_cnn)    run_train_cnn ;;
        train_rnn)    run_train_rnn ;;
        plot_metrics) run_plot_metrics ;;
        xai_attr)     run_xai_attr ;;
        xai_plots)    run_xai_plots ;;
        stats)        run_stats ;;
        report)       run_report ;;
        step_rf)      run_step_rf ;;
        step_knn)     run_step_knn ;;
        step_xgb)     run_step_xgb ;;
        step_cnn)     run_step_cnn ;;
        step_rnn)     run_step_rnn ;;
        all_rf)       run_all_rf ;;
        all_knn)      run_all_knn ;;
        all_xgb)      run_all_xgb ;;
        all_cnn)      run_all_cnn ;;
        all_rnn)      run_all_rnn ;;
        *)
            echo "[ERREUR] Cible inconnue : '$target'"
            exit 1
            ;;
    esac
done

echo ""
echo "=== Pipeline terminé avec succès ==="
