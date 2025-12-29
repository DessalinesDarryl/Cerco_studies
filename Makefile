PY=python

RAW_ROOT ?= ../documents/EEG/raw
PROC_ROOT ?= ../documents/EEG/preprocessed/XAI/data

LOG_DIR = logs
NWORKERS ?= 10

.DEFAULT_GOAL := all_rf

# Assure que le dossier logs existe pour chaque règle
$(LOG_DIR):
	mkdir -p $(LOG_DIR)

# ======================== Préparation des données ========================
preprocess: | $(LOG_DIR)
	$(PY) scripts/preprocess.py \
		--config configs/preproc/default.yaml \
		--n_workers $(NWORKERS) \
		2>&1 | tee $(LOG_DIR)/logs_preprocess.txt

segment_rem: | $(LOG_DIR)
	$(PY) scripts/segment_rem.py \
		--config configs/preproc/segment_rem.yaml \
		--n_workers $(NWORKERS) \
		2>&1 | tee $(LOG_DIR)/logs_segment_rem.txt

rbd_emg: | $(LOG_DIR)
	$(PY) src/features/emg_rbd.py \
		--emg_channels Menton JAMBG JAMBD EMG1 EMG2 \
		2>&1 | tee $(LOG_DIR)/logs_rbd_emg.txt

# Add features : cohérence / corrélation mvt oculaires+EMG / 
features: | $(LOG_DIR)
	$(PY) scripts/extract_features.py \
		--config configs/preproc/features.yaml \
		--n_workers $(NWORKERS) \
		2>&1 | tee $(LOG_DIR)/logs_extract_features.txt

dataset: | $(LOG_DIR)
	$(PY) scripts/build_dataset.py \
		--eeg-features data/processed/features/features.csv \
		--emg-rbd data/processed/rbd/rbd_emg_events_and_summary_4s_per_channel.csv \
		--out-csv data/processed/features/dataset_final.csv \
		2>&1 | tee $(LOG_DIR)/logs_dataset.txt

# Commande à lancer pour préparer les données
prepa_data: preprocess segment_rem rbd_emg features dataset

# ======================== Entrainement des modèles ========================

### Modèles tabulaires
train_rf: | $(LOG_DIR)
	$(PY) scripts/train_tabular_models.py \
		--config configs/eval/train_rf.yaml \
		2>&1 | tee $(LOG_DIR)/logs_train_rf.txt

train_knn: | $(LOG_DIR)
	$(PY) scripts/train_tabular_models.py \
		--config configs/eval/train_knn.yaml \
		2>&1 | tee $(LOG_DIR)/logs_train_knn.txt

train_xgb: | $(LOG_DIR)
	$(PY) scripts/train_tabular_models.py \
		--config configs/eval/train_xgb.yaml \
		2>&1 | tee $(LOG_DIR)/logs_train_xgb.txt

### Modèles Deep
train_cnn: | $(LOG_DIR)
	$(PY) scripts/train_deep_models.py \
		--config configs/eval/train_cnn.yaml \
		2>&1 | tee $(LOG_DIR)/logs_train_cnn.txt

train_rnn: | $(LOG_DIR)
	$(PY) scripts/train_deep_models.py \
		--config configs/eval/train_rnn.yaml \
		2>&1 | tee $(LOG_DIR)/logs_train_rnn.txt


# Evaluation et XAI

plot_metrics: | $(LOG_DIR)
	$(PY) scripts/plot_roc_cm.py \
		--config configs/eval/plots_metrics.yaml \
		2>&1 | tee $(LOG_DIR)/logs_plot_metrics.txt

xai_attr: | $(LOG_DIR) # prévu pour RF / XGB (TreeExplainer)
	$(PY) scripts/xai_attributions.py \
		--config configs/xai/attributions.yaml \
		2>&1 | tee $(LOG_DIR)/logs_xai_attr.txt

xai_plots: | $(LOG_DIR)
	$(PY) scripts/xai_plots.py \
		--config configs/xai/plots.yaml \
		2>&1 | tee $(LOG_DIR)/logs_xai_plots.txt

stats: | $(LOG_DIR)
	$(PY) scripts/group_stats.py \
		--config configs/eval/stats.yaml \
		2>&1 | tee $(LOG_DIR)/logs_stats.txt

report: | $(LOG_DIR)
	$(PY) scripts/build_reports.py \
		--xai-root outputs/xai \
		--stats-root outputs/stats \
		--fig-root outputs/figures \
		--out outputs/reports \
		2>&1 | tee $(LOG_DIR)/logs_report.txt



## Commande à lancer pour entrainer et évaluer les modèles
step_rf: train_rf plot_metrics xai_attr xai_plots stats report
step_knn: train_knn plot_metrics xai_attr xai_plots stats report
step_xgb: train_xgb plot_metrics xai_attr xai_plots stats report

step_cnn: train_cnn plot_metrics report
step_rnn: train_rnn plot_metrics report

## Pipeline complet 
all_rf: prepa_data step_rf
all_knn: prepa_data step_knn
all_xgb: prepa_data step_xgb
all_cnn: prepa_data step_cnn
all_rnn: prepa_data step_rnn