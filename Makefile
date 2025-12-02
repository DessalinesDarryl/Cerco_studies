PY=python

RAW_ROOT ?= ../documents/EEG/raw
PROC_ROOT ?= ../documents/EEG/preprocessed/XAI/data

LOG_DIR = logs
NWORKERS ?= 10

# Assure que le dossier logs existe pour chaque règle
$(LOG_DIR):
	mkdir -p $(LOG_DIR)


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

# Utiliser plutot le modele KNN puis tester avec d'autres modèles : XGBoost, CNN, RNN
train: | $(LOG_DIR)
	$(PY) scripts/train.py \
		--config configs/eval/train_rf.yaml \
		2>&1 | tee $(LOG_DIR)/logs_train.txt

infer: | $(LOG_DIR)
	$(PY) scripts/infer.py \
		--config configs/eval/infer.yaml \
		2>&1 | tee $(LOG_DIR)/logs_infer.txt

xai_attr: | $(LOG_DIR)
	$(PY) scripts/xai_attributions.py \
		--config configs/xai/attributions.yaml \
		2>&1 | tee $(LOG_DIR)/logs_xai_attr.txt

xai_plots: | $(LOG_DIR)
	$(PY) scripts/xai_plots.py \
		--config configs/xai/plots.yaml \
		2>&1 | tee $(LOG_DIR)/logs_xai_plots.txt

plot_metrics: | $(LOG_DIR)
	$(PY) scripts/plot_roc_cm.py \
		--config configs/eval/plots_metrics.yaml \
		2>&1 | tee $(LOG_DIR)/logs_plot_metrics.txt

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

# Pipeline principal
all: preprocess segment_rem rbd_emg features dataset train plot_metrics xai_attr xai_plots stats report

# Pipeline sans preprocessing
all_no_preproc: segment_rem rbd_emg features dataset train plot_metrics xai_attr xai_plots stats report