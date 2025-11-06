PY=python

RAW_ROOT ?= /data/lab/projet_X/raw     
PROC_ROOT ?= data/processed

preprocess:
	$(PY) scripts/preprocess.py --raw-root $(RAW_ROOT) --out-root $(PROC_ROOT) | tee logs/logs_remove_artefacts.txt

segment_rem:
	$(PY) scripts/segment_rem.py --in-root $(PROC_ROOT) --out-root $(PROC_ROOT)/rem_segments | tee logs/logs_segment_rem.txt

features:
	$(PY) scripts/extract_features.py --in-root $(PROC_ROOT)/rem_segments --out-root $(PROC_ROOT)/features | tee logs/logs_compute_psd.txt

train:
	$(PY) scripts/train.py --features-root $(PROC_ROOT)/features --out models/ | tee logs/logs.txt

infer:
	$(PY) scripts/infer.py --features-root $(PROC_ROOT)/features --ckpt models/checkpoint.pt --out outputs/predictions/preds.csv

xai_attr:
	$(PY) scripts/xai_attributions.py --ckpt models/checkpoint.pt --features-root $(PROC_ROOT)/features --out outputs/xai/attributions

xai_attn:
	$(PY) scripts/xai_attention_maps.py --ckpt models/checkpoint.pt --features-root $(PROC_ROOT)/features --out outputs/xai/attention

stats:
	$(PY) scripts/group_stats.py --xai-root outputs/xai --out outputs/stats | tee logs/logs_stats.txt

report:
	$(PY) scripts/build_reports.py --xai-root outputs/xai --stats-root outputs/stats --fig-root outputs/figures --out outputs/reports

all: preprocess segment_rem features train infer xai_attr xai_attn stats report
