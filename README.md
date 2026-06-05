# RBD EEG/EMG Pipeline (Preprocessing + Features)

## Overview

This branch contains only the steps up to dataset assembly:

- EEG/EMG preprocessing
- clean REM segmentation
- RSWA/EMG marker extraction
- EEG feature extraction
- dataset assembly

All downstream components were removed from this branch.

## Current Structure

```text
configs/
  preproc/
    default.yaml
    features.yaml
    segment_rem.yaml

data/
  raw/
  processed/
    preprocessed/
    rem_epochs/
    rbd/
    features/

scripts/
  preprocess.py
  segment_rem.py
  segment_rswa.py
  extract_features.py
  build_dataset.py
  filter_features_rswa_epochs.py
  compute_psd.py
  compute_eye_emg_corr_raw.py

  script_ok/
    preprocessed/
    eeg/
    emg/

src/
  data/
  features/
  segmentation/
  utils/

results/
  eeg/
    boxplots/
    csv/
  emg/
    csv/

dvc.yaml
Makefile
run_pipeline.sh
```

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run full pipeline

```bash
bash run_pipeline.sh prepa_data
```

Equivalent with Make:

```bash
make prepa_data
```

## Pipeline Stages

1. preprocess

```bash
python scripts/preprocess.py --config configs/preproc/default.yaml
```

2. segment_rem

```bash
python scripts/segment_rem.py --config configs/preproc/segment_rem.yaml
```

3. rbd_emg

```bash
python src/features/emg_rbd.py --emg_channels Menton JAMBG JAMBD EMG1 EMG2
```

4. features

```bash
python scripts/extract_features.py --config configs/preproc/features.yaml
```

5. dataset

```bash
python scripts/build_dataset.py \
  --eeg-features data/processed/features/features.csv \
  --emg-rbd data/processed/rbd/rbd_emg_events_and_summary_4s_per_channel.csv \
  --out-csv data/processed/features/dataset_final.csv
```

## Helper Scripts for Requested Architecture

Preprocessed synchronization:

```bash
bash scripts/script_ok/preprocessed/run_preprocessed.sh
```

EEG export to requested results tree:

```bash
bash scripts/script_ok/eeg/run_eeg.sh
```

EMG export to requested results tree:

```bash
bash scripts/script_ok/emg/run_emg.sh
```

## Notes

- This branch intentionally stops at dataset creation.
- If you need downstream stages, use another branch or reintroduce those components explicitly.
