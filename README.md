# RBD-EEG-EMG Pipeline: Automated Detection and Classification of REM Sleep Behavior Disorder

## Overview

This repository contains a comprehensive, reproducible computational pipeline for analyzing polysomnographic (PSG) recordings in Rapid Eye Movement (REM) Sleep Behavior Disorder (RBD). The pipeline integrates signal processing, feature engineering, machine learning, and explainable AI (XAI) to enable reliable detection and characterization of REM Sleep Without Atonia (RSWA) markers—the hallmark of RBD.

### Key Features

- **Signal Preprocessing**: Montage construction (bipolarity), digital filtering (EEG: 0.5–80 Hz, EMG: 30–100 Hz), artifact detection via covariance method
- **REM Segmentation**: Automated extraction of artifact-free REM sleep epochs (4 s windows) using source hypnograms
- **RSWA Detection**: EMG-based phasic and tonic activity markers quantified per channel and epoch during REM sleep
- **Multimodal Feature Extraction**: 
  - EEG spectral (bandpowers, ratios, entropy, centroid)
  - EEG temporal (Hjorth parameters, zero-crossing rate, statistics)
  - EEG–EMG coupling (coherence, cross-correlation)
  - EMG RBD-specific markers (tonic ratio, phasic density, EOG-EMG correlation)
- **Classifier Training**: RandomForest, XGBoost, and Deep Learning models (CNN, GRU) with GroupKFold cross-validation
- **Explainability**: SHAP attribution plots and permutation feature importance at global and patient levels
- **Automated Reporting**: Markdown-based synthesis of metrics, ROC curves, confusion matrices, and XAI visualizations

---

## Table of Contents

1. [Scientific Context](#scientific-context)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Data Format and Expected Inputs](#data-format-and-expected-inputs)
5. [Pipeline Architecture](#pipeline-architecture)
6. [Detailed Workflow](#detailed-workflow)
7. [Usage Examples](#usage-examples)
8. [Configuration](#configuration)
9. [Outputs and Interpretation](#outputs-and-interpretation)
10. [Reproducibility](#reproducibility)
11. [References](#references)

---

## Scientific Context

### REM Sleep Behavior Disorder (RBD)

RBD is a parasomnia characterized by the loss of normal REM sleep atonia, leading to excessive EMG activity and dream-enactment behavior. RSWA (REM Sleep Without Atonia) is the polysomnographic hallmark of RBD and is defined as:

- **Phasic EMG activity**: Rapid, bursting muscle activation during REM sleep
- **Tonic EMG activity**: Sustained, elevated baseline EMG during REM sleep, typically quantified as an increase relative to NREM baseline

This pipeline implements evidence-based RSWA detection algorithms to support both clinical diagnosis and research investigations.

### Key Methodological Choices

1. **EMG Preprocessing**: Raw EMG is bandpass-filtered (30–100 Hz) and envelope-detected via moving average (MA 1 s)
2. **Phasic Threshold**: 95th percentile of NREM reference window (30 s prior to REM episode onset, with margin)
3. **Tonic Ratio**: Median EMG envelope (REM, artifact-masked) / median EMG envelope (NREM reference)
4. **NREM–REM Pairing**: Each REM episode is paired with the nearest preceding NREM episode (≥10 s, ≤2 s gap tolerance)
5. **Epoch-Level RSWA**: Binary classification per 4 s epoch using heuristic (tonic ratio > 1.3 OR EOG phasicity criterion)

---

## Installation

### Prerequisites

- Python ≥ 3.11
- Conda or venv virtual environment manager
- Approximately 50 GB disk space for full dataset + outputs

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd Cerco_studies

# Create and activate virtual environment
conda create -n cerco_studies python=3.11
conda activate cerco_studies

# Install dependencies
pip install -r requirements.txt

# (Optional) Install dev tools
pip install -r requirements-dev.txt
```

For GPU acceleration (PyTorch):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Project Structure

```
Cerco_studies/
├── README.md                          # This file
├── requirements.txt                   # Production dependencies (pinned versions)
├── requirements-dev.txt               # Development/testing dependencies
├── pyproject.toml                     # Modern Python packaging config
├── Makefile                           # Common task automation
├── dvc.yaml                           # DVC pipeline (optional, for experiment tracking)
├── .gitignore                         # Git exclusions
├── .vscode/settings.json              # VS Code editor settings
│
├── configs/                           # YAML configuration files
│   ├── preproc/
│   │   ├── default.yaml              # Primary preprocessing params
│   │   ├── features.yaml             # Feature extraction config
│   │   └── segment_rem.yaml          # REM segmentation config
│   ├── eval/
│   │   ├── train_rf.yaml             # RandomForest config
│   │   ├── train_xgb.yaml            # XGBoost config
│   │   ├── train_knn.yaml            # KNN config
│   │   ├── train_cnn.yaml            # CNN (deep) config
│   │   ├── train_rnn.yaml            # GRU/RNN config
│   │   ├── report.yaml               # Report generation config
│   │   ├── stats.yaml                # Statistical analysis config
│   │   └── infer.yaml                # Inference config
│   └── xai/
│       ├── attributions.yaml         # SHAP/Permutation importance config
│       └── plots.yaml                # XAI visualization config
│
├── scripts/                           # Orchestration entry points
│   ├── preprocess.py                 # [1] Raw → preprocessed FIF
│   ├── segment_rem.py                # [2] Preprocessed FIF → REM epochs
│   ├── segment_rswa.py               # [2b] REM epochs → RSWA markers CSV
│   ├── extract_features.py           # [3] Epochs + RSWA → features CSV
│   ├── build_dataset.py              # [4] Features + labels → dataset
│   ├── train_tabular_models.py       # [5a] Dataset → RF/XGB/KNN models
│   ├── train_deep_models.py          # [5b] Dataset → CNN/GRU models
│   ├── infer.py                      # [6] Model + features → predictions
│   ├── plot_roc_cm.py                # [7] Predictions → ROC/CM plots
│   ├── group_stats.py                # [8] Band powers → group comparisons
│   ├── xai_attributions.py           # [9] Model + features → SHAP/permutation
│   ├── xai_plots.py                  # [10] Attributions → visualization
│   └── build_reports.py              # [11] Aggregate & synthesize results
│
├── src/                               # Reusable library modules
│   ├── data/
│   │   ├── __init__.py
│   │   ├── io.py                     # EDF/FIF I/O utilities
│   │   └── preprocessing.py          # Core preprocessing logic
│   ├── features/
│   │   ├── __init__.py
│   │   ├── spectral.py               # Generic spectral features
│   │   ├── spectral_eeg.py           # EEG-specific spectral features
│   │   ├── temporal_eeg.py           # EEG temporal features
│   │   ├── connectivity.py           # EEG–EEG coherence
│   │   ├── eeg_emg_coupling.py       # EEG–EMG coupling
│   │   └── emg_rbd.py                # RBD-specific EMG features
│   ├── segmentation/
│   │   ├── __init__.py
│   │   └── rem.py                    # REM epoch extraction logic
│   ├── stats/
│   │   ├── __init__.py
│   │   ├── anova_kruskal.py          # Group comparisons
│   │   └── graphs_metrics.py         # Topological metrics
│   ├── training/
│   │   ├── __init__.py
│   │   └── utils.py                  # Metrics computation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py                 # YAML config loading
│   │   ├── logging.py                # Logger factory
│   │   ├── metrics_logger.py         # Metrics export
│   │   └── seed.py                   # Reproducibility (random seeds)
│   ├── viz/
│   │   ├── __init__.py
│   │   ├── plots.py                  # Feature importance plots
│   │   ├── metrics_plots.py          # Confusion matrix, ROC
│   │   └── reports.py                # Report generation
│   └── xai/
│       ├── __init__.py
│       ├── aggregation.py            # Attribution aggregation
│       ├── deeplift.py               # DeepLIFT (PyTorch)
│       └── integrated_gradients.py   # Integrated Gradients (PyTorch)
│
├── data/
│   ├── patients_label.txt            # Patient ID ↔ group mapping
│   └── processed/
│       ├── rbd/
│       │   └── rbd_emg_events_and_summary_4s_per_channel.csv
│       ├── features/
│       │   └── dataset_final.csv     # Consolidated feature matrix + labels
│       └── ...
│
├── models/                            # Trained model checkpoints
│   ├── checkpoint_rf.joblib
│   ├── checkpoint_xgb.joblib
│   └── checkpoint_cnn.pt
│
├── outputs/
│   ├── figures/
│   │   ├── metrics/
│   │   └── xai/
│   ├── reports/
│   │   └── report.md
│   ├── stats/
│   │   ├── train_metrics_rf.csv
│   │   └── ...
│   ├── infer/
│   │   └── predictions_rf.csv
│   └── xai/
│       └── attributions/
│
├── logs/
│   ├── logs_preprocess.txt
│   ├── logs_segment_rem.txt
│   └── ...
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Pytest fixtures
│   ├── test_preproc.py               # Unit tests
│   ├── test_stats.py
│   └── test_xai.py
│
└── notebooks/
    ├── 00_qc_exploration.ipynb
    ├── 10_preproc_demo.ipynb
    ├── 90_figures_paper.ipynb
    └── Boxplots.ipynb
```

---

## Data Format and Expected Inputs

### Input Raw Data

- **Format**: EDF, GDF, BDF, or FIF files
- **Sampling Rate**: Typically 100–500 Hz
- **Channels Expected**:
  - **EEG**: Fpz–T3, Fp1–C3, T3–O1 (and symmetric pairs)
  - **EMG**: Chin (mentalis), bilateral leg (anterior tibialis)
  - **EOG**: Left/Right eye movements

### Hypnogram Format

Supported formats (auto-detected):
1. **Text (EXP)**: Space-separated [seconds, HH:MM:SS, stage, code]
2. **CSV**: Columns `onset_sec`, `duration_sec`, `stage`
3. **MNE Annotations**: Embedded in Raw objects (REM, N1, N2, N3, W)

### Clinical Labels

- **Format**: Excel/CSV with columns `patient_id`, `diagnosis`, optionally `group`, `age`, `gender`
- **Label Mapping**: Auto-inferred; can be manually specified in config

---

## Pipeline Architecture

### High-Level Flow

```
Raw PSG Files
      ↓
[1] Preprocessing (Montage + Filters + Artifact Detection + REM Annotation)
      ↓
Preprocessed FIF Files
      ↓
[2] REM Segmentation + [2b] RSWA Detection
      ↓
[3] Feature Extraction (EEG Spectral/Temporal + EMG-RBD + Coupling)
      ↓
[4] Dataset Construction (Features + Labels)
      ↓
[5] Model Training (Tabular: RF/XGB/KNN, OR Deep: CNN/GRU)
      ↓
[6] Inference (Predictions)
      ↓
[7–10] Evaluation + XAI Visualization
      ↓
[11] Final Report Generation
```

---

## Detailed Workflow

### Stage 1: Preprocessing

**Script**: `scripts/preprocess.py`

Operations:
1. **Montage (GP2)**: 6 bipolar EEG pairs + raw EMG/ECG/EOG
2. **Filtering**:
   - EEG: 0.5–80 Hz bandpass + 50 Hz notch
   - EMG: 30–100 Hz bandpass + 50 Hz notch
3. **Hypnogram Loading**: Parse and harmonize stage labels
4. **REM Annotation**: Add MNE annotations for REM epochs
5. **Artifact Detection** (YASA): Covariance-based; threshold 3 SD
6. **Output**: FIF file with annotations

### Stage 2: REM Segmentation & RSWA Detection

**Scripts**: `scripts/segment_rem.py` (epochs), `scripts/segment_rswa.py` (markers)

- Extract artifact-free 4 s REM windows
- Detect phasic (>threshold, >3 s) and tonic (<25 µV EOG) EMG activities
- Compute per-epoch RSWA markers (ratio, phasic density, EOG criterion)
- Output: Epochs object + CSV with per-channel summaries

### Stage 3: Feature Extraction

**Script**: `scripts/extract_features.py`

Features extracted per channel/epoch:
- **EEG Spectral**: Bandpowers (delta–gamma), ratios, entropy, centroid
- **EEG Temporal**: Mean, variance, Hjorth, zero-crossing rate
- **Connectivity**: EEG–EEG correlation / coherence; EEG–EMG coupling
- **EMG-RBD**: Phasic ratio, tonic ratio, marker flags

**Output**: Wide CSV (~200–500 features)

### Stage 4: Dataset Construction

**Script**: `scripts/build_dataset.py`

- Merge features + clinical labels by patient ID
- Filter to patients with both data and labels
- Handle missing values
- Save consolidated dataset: `dataset_final.csv`

### Stage 5: Model Training

#### 5a. Tabular Models (RF, XGB, KNN)

**Script**: `scripts/train_tabular_models.py`

- **GroupKFold** CV (5 folds, group by patient_id to avoid leakage)
- Per fold: train → evaluate on Balanced Accuracy, F1
- Report mean ± SD
- Train final model on full dataset
- Checkpoint: `checkpoint_rf.joblib`

#### 5b. Deep Learning (CNN, GRU)

**Script**: `scripts/train_deep_models.py`

- **Architecture**: CNN1D (conv+pool+dense) or GRU
- **Training**: GroupKFold CV, Adam optimizer, n_epochs=50
- **Checkpoint**: `checkpoint_cnn.pt` (PyTorch state dict)

### Stage 6–11: Inference, Evaluation, XAI, Reporting

See detailed workflow section in full documentation or inline script comments for specifics on inference, ROC/CM plotting, statistical analysis, SHAP/permutation importance, and markdown report generation.

---

## Usage Examples

### Full Pipeline

```bash
conda activate cerco_studies

# 1. Preprocess
python scripts/preprocess.py --config configs/preproc/default.yaml

# 2. REM segmentation + RSWA detection
python scripts/segment_rem.py --config configs/preproc/segment_rem.yaml
python scripts/segment_rswa.py --config configs/preproc/segment_rem.yaml

# 3. Extract features
python scripts/extract_features.py --config configs/preproc/features.yaml

# 4. Build dataset
python scripts/build_dataset.py --config configs/preproc/default.yaml

# 5. Train model (RandomForest example)
python scripts/train_tabular_models.py --config configs/eval/train_rf.yaml

# 6–11. Inference → Evaluation → XAI → Report
python scripts/infer.py --config configs/eval/infer.yaml
python scripts/plot_roc_cm.py --config configs/eval/infer.yaml
python scripts/xai_attributions.py --config configs/xai/attributions.yaml
python scripts/xai_plots.py --config configs/xai/plots.yaml
python scripts/build_reports.py --config configs/eval/report.yaml
```

### Test Suite

```bash
pytest tests/ -v
pytest tests/test_preproc.py -v   # Test preprocessing module
```

---

## Configuration

All behavior controlled via YAML in `configs/`. Key sections:

### `configs/preproc/default.yaml`

```yaml
raw_root: /data/raw_edf
out_root: data/processed/preproc
hypno_root: /data/hypnograms

eeg:
  l_freq: 0.5
  h_freq: 80.0
  notch: 50.0

emg:
  hp: 30.0
  lp: 100.0
  notch: 50.0

yasa:
  win_sec: 4.0
  method: covar
  threshold: 3.0
  include: sleep
```

### `configs/eval/train_rf.yaml`

```yaml
features_csv: data/processed/features/dataset_final.csv
label_col: label_id

model_type: rf

out_dir: models
ckpt_name: checkpoint_rf.joblib
metrics_out: outputs/stats/train_metrics_rf.csv

cv:
  n_splits: 5
  group_by_patient: true
  random_state: 42

rf_params:
  n_estimators: 300
  max_depth: null
  class_weight: balanced
  n_jobs: -1
```

---

## Outputs and Interpretation

### Training Metrics

`outputs/stats/train_metrics_rf.csv`:

| fold | balanced_accuracy | f1_macro | recall_macro |
|------|------------------|----------|--------------|
| 1–5  | 0.87 ± 0.03      | 0.85 ± 0.04  | 0.88 ± 0.03  |

### ROC Curves

Multiclass one-vs-rest ROC; per-class and macro AUC saved to `outputs/figures/metrics/`

### Feature Importance (XAI)

Top features typically align with known RSWA markers:
- Phasic EMG density / time
- Tonic EMG ratio
- EEG alpha/beta ratios
- Theta power

---

## Reproducibility

### Random Seed Fixing

All randomness controlled via `src/utils/seed.py::set_seed(42)`:
- Python, NumPy, PyTorch, CuDNN

### Dependency Pinning

`requirements.txt` specifies exact versions:
```
mne==1.6.1
pandas==2.1.4
numpy==2.2.0
scikit-learn==1.4.2
xgboost==2.0.3
torch==2.1.2
```

### Environment Reproduction

```bash
pip install -r requirements.txt
```

---

## References

### RBD / RSWA

- Högl, B., & Stefani, A. (2017). REM sleep behaviour disorder. *Nat. Rev. Neurol.*, 14(5), 293–309.
- Garcia-Borreguero, D., & Santamaria, J. (2020). REM sleep behavior disorder and its relationship to neurodegeneration. *Curr. Neurol. Neurosci. Rep.*, 20(2), 9.
- International Classification of Sleep Disorders (ICSD-3). American Academy of Sleep Medicine.

### Software

- MNE-Python: https://mne.tools/
- YASA: https://raphaelvallat.com/yasa/
- scikit-learn: https://scikit-learn.org/
- SHAP: https://shap.readthedocs.io/
- PyTorch: https://pytorch.org/

---

## License

[Specify your license here]

---

## Contact

For questions or contributions, please contact [your contact info].

---

**Last Updated**: April 24, 2026  
**Version**: 1.0
