# RBD-EEG-EMG Pipeline (REM Sleep Behavior Disorder)

Ce dépôt contient un pipeline complet pour :

- Prétraiter des enregistrements EEG/EMG de sommeil
- Extraire des segments REM sans artefacts
- Détecter des marqueurs EMG de RBD (RSWA phasique/tonique)
- Extraire des features EEG, EMG et couplage EEG–EMG
- Fusionner ces features avec des labels cliniques (Excel)
- Entraîner des modèles classiques (RandomForest, SVM, XGBoost)
- Évaluer les performances (ROC, F1, balanced accuracy)
- Expliquer les modèles avec XAI (SHAP + permutation importance)
- Générer des rapports avec figures (perf + XAI)

---

## Arborescence principale

- `configs/`
  - `preproc/default.yaml` : chemins raw/processed + params prétraitement
  - `preproc/segment_rem.yaml` : chemins pour segmentation REM
  - `preproc/features.yaml` : chemins pour features
  - `eval/train_rf.yaml` : config du modèle (RF/SVM/XGB, split, métriques)
  - `eval/report.yaml` : config du rapport final
  - `xai/attributions.yaml` : config XAI (SHAP, permutation)
  - `xai/plots.yaml` : config plots XAI
- `scripts/`
  - `preprocess.py` : prétraitement (montage gp2 + filtres + artefacts + REM)
  - `segment_rem.py` : extraction d’époques REM (4 s) sans artefacts
  - `extract_features.py` : features EEG, EMG, couplage, + fusion EMG-RBD
  - `build_dataset.py` : fusion features + labels Excel → dataset final
  - `train.py` : entraînement RF/SVM/XGB + métriques
  - `infer.py` : prédictions sur dataset final
  - `plot_roc_cm.py` : ROC + matrice de confusion
  - `xai_attributions.py` : SHAP + permutation importance
  - `xai_plots.py` : figures XAI (barplots)
  - `group_stats.py` : stats groupales (optionnel)
  - `build_reports.py` : rapport final (Markdown) avec figures
- `src/`
  - `data/preprocessing.py` : logique détaillée du prétraitement (montage gp2, filtres, hypno, YASA)
  - `features/` : extraction de features (EEG, EMG, EMG-RBD, couplage EEG–EMG)
  - `viz/metrics_plots.py`, `viz/plots.py`, `viz/reports.py` : utilitaires de visualisation / rapports
  - `utils/config.py`, `utils/logging.py`, `utils/seed.py` : utilitaires

---

## 1. Prétraitement

### 1.1. Config

`configs/preproc/default.yaml` (exemple) :

```yaml
raw_root: /data/lab/projet_X/raw
out_root: data/processed/preproc
hypno_root: /data/lab/projet_X/hypnos

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
