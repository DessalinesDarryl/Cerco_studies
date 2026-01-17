# ==============================================================================
# Makefile
# ==============================================================================
#
# Ce Makefile sert de "table de commandes" (pipeline CLI) pour enchaîner :
#   1) Préparation des données (préproc EEG -> epochs REM -> features EEG/EMG -> dataset final)
#   2) Entraînement de modèles (tabulaires et deep)
#   3) Évaluation (ROC / confusion matrix) + XAI + stats + rapport
#
# Il vise 2 objectifs :
#   - Reproductibilité : relancer les mêmes commandes avec les mêmes configs YAML.
#   - Ergonomie : une seule commande ("make all_rf") lance toute la chaîne.
#
# IMPORTANT (limites actuelles de ce Makefile) :
#   - Les règles ne déclarent pas de dépendances fichiers (pas de "targets outputs").
#     => Make ne sait pas si une étape est "déjà faite". Il relancera l’étape à chaque appel.
#   - Les variables RAW_ROOT et PROC_ROOT sont définies mais NE SONT PAS UTILISÉES
#     par les commandes (donc "superflu" tant que les scripts ne s’appuient pas dessus).
#   - Les logs sont capturés via "tee" dans logs/*.txt.
#
# Astuce d’usage :
#   - Pour changer le nombre de workers :
#       make preprocess NWORKERS=20
#   - Pour changer l’interpréteur python :
#       make train_rf PY=python3.10
#
# ==============================================================================

# ------------------------------------------------------------------------------
# Variables d'environnement / paramètres modifiables
# ------------------------------------------------------------------------------

# Interpréteur Python utilisé pour exécuter les scripts.
# ON peut le surcharger au moment de l'appel :
#   make train_rf PY=python3
PY=python

# Répertoires "raw" et "processed" (valeurs par défaut).
# NOTE: Actuellement ces variables ne sont PAS utilisées dans les règles,
# car les scripts lisent leurs chemins via leurs YAML / valeurs internes.
# => Tant que les scripts n'ont pas une option --raw-root / --proc-root,
# ces variables sont "superflues".
RAW_ROOT ?= ../documents/EEG/raw
PROC_ROOT ?= ../documents/EEG/preprocessed/XAI/data

# Dossier de logs (toutes les sorties STDOUT/STDERR y sont enregistrées).
LOG_DIR = logs

# Nombre de workers par défaut pour les scripts qui supportent --n_workers.
# Surcharge typique:
#   make prepa_data NWORKERS=30
NWORKERS ?= 10

# Cible par défaut si tu fais juste "make" sans argument.
# Ici, ça lance le pipeline complet basé sur le modèle RF.
.DEFAULT_GOAL := all_rf


# ------------------------------------------------------------------------------
# Règle utilitaire : créer le dossier logs (order-only prerequisite)
# ------------------------------------------------------------------------------

# Cette "cible" ne construit pas un fichier, elle crée un dossier.
# Elle est utilisée comme dépendance "order-only" via le symbole "|".
# => Cela signifie : "avant de lancer la recette, assure-toi que logs/ existe",
# sans considérer logs/ comme un "vrai" input dont la modification retrigger l'étape.
$(LOG_DIR):
	mkdir -p $(LOG_DIR)


# ==============================================================================
# 1) Préparation des données
# ==============================================================================

# ------------------------------------------------------------------------------
# preprocess
# ------------------------------------------------------------------------------
# Lance le preprocessing EEG à partir de la config YAML:
#   configs/preproc/default.yaml
#
# Objectif typique:
#   - chargement des données brutes
#   - filtrage / notch
#   - ICA / nettoyage (selon la config)
#   - ajout éventuel d’annotations d’artefacts
#   - export des fichiers preprocessed (souvent .fif)
#
# Le log complet est écrit dans logs/logs_preprocess.txt
preprocess: | $(LOG_DIR)
	$(PY) scripts/preprocess.py \
		--config configs/preproc/default.yaml \
		--n_workers $(NWORKERS) \
		2>&1 | tee $(LOG_DIR)/logs_preprocess.txt


# ------------------------------------------------------------------------------
# segment_rem
# ------------------------------------------------------------------------------
# Segmente uniquement le REM propre (REM - ARTEFACT) en epochs fixes (souvent 4s),
# via la config YAML:
#   configs/preproc/segment_rem.yaml
#
# Produit typiquement:
#   - fichiers *_REM-epo.fif
#
# Le log complet est écrit dans logs/logs_segment_rem.txt
segment_rem: | $(LOG_DIR)
	$(PY) scripts/segment_rem.py \
		--config configs/preproc/segment_rem.yaml \
		--n_workers $(NWORKERS) \
		2>&1 | tee $(LOG_DIR)/logs_segment_rem.txt

# TODO : Ajouter la segmentation RSWA si tu veux générer des fichiers RSWA-only
# (par exemple segment_rswa.py / segment_rswa_from_csv.py).
# Exemple (si tu ajoutes un script segment_rswa.py) :
# segment_rswa: | $(LOG_DIR)
# 	$(PY) scripts/segment_rswa.py --config ... 2>&1 | tee $(LOG_DIR)/logs_segment_rswa.txt


# ------------------------------------------------------------------------------
# rbd_emg
# ------------------------------------------------------------------------------
# Calcule les marqueurs RSWA / phasic / tonic basés sur l’EMG pendant le REM.
#
# Particularités :
#   - Il pointe directement vers src/features/emg_rbd.py (script exécutable)
#   - La liste de canaux EMG est fournie en dur ici.
#
# Attention :
#   - Si la liste de canaux change selon les patients, mieux vaut passer
#     cette liste via YAML (config) plutôt que la coder ici.
rbd_emg: | $(LOG_DIR)
	$(PY) src/features/emg_rbd.py \
		--emg_channels Menton JAMBG JAMBD EMG1 EMG2 \
		2>&1 | tee $(LOG_DIR)/logs_rbd_emg.txt


# ------------------------------------------------------------------------------
# features
# ------------------------------------------------------------------------------
# Extrait les features EEG (et éventuellement fusionne des features EMG patient-level
# si l'extract_features.py le fait).
#
# Config YAML attendue:
#   configs/preproc/features.yaml
#
# Produit typiquement:
#   data/processed/features/features.csv  (ou features.csv selon le script)
features: | $(LOG_DIR)
	$(PY) scripts/extract_features.py \
		--config configs/preproc/features.yaml \
		--n_workers $(NWORKERS) \
		2>&1 | tee $(LOG_DIR)/logs_extract_features.txt


# ------------------------------------------------------------------------------
# dataset
# ------------------------------------------------------------------------------
# Fusionne features EEG + EMG-RBD + labels pour produire dataset_final.csv.
#
# Points importants :
#   - Ici tu fournis des chemins explicites (--eeg-features, --emg-rbd, --out-csv)
#   - Tu ne fournis PAS --labels-txt (donc build_dataset.py utilisera son défaut)
#     => Potentiel problème si le défaut n’est pas le bon.
#
# Reco :
#   - Ajoute explicitement --labels-txt data/patients_label.txt
#     pour éviter les surprises.
dataset: | $(LOG_DIR)
	$(PY) scripts/build_dataset.py \
		--eeg-features data/processed/features/features.csv \
		--emg-rbd data/processed/rbd/rbd_emg_events_and_summary_4s_per_channel.csv \
		--out-csv data/processed/features/dataset_final.csv \
		2>&1 | tee $(LOG_DIR)/logs_dataset.txt


# ------------------------------------------------------------------------------
# prepa_data
# ------------------------------------------------------------------------------
# "Meta-cible" : enchaîne toutes les étapes nécessaires pour construire le dataset.
#
# Ordre choisi :
#   preprocess -> segment_rem -> rbd_emg -> features -> dataset
#
# Hypothèses implicites :
#   - preprocess produit les fichiers que segment_rem attend
#   - segment_rem produit les *_REM-epo.fif que extract_features attend
#   - rbd_emg produit le CSV EMG-RBD que build_dataset attend
prepa_data: preprocess segment_rem rbd_emg features dataset


# ==============================================================================
# 2) Entraînement des modèles
# ==============================================================================

# ------------------------------------------------------------------------------
# Modèles tabulaires
# ------------------------------------------------------------------------------
# Le même script train_tabular_models.py est utilisé pour RF / KNN / XGB,
# piloté par une config YAML différente.

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


# ------------------------------------------------------------------------------
# Modèles deep
# ------------------------------------------------------------------------------
# Entraînement CNN/RNN sur données tabulaires (features) en CV, puis fit final.

train_cnn: | $(LOG_DIR)
	$(PY) scripts/train_deep_models.py \
		--config configs/eval/train_cnn.yaml \
		2>&1 | tee $(LOG_DIR)/logs_train_cnn.txt

train_rnn: | $(LOG_DIR)
	$(PY) scripts/train_deep_models.py \
		--config configs/eval/train_rnn.yaml \
		2>&1 | tee $(LOG_DIR)/logs_train_rnn.txt


# ==============================================================================
# 3) Évaluation, XAI, stats, rapport
# ==============================================================================

# ------------------------------------------------------------------------------
# plot_metrics
# ------------------------------------------------------------------------------
# Génère des figures de métriques (ROC multiclass + confusion matrix).
# Dépend des outputs de training (prédictions / checkpoint / etc.) selon l'implémentation.
plot_metrics: | $(LOG_DIR)
	$(PY) scripts/plot_roc_cm.py \
		--config configs/eval/plots_metrics.yaml \
		2>&1 | tee $(LOG_DIR)/logs_plot_metrics.txt


# ------------------------------------------------------------------------------
# xai_attr
# ------------------------------------------------------------------------------
# Lance les attributions globales XAI (Permutation importance + SHAP).
# Remarque : c’est "prévu pour RF/XGB" car SHAP TreeExplainer est rapide sur les arbres.
xai_attr: | $(LOG_DIR)
	$(PY) scripts/xai_attributions.py \
		--config configs/xai/attributions.yaml \
		2>&1 | tee $(LOG_DIR)/logs_xai_attr.txt


# ------------------------------------------------------------------------------
# xai_plots
# ------------------------------------------------------------------------------
# Génère les figures XAI (barplots top-k) à partir des CSV produits par xai_attr.
xai_plots: | $(LOG_DIR)
	$(PY) scripts/xai_plots.py \
		--config configs/xai/plots.yaml \
		2>&1 | tee $(LOG_DIR)/logs_xai_plots.txt


# ------------------------------------------------------------------------------
# stats
# ------------------------------------------------------------------------------
# Lance les analyses statistiques de groupe (ANOVA/Kruskal/Dunn/Tukey, etc.)
stats: | $(LOG_DIR)
	$(PY) scripts/group_stats.py \
		--config configs/eval/stats.yaml \
		2>&1 | tee $(LOG_DIR)/logs_stats.txt


# ------------------------------------------------------------------------------
# report
# ------------------------------------------------------------------------------
# Construit un rapport (markdown) en agrégeant :
#   - sorties XAI
#   - stats
#   - figures
#
# Ici les chemins sont passés en CLI (pas via YAML).
report: | $(LOG_DIR)
	$(PY) scripts/build_reports.py \
		--xai-root outputs/xai \
		--stats-root outputs/stats \
		--fig-root outputs/figures \
		--out outputs/reports \
		2>&1 | tee $(LOG_DIR)/logs_report.txt


# ==============================================================================
# 4) Pipelines "macro" par modèle
# ==============================================================================

# ------------------------------------------------------------------------------
# step_* : Entraîner + évaluer + produire XAI/stats/rapport pour un modèle
# ------------------------------------------------------------------------------
# Note : actuellement xai_attr/xai_plots sont aussi appelés dans step_knn,
# mais KNN n’est pas un modèle "tree" => SHAP KernelExplainer serait très lent.
# Donc, pour KNN, xai_attr est potentiellement superflu (ou à configurer pour skip).
step_rf:  train_rf  plot_metrics xai_attr xai_plots stats report
step_knn: train_knn plot_metrics xai_attr xai_plots stats report
step_xgb: train_xgb plot_metrics xai_attr xai_plots stats report

# Pour deep, tu n’as pas branché xai/stats (choix OK).
# Si tu veux des stats/rapport, tu les gardes.
step_cnn: train_cnn plot_metrics report
step_rnn: train_rnn plot_metrics report


# ------------------------------------------------------------------------------
# all_* : Pipeline complet (data + modèle)
# ------------------------------------------------------------------------------
# Lance prepa_data puis le step_ correspondant.
all_rf:  prepa_data step_rf
all_knn: prepa_data step_knn
all_xgb: prepa_data step_xgb
all_cnn: prepa_data step_cnn
all_rnn: prepa_data step_rnn
