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
#   - Ergonomie : une seule commande ("make prepa_data") lance la chaîne pré-ML.
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
#       make preprocess PY=python3.10
#
# ==============================================================================

# ------------------------------------------------------------------------------
# Variables d'environnement / paramètres modifiables
# ------------------------------------------------------------------------------

# Interpréteur Python utilisé pour exécuter les scripts.
# ON peut le surcharger au moment de l'appel :
#   make preprocess PY=python3
PY=python

# Répertoires "raw" et "processed" (valeurs par défaut).
# NOTE: Actuellement ces variables ne sont PAS utilisées dans les règles,
# car les scripts lisent leurs chemins via leurs YAML / valeurs internes.
# => Tant que les scripts n'ont pas une option --raw-root / --proc-root,
# ces variables sont "superflues".
RAW_ROOT ?= ../documents/EEG/raw
PROC_ROOT ?= ../documents/EEG/preprocessed/XAI/data

# Dossier de logs (toutes les sorties STDOUT/STDERR y sont enregistrées).
LOG_DIR = data/logs

# Nombre de workers par défaut pour les scripts qui supportent --n_workers.
# Surcharge typique:
#   make prepa_data NWORKERS=30
NWORKERS ?= 10

# Cible par défaut si tu fais juste "make" sans argument.
# Ici, ça lance uniquement le pipeline pré-ML.
.DEFAULT_GOAL := prepa_data


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
# Fin du Makefile: uniquement pipeline pré-ML dans cette branche.
# ==============================================================================
