from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
05_build_dataset.py  -  Construction du dataset final
=============================================================================

Ce script est AUTONOME : il ne dépend d'aucun dossier src/, ni de fichier YAML.

Ce qu'il fait :
  1. Charge les features EEG (CSV sortie de 04_extract_features.py)
  2. Charge les marqueurs EMG-RBD (CSV sortie de 03_segment_rswa.py) - optionnel
  3. Fusionne EEG + EMG par epoch (alignement sur epoch_start_sec)
  4. Charge et fusionne les labels patients (fichier TXT/CSV)
  5. Encode les labels en entier (SYN=0, Narco=1, TCSPi=2, EAI=3)
  6. Sauvegarde le dataset final

Sortie :
  - OUTPUT_CSV      : dataset final (1 ligne par patient ou par epoch)
  - *_alignment_report.csv : rapport d'alignement EEG/EMG
  - patients_without_labels.txt : patients sans label (si existants)

Comment utiliser ce script :
  1. Modifier les chemins dans la section "PARAMÈTRES" ci-dessous
  2. Lancer :  python scripts/scripts_fonctionnels/05_build_dataset.py 2>&1 | tee logfiles/log_05_build_dataset.txt

Dépendances requises :
  pip install pandas numpy

=============================================================================

INPUTS
------
EEG_FEATURES_CSV : features EEG patient-level (sortie de 04_extract_features.py)
                   Exemple : results/eeg/eeg_features.csv
 
EMG_RBD_CSV      : marqueurs RSWA/EMG epoch-level (sortie de 03_segment_rswa.py)
                   Exemple : results/emg/rbd_emg_events_and_summary_4s_per_channel.csv
                   (fichier absent > dataset sans features EMG)
 
LABELS_TXT       : labels cliniques patients
                   Format   : CSV avec colonnes patient_id, label_str
                              OU TXT sans header (2 colonnes séparées par virgule)
                   Exemple  : data/patients_label.txt
                   Classes  : SYN=0 | Narco=1 | TCSPi=2 | EAI=3
 
OUTPUTS
-------
OUTPUT_CSV : dataset final prêt pour l'entraînement
             Chemin  : results/dataset_final.csv
             Colonnes : toutes les features EEG + EMG + label_str + label_id
 
<OUTPUT_CSV_stem>_alignment_report.csv
             Rapport d'alignement EEG/EMG (patient_id, epoch_start_sec, label_str)
 
patients_without_labels.txt
             Liste des patient_id sans label dans LABELS_TXT (si applicable)
 
DÉPENDANCES PIPELINE
--------------------
Ce script est l'ÉTAPE 5 (dernière). Il requiert les sorties de 03 et 04.
Sa sortie alimente :
  > 05_build_dataset.py > scripts d'entraînement (train_tabular_models, etc.)
  > NB_C_classification.ipynb (visualisation et classification)
"""

# ============================================================================
# PARAMÈTRES  <<<  À MODIFIER SELON VOTRE LA CONFIGURATION SOUHAITÉE
# ============================================================================

# CSV des features EEG (sortie de 04_extract_features.py)
EEG_FEATURES_CSV = r"c:\dev\Cerco_studies\data\eeg_features.csv"

# CSV des marqueurs EMG-RBD (sortie de 03_segment_rswa.py)
EMG_RBD_CSV = r"c:\dev\Cerco_studies\data\rbd_emg_events_and_summary_4s_per_channel.csv"

# Fichier de labels patients (colonnes : patient_id, label_str)
# Formats acceptés : CSV avec ou sans header, TXT séparé par virgule
LABELS_TXT = r"c:\dev\Cerco_studies\data\patients_label.txt"

# Fichier CSV de sortie (dataset final)
OUTPUT_CSV = r"c:\dev\Cerco_studies\data\dataset_final.csv"

# Tolérance (secondes) pour l'alignement EEG/EMG sur epoch_start_sec
TIME_TOL = 0.25

# ============================================================================
# IMPORTS
# ============================================================================


import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ============================================================================
# LOGGER
# ============================================================================

def _get_logger(name: str = "build_dataset") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


# ============================================================================
# CHARGEMENT ET NORMALISATION DES LABELS
# ============================================================================

def _load_labels(txt_path: Path) -> pd.DataFrame:
    """
    Charge un fichier de labels patients et encode les classes.

    Format accepté :
      - CSV avec colonnes patient_id / label_str (ou diagnostic / label)
      - TXT sans header : 2 colonnes (patient_id, label_str)

    Encodage des macro-classes (ordre fixe) :
      SYN=0, Narco=1, TCSPi=2, EAI=3

    Mapping tolérant :
      PARK / MPI / AMS / DCL / DLB / PAF > SYN
      NARCO                               > Narco
      TCSP / RBDI                         > TCSPi
      EAI / ENCEPHALITE                   > EAI
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"Fichier labels introuvable : {txt_path}")

    # Tentative avec header
    try:
        df = pd.read_csv(txt_path)
        cols_lower = {c.lower(): c for c in df.columns}

        if "patient_id" not in cols_lower and "identifiant" not in cols_lower:
            raise ValueError("Pas de colonne patient_id.")

        id_col = cols_lower.get("patient_id") or cols_lower.get("identifiant")

        if "label_str" in cols_lower:
            label_col = cols_lower["label_str"]
        elif "diagnostic" in cols_lower:
            label_col = cols_lower["diagnostic"]
        elif "label" in cols_lower:
            label_col = cols_lower["label"]
        else:
            other = [c for c in df.columns if c != id_col]
            if len(other) != 1:
                raise ValueError("Impossible d'inférer la colonne label.")
            label_col = other[0]

        out = pd.DataFrame({
            "patient_id": df[id_col].astype(str).str.strip(),
            "label_str":  df[label_col].astype(str).str.strip(),
        })

    except Exception:
        # Lecture sans header
        df = pd.read_csv(txt_path, header=None, names=["patient_id", "label_str"])
        out = pd.DataFrame({
            "patient_id": df["patient_id"].astype(str).str.strip(),
            "label_str":  df["label_str"].astype(str).str.strip(),
        })

    out = out.drop_duplicates(subset=["patient_id"], keep="first")

    # Mapping vers macro-classes
    macro_order = ["SYN", "Narco", "TCSPi", "EAI"]

    def _to_macro(s_raw: str) -> str:
        s_up = s_raw.upper()
        if any(k in s_up for k in ("PARK", "MPI", "AMS", "DCL", "DLB", "PAF")):
            return "SYN"
        if "NARCO" in s_up:
            return "Narco"
        if "TCSP" in s_up or "RBDI" in s_up:
            return "TCSPi"
        if "EAI" in s_up or "ENCEPHALITE" in s_up:
            return "EAI"
        if s_up in {"SYN", "NARCO", "TCSPI", "EAI"}:
            return "TCSPi" if s_up == "TCSPI" else s_raw
        return s_raw

    out["label_str"] = out["label_str"].map(_to_macro)
    cats = pd.Categorical(out["label_str"], categories=macro_order)
    out["label_id"] = cats.codes.astype(int)

    return out


# ============================================================================
# AGRÉGATION EMG-RBD PAR EPOCH
# ============================================================================

def _aggregate_emg_rbd(emg_df: pd.DataFrame, log, time_tol: float = 0.25) -> pd.DataFrame:
    """
    Réduit le CSV EMG long (par canal) à 1 ligne par epoch (agrégation sur les canaux).

    Colonnes produites :
      epoch_start_sec, epoch_end_sec, emg_phasic_ratio_mean,
      emg_tonic_ratio_mean, emg_phasic_count_sum, emg_rswa_any,
      emg_very_phasic_any, emg_n_channels
    """
    if emg_df.empty or "type" not in emg_df.columns:
        return pd.DataFrame()

    df = emg_df[emg_df["type"] == "REM_EPOCH_4S"].copy()
    if df.empty:
        return df

    required = ["patient_id", "epoch_start_sec", "epoch_end_sec",
                "phasic_ratio", "tonic_ratio", "rswa", "very_phasic",
                "phasic_count", "channel"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV EMG : {missing}")

    df["patient_id"]    = df["patient_id"].astype(str).str.strip()
    df["epoch_start_sec"] = df["epoch_start_sec"].astype(float)
    df["epoch_end_sec"]   = df["epoch_end_sec"].astype(float)
    df["phasic_ratio"]    = df["phasic_ratio"].astype(float)
    df["tonic_ratio"]     = df["tonic_ratio"].astype(float)
    df["phasic_count"]    = df["phasic_count"].astype(float)

    for col in ("rswa", "very_phasic"):
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.strip().str.lower().isin(
                ["1", "true", "t", "yes", "y"])

    df["epoch_start_round"] = (df["epoch_start_sec"] / float(time_tol)).round().astype(int)

    agg = df.groupby(["patient_id", "epoch_start_round"], as_index=False).agg(
        epoch_start_sec     =("epoch_start_sec",  "mean"),
        epoch_end_sec       =("epoch_end_sec",    "mean"),
        emg_phasic_ratio_mean=("phasic_ratio",    "mean"),
        emg_tonic_ratio_mean =("tonic_ratio",     "mean"),
        emg_phasic_count_sum =("phasic_count",    "sum"),
        emg_rswa_any         =("rswa",            "any"),
        emg_very_phasic_any  =("very_phasic",     "any"),
        emg_n_channels       =("channel",         "nunique"),
    )

    return agg


# ============================================================================
# CONSTRUCTION DU DATASET FINAL
# ============================================================================

def build_dataset() -> None:
    log = _get_logger()

    eeg_features_csv = Path(EEG_FEATURES_CSV)
    emg_rbd_csv      = Path(EMG_RBD_CSV)
    labels_txt       = Path(LABELS_TXT)
    out_csv          = Path(OUTPUT_CSV)
    time_tol         = float(TIME_TOL)

    # 1) EEG features
    log.info(f"Lecture EEG features : {eeg_features_csv}")
    if not eeg_features_csv.exists():
        log.error(f"Fichier EEG features introuvable : {eeg_features_csv}")
        sys.exit(1)

    eeg = pd.read_csv(eeg_features_csv)
    eeg["patient_id"] = eeg["patient_id"].astype(str).str.strip()
    has_time        = "epoch_start_sec" in eeg.columns
    has_epoch_index = "epoch_index" in eeg.columns

    if has_time:
        eeg["epoch_start_sec"] = eeg["epoch_start_sec"].astype(float)

    log.info(f"EEG : {eeg.shape} | has_time={has_time} | has_epoch_index={has_epoch_index}")

    # 2) EMG-RBD
    emg_agg = None
    if emg_rbd_csv.exists():
        log.info(f"Lecture EMG-RBD : {emg_rbd_csv}")
        emg = pd.read_csv(emg_rbd_csv)
        if not emg.empty:
            emg_agg = _aggregate_emg_rbd(emg, log, time_tol=time_tol)
            if emg_agg is not None and not emg_agg.empty:
                log.info(f"EMG agrégé : {emg_agg.shape}")
    else:
        log.warning(f"CSV EMG-RBD introuvable : {emg_rbd_csv} > EMG ignoré.")

    # 3) Labels
    log.info(f"Lecture labels : {labels_txt}")
    labels = _load_labels(labels_txt)
    log.info(f"Labels : {labels['patient_id'].nunique()} patients, "
             f"{labels['label_str'].nunique()} classes.")

    # 4) Fusion EEG + EMG
    if emg_agg is not None and not emg_agg.empty:
        if has_time:
            emg_agg["epoch_start_round"] = (
                emg_agg["epoch_start_sec"] / float(time_tol)).round().astype(int)
            eeg["epoch_start_round"] = (
                eeg["epoch_start_sec"] / float(time_tol)).round().astype(int)

            dataset = pd.merge(
                eeg,
                emg_agg.drop(columns=["epoch_start_sec", "epoch_end_sec"]),
                on=["patient_id", "epoch_start_round"],
                how="left",
            )
            dataset.drop(columns=["epoch_start_round"], inplace=True)
            log.info("Fusion EEG+EMG via (patient_id, epoch_start_sec ~).")

        elif has_epoch_index and "epoch_index" in emg_agg.columns:
            dataset = pd.merge(eeg, emg_agg, on=["patient_id", "epoch_index"],
                               how="left", suffixes=("", "_emg"))
            log.info("Fusion EEG+EMG via (patient_id, epoch_index).")

        else:
            log.warning("Impossible d'aligner EEG et EMG > EMG ignoré.")
            dataset = eeg.copy()
    else:
        dataset = eeg.copy()
        log.info("Pas d'EMG à fusionner.")

    # 5) Fusion labels
    dataset = pd.merge(dataset, labels, on="patient_id", how="left")

    # 6) Rapport patients sans label
    n_no_label = int(dataset["label_str"].isna().sum()) if "label_str" in dataset.columns else len(dataset)
    if n_no_label > 0:
        log.warning(f"{n_no_label}/{len(dataset)} lignes sans label.")
        missing_patients = sorted(dataset.loc[dataset["label_str"].isna(), "patient_id"]
                                  .astype(str).dropna().unique())
        missing_path = out_csv.parent / "patients_without_labels.txt"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(missing_path, "w", encoding="utf-8") as f:
            for pid in missing_patients:
                f.write(f"{pid}\n")
        log.info(f"Patients sans label > {missing_path}")
    else:
        log.info("Tous les patients ont un label.")

    # 7) Sauvegarde
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out_csv, index=False)
    log.info(f"Dataset final > {out_csv} {dataset.shape}")

    # 8) Rapport d'alignement
    rep_cols = ["patient_id"]
    for c in ["epoch_index", "epoch_start_sec", "emg_phasic_ratio_mean",
              "emg_tonic_ratio_mean", "emg_rswa_any", "label_str"]:
        if c in dataset.columns:
            rep_cols.append(c)

    report_path = out_csv.parent / (out_csv.stem + "_alignment_report.csv")
    dataset[rep_cols].to_csv(report_path, index=False)
    log.info(f"Rapport d'alignement > {report_path}")


# ============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    build_dataset()