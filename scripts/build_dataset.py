#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_dataset.py
================

But
---
Construire un **dataset final prêt pour l’entraînement** en fusionnant :

1) **Features EEG** (CSV)
   - soit **patient-level** (1 ligne par patient)
   - soit **epoch-level** (1 ligne par epoch REM, par patient)
   -> Le script s’adapte selon la présence de colonnes d’alignement (temps ou epoch_index).

2) **Indicateurs EMG-RBD** (CSV) issus de `emg_rbd.py`
   - On utilise uniquement les lignes `type == "REM_EPOCH_4S"` (résumés d’epochs REM de 4s)
   - On agrège d’abord les canaux EMG (moyennes / any / somme), puis on fusionne à l’EEG.

3) **Labels patients** (TXT/CSV)
   - Deux colonnes `patient_id,label_str` (avec ou sans header)
   - On normalise et on encode `label_id` selon un ordre macro fixe :
       SYN=0, Narco=1, TCSPi=2, EAI=3

Entrées (par défaut via CLI)
----------------------------
--eeg-features : data/processed/features/eeg_features.csv
--emg-rbd      : data/processed/rbd/rbd_emg_events_and_summary_4s_per_channel.csv
--labels-txt   : data/patients_label.txt
--out-csv      : data/processed/features/dataset_final.csv
--time-tol     : 0.25  (tolérance en secondes pour l’alignement par temps)

Sorties
-------
1) Dataset final : `--out-csv`
2) Rapport d’alignement : `<out_csv_stem>_alignment_report.csv`
3) Si des patients sont sans label :
   - `patients_without_labels.txt` dans le même dossier que out_csv

Alignement EEG / EMG
--------------------
Le script essaie d’aligner EEG et EMG de la façon la plus fiable possible, selon les colonnes présentes :

Cas A) Alignement par temps (recommandé)
    - Si EEG possède `epoch_start_sec` :
        -> On arrondit `epoch_start_sec` par pas de `time_tol` (ex: 0.25 s)
        -> On fusionne sur (patient_id, epoch_start_round)

Cas B) Alignement par index d’epoch
    - Si EEG ne possède pas `epoch_start_sec` mais possède `epoch_index`
      ET si EMG possède aussi `epoch_index` :
        -> On fusionne sur (patient_id, epoch_index)

Cas C) Impossible d’aligner
    - Si aucune clé fiable n’est disponible :
        -> EMG est ignoré (dataset = EEG + labels)

Hypothèses / points d’attention
-------------------------------
- Le fichier EMG doit contenir (au minimum) :
    patient_id, type, epoch_start_sec, epoch_end_sec, channel,
    phasic_ratio, tonic_ratio, rswa, very_phasic, phasic_count
- L’agrégation EMG par epoch résume les canaux :
    - phasic_ratio_mean : moyenne sur canaux
    - tonic_ratio_mean  : moyenne sur canaux
    - phasic_count_sum  : somme sur canaux
    - rswa_any          : True si au moins un canal RSWA
    - very_phasic_any   : True si au moins un canal très phasique
    - n_channels        : nb canaux EMG contribuant à cette epoch

- Le mapping labels -> macros est volontairement "tolérant" :
    certains libellés (PARK, AMS, DCL/DLB, etc.) sont regroupés en SYN.

Exécution
---------
python scripts/build_dataset.py \
  --eeg-features data/processed/features/eeg_features.csv \
  --emg-rbd data/processed/rbd/rbd_emg_events_and_summary_4s_per_channel.csv \
  --labels-txt data/patients_label.txt \
  --out-csv data/processed/features/dataset_final.csv \
  --time-tol 0.25
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse
import numpy as np
import pandas as pd

from src.utils.logging import get_logger


# ---------------------------------------------------------------------
# Chargement et normalisation des labels patients
# ---------------------------------------------------------------------
def load_labels_from_txt(txt_path: Path) -> pd.DataFrame:
    """
    Charge un fichier de labels (TXT/CSV) et retourne un DataFrame standardisé.

    Format attendu
    --------------
    Le fichier doit contenir au moins deux colonnes :
      - patient_id
      - label_str

    Ce loader accepte :
    - un fichier AVEC header (patient_id, label_str, diagnostic, label, etc.)
    - un fichier SANS header (2 colonnes dans l’ordre : patient_id,label_str)

    Normalisations effectuées
    -------------------------
    - patient_id : cast str + strip
    - label_str  : cast str + strip
    - suppression des doublons : 1 label par patient (keep first)

    Encodage label_id
    -----------------
    On force un ordre macro pour produire un label numérique stable :
        SYN=0, Narco=1, TCSPi=2, EAI=3

    Tout label non reconnu peut aboutir à -1 (cats.codes) si hors catégories.

    Mapping "tolérant" vers macros
    ------------------------------
    Si label_str contient certains mots-clés, on mappe vers une macro :
      - PARK, MPI, AMS, DCL, DLB, PAF => SYN
      - NARCO => Narco
      - TCSP, RBDI => TCSPi
      - EAI, ENCEPHALITE => EAI

    Paramètres
    ----------
    txt_path : Path
        Chemin vers le fichier labels.

    Retours
    -------
    pd.DataFrame
        Colonnes :
          - patient_id
          - label_str (macro normalisée)
          - label_id  (int)
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"Fichier labels TXT introuvable : {txt_path}")

    # 1) Tentative de lecture avec header explicite
    try:
        df = pd.read_csv(txt_path)
        cols_lower = {c.lower(): c for c in df.columns}

        # Si aucune colonne patient_id/identifiant n’est trouvée, on bascule en lecture brute
        if "patient_id" not in cols_lower and "identifiant" not in cols_lower:
            raise ValueError("Pas de colonnes patient_id/identifiant détectées, on tente sans header.")

        # Détection de la colonne identifiant patient
        id_col = cols_lower.get("patient_id") or cols_lower.get("identifiant")

        # Détection de la colonne label clinique
        if "label_str" in cols_lower:
            label_col = cols_lower["label_str"]
        elif "diagnostic" in cols_lower:
            label_col = cols_lower["diagnostic"]
        elif "label" in cols_lower:
            label_col = cols_lower["label"]
        else:
            # Heuristique : s’il ne reste qu’une seule autre colonne, on l’utilise comme label
            other_cols = [c for c in df.columns if c != id_col]
            if len(other_cols) != 1:
                raise ValueError("Impossible d'inférer la colonne de label dans le fichier TXT.")
            label_col = other_cols[0]

        out = pd.DataFrame()
        out["patient_id"] = df[id_col].astype(str).str.strip()
        out["label_str"] = df[label_col].astype(str).str.strip()

    except Exception:
        # 2) Lecture brute sans header : patient_id,label_str
        df = pd.read_csv(txt_path, header=None, names=["patient_id", "label_str"])
        out = pd.DataFrame()
        out["patient_id"] = df["patient_id"].astype(str).str.strip()
        out["label_str"] = df["label_str"].astype(str).str.strip()

    # 3) Une seule ligne de label conservée par patient
    out = out.drop_duplicates(subset=["patient_id"], keep="first")

    # 4) Définition de l’ordre fixe des macro-classes
    macro_order = ["SYN", "Narco", "TCSPi", "EAI"]

    norm = out["label_str"].astype(str).str.strip()
    norm_upper = norm.str.upper()

    def _to_macro(s_up: str, s_raw: str) -> str:
        """
        Mappe un label texte (potentiellement hétérogène) vers une macro-classe.

        Args:
            s_up: label upper-case
            s_raw: label original (non upper) (sert à conserver une casse “propre”)

        Returns:
            Une chaîne label_str normalisée (macro si reconnue, sinon raw).
        """
        if "PARK" in s_up or "MPI" in s_up or "AMS" in s_up or "DCL" in s_up or "DLB" in s_up or "PAF" in s_up:
            return "SYN"
        if "NARCO" in s_up:
            return "Narco"
        if "TCSP" in s_up or "RBDI" in s_up:
            return "TCSPi"
        if "EAI" in s_up or "ENCEPHALITE" in s_up:
            return "EAI"

        # Si déjà une macro, on la garde (en corrigeant TCSPi)
        if s_up in {"SYN", "NARCO", "TCSPI", "EAI"}:
            return "TCSPi" if s_up == "TCSPI" else s_raw

        return s_raw

    out["label_str"] = [_to_macro(up, raw) for up, raw in zip(norm_upper, norm)]

    # Encodage en catégories avec ordre imposé
    cats = pd.Categorical(out["label_str"], categories=macro_order)
    out["label_id"] = cats.codes.astype(int)

    return out


# ---------------------------------------------------------------------
# Fusion EMG-RBD (par époque) -> agrégats par patient + temps/epoch
# ---------------------------------------------------------------------
def aggregate_emg_rbd(emg_df: pd.DataFrame, log, time_tol: float = 0.25) -> pd.DataFrame:
    """
    Agrège le CSV EMG-RBD (sortie `emg_rbd.py`) à lRemember: by epoch.

    Objectif
    --------
    Le fichier EMG est typiquement "long" et contient :
      - plusieurs canaux (channel)
      - plusieurs types de lignes (PHASIC, REM_EPOCH_4S, etc.)
    Ici on veut une représentation **par epoch** (REM_EPOCH_4S),
    indépendante du canal, afin de pouvoir fusionner avec l’EEG.

    Étapes
    ------
    1) Filtrer `type == "REM_EPOCH_4S"`
    2) Vérifier les colonnes requises
    3) Nettoyer les types (float/bool)
    4) Créer une clé robuste d’alignement temporel :
         epoch_start_round = round(epoch_start_sec / time_tol)
       Exemple : time_tol=0.25 -> arrondi au quart de seconde
    5) groupby(patient_id, epoch_start_round) et agrégation :
       - epoch_start_sec mean (moyenne sur canaux)
       - epoch_end_sec mean
       - phasic_ratio mean
       - tonic_ratio mean
       - phasic_count sum (somme des canaux)
       - rswa any (True si au moins un canal RSWA)
       - very_phasic any
       - n_channels nunique

    Paramètres
    ----------
    emg_df : pd.DataFrame
        DataFrame brut du CSV EMG-RBD.
    log : logger
        Logger projet.
    time_tol : float
        Tolérance (s) pour construire une clé temporelle discrète.

    Retours
    -------
    pd.DataFrame
        Colonnes:
          - patient_id
          - epoch_start_round
          - epoch_start_sec, epoch_end_sec
          - emg_phasic_ratio_mean
          - emg_tonic_ratio_mean
          - emg_phasic_count_sum
          - emg_rswa_any
          - emg_very_phasic_any
          - emg_n_channels

    Raises
    ------
    ValueError
        Si des colonnes indispensables sont manquantes.
    """
    if emg_df.empty:
        log.warning("EMG-RBD DataFrame vide.")
        return emg_df

    if "type" not in emg_df.columns:
        log.warning("EMG-RBD sans colonne 'type'.")
        return pd.DataFrame()

    df_ep = emg_df[emg_df["type"] == "REM_EPOCH_4S"].copy()
    if df_ep.empty:
        log.warning("Aucune ligne type 'REM_EPOCH_4S' trouvée dans EMG-RBD.")
        return df_ep

    required = [
        "patient_id",
        "epoch_start_sec",
        "epoch_end_sec",
        "phasic_ratio",
        "tonic_ratio",
        "rswa",
        "very_phasic",
        "phasic_count",
        "channel",
    ]
    missing = [c for c in required if c not in df_ep.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans EMG-RBD (REM_EPOCH_4S) : {missing}")

    # Nettoyage types
    df_ep["patient_id"] = df_ep["patient_id"].astype(str).str.strip()

    df_ep["epoch_start_sec"] = df_ep["epoch_start_sec"].astype(float)
    df_ep["epoch_end_sec"] = df_ep["epoch_end_sec"].astype(float)
    df_ep["phasic_ratio"] = df_ep["phasic_ratio"].astype(float)
    df_ep["tonic_ratio"] = df_ep["tonic_ratio"].astype(float)
    df_ep["phasic_count"] = df_ep["phasic_count"].astype(float)

    # bool tolérant (au cas où c’est 0/1 ou "true"/"false")
    for col in ("rswa", "very_phasic"):
        if df_ep[col].dtype != bool:
            df_ep[col] = df_ep[col].astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes", "y"])

    # Clé temporelle discrète
    df_ep["epoch_start_round"] = (df_ep["epoch_start_sec"] / float(time_tol)).round().astype(int)

    grp = df_ep.groupby(["patient_id", "epoch_start_round"], as_index=False)

    agg = grp.agg(
        epoch_start_sec=("epoch_start_sec", "mean"),
        epoch_end_sec=("epoch_end_sec", "mean"),
        emg_phasic_ratio_mean=("phasic_ratio", "mean"),
        emg_tonic_ratio_mean=("tonic_ratio", "mean"),
        emg_phasic_count_sum=("phasic_count", "sum"),
        emg_rswa_any=("rswa", "any"),
        emg_very_phasic_any=("very_phasic", "any"),
        emg_n_channels=("channel", "nunique"),
    )

    return agg


# ---------------------------------------------------------------------
# Fusion EEG + EMG + labels
# ---------------------------------------------------------------------
def build_dataset(
    eeg_features_csv: Path,
    emg_rbd_csv: Path,
    labels_txt: Path,
    out_csv: Path,
    time_tol: float,
    log,
) -> None:
    """
    Construit le dataset final en fusionnant EEG, EMG et labels.

    Pipeline détaillé
    -----------------
    1) Lecture EEG features
       - vérifie patient_id
       - détecte si on peut aligner par temps (`epoch_start_sec`) ou par index (`epoch_index`)

    2) Lecture EMG-RBD
       - si fichier absent/vide -> EMG ignoré
       - sinon agrégation par epoch via `aggregate_emg_rbd`

    3) Lecture labels
       - via `load_labels_from_txt`
       - produit patient_id, label_str, label_id

    4) Fusion EEG + EMG
       - si EEG possède epoch_start_sec :
            merge sur (patient_id, epoch_start_round)
         sinon si EEG et EMG possèdent epoch_index :
            merge sur (patient_id, epoch_index)
         sinon :
            EMG ignoré

    5) Fusion dataset + labels (patient-level)
       - merge sur patient_id (left join)
       - log et export des patients sans labels (si existants)

    6) Sauvegarde dataset final + rapport d’alignement

    Paramètres
    ----------
    eeg_features_csv : Path
        CSV des features EEG.
    emg_rbd_csv : Path
        CSV EMG-RBD (sortie emg_rbd.py).
    labels_txt : Path
        Fichier de labels (patient_id,label_str).
    out_csv : Path
        Chemin de sortie dataset final.
    time_tol : float
        Tolérance temporelle (s) pour alignement sur epoch_start_sec.
    log : logger
        Logger projet.

    Returns
    -------
    None
    """
    # 1) EEG features
    log.info(f"Lecture EEG features : {eeg_features_csv}")
    eeg = pd.read_csv(eeg_features_csv)

    required_eeg = ["patient_id"]
    missing_eeg = [c for c in required_eeg if c not in eeg.columns]
    if missing_eeg:
        raise ValueError(f"Colonnes manquantes dans EEG features : {missing_eeg}")

    eeg["patient_id"] = eeg["patient_id"].astype(str).str.strip()

    has_time = "epoch_start_sec" in eeg.columns
    has_epoch_index = "epoch_index" in eeg.columns

    if has_time:
        eeg["epoch_start_sec"] = eeg["epoch_start_sec"].astype(float)

    log.info(f"EEG features : {eeg.shape[0]} lignes, {eeg.shape[1]} colonnes.")
    log.info(f"EEG alignement: has_time={has_time} | has_epoch_index={has_epoch_index}")

    # 2) EMG-RBD
    log.info(f"Lecture EMG-RBD : {emg_rbd_csv}")
    if emg_rbd_csv.exists():
        emg = pd.read_csv(emg_rbd_csv)
    else:
        log.warning(f"Fichier EMG-RBD introuvable : {emg_rbd_csv} → EMG ignoré.")
        emg = pd.DataFrame()

    if emg.empty:
        log.warning("EMG-RBD CSV vide → dataset sans features EMG.")
        emg_agg = None
    else:
        emg_agg = aggregate_emg_rbd(emg, log, time_tol=float(time_tol))
        log.info(f"EMG-RBD agrégé : {emg_agg.shape[0]} lignes, {emg_agg.shape[1]} colonnes.")

    # 3) Labels
    log.info(f"Lecture labels TXT : {labels_txt}")
    labels = load_labels_from_txt(labels_txt)

    log.info(
        f"Labels : {labels['patient_id'].nunique()} patients, "
        f"{labels['label_str'].nunique()} classes."
    )

    # 4) Merge EEG + EMG
    if emg_agg is not None and not emg_agg.empty:
        if has_time:
            # Alignement par temps discretisé
            emg_agg["epoch_start_round"] = (emg_agg["epoch_start_sec"] / float(time_tol)).round().astype(int)
            eeg["epoch_start_round"] = (eeg["epoch_start_sec"] / float(time_tol)).round().astype(int)

            dataset = pd.merge(
                eeg,
                emg_agg.drop(columns=["epoch_start_sec", "epoch_end_sec"]),
                on=["patient_id", "epoch_start_round"],
                how="left",
            )
            dataset.drop(columns=["epoch_start_round"], inplace=True)
            log.info("Fusion EEG+EMG réalisée via (patient_id, epoch_start_sec ~).")

        else:
            # Pas de temps dans EEG, tentative alignement par epoch_index
            if (not has_epoch_index) or ("epoch_index" not in emg.columns):
                log.warning(
                    "Pas de epoch_start_sec dans EEG features et/ou pas de epoch_index dans EMG-RBD. "
                    "Impossible d'aligner EEG et EMG par époque. EMG ignoré."
                )
                dataset = eeg.copy()
            else:
                emg_epochs = emg[emg["type"] == "REM_EPOCH_4S"].copy()
                emg_epochs["patient_id"] = emg_epochs["patient_id"].astype(str).str.strip()

                dataset = pd.merge(
                    eeg,
                    emg_epochs,
                    on=["patient_id", "epoch_index"],
                    how="left",
                    suffixes=("", "_emg"),
                )
                log.info("Fusion EEG+EMG réalisée via (patient_id, epoch_index).")
    else:
        dataset = eeg.copy()
        log.info("Pas d'EMG à fusionner : dataset = EEG uniquement (pour l’instant).")

    # 5) Merge labels
    dataset = pd.merge(dataset, labels, on="patient_id", how="left")

    # 6) Validation couverture labels + export patients manquants
    n_total = dataset.shape[0]
    if "label_str" not in dataset.columns:
        log.warning("Aucune colonne label_str après merge labels (problème fichier labels ?).")
        n_no_label = n_total
    else:
        n_no_label = int(dataset["label_str"].isna().sum())

    if n_no_label > 0:
        log.warning(f"{n_no_label}/{n_total} lignes sans label patient (labels TXT).")

        missing_patients = (
            dataset.loc[dataset["label_str"].isna(), "patient_id"]
            .astype(str)
            .dropna()
            .unique()
        )
        missing_patients = sorted(missing_patients)

        log.info(f"Patients sans label ({len(missing_patients)}): " + ", ".join(missing_patients))

        missing_path = out_csv.parent / "patients_without_labels.txt"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(missing_path, "w", encoding="utf-8") as f:
            for pid in missing_patients:
                f.write(f"{pid}\n")
        log.info(f"Liste des patients sans label écrite dans : {missing_path}")
    else:
        log.info("Tous les patients du dataset ont un label.")

    # 7) Sauvegarde dataset
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out_csv, index=False)
    log.info(f"Dataset final écrit : {out_csv} (shape = {dataset.shape})")

    # 8) Rapport d’alignement (utile debug)
    report_path = out_csv.parent / (out_csv.stem + "_alignment_report.csv")

    rep_cols = ["patient_id"]
    if "epoch_index" in dataset.columns:
        rep_cols.append("epoch_index")
    if "epoch_start_sec" in dataset.columns:
        rep_cols.append("epoch_start_sec")

    for c in ["emg_phasic_ratio_mean", "emg_tonic_ratio_mean", "emg_rswa_any"]:
        if c in dataset.columns:
            rep_cols.append(c)

    if "label_str" in dataset.columns:
        rep_cols.append("label_str")

    report = dataset[rep_cols].copy()
    report.to_csv(report_path, index=False)
    log.info(f"Rapport d’alignement écrit : {report_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    """
    Parse les arguments de la ligne de commande.

    Returns
    -------
    argparse.Namespace
        Attributs :
          - eeg_features : str
          - emg_rbd      : str
          - labels_txt   : str
          - out_csv      : str
          - time_tol     : float
    """
    p = argparse.ArgumentParser(description="Construire le dataset final EEG + EMG-RBD + labels (TXT).")

    p.add_argument(
        "--eeg-features",
        type=str,
        default="data/processed/features/eeg_features.csv",
        help="CSV des features EEG (par patient ou par époque).",
    )
    p.add_argument(
        "--emg-rbd",
        type=str,
        default="data/processed/rbd/rbd_emg_events_and_summary_4s_per_channel.csv",
        help="CSV EMG-RBD produit par le script emg_rbd.",
    )
    p.add_argument(
        "--labels-txt",
        type=str,
        default="data/patients_label.txt",
        help="Fichier des labels patients (patient_id,label_str).",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default="data/processed/features/dataset_final.csv",
        help="Chemin de sortie du dataset fusionné.",
    )
    p.add_argument(
        "--time-tol",
        type=float,
        default=0.25,
        help="Tolérance (s) pour l’alignement par temps (epoch_start_sec).",
    )
    return p.parse_args()


def main():
    """
    Point d’entrée CLI :
    - initialise le logger
    - parse args
    - appelle build_dataset
    """
    log = get_logger("build_dataset")

    args = parse_args()
    eeg_csv = Path(args.eeg_features)
    emg_csv = Path(args.emg_rbd)
    labels_txt = Path(args.labels_txt)
    out_csv = Path(args.out_csv)

    log.info(f"EEG features : {eeg_csv}")
    log.info(f"EMG-RBD      : {emg_csv}")
    log.info(f"Labels TXT   : {labels_txt}")
    log.info(f"Sortie       : {out_csv}")
    log.info(f"Tolérance temps (s) : {args.time_tol}")

    build_dataset(
        eeg_features_csv=eeg_csv,
        emg_rbd_csv=emg_csv,
        labels_txt=labels_txt,
        out_csv=out_csv,
        time_tol=float(args.time_tol),
        log=log,
    )


if __name__ == "__main__":
    main()
