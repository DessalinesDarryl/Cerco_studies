#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_dataset.py

Fusionne :
  - features EEG par époque REM
  - indicateurs EMG-RBD (phasic/tonic/RSWA) par époque REM
  - labels patients (Excel)
et produit un dataset final prêt pour l'entraînement.

Entrées (par défaut) :
  --eeg-features  : data/processed/features/eeg_features.csv
  --emg-rbd       : data/rbd_emg_events_and_summary_4s_per_channel.csv
  --labels-xlsx   : data/BDD_RBD_patients_updated.xlsx
  --labels-sheet  : "classification"
  --out-csv       : data/processed/features/dataset_final.csv

Hypothèses :
  - EEG features : une ligne par (patient_id, epoch_index)
      colonnes minimales attendues :
        * patient_id
        * epoch_index
        * (optionnel) epoch_start_sec
  - EMG-RBD CSV : produit par ton script emg_rbd, avec lignes "REM_EPOCH_4S"
      colonnes clés :
        * patient_id
        * type == "REM_EPOCH_4S"
        * epoch_index
        * epoch_start_sec, epoch_end_sec
        * phasic_ratio, tonic_ratio, rswa, phasic_count, very_phasic, ...
  - Labels Excel : colonne "identifiant" pour l'ID patient
                   + une colonne de catégorie (voir heuristique plus bas).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import get_logger


# ---------------------------------------------------------------------
# Utilitaires labels
# ---------------------------------------------------------------------


def load_labels_from_excel(xlsx_path: Path, sheet: str = "classification"):
    """
    Lit l'Excel des patients et retourne un DataFrame avec :
      - patient_id (str)
      - label_str  (str, catégorie textuelle)
      - label_id   (int, encodage 0..K-1)
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet)

    cols = {c.lower(): c for c in df.columns}

    # ID patient : on cherche des choses type "identifiant", "id", "patient"
    id_col = (
        cols.get("identifiant")
        or cols.get("id_patient")
        or cols.get("patient_id")
        or cols.get("patient")
        or cols.get("id")
    )
    if id_col is None:
        raise ValueError(
            f"Aucune colonne ID patient trouvée dans {xlsx_path}. "
            f"Colonnes disponibles : {list(df.columns)}"
        )

    # Colonne de catégorie : on teste plusieurs noms usuels
    label_col = (
        cols.get("categorie")
        or cols.get("catégorie")
        or cols.get("category")
        or cols.get("groupe")
        or cols.get("group")
        or cols.get("diagnostic")
        or cols.get("diag")
    )
    if label_col is None:
        raise ValueError(
            f"Aucune colonne de label trouvée dans {xlsx_path}. "
            f"Essaye de renommer en 'categorie' ou 'groupe'."
        )

    out = pd.DataFrame()
    out["patient_id"] = df[id_col].astype(str).str.strip()
    out["label_str"] = df[label_col].astype(str).str.strip()

    # Encodage numérique des labels
    cats = pd.Categorical(out["label_str"])
    out["label_id"] = cats.codes.astype(int)

    return out


# ---------------------------------------------------------------------
# Fusion EMG-RBD (par époque) -> agrégats par patient + epoch_index
# ---------------------------------------------------------------------


def aggregate_emg_rbd(emg_df: pd.DataFrame, log, time_tol: float = 0.25):
    """
    Prend le CSV RBD EMG (sortie emg_rbd) et agrège les lignes REM_EPOCH_4S
    par (patient_id, epoch_start_sec) en résumant les canaux EMG.

    Sortie :
      DataFrame avec colonnes :
        - patient_id
        - epoch_start_sec
        - epoch_end_sec
        - emg_phasic_ratio_mean
        - emg_tonic_ratio_mean
        - emg_phasic_count_sum
        - emg_rswa_any
        - emg_very_phasic_any
        - emg_n_channels
    """
    if emg_df.empty:
        log.warning("EMG-RBD DataFrame vide.")
        return emg_df

    # On ne garde que les lignes par époque REM
    df_ep = emg_df[emg_df["type"] == "REM_EPOCH_4S"].copy()
    if df_ep.empty:
        log.warning("Aucune ligne type 'REM_EPOCH_4S' trouvée dans EMG-RBD.")
        return df_ep

    # Colonnes indispensables
    required = [
        "patient_id",
        "epoch_start_sec",
        "epoch_end_sec",
        "phasic_ratio",
        "tonic_ratio",
        "rswa",
        "very_phasic",
        "phasic_count",
    ]
    missing = [c for c in required if c not in df_ep.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans EMG-RBD (REM_EPOCH_4S) : {missing}")

    # On force types
    df_ep["patient_id"] = df_ep["patient_id"].astype(str).str.strip()
    df_ep["epoch_start_sec"] = df_ep["epoch_start_sec"].astype(float)
    df_ep["epoch_end_sec"] = df_ep["epoch_end_sec"].astype(float)
    df_ep["phasic_ratio"] = df_ep["phasic_ratio"].astype(float)
    df_ep["tonic_ratio"] = df_ep["tonic_ratio"].astype(float)
    df_ep["phasic_count"] = df_ep["phasic_count"].astype(float)
    df_ep["rswa"] = df_ep["rswa"].astype(bool)
    df_ep["very_phasic"] = df_ep["very_phasic"].astype(bool)

    # Agrégation sur (patient_id, epoch_start_sec arrondi)
    # -> ça gère le cas où plusieurs canaux EMG existent
    df_ep["epoch_start_round"] = (df_ep["epoch_start_sec"] / time_tol).round().astype(int)

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
    labels_xlsx: Path,
    labels_sheet: str,
    out_csv: Path,
    time_tol: float,
    log,
):
    # 1) EEG features
    log.info(f"Lecture EEG features : {eeg_features_csv}")
    eeg = pd.read_csv(eeg_features_csv)

    required_eeg = ["patient_id", "epoch_index"]
    missing_eeg = [c for c in required_eeg if c not in eeg.columns]
    if missing_eeg:
        raise ValueError(f"Colonnes manquantes dans EEG features : {missing_eeg}")

    eeg["patient_id"] = eeg["patient_id"].astype(str).str.strip()

    has_time = "epoch_start_sec" in eeg.columns
    if has_time:
        eeg["epoch_start_sec"] = eeg["epoch_start_sec"].astype(float)

    log.info(f"EEG features : {eeg.shape[0]} lignes, {eeg.shape[1]} colonnes.")

    # 2) EMG-RBD
    log.info(f"Lecture EMG-RBD : {emg_rbd_csv}")
    emg = pd.read_csv(emg_rbd_csv)
    if emg.empty:
        log.warning("EMG-RBD CSV vide → dataset sans features EMG.")
        emg_agg = None
    else:
        emg_agg = aggregate_emg_rbd(emg, log, time_tol=time_tol)
        log.info(f"EMG-RBD agrégé : {emg_agg.shape[0]} lignes.")

    # 3) Labels
    log.info(f"Lecture labels Excel : {labels_xlsx} (feuille '{labels_sheet}')")
    labels = load_labels_from_excel(labels_xlsx, sheet=labels_sheet)
    log.info(f"Labels : {labels['patient_id'].nunique()} patients, "
             f"{labels['label_str'].nunique()} classes.")

    # 4) Merge EEG + EMG (si EMG dispo)
    if emg_agg is not None and not emg_agg.empty:
        if has_time:
            # Merge par patient + temps approx (via epoch_start_round)
            emg_agg["epoch_start_round"] = (emg_agg["epoch_start_sec"] / time_tol).round().astype(int)
            eeg["epoch_start_round"] = (eeg["epoch_start_sec"] / time_tol).round().astype(int)

            dataset = pd.merge(
                eeg,
                emg_agg.drop(columns=["epoch_start_sec", "epoch_end_sec"]),
                on=["patient_id", "epoch_start_round"],
                how="left",
                validate="m:1",
            )
            dataset.drop(columns=["epoch_start_round"], inplace=True)
            log.info("Fusion EEG+EMG réalisée via (patient_id, epoch_start_sec ~).")
        else:
            # Fallback : merge sur (patient_id, epoch_index)
            if "epoch_index" not in emg.columns:
                log.warning(
                    "Pas de epoch_start_sec dans EEG features et pas de epoch_index dans EMG-RBD. "
                    "Impossible d'aligner correctement EEG et EMG. EMG sera ignoré."
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

    # 5) Merge avec labels patients
    dataset = pd.merge(
        dataset,
        labels,
        on="patient_id",
        how="left",
        validate="m:1",
    )

    # 6) Validation de la couverture
    n_total = dataset.shape[0]
    n_no_label = dataset["label_str"].isna().sum()
    if n_no_label > 0:
        log.warning(f"{n_no_label}/{n_total} lignes sans label patient (Excel).")

    if emg_agg is not None:
        cols_emg = [c for c in dataset.columns if c.startswith("emg_")]
        if cols_emg:
            n_no_emg = dataset[cols_emg].isna().all(axis=1).sum()
            log.info(f"{n_no_emg}/{n_total} lignes sans features EMG-RBD (alignement manquant ou EMG absent).")

    # 7) Sauvegarde
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out_csv, index=False)
    log.info(f"Dataset final écrit : {out_csv} (shape = {dataset.shape})")

    # 8) Petit rapport d’alignement
    report_path = out_csv.with_suffix("_alignment_report.csv")
    rep_cols = ["patient_id", "epoch_index"]
    if "epoch_start_sec" in dataset.columns:
        rep_cols.append("epoch_start_sec")
    if "emg_phasic_ratio_mean" in dataset.columns:
        rep_cols += ["emg_phasic_ratio_mean", "emg_tonic_ratio_mean", "emg_rswa_any"]
    if "label_str" in dataset.columns:
        rep_cols.append("label_str")

    report = dataset[rep_cols].copy()
    report.to_csv(report_path, index=False)
    log.info(f"Rapport d’alignement écrit : {report_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Construire le dataset final EEG + EMG-RBD + labels.")
    p.add_argument(
        "--eeg-features",
        type=str,
        default="data/processed/features/eeg_features.csv",
        help="CSV des features EEG par époque.",
    )
    p.add_argument(
        "--emg-rbd",
        type=str,
        default="data/rbd_emg_events_and_summary_4s_per_channel.csv",
        help="CSV EMG-RBD produit par le script emg_rbd.",
    )
    p.add_argument(
        "--labels-xlsx",
        type=str,
        default="data/BDD_RBD_patients_updated.xlsx",
        help="Fichier Excel des labels patients.",
    )
    p.add_argument(
        "--labels-sheet",
        type=str,
        default="classification",
        help="Nom de la feuille Excel contenant les labels.",
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
    log = get_logger("build_dataset")

    args = parse_args()
    eeg_csv = Path(args.eeg_features)
    emg_csv = Path(args.emg_rbd)
    labels_xlsx = Path(args.labels_xlsx)
    out_csv = Path(args.out_csv)

    log.info(f"EEG features : {eeg_csv}")
    log.info(f"EMG-RBD      : {emg_csv}")
    log.info(f"Labels Excel : {labels_xlsx}")
    log.info(f"Sortie       : {out_csv}")
    log.info(f"Tolérance temps (s) : {args.time_tol}")

    build_dataset(
        eeg_features_csv=eeg_csv,
        emg_rbd_csv=emg_csv,
        labels_xlsx=labels_xlsx,
        labels_sheet=args.labels_sheet,
        out_csv=out_csv,
        time_tol=args.time_tol,
        log=log,
    )


if __name__ == "__main__":
    main()
