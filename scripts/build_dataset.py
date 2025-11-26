#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_dataset.py

Fusionne :
  - features EEG par époque REM
  - indicateurs EMG-RBD (phasic/tonic/RSWA) par époque REM
  - labels patients (fichier texte patient_id,label_str)
et produit un dataset final prêt pour l'entraînement.

Entrées (par défaut) :
  --eeg-features  : data/processed/features/eeg_features.csv
  --emg-rbd       : data/rbd_emg_events_and_summary_4s_per_channel.csv
  --labels-txt    : data/labels_macro.txt
  --out-csv       : data/processed/features/dataset_final.csv

Hypothèses :
  - EEG features : une ligne par patient (ou par (patient_id, epoch_index))
      colonnes minimales attendues :
        * patient_id
        * (optionnel) epoch_index
        * (optionnel) epoch_start_sec
  - EMG-RBD CSV : produit par ton script emg_rbd, avec lignes "REM_EPOCH_4S"
      colonnes clés :
        * patient_id
        * type == "REM_EPOCH_4S"
        * epoch_index (optionnel si alignement par temps)
        * epoch_start_sec, epoch_end_sec
        * phasic_ratio, tonic_ratio, rswa, phasic_count, very_phasic, ...
  - Labels TXT : 2 colonnes patient_id,label_str (éventuellement sans header)
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
# Utilitaires labels (depuis un fichier .txt / .csv)
# ---------------------------------------------------------------------


def load_labels_from_txt(txt_path: Path) -> pd.DataFrame:
    """
    Lit un fichier texte / CSV contenant au moins :
        patient_id,label_str

    - Accepte un fichier AVEC ou SANS header.
    - Normalise l'ID patient (string, strip).
    - Crée label_id en fixant l'ordre des classes :
        SYN=0, Narco=1, TCSPi=2, EAI=3
      Les autres labels éventuels auront un code -1.
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"Fichier labels TXT introuvable : {txt_path}")

    # On essaie d'abord avec header, sinon sans header
    try:
        df = pd.read_csv(txt_path)
        cols_lower = {c.lower(): c for c in df.columns}
        if "patient_id" not in cols_lower and "identifiant" not in cols_lower:
            # Pas de colonnes explicites → on relit sans header
            raise ValueError("Pas de colonnes patient_id/identifiant détectées, on tente sans header.")
        # On mappe vers noms canoniques
        id_col = cols_lower.get("patient_id") or cols_lower.get("identifiant")
        if "label_str" in cols_lower:
            label_col = cols_lower["label_str"]
        elif "diagnostic" in cols_lower:
            label_col = cols_lower["diagnostic"]
        elif "label" in cols_lower:
            label_col = cols_lower["label"]
        else:
            # si une seule autre colonne, on la prend comme label
            other_cols = [c for c in df.columns if c != id_col]
            if len(other_cols) != 1:
                raise ValueError("Impossible d'inférer la colonne de label dans le fichier TXT.")
            label_col = other_cols[0]

        out = pd.DataFrame()
        out["patient_id"] = df[id_col].astype(str).str.strip()
        out["label_str"] = df[label_col].astype(str).str.strip()

    except Exception:
        # Lecture brute sans header : patient_id,label_str
        df = pd.read_csv(txt_path, header=None, names=["patient_id", "label_str"])
        out = pd.DataFrame()
        out["patient_id"] = df["patient_id"].astype(str).str.strip()
        out["label_str"] = df["label_str"].astype(str).str.strip()

    # On garde un seul label par patient (au cas où)
    out = out.drop_duplicates(subset=["patient_id"], keep="first")

    # Encodage numérique des labels avec ordre fixe
    macro_order = ["SYN", "Narco", "TCSPi", "EAI"]
    # On met tout en forme propre pour la catégorisation
    norm = out["label_str"].astype(str).str.strip()
    # On force la casse pour la catégorisation mais on garde label_str tel quel
    norm_upper = norm.str.upper()

    # On mappe vers macro (au cas où certains labels seraient déjà "PARK", etc.)
    def _to_macro(s_up: str, s_raw: str) -> str:
        if "PARK" in s_up or "MPI" in s_up or "AMS" in s_up or "DCL" in s_up or "DLB" in s_up or "PAF" in s_up:
            return "SYN"
        if "NARCO" in s_up:
            return "Narco"
        if "TCSP" in s_up or "RBDI" in s_up:
            return "TCSPi"
        if "EAI" in s_up or "ENCEPHALITE" in s_up:
            return "EAI"
        # Si déjà une macro propre, on la garde
        if s_up in {"SYN", "NARCO", "TCSPi".upper(), "EAI"}:
            return "TCSPi" if s_up == "TCSPI" else s_raw
        return s_raw

    out["label_str"] = [
        _to_macro(up, raw) for up, raw in zip(norm_upper, norm)
    ]

    cats = pd.Categorical(out["label_str"], categories=macro_order)
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
    ]
    missing = [c for c in required if c not in df_ep.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans EMG-RBD (REM_EPOCH_4S) : {missing}")

    df_ep["patient_id"] = df_ep["patient_id"].astype(str).str.strip()
    df_ep["epoch_start_sec"] = df_ep["epoch_start_sec"].astype(float)
    df_ep["epoch_end_sec"] = df_ep["epoch_end_sec"].astype(float)
    df_ep["phasic_ratio"] = df_ep["phasic_ratio"].astype(float)
    df_ep["tonic_ratio"] = df_ep["tonic_ratio"].astype(float)
    df_ep["phasic_count"] = df_ep["phasic_count"].astype(float)
    df_ep["rswa"] = df_ep["rswa"].astype(bool)
    df_ep["very_phasic"] = df_ep["very_phasic"].astype(bool)

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
    labels_txt: Path,
    out_csv: Path,
    time_tol: float,
    log,
):
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
        emg_agg = aggregate_emg_rbd(emg, log, time_tol=time_tol)
        log.info(f"EMG-RBD agrégé : {emg_agg.shape[0]} lignes.")

    # 3) Labels (depuis TXT)
    log.info(f"Lecture labels TXT : {labels_txt}")
    labels = load_labels_from_txt(labels_txt)
    log.info(
        f"Labels : {labels['patient_id'].nunique()} patients, "
        f"{labels['label_str'].nunique()} classes."
    )

    # 4) Merge EEG + EMG (si EMG dispo)
    if emg_agg is not None and not emg_agg.empty:
        if has_time:
            emg_agg["epoch_start_round"] = (emg_agg["epoch_start_sec"] / time_tol).round().astype(int)
            eeg["epoch_start_round"] = (eeg["epoch_start_sec"] / time_tol).round().astype(int)

            dataset = pd.merge(
                eeg,
                emg_agg.drop(columns=["epoch_start_sec", "epoch_end_sec"]),
                on=["patient_id", "epoch_start_round"],
                how="left",
            )
            dataset.drop(columns=["epoch_start_round"], inplace=True)
            log.info("Fusion EEG+EMG réalisée via (patient_id, epoch_start_sec ~).")
        else:
            if (not has_epoch_index) or ("epoch_index" not in emg.columns):
                log.warning(
                    "Pas de epoch_start_sec dans EEG features ou pas de epoch_index dans EEG/EMG-RBD. "
                    "Impossible d'aligner correctement EEG et EMG par époque. EMG sera ignoré."
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

    # 5) Merge avec labels patients (TXT)
    dataset = pd.merge(
        dataset,
        labels,
        on="patient_id",
        how="left",
    )

    # 6) Validation de la couverture
    n_total = dataset.shape[0]
    n_no_label = dataset["label_str"].isna().sum()
    if n_no_label > 0:
        log.warning(f"{n_no_label}/{n_total} lignes sans label patient (labels TXT).")

        # --- Liste des patients sans label ---
        missing_patients = (
            dataset.loc[dataset["label_str"].isna(), "patient_id"]
            .astype(str)
            .dropna()
            .unique()
        )
        missing_patients = sorted(missing_patients)

        log.info(f"Patients sans label ({len(missing_patients)}): "
                 + ", ".join(missing_patients))

        # Sauvegarde dans un fichier texte à côté du dataset
        missing_path = out_csv.parent / "patients_without_labels.txt"
        with open(missing_path, "w", encoding="utf-8") as f:
            for pid in missing_patients:
                f.write(f"{pid}\n")
        log.info(f"Liste des patients sans label écrite dans : {missing_path}")
    else:
        log.info("Tous les patients du dataset ont un label.")


    # 7) Sauvegarde
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out_csv, index=False)
    log.info(f"Dataset final écrit : {out_csv} (shape = {dataset.shape})")

    # 8) Petit rapport d’alignement
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
    p = argparse.ArgumentParser(
        description="Construire le dataset final EEG + EMG-RBD + labels (TXT)."
    )
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
        default="data/patient_list.txt",
        help="Fichier texte des labels patients (patient_id,label_str).",
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
        time_tol=args.time_tol,
        log=log,
    )


if __name__ == "__main__":
    main()
