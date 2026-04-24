#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filtre les epochs RSWA (fenêtres REM de 4 s) à partir du dataset final RBD.

Entrée principale
-----------------
- data/processed/rbd/rbd_emg_events_and_summary_4s_per_channel.csv

Ce fichier contient notamment :
- des lignes de type == "PHASIC" (événements)
- des lignes de type == "REM_EPOCH_4S" (epochs REM de 4 s) par canal EMG
- un drapeau `rswa` par epoch/canal
- plusieurs métriques associées : `tonic_ratio`, `phasic_ratio`, `eye_emg_corr`, etc.

Sorties
-------
1) epochs RSWA seules au niveau canal
2) epochs RSWA agrégées au niveau de l'epoch
3) petits exports de contrôle qualité

Usage
-----
    python filter_rswa_epochs.py \
        --input_csv data/processed/rbd/rbd_emg_events_and_summary_4s_per_channel.csv \
        --out_dir data/processed/rbd/_rswa_only
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# -------------------------
# Fonctions utilitaires
# -------------------------
def _to_bool_series(s: pd.Series) -> pd.Series:
    """Robust conversion of rswa column to boolean."""
    if s.dtype == bool:
        return s
    # handle 0/1, "True"/"False", "true"/"false", etc.
    return s.astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes", "y"])


def _safe_agg_dict(df: pd.DataFrame) -> dict:
    """
    Builds an aggregation dict only with columns that exist in df.
    We use robust choices for RSWA research:
    - corr: median (robust to outliers)
    - phasic_ratio/time: max (if any channel sees phasic, keep it)
    - tonic_ratio: median (robust across channels)
    - phasic_count: sum (events across channels)
    - boolean flags: max (OR)
    """
    desired = {
        "eye_emg_corr": "median",
        "eye_emg_corr_raw": "median",
        "eye_emg_corrmax_raw": "median",
        "eye_emg_lag_ms_raw": "median",

        "phasic_ratio": "max",
        "phasic_time_sec": "max",
        "phasic_count": "sum",

        "tonic_ratio": "median",
        "tonic_eog": "max",
        "very_phasic": "max",
        "tonic_excluded": "max",
        "rswa": "max",
    }
    return {k: v for k, v in desired.items() if k in df.columns}


def _pick_epoch_key_cols(df: pd.DataFrame):
    """
    Prefer stable keys: patient_id + episode_index + epoch_index.
    Fall back to patient_id + epoch_start_sec + epoch_end_sec.
    """
    if all(c in df.columns for c in ["patient_id", "episode_index", "epoch_index"]):
        key = ["patient_id", "episode_index", "epoch_index"]
        # keep times too if present
        if "epoch_start_sec" in df.columns:
            key.append("epoch_start_sec")
        if "epoch_end_sec" in df.columns:
            key.append("epoch_end_sec")
        return key

    if all(c in df.columns for c in ["patient_id", "epoch_start_sec", "epoch_end_sec"]):
        return ["patient_id", "epoch_start_sec", "epoch_end_sec"]

    raise RuntimeError(
        "Impossible d'identifier une clé d'epoch. "
        "Il faut au moins (patient_id, episode_index, epoch_index) OU "
        "(patient_id, epoch_start_sec, epoch_end_sec)."
    )


# -------------------------
# Point d’entrée principal
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Filter RSWA-only REM 4s epochs from final RBD dataset CSV.")
    ap.add_argument(
        "--input_csv",
        type=str,
        default="data/processed/rbd/rbd_emg_events_and_summary_4s_per_channel.csv",
        help="CSV final (per_channel) issu de ton pipeline RSWA.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="data/processed/rbd/_rswa_only",
        help="Dossier de sortie.",
    )
    ap.add_argument(
        "--type_col",
        type=str,
        default="type",
        help="Nom de colonne indiquant le type de ligne (PHASIC vs REM_EPOCH_4S).",
    )
    ap.add_argument(
        "--epoch_type_value",
        type=str,
        default="REM_EPOCH_4S",
        help="Valeur de type correspondant aux epochs REM (4s).",
    )
    ap.add_argument(
        "--rswa_col",
        type=str,
        default="rswa",
        help="Nom de colonne RSWA booléen.",
    )
    ap.add_argument(
        "--keep_only_rswa",
        action="store_true",
        help="Si activé: garde uniquement RSWA==True. Sinon exporte aussi non-RSWA (mais séparé).",
    )
    ap.add_argument(
        "--round_sec",
        type=int,
        default=None,
        help="Optionnel: arrondir epoch_start_sec/epoch_end_sec (ex: 3 pour 0.001s).",
    )
    args = ap.parse_args()

    in_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_csv.is_file():
        raise FileNotFoundError(f"CSV introuvable: {in_csv}")

    df = pd.read_csv(in_csv)

    if args.type_col not in df.columns:
        raise RuntimeError(f"Colonne '{args.type_col}' absente du CSV.")
    if args.rswa_col not in df.columns:
        raise RuntimeError(f"Colonne '{args.rswa_col}' absente du CSV.")

    # 1) Conservation des lignes correspondant uniquement aux epochs REM
    df_epochs = df.loc[df[args.type_col].astype(str) == args.epoch_type_value].copy()
    
    if df_epochs.empty:
        raise RuntimeError(f"Aucune ligne trouvée avec {args.type_col} == '{args.epoch_type_value}'.")

    # 2) Normalisation de la colonne RSWA en booléen robuste
    df_epochs[args.rswa_col] = _to_bool_series(df_epochs[args.rswa_col])

    # 3) Arrondi optionnel des temps pour stabiliser les clés d'agrégation
    if args.round_sec is not None:
        for c in ["epoch_start_sec", "epoch_end_sec"]:
            if c in df_epochs.columns:
                df_epochs[c] = pd.to_numeric(df_epochs[c], errors="coerce").round(args.round_sec)

    # 4) Séparation des epochs RSWA et non-RSWA
    df_rswa = df_epochs.loc[df_epochs[args.rswa_col] == True].copy()
    df_non  = df_epochs.loc[df_epochs[args.rswa_col] == False].copy()

    print("[INFO] Input rows:", df.shape)
    print("[INFO] Epoch rows:", df_epochs.shape)
    print("[INFO] RSWA epochs:", df_rswa.shape)
    print("[INFO] non-RSWA epochs:", df_non.shape)

    # 5) Export des fichiers au niveau canal
    out_rswa_per_ch = out_dir / "rbd_emg_epochs_4s_rswa_only_per_channel.csv"
    df_rswa.to_csv(out_rswa_per_ch, index=False)
    print("[OK] Écrit:", out_rswa_per_ch)

    if not args.keep_only_rswa:
        out_non_per_ch = out_dir / "rbd_emg_epochs_4s_non_rswa_per_channel.csv"
        df_non.to_csv(out_non_per_ch, index=False)
        print("[OK] Écrit:", out_non_per_ch)

    # 6) Agrégation au niveau epoch pour les analyses de groupe
    key_cols = _pick_epoch_key_cols(df_epochs)

    agg_dict = _safe_agg_dict(df_epochs)
    if not agg_dict:
        print("[WARN] Aucune colonne connue à agréger trouvée. "
              "Je vais agréger uniquement rswa (max) si possible.")
        agg_dict = {args.rswa_col: "max"}

    # 7) Conservation de quelques métadonnées descriptives
    meta_candidates = [
        "label", "label_id", "diagnosis", "group",
        "epoch_len_sec", "sfreq", "night", "session"
    ]
    meta_cols = [c for c in meta_candidates if c in df_epochs.columns]
    meta_aggs = {c: "first" for c in meta_cols}

    # 8) Suppression implicite de la dimension canal par agrégation
    df_rswa_agg = (
        df_rswa
        .groupby(key_cols, as_index=False)
        .agg({**meta_aggs, **agg_dict})
    )

    out_rswa_agg = out_dir / "rbd_emg_epochs_4s_rswa_only_aggregated.csv"
    df_rswa_agg.to_csv(out_rswa_agg, index=False)
    print("[OK] Écrit:", out_rswa_agg, "| shape:", df_rswa_agg.shape)

    if not args.keep_only_rswa:
        df_non_agg = (
            df_non
            .groupby(key_cols, as_index=False)
            .agg({**meta_aggs, **agg_dict})
        )
        out_non_agg = out_dir / "rbd_emg_epochs_4s_non_rswa_aggregated.csv"
        df_non_agg.to_csv(out_non_agg, index=False)
        print("[OK] Écrit:", out_non_agg, "| shape:", df_non_agg.shape)

    # QC report
    qc = []
    if "patient_id" in df_rswa.columns:
        vc = df_rswa["patient_id"].value_counts()
        qc.append(pd.DataFrame({"patient_id": vc.index, "n_rswa_rows_per_channel": vc.values}))
    if qc:
        qc_df = qc[0]
        qc_path = out_dir / "qc_rswa_counts_per_patient.csv"
        qc_df.to_csv(qc_path, index=False)
        print("[OK] QC:", qc_path)

    # Basic summary print
    if "label_id" in df_rswa.columns:
        print("\n[SUMMARY] RSWA per-channel label_id counts:")
        print(df_rswa["label_id"].value_counts(dropna=False).to_string())

    print("\n[DONE] RSWA filtering complete.")


if __name__ == "__main__":
    main()
