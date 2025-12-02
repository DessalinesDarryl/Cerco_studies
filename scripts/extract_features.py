#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_features.py

Lit :
  - fichiers *_REM-epo.fif (époques REM 4 s) sous in_root
  - un CSV RBD EMG (par canal, 4 s) si fourni

Sortie :
  - features.csv : 1 ligne par patient, avec :
        * agrégats EEG (moyenne / std sur les époques REM)
        * agrégats EMG-RBD (RSWA, phasic_ratio, tonic_ratio, etc.)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import numpy as np
import pandas as pd
import mne

from src.utils.config import load_yaml, add_common_args
from src.utils.logging import get_logger
from src.features.spectral_eeg import compute_spectral_eeg_features


def compute_time_eeg_features(epochs: mne.Epochs):
    picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    if len(picks) == 0:
        return None, []

    data = epochs.get_data()[:, picks, :]  # (n_epochs, n_ch, n_times)

    mean = data.mean(axis=-1)
    var = data.var(axis=-1)
    rms = np.sqrt((data ** 2).mean(axis=-1))
    zc = ((np.diff(np.sign(data), axis=-1) != 0) & (np.abs(np.diff(data, axis=-1)) > 0)).sum(axis=-1)

    feats = np.stack([mean, var, rms, zc], axis=2)  # (n_epochs, n_ch, 4)

    n_epochs, n_ch, n_f = feats.shape
    feats = feats.reshape(n_epochs, n_ch * n_f)

    names: List[str] = []
    for ch_idx in picks:
        ch_name = epochs.ch_names[ch_idx]
        names.extend([
            f"eeg_{ch_name}_mean",
            f"eeg_{ch_name}_var",
            f"eeg_{ch_name}_rms",
            f"eeg_{ch_name}_zc",
        ])
    return feats, names



def aggregate_patient_features(X: np.ndarray, names: List[str]) -> Dict[str, float]:
    row = {}
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    for i, name in enumerate(names):
        row[f"{name}_mean"] = float(mean[i])
        row[f"{name}_std"] = float(std[i])
    return row


def load_rbd_emg_features_patient_level(rbd_csv: Path, log):
    if not rbd_csv or not rbd_csv.exists():
        log.warning(f"CSV RBD EMG introuvable ({rbd_csv}), pas de features EMG-RBD.")
        return None

    df = pd.read_csv(rbd_csv)
    if "type" not in df.columns:
        log.warning("CSV RBD EMG sans colonne 'type' >>> ignoré.")
        return None

    df_ep = df[df["type"] == "REM_EPOCH_4S"].copy()
    if df_ep.empty:
        log.warning("CSV RBD EMG sans lignes REM_EPOCH_4S >>> ignoré.")
        return None

    agg = df_ep.groupby(["patient_id", "channel"]).agg(
        rswa_fraction=("rswa", "mean"),
        phasic_ratio_mean=("phasic_ratio", "mean"),
        phasic_ratio_std=("phasic_ratio", "std"),
        tonic_ratio_mean=("tonic_ratio", "mean"),
        tonic_ratio_std=("tonic_ratio", "std"),
        very_phasic_fraction=("very_phasic", "mean"),
        phasic_count_mean=("phasic_count", "mean"),
    ).reset_index()

    agg_pat = agg.groupby("patient_id").agg(
        emg_rswa_fraction_mean=("rswa_fraction", "mean"),
        emg_phasic_ratio_mean=("phasic_ratio_mean", "mean"),
        emg_phasic_ratio_std=("phasic_ratio_mean", "std"),
        emg_tonic_ratio_mean=("tonic_ratio_mean", "mean"),
        emg_tonic_ratio_std=("tonic_ratio_mean", "std"),
        emg_very_phasic_fraction_mean=("very_phasic_fraction", "mean"),
        emg_phasic_count_mean=("phasic_count_mean", "mean"),
    ).reset_index()

    return agg_pat


def _process_one_feature_file(fpath: Path):
    """Calcule les features EEG agrégées pour un seul fichier *_REM-epo.fif."""
    log = get_logger("extract_features")

    base = fpath.stem  # ex: AN166_raw_REM-epo
    patient_id = base.split("_")[0]

    log.info(f"Features EEG pour {patient_id} ({fpath.name})")
    epochs = mne.read_epochs(fpath, preload=True, verbose=False)

    X_time, names_time = compute_time_eeg_features(epochs)
    X_spec, names_spec = compute_spectral_eeg_features(epochs)

    feat_parts = []
    feat_names = []

    if X_time is not None:
        feat_parts.append(X_time)
        feat_names.extend(names_time)
    if X_spec is not None:
        feat_parts.append(X_spec)
        feat_names.extend(names_spec)

    if not feat_parts:
        log.warning(f"[{patient_id}] aucune feature EEG calculée >>> skip")
        return None

    X_all = np.concatenate(feat_parts, axis=1)  # (n_epochs, n_features)
    row = {"patient_id": patient_id}
    row.update(aggregate_patient_features(X_all, feat_names))
    return row


def main(cfg):
    log = get_logger("extract_features")

    in_root = Path(cfg["in_root"])
    out_csv = Path(cfg["out_csv"])
    rbd_emg_csv = Path(cfg["rbd_emg_csv"]) if cfg.get("rbd_emg_csv") else None
    n_workers = int(cfg.get("n_workers", 15))

    fif_files = sorted(in_root.glob("**/*_REM-epo.fif"))
    if not fif_files:
        log.warning(f"Aucun fichier *_REM-epo.fif trouvé dans {in_root}")
        return

    log.info(f"{len(fif_files)} fichiers d'epochs trouvés. Lancement avec n_workers={n_workers}.")

    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_process_one_feature_file, f): f for f in fif_files}
        for fut in as_completed(futures):
            f = futures[fut]
            try:
                row = fut.result()
                if row is not None:
                    rows.append(row)
            except Exception as e:
                log.error(f"[ERREUR] lors du calcul de features pour {f}: {e}")

    if not rows:
        log.warning("Aucune feature EEG n'a été extraite.")
        return

    df_eeg = pd.DataFrame(rows).drop_duplicates(subset=["patient_id"])
    log.info(f"Features EEG : {df_eeg.shape[0]} patients, {df_eeg.shape[1]-1} features.")

    df_final = df_eeg

    df_emg_rbd = load_rbd_emg_features_patient_level(rbd_emg_csv, log) if rbd_emg_csv else None
    if df_emg_rbd is not None:
        df_final = df_final.merge(df_emg_rbd, on="patient_id", how="left")
        log.info(f"Fusion EEG + EMG-RBD : {df_final.shape[0]} patients, {df_final.shape[1]-1} features.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_csv, index=False)
    log.info(f"Features sauvegardées dans {out_csv}")


if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument(
        "--n_workers",
        type=int,
        default=15,
        help="Nombre de workers en parallèle (par défaut: 15).",
    )
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    cfg["n_workers"] = args.n_workers
    main(cfg)
