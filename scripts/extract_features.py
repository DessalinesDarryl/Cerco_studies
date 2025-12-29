#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_features.py

Lit :
  - fichiers *_REM-epo.fif (époques REM 4 s) sous in_root
  - un CSV RBD EMG (par canal, 4 s) si fourni

Sortie :
  - features.csv : 1 ligne par patient, avec :
        * agrégats EEG (temps + spectral)
        * agrégats EMG-RBD classiques
        * couplage fonctionnel EOG–EMG
        * corrélation statistique EOG–EMG
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import mne

from src.utils.config import load_yaml, add_common_args
from src.utils.logging import get_logger
from src.features.spectral_eeg import compute_spectral_eeg_features
from scipy.stats import skew, kurtosis

# ======================================================
# EEG FEATURES
# ======================================================
from scipy.stats import skew, kurtosis

def compute_time_eeg_features(epochs: mne.Epochs):
    picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    if len(picks) == 0:
        return None, []

    # (n_epochs, n_ch, n_times)
    data = epochs.get_data()[:, picks, :]

    mean = data.mean(axis=-1)
    std  = data.std(axis=-1)
    var  = data.var(axis=-1)
    rms  = np.sqrt((data ** 2).mean(axis=-1))

    # Zero crossing rate
    zc = ((np.diff(np.sign(data), axis=-1) != 0)).sum(axis=-1)

    # Skewness & Kurtosis
    skw = skew(data, axis=-1, nan_policy="omit")
    krt = kurtosis(data, axis=-1, nan_policy="omit")

    # Peak-to-peak amplitude
    ptp = np.ptp(data, axis=-1)

    # Line length
    ll = np.sum(np.abs(np.diff(data, axis=-1)), axis=-1)

    # Stack features
    feats = np.stack(
        [mean, std, var, rms, zc, skw, krt, ptp, ll],
        axis=2
    )  # (n_epochs, n_ch, n_features)

    n_epochs, n_ch, n_f = feats.shape
    feats = feats.reshape(n_epochs, n_ch * n_f)

    names = []
    for ch_idx in picks:
        ch = epochs.ch_names[ch_idx]
        names.extend([
            f"eeg_{ch}_mean",
            f"eeg_{ch}_std",
            f"eeg_{ch}_var",
            f"eeg_{ch}_rms",
            f"eeg_{ch}_zc",
            f"eeg_{ch}_skew",
            f"eeg_{ch}_kurt",
            f"eeg_{ch}_ptp",
            f"eeg_{ch}_linelen",
        ])

    return feats, names


def aggregate_patient_features(X: np.ndarray, names: List[str]) -> Dict[str, float]:
    row = {}
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    for i, name in enumerate(names):
        row[f"{name}_mean"] = float(mean[i])
        row[f"{name}_std"]  = float(std[i])
    return row


# ======================================================
# EMG / EOG FEATURES (PATIENT LEVEL)
# ======================================================
def load_rbd_emg_features_patient_level(rbd_csv: Path, log):
    if not rbd_csv or not rbd_csv.exists():
        log.warning("CSV RBD EMG introuvable >>> EMG ignoré")
        return None

    df = pd.read_csv(rbd_csv)
    if "type" not in df.columns:
        log.warning("CSV EMG sans colonne 'type'")
        return None

    df = df[df["type"] == "REM_EPOCH_4S"].copy()
    if df.empty:
        log.warning("CSV EMG sans REM_EPOCH_4S")
        return None

    # -------- agrégation par canal --------
    agg_ch = df.groupby(["patient_id", "channel"]).agg(
        rswa_fraction=("rswa", "mean"),
        phasic_ratio_mean=("phasic_ratio", "mean"),
        tonic_ratio_mean=("tonic_ratio", "mean"),
        tonic_eog_fraction=("tonic_eog", "mean"),
        eye_emg_corr_mean=("eye_emg_corr", "mean"),
        eye_emg_corr_abs_max=("eye_emg_corr", lambda x: np.nanmax(np.abs(x))),
    ).reset_index()

    # -------- agrégation patient --------
    agg_pat = agg_ch.groupby("patient_id").agg(
        emg_rswa_fraction_mean=("rswa_fraction", "mean"),
        emg_phasic_ratio_mean=("phasic_ratio_mean", "mean"),
        emg_tonic_ratio_mean=("tonic_ratio_mean", "mean"),
        emg_tonic_eog_fraction_mean=("tonic_eog_fraction", "mean"),
        emg_eye_corr_mean=("eye_emg_corr_mean", "mean"),
        emg_eye_corr_abs_max=("eye_emg_corr_abs_max", "max"),
    ).reset_index()

    return agg_pat


# ======================================================
# PER FILE PROCESSING
# ======================================================
def _process_one_feature_file(fpath: Path):
    log = get_logger("extract_features")
    patient_id = fpath.stem.split("_")[0]

    log.info(f"EEG features >>> {patient_id}")
    epochs = mne.read_epochs(fpath, preload=True, verbose=False)

    X_time, names_time = compute_time_eeg_features(epochs)
    X_spec, names_spec = compute_spectral_eeg_features(epochs)

    parts, names = [], []
    if X_time is not None:
        parts.append(X_time); names += names_time
    if X_spec is not None:
        parts.append(X_spec); names += names_spec

    if not parts:
        return None

    X = np.concatenate(parts, axis=1)
    row = {"patient_id": patient_id}
    row.update(aggregate_patient_features(X, names))
    return row


# ======================================================
# MAIN
# ======================================================
def main(cfg):
    log = get_logger("extract_features")

    in_root = Path(cfg["in_root"])
    out_csv = Path(cfg["out_csv"])
    rbd_emg_csv = Path(cfg["rbd_emg_csv"]) if cfg.get("rbd_emg_csv") else None
    n_workers = int(cfg.get("n_workers", 15))

    fif_files = sorted(in_root.glob("**/*_REM-epo.fif"))
    if not fif_files:
        log.warning("Aucun fichier *_REM-epo.fif trouvé")
        return

    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_process_one_feature_file, f): f for f in fif_files}
        for fut in as_completed(futures):
            row = fut.result()
            if row is not None:
                rows.append(row)

    df_eeg = pd.DataFrame(rows).drop_duplicates("patient_id")
    log.info(f"EEG features : {df_eeg.shape}")

    df_final = df_eeg
    df_emg = load_rbd_emg_features_patient_level(rbd_emg_csv, log)
    if df_emg is not None:
        df_final = df_final.merge(df_emg, on="patient_id", how="left")
        log.info(f"EEG + EMG features : {df_final.shape}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_csv, index=False)
    log.info(f"Features écrites >>> {out_csv}")


if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--n_workers", type=int, default=15)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    cfg["n_workers"] = args.n_workers
    main(cfg)
