"""Extraction de features spectrales génériques sur epochs MNE.

Ce module propose des fonctions utilitaires pour calculer des puissances
de bandes et des statistiques dérivées par epoch/canal.
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

import mne


# ======================================================================
# Calcul de puissance spectrale en bandes de fréquences
# ======================================================================

def bandpow(x, sfreq, fmin, fmax):
    """Retourne la puissance moyenne (dB) dans la bande [fmin, fmax]."""
    psd, _ = mne.time_frequency.psd_array_welch(x, sfreq, fmin=fmin, fmax=fmax, verbose=False)
    return np.mean(10 * np.log10(psd + 1e-12), axis=1)


# ======================================================================
# Extraction de features spectrales au niveau epoch
# ======================================================================

def extract_epoch_features(epochs_fif: str, bands: dict, stats=("mean", "var", "skew", "kurt")):
    """Extrait des features spectrales par channel/band et des statistiques globales par epoch.

    Paramètres
    ----------
    epochs_fif : str
        Chemin du fichier Epochs MNE.
    bands : dict
        Dictionnaire {band_name: (fmin, fmax)}.
    stats : tuple
        Statistiques agrégées à calculer sur toutes les bandes.

    Retours
    -------
    df, meta : (DataFrame, DataFrame)
        Respectivement features et métadonnées par epoch.
    """
    epochs = mne.read_epochs(epochs_fif, preload=True, verbose=False)
    X = []
    info = []

    for i, ep in enumerate(epochs):
        row = {}
        # 1) Puissance par canal et par bande
        for ch_idx, ch_name in enumerate(epochs.ch_names):
            sig = ep[ch_idx]
            sf = epochs.info["sfreq"]
            for bname, (f1, f2) in bands.items():
                p = bandpow(sig, sf, f1, f2).mean()
                row[f"{ch_name}_{bname}_db"] = float(p)

        # 2) Statistiques agrégées (moyennes de toutes les bandes)
        v = np.array(list(row.values()))
        if "mean" in stats:
            row["feat_mean"] = float(v.mean())
        if "var" in stats:
            row["feat_var"] = float(v.var())
        if "skew" in stats:
            row["feat_skew"] = float(skew(v))
        if "kurt" in stats:
            row["feat_kurt"] = float(kurtosis(v))

        X.append(row)
        info.append({"epoch_idx": i})

    df = pd.DataFrame(X)
    meta = pd.DataFrame(info)
    return df, meta
