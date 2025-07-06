"""
EEG/EMG/ECG features
"""
import mne
import numpy as np
import pandas as pd
from utils import get_standard_bands, filter_band

def compute_band_power(raw):
    """
    Calcule la puissance moyenne dans chaque bande EEG standard pour chaque canal.

    Args:
        raw (mne.io.Raw): Signal EEG brut (idéalement segmenté en phase REM).

    Returns:
        dict: {"bande": moyenne de puissance sur tous les canaux}
    """
    powers = {}
    bands = get_standard_bands()

    for name, (l_freq, h_freq) in bands.items():
        raw_filt = filter_band(raw, l_freq, h_freq)
        data, _ = raw_filt.get_data(return_times=True)
        power = np.mean(data ** 2)  # Puissance moyenne
        powers[name] = power # Attention puissance pour trigger EEG energy changes upon state changes, such as sleep stage changes, seizures, and emotional changes.

    return powers

def extract_features_from_directory(rem_dir):
    """
    Extrait les puissances par bande pour chaque fichier .fif dans un dossier.

    Args:
        rem_dir (str): Dossier contenant les fichiers *_REM_raw.fif

    Returns:
        pd.DataFrame: Une ligne par fichier, une colonne par bande
    """
    import os
    import glob

    results = []
    files = glob.glob(os.path.join(rem_dir, "*_REM_raw.fif"))

    for fpath in files:
        raw = mne.io.read_raw_fif(fpath, preload=True)
        raw.pick_types(eeg=True)

        features = compute_band_power(raw)
        features["patient"] = os.path.basename(fpath).replace("_REM_raw.fif", "")
        results.append(features)

    return pd.DataFrame(results)
