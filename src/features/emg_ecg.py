"""Features de base pour signaux EMG et ECG.

Le module fournit des descripteurs simples (RMS, énergie, etc.)
utilisés comme briques de base dans le pipeline de features.
"""

import numpy as np, pandas as pd
def emg_basic_features(epoch_array, sfreq):
    # Placeholder: RMS
    rms = np.sqrt(np.mean(epoch_array**2))
    return {"emg_rms": float(rms)}
def ecg_basic_features(epoch_array, sfreq):
    # Placeholder: energy
    ene = float(np.sum(epoch_array**2))
    return {"ecg_energy": ene}
