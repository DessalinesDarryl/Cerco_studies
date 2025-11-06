# src/features/emg_ecg.py
import numpy as np, pandas as pd
def emg_basic_features(epoch_array, sfreq):
    # Placeholder: RMS
    rms = np.sqrt(np.mean(epoch_array**2))
    return {"emg_rms": float(rms)}
def ecg_basic_features(epoch_array, sfreq):
    # Placeholder: energy
    ene = float(np.sum(epoch_array**2))
    return {"ecg_energy": ene}
