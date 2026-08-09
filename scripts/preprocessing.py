"""
preprocessing.py
-----------------
Pretraitement du signal EMG submentalis/membre pour l'analyse RSWA
(REM Sleep Without Atonia).

References
----------
- Ferri et al. 2008, J Sleep Res 17:89-100      (filtre 10-100 Hz, notch 50 Hz)
- Ferri et al. 2010, Sleep Med 11:947-949       (correction de bruit)
- Frauscher et al. 2012, Sleep 35:835-847       (fs=256 Hz, mini-epochs 3 s)
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(signal, fs, low=10.0, high=100.0, order=4):
    """Filtre passe-bande Butterworth 10-100 Hz (convention Ferri/Frauscher/
    Iranzo/McCarter)."""
    nyq = fs / 2.0
    high = min(high, nyq * 0.99)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def notch_filter(signal, fs, freq=50.0, quality=30.0):
    """Filtre coupe-bande secteur (50 Hz Europe, 60 Hz Amerique du Nord)."""
    b, a = iirnotch(freq / (fs / 2.0), quality)
    return filtfilt(b, a, signal)


def rectify(signal):
    """Redressement (valeur absolue)."""
    return np.abs(signal)


def preprocess_emg(raw_signal, fs, mains_freq=50.0, low=10.0, high=100.0):
    """Pipeline complet : bandpass -> notch -> redressement."""
    filtered = bandpass_filter(raw_signal, fs, low=low, high=high)
    filtered = notch_filter(filtered, fs, freq=mains_freq)
    return rectify(filtered)


def mini_epoch_amplitude(rectified_signal, fs, epoch_duration=1.0):
    """
    Calcule l'amplitude moyenne redressee par mini-epoque.

    epoch_duration : duree d'une mini-epoque en secondes
        - 1.0 s : convention Ferri 2008/2010 (Atonia Index)
        - 2.0 s : convention Khalil 2013
        - 3.0 s : convention Frauscher/Iranzo/McCarter (SINBAR)
    """
    n_samples_per_epoch = int(round(fs * epoch_duration))
    if n_samples_per_epoch < 1:
        raise ValueError("epoch_duration trop courte pour cette frequence d'echantillonnage")
    n_epochs = len(rectified_signal) // n_samples_per_epoch
    trimmed = rectified_signal[: n_epochs * n_samples_per_epoch]
    reshaped = trimmed.reshape(n_epochs, n_samples_per_epoch)
    return reshaped.mean(axis=1)
