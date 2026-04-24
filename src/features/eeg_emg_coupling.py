"""Extraction de features de couplage EEG-EMG.

Ce module calcule des indicateurs de cohérence/corrélation entre canaux
EEG et signaux EMG pour la caractérisation des épisodes REM.
"""

import numpy as np
import mne
from scipy.signal import coherence, correlate

from .temporal_emg import _get_emg_like_picks


def compute_eeg_emg_coupling_features(epochs: mne.Epochs,
                                      do_coherence: bool = True,
                                      do_cross_corr: bool = True):
    """Extrait des features de couplage EEG-EMG.

    Mesures incluses
    ----------------
    - Cohérence dans les bandes bêta (12-30 Hz) et gamma (30-80 Hz)
    - Corrélation croisée maximale par paire EEG-EMG

    Retours
    -------
    X : ndarray, shape (n_epochs, n_features)
    names : list of str
    """
    eeg_picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    emg_picks = _get_emg_like_picks(epochs)
    if len(eeg_picks) == 0 or len(emg_picks) == 0:
        return None, []

    data = epochs.get_data()
    sf = epochs.info["sfreq"]
    n_epochs, _, n_times = data.shape

    feats = []
    names = []

    # 1) Cohérence bêta/gamma moyenne (EEG-EMG) par paire
    if do_coherence:
        for epick in eeg_picks:
            for mpick in emg_picks:
                x = data[:, epick, :]
                y = data[:, mpick, :]
                coh_beta = []
                coh_gamma = []
                for e in range(n_epochs):
                    f, Cxy = coherence(
                        x[e], y[e], fs=sf, nperseg=min(256, n_times)
                    )
                    mask_beta = (f >= 12.0) & (f <= 30.0)
                    mask_gamma = (f >= 30.0) & (f <= 80.0)
                    coh_beta.append(Cxy[mask_beta].mean() if mask_beta.any() else 0.0)
                    coh_gamma.append(Cxy[mask_gamma].mean() if mask_gamma.any() else 0.0)
                coh_beta = np.array(coh_beta)
                coh_gamma = np.array(coh_gamma)

                feats.append(coh_beta[:, None])
                feats.append(coh_gamma[:, None])

                e_name = epochs.ch_names[epick]
                m_name = epochs.ch_names[mpick]
                names.append(f"cpl_coh_beta_{e_name}_{m_name}")
                names.append(f"cpl_coh_gamma_{e_name}_{m_name}")

    # 2) Corrélation croisée maximale par paire (normalisée)
    if do_cross_corr:
        max_lags = []
        for epick in eeg_picks:
            for mpick in emg_picks:
                x = data[:, epick, :]
                y = data[:, mpick, :]
                max_corr = []
                for e in range(n_epochs):
                    x0 = x[e] - x[e].mean()
                    y0 = y[e] - y[e].mean()
                    corr = correlate(x0, y0, mode="full")
                    corr /= (np.sqrt((x0**2).sum()) * np.sqrt((y0**2).sum()) + 1e-20)
                    max_corr.append(corr.max())
                max_corr = np.array(max_corr)
                max_lags.append(max_corr)

                e_name = epochs.ch_names[epick]
                m_name = epochs.ch_names[mpick]
                feats.append(max_corr[:, None])
                names.append(f"cpl_xcorrmax_{e_name}_{m_name}")

    if not feats:
        return None, []

    X = np.concatenate(feats, axis=1)
    return X, names
