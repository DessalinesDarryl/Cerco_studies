"""Features temporelles pour signaux EEG.

Le module regroupe des descripteurs de dynamique temporelle (Hjorth,
zéro-crossing, moments statistiques) calculés par canal/epoch.
"""

import numpy as np
import mne
from scipy.stats import kurtosis, skew


# ======================================================================
# Fonctions utilitaires : taux de zéro-crossing et paramètres Hjorth
# ======================================================================

def _zero_crossing_rate(sig):
    """Retourne le taux de changements de signe dans le signal."""
    return ((sig[:-1] * sig[1:]) < 0).mean()


def _hjorth_parameters(sig):
    """Calcule les paramètres de Hjorth (activité, mobilité, complexité).

    Retours
    -------
    activity, mobility, complexity : float
    """
    # Activité = variance du signal
    activity = np.var(sig)
    
    # Mobilité = racine du ratio variance(dérivée)/variance(signal)
    diff1 = np.diff(sig)
    var_diff1 = np.var(diff1)
    mobility = np.sqrt(var_diff1 / (activity + 1e-20))
    
    # Complexité = ratio mobilité(dérivée)/mobilité(signal)
    diff2 = np.diff(diff1)
    var_diff2 = np.var(diff2)
    mobility_diff = np.sqrt(var_diff2 / (var_diff1 + 1e-20))
    complexity = mobility_diff / (mobility + 1e-20)
    
    return activity, mobility, complexity


# ======================================================================
# Extraction de features temporelles EEG
# ======================================================================

def compute_temporal_eeg_features(epochs: mne.Epochs):
    """Extrait 15 features temporelles par canal EEG.

    Features
    --------
    - Statistiques : mean, median, var, std, energy, rms
    - Moments : kurtosis, skewness
    - Morphologie : max, min, amplitude
    - Dynamique : zero-crossing rate, Hjorth (activity, mobility, complexity)

    Retours
    -------
    X : ndarray, shape (n_epochs, n_channels * 15)
    names : list of str
    """
    picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    if len(picks) == 0:
        return None, []

    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs = data.shape[0]

    feat_list = []
    names = []

    for ch_idx in picks:
        ch_name = epochs.ch_names[ch_idx]
        sig = data[:, ch_idx, :]  # (n_epochs, n_times)

        # 1) Statistiques de base
        mean = sig.mean(axis=-1)
        median = np.median(sig, axis=-1)
        var = sig.var(axis=-1)
        std = sig.std(axis=-1)
        energy = (sig**2).sum(axis=-1)
        rms = np.sqrt((sig**2).mean(axis=-1))
        
        # 2) Moments statistiques
        kurt = kurtosis(sig, axis=-1, fisher=True, bias=False)
        skewn = skew(sig, axis=-1, bias=False)
        
        # 3) Morphologie du signal
        maxv = sig.max(axis=-1)
        minv = sig.min(axis=-1)
        ampl = maxv - minv

        # 4) Dynamique temporelle
        zcr = np.array([_zero_crossing_rate(s) for s in sig])

        # 5) Paramètres de Hjorth
        hj_act = np.zeros(n_epochs)
        hj_mob = np.zeros(n_epochs)
        hj_com = np.zeros(n_epochs)
        for i in range(n_epochs):
            a, m, c = _hjorth_parameters(sig[i])
            hj_act[i] = a
            hj_mob[i] = m
            hj_com[i] = c

        # Concaténation des 15 features pour ce canal
        ch_feats = np.vstack([
            mean, median, var, std, energy, rms,
            kurt, skewn, maxv, minv, ampl,
            zcr, hj_act, hj_mob, hj_com
        ]).T  # (n_epochs, 15)

        feat_list.append(ch_feats)

        base = f"eeg_{ch_name}"
        names.extend([
            f"{base}_mean",
            f"{base}_median",
            f"{base}_var",
            f"{base}_std",
            f"{base}_energy",
            f"{base}_rms",
            f"{base}_kurtosis",
            f"{base}_skewness",
            f"{base}_max",
            f"{base}_min",
            f"{base}_amplitude",
            f"{base}_zcr",
            f"{base}_hj_activity",
            f"{base}_hj_mobility",
            f"{base}_hj_complexity",
        ])

    X = np.concatenate(feat_list, axis=1)  # (n_epochs, n_channels * nb_feats)
    return X, names
