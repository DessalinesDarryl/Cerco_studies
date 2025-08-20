# -*- coding: utf-8 -*-
"""
Fonctions d'extraction de caractéristiques EEG (spectrales) pour les époques REM.
- Calcul robuste des puissances de bandes par intégration de la PSD (V²/Hz) → V².
- Bandes: delta (1–4), theta (4–8), alpha (8–12), beta (12–30), gamma (30–45) Hz.
- Ratio principal retourné: gamma/beta.
- Deux API:
  1) extract_spectral_features(epoch)           # une sous-Epochs (n_epochs=1)
  2) compute_band_powers_epochs(epochs)         # vectorisé, toutes les époques d'un Epochs
"""

from __future__ import annotations
from typing import Dict, List, Iterable, Tuple, Optional
import numpy as np
from mne.time_frequency import psd_array_welch
import mne


BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta":  (12.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _integrate_band_power(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float):
    """
    Intègre la PSD sur [fmin, fmax] (approx. trapézoïdale).
    psd: (..., n_freqs), freqs: (n_freqs,)
    Retour: (...,) puissance intégrée.
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        # Si aucun bin dans la bande, retourne 0 (même shape sans l'axe freq)
        return np.zeros(psd.shape[:-1], dtype=float)
    return np.trapz(psd[..., mask], freqs[mask], axis=-1)


def extract_spectral_features(epoch: mne.Epochs):
    """
    Calcule un score gamma/bêta pour UNE sous-epoch (n_epochs=1) en intégrant la PSD.

    Paramètres
    ----------
    epoch : mne.Epochs
        Objet Epochs contenant exactement 1 époque avec des canaux EEG.

    Retour
    ------
    score : float
        Ratio puissance_gamma / puissance_bêta, borné à [0, 1].
    """
    ep = epoch.copy().pick_types(eeg=True)
    data = ep.get_data()                  # (1, n_channels, n_times)
    sfreq = float(ep.info['sfreq'])

    # PSD: (n_epochs, n_channels, n_freqs)
    psd, freqs = psd_array_welch(
        data, sfreq=sfreq, fmin=1, fmax=45, n_fft=256, average='mean'
    )
    psd = np.squeeze(psd, axis=0)         # (n_channels, n_freqs)

    beta_power_chan  = _integrate_band_power(psd, freqs, *BANDS["beta"])   # (n_channels,)
    gamma_power_chan = _integrate_band_power(psd, freqs, *BANDS["gamma"])  # (n_channels,)

    beta_power  = float(beta_power_chan.mean())
    gamma_power = float(gamma_power_chan.mean())

    score = gamma_power / (beta_power + 1e-12)
    return float(np.clip(score, 0.0, 1.0))


def compute_band_powers_epochs(epochs: mne.Epochs):
    """
    Version vectorisée: calcule les puissances de bandes pour TOUTES les époques d'un objet Epochs.

    Paramètres
    ----------
    epochs : mne.Epochs
        Epochs EEG (shape implicite: n_epochs x n_channels x n_times)

    Retour
    ------
    band_powers : dict
        Dictionnaire: clé = nom de bande, valeur = np.ndarray (n_epochs,)
        Moyenne inter-canaux après intégration par bande.
        Ajoute aussi la clé 'score' = gamma/beta (borné [0,1]).
    """
    ep = epochs.copy().pick_types(eeg=True)
    data = ep.get_data()                   # (n_epochs, n_channels, n_times)
    sfreq = float(ep.info['sfreq'])

    psd, freqs = psd_array_welch(
        data, sfreq=sfreq, fmin=1, fmax=45, n_fft=256, average='mean'
    )                                      # (n_epochs, n_channels, n_freqs)

    band_powers: Dict[str, np.ndarray] = {}
    for band, (fmin, fmax) in BANDS.items():
        band_power_ch = _integrate_band_power(psd, freqs, fmin, fmax)      # (n_epochs, n_channels)
        band_powers[band] = band_power_ch.mean(axis=1)                     # moyenne inter-canaux → (n_epochs,)

    # Score gamma/bêta par époque
    score = band_powers["gamma"] / (band_powers["beta"] + 1e-12)
    band_powers["score"] = np.clip(score, 0.0, 1.0)
    return band_powers


def compute_microstate_features(labels: Iterable[str], window_sec: float = 4.0):
    """
    Calcule des indicateurs résumant la dynamique REM (tonic/phasic) à partir d'une séquence de labels.

    labels : séquence de "phasic" / "tonic" (autres ignorés).
    window_sec : durée d'une fenêtre (s).

    Renvoie un dict avec, entre autres:
      - rem_n_windows
      - rem_duration_sec
      - phasic_count / tonic_count
      - phasic_ratio / tonic_ratio
      - alternations, alternation_rate_per_min
      - mean_run_len, max_run_len
    """
    labels = [lab for lab in labels if lab in ("phasic", "tonic")]
    n = len(labels)

    feats = {
        "rem_n_windows": float(n),
        "rem_duration_sec": float(n) * float(window_sec),
        "phasic_count": 0.0,
        "tonic_count": 0.0,
        "phasic_ratio": 0.0,
        "tonic_ratio": 0.0,
        "alternations": 0.0,
        "alternation_rate_per_min": 0.0,
        "mean_run_len": 0.0,
        "max_run_len": 0.0,
    }
    if n == 0:
        return feats

    # Comptes
    phasic_count = sum(1 for x in labels if x == "phasic")
    tonic_count = n - phasic_count
    feats["phasic_count"] = float(phasic_count)
    feats["tonic_count"] = float(tonic_count)
    feats["phasic_ratio"] = phasic_count / float(n)
    feats["tonic_ratio"] = tonic_count / float(n)

    # Alternances
    alternations = sum(1 for i in range(1, n) if labels[i] != labels[i - 1])
    feats["alternations"] = float(alternations)
    duration_min = (n * window_sec) / 60.0
    feats["alternation_rate_per_min"] = alternations / (duration_min + 1e-12)

    # Runs
    run_lengths: List[int] = []
    cur = 1
    for i in range(1, n):
        if labels[i] == labels[i - 1]:
            cur += 1
        else:
            run_lengths.append(cur)
            cur = 1
    run_lengths.append(cur)

    feats["mean_run_len"] = float(np.mean(run_lengths))
    feats["max_run_len"] = float(np.max(run_lengths))
    return feats
