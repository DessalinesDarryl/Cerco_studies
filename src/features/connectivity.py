"""Extraction de descripteurs de connectivité EEG.

Le module calcule des métriques de connectivité entre paires de canaux
à partir de signaux EEG par bandes de fréquences.
"""

import numpy as np
import mne

DEFAULT_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta":  (12.0, 30.0),
    "gamma_bas": (30.0, 50.0),
    "gamma_haut": (50.0, 80.0),
}


# ======================================================================
# Utilitaires pour énumération des paires de canaux
# ======================================================================

def _pair_indices(n):
    """Retourne tous les couples (i, j) avec i < j pour n canaux."""
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    return pairs


# ======================================================================
# Calcul des features de connectivité EEG
# ======================================================================

def compute_connectivity_features(epochs: mne.Epochs, bands=None):
    """Extrait des features de connectivité EEG par paire de canaux.

    Mesures incluses
    ----------------
    - Corrélation de Pearson temporelle
    - Cohérence magnitude-squared par bande

    Attention
    ---------
    Le nombre de features peut croître rapidement avec le nombre de canaux.
    Préférer une sélection ROI si besoin.
    """
    picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    if len(picks) < 2:
        return None, []

    if bands is None:
        bands = DEFAULT_BANDS
    else:
        bands = {k: tuple(v) for k, v in bands.items()}

    data = epochs.get_data()[:, picks, :]  # (n_epochs, n_ch, n_times)
    sf = epochs.info["sfreq"]
    n_epochs, n_ch, n_times = data.shape

    pairs = _pair_indices(n_ch)
    if not pairs:
        return None, []

    feat_list = []
    names = []

    # 1) Corrélation de Pearson temporelle par paire de canaux
    corr_feats = []
    for (i, j) in pairs:
        x = data[:, i, :]  # (n_epochs, n_times)
        y = data[:, j, :]
        # corrélation epoch-wise : corr(x_t, y_t)
        num = (x * y).mean(axis=-1) - x.mean(axis=-1) * y.mean(axis=-1)
        den = x.std(axis=-1) * y.std(axis=-1) + 1e-20
        r = num / den  # (n_epochs,)
        corr_feats.append(r)

        ch_i = epochs.ch_names[picks[i]]
        ch_j = epochs.ch_names[picks[j]]
        names.append(f"conn_corr_{ch_i}_{ch_j}")

    corr_arr = np.stack(corr_feats, axis=1)  # (n_epochs, n_pairs)
    feat_list.append(corr_arr)

    # 2) Cohérence par bande (moyenne intra-bande) pour chaque paire
    from scipy.signal import coherence

    for band_name, (fmin, fmax) in bands.items():
        band_coh = []
        for (i, j) in pairs:
            x = data[:, i, :]
            y = data[:, j, :]
            coh_vals = []
            for e in range(n_epochs):
                f, Cxy = coherence(
                    x[e], y[e], fs=sf, nperseg=min(256, n_times)
                )
                mask = (f >= fmin) & (f <= fmax)
                if mask.any():
                    coh_vals.append(Cxy[mask].mean())
                else:
                    coh_vals.append(0.0)
            band_coh.append(np.array(coh_vals))
            ch_i = epochs.ch_names[picks[i]]
            ch_j = epochs.ch_names[picks[j]]
            names.append(f"conn_coh_{band_name}_{ch_i}_{ch_j}")

        band_coh_arr = np.stack(band_coh, axis=1)  # (n_epochs, n_pairs)
        feat_list.append(band_coh_arr)

    X = np.concatenate(feat_list, axis=1)
    return X, names
