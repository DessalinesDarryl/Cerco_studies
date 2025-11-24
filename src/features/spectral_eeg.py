# src/features/spectral_eeg.py
import numpy as np
import mne
from scipy.stats import entropy

DEFAULT_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta":  (12.0, 30.0),
    "gamma_bas": (30.0, 50.0),
    "gamma_haut": (50.0, 80.0),
}


def _spectral_base(epochs, fmin=0.5, fmax=80.0, picks=None):
    psds, freqs = mne.time_frequency.psd_welch(
        epochs,
        fmin=fmin,
        fmax=fmax,
        picks=picks,
        n_fft=None,
        n_overlap=0,
        average="mean",
        verbose=False,
    )
    # psds : (n_epochs, n_channels, n_freqs)
    return psds, freqs


def compute_spectral_eeg_features(epochs: mne.Epochs,
                                  bands=None):
    """
    Pour chaque canal EEG et epoch :
      - Bandpower absolue et relative par bande
      - Ratios inter-bandes (α/β, θ/α, θ/β, gamma_bas/β, gamma_haut/β) moyennés sur les canaux
      - Spectral centroid, spectral entropy, peak frequency, bandwidth
        (sur [0.5,80] Hz)
    """
    picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    if len(picks) == 0:
        return None, []

    if bands is None:
        bands = DEFAULT_BANDS
    else:
        bands = {k: tuple(v) for k, v in bands.items()}

    # PSD global pour 0.5-80
    psds_full, freqs = _spectral_base(epochs, fmin=0.5, fmax=80.0, picks=picks)
    # shape : (n_epochs, n_channels, n_freqs)
    psd_sum = psds_full.sum(axis=-1, keepdims=True) + 1e-20

    n_epochs, n_ch, _ = psds_full.shape

    feat_list = []
    names = []

    # --- Band powers (abs + rel) par canal ---
    band_powers_abs = []
    band_powers_rel = []
    band_names = list(bands.keys())

    for band_name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        bp = psds_full[:, :, mask].sum(axis=-1)   # (n_epochs, n_ch)
        bp_rel = bp / (psd_sum.squeeze(-1))       # relatif au total
        band_powers_abs.append(bp)
        band_powers_rel.append(bp_rel)

    # concat : (n_epochs, n_ch * nb_bands)
    bp_abs_cat = np.concatenate(band_powers_abs, axis=1)
    bp_rel_cat = np.concatenate(band_powers_rel, axis=1)

    # noms
    for ch_idx in picks:
        ch_name = epochs.ch_names[ch_idx]
        for b in band_names:
            names.append(f"eeg_{ch_name}_{b}_bp_abs")
        for b in band_names:
            names.append(f"eeg_{ch_name}_{b}_bp_rel")

    feat_list.extend([bp_abs_cat, bp_rel_cat])

    # --- Moyenne par bande sur les canaux (pour les ratios) ---
    band_means = {}
    for i, b in enumerate(band_names):
        bp_b = band_powers_abs[i]          # (n_epochs, n_ch)
        band_means[b] = bp_b.mean(axis=1)  # (n_epochs,)

    # --- Ratios inter-bandes (basés sur moyenne des canaux) ---
    ratios = []
    ratio_names = []

    def add_ratio(num, den):
        r = band_means[num] / (band_means[den] + 1e-20)
        ratios.append(r)
        ratio_names.append(f"eeg_ratio_{num}_over_{den}")

    if "alpha" in band_means and "beta" in band_means:
        add_ratio("alpha", "beta")

    if "theta" in band_means and "alpha" in band_means:
        add_ratio("theta", "alpha")

    if "theta" in band_means and "beta" in band_means:
        add_ratio("theta", "beta")

    if "gamma_bas" in band_means and "beta" in band_means:
        add_ratio("gamma_bas", "beta")

    if "gamma_haut" in band_means and "beta" in band_means:
        add_ratio("gamma_haut", "beta")

    if ratios:
        ratios_arr = np.stack(ratios, axis=1)  # (n_epochs, n_ratios)
        feat_list.append(ratios_arr)
        names.extend(ratio_names)

    # --- Centroid, entropy, peak freq, bandwidth (par canal) ---
    p_norm = psds_full / psd_sum  # normalisation
    # spectral entropy
    spec_entropy = entropy(p_norm + 1e-20, base=2, axis=-1)  # (n_epochs, n_ch)
    # spectral centroid
    centroid = (p_norm * freqs[np.newaxis, np.newaxis, :]).sum(axis=-1)
    # peak freq
    peak_idx = psds_full.argmax(axis=-1)  # indices max
    peak_freq = freqs[peak_idx]
    # bandwidth (écart-type fréquentiel)
    mean_f = centroid
    var_f = (p_norm * (freqs[np.newaxis, np.newaxis, :] - mean_f[..., np.newaxis])**2).sum(axis=-1)
    bandwidth = np.sqrt(var_f)

    spec_feats = np.concatenate([
        centroid, spec_entropy, peak_freq, bandwidth
    ], axis=1)  # (n_epochs, n_ch * 4)

    for ch_idx in picks:
        ch_name = epochs.ch_names[ch_idx]
        names.extend([
            f"eeg_{ch_name}_spec_centroid",
            f"eeg_{ch_name}_spec_entropy",
            f"eeg_{ch_name}_peak_freq",
            f"eeg_{ch_name}_bandwidth",
        ])

    feat_list.append(spec_feats)

    X = np.concatenate(feat_list, axis=1)
    return X, names
