import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann

"""
EEG PREPROCESSING PIPELINE
==========================
Filtrage → Référence → ICA (FastICA | Picard) → (optionnel) Rejection by
Projection on the Riemannian manifold (RPF)

- Compatible gros fichiers (fenêtrage `tstep`, décimation temporaire)
- Picard multithreadé, FastICA mono‑thread (n_jobs retiré de scikit‑learn)
- RPF « memory‑safe » : traitement par batchs pour éviter les MemoryError

Version : 2025‑07‑11
"""

# =================================================
# 1. FILTRAGE & RÉFÉRENCEMENT
# =================================================

def preprocess_raw(raw, l_freq: float = 10, h_freq: float = 100, notch: float = 50, ref: str = "A1"):
    """Band‑pass 10‑100 Hz, notch 50 Hz, référence A1 ou average."""
    raw_f = raw.copy().filter(l_freq, h_freq, fir_design="firwin")
    raw_f.notch_filter(notch)
    if ref in raw_f.ch_names:
        raw_f.set_eeg_reference([ref])
    else:
        raw_f.set_eeg_reference("average", projection=False)
    return raw_f

# =================================================
# 2. ICA (FASTICA / PICARD)
# =================================================

def run_ica(
    raw: mne.io.BaseRaw,
    method: str = "fastica",
    n_comp: float | int = 0.99,
    random_state: int = 42,
    tstep: float = 30.0,
    resample_hz: float | None = None,
):
    """Retourne raw nettoyé + objet ICA."""
    if method not in {"fastica", "picard"}:
        raise ValueError("method doit être 'fastica' ou 'picard'")

    raw_fit = raw.copy()
    if resample_hz is not None and resample_hz < raw.info["sfreq"]:
        raw_fit = raw_fit.resample(resample_hz, npad="auto")

    fit_params = {}
    if method == "picard":
        fit_params = dict(ortho=False)

    ica = ICA(method=method, n_components=n_comp, random_state=random_state, fit_params=fit_params)
    ica.fit(raw_fit, tstep=tstep)

    # Détection EOG/ECG
    eog_inds, ecg_inds = [], []
    eog_ch = next((c for c in raw.ch_names if "EOG" in c.upper()), None)
    if eog_ch:
        eog_inds, _ = ica.find_bads_eog(raw_fit, ch_name=eog_ch)
    ecg_ch = next((c for c in raw.ch_names if "ECG" in c.upper()), None)
    if ecg_ch:
        ecg_inds, _ = ica.find_bads_ecg(raw_fit, ch_name=ecg_ch)
    ica.exclude = eog_inds + ecg_inds

    return ica.apply(raw.copy()), ica

# =================================================
# 3. SEGMENTATION FIXE
# =================================================

def segment_fixed_epochs(raw: mne.io.BaseRaw, dur: float = 4.0):
    events = mne.make_fixed_length_events(raw, 1, dur)
    epochs = mne.Epochs(raw, events, 1, 0, dur, baseline=None, preload=True)
    epochs._data = epochs.get_data().astype("float32")  # cast RAM ½
    return epochs

# =================================================
# 4. RPF « MEMORY‑SAFE » PAR BATCHS
# =================================================

def _reg_cov(c, lam: float):
    return c + lam * np.eye(c.shape[-1], dtype=c.dtype)

def _epoch_cov(epoch_data: np.ndarray, lam: float):
    return _reg_cov(np.cov(epoch_data), lam)

def apply_rpf_batches(
    epochs: mne.Epochs,
    z: float = 2.0,
    lam: float = 1e-6,
    batch_size: int = 5000,
):
    """RPF en batchs, compatible anciennes versions MNE."""
    n_epochs = len(epochs)
    covs_list: list[np.ndarray] = []

    for start in range(0, n_epochs, batch_size):
        stop = min(start + batch_size, n_epochs)
        batch_data = epochs[start:stop].get_data()  # slicing → version‑safe
        for ep in batch_data:
            covs_list.append(_epoch_cov(ep, lam))
        del batch_data  # libère RAM

    covs = np.stack(covs_list)
    mu = mean_riemann(covs)

    dist = np.empty(n_epochs, dtype=np.float32)
    idx = 0
    for start in range(0, n_epochs, batch_size):
        stop = min(start + batch_size, n_epochs)
        for c in covs_list[start:stop]:
            dist[idx] = distance_riemann(c, mu)
            idx += 1

    thr = float(dist.mean() + z * dist.std())
    keep = dist < thr
    return epochs[keep], dist, thr

# =================================================
# 5. VISUALISATION RAPIDE
# =================================================

def quick_plot(
    raw0: mne.io.BaseRaw,
    raw_ica: mne.io.BaseRaw,
    epochs_rpf: mne.Epochs | None = None,
    n_ch: int = 5,
    dur: int | None = None,
):
    """Affiche les *n_ch* premiers canaux (hors bads) sur toute la durée ou *dur* secondes.

    - Retourne également la liste `(picks, ch_names)` pour savoir quels canaux sont tracés.
    """
    sf = raw0.info["sfreq"]
    picks_all = mne.pick_types(raw0.info, eeg=True, exclude="bads")
    picks = picks_all[1 : n_ch + 1] if len(picks_all) > n_ch else picks_all[:n_ch]
    ch_names = [raw0.ch_names[p] for p in picks]

    # Portion temporelle
    if dur is None:
        t = slice(0, raw0.n_times)  # signal complet
    else:
        t = slice(0, int(dur * sf))

    rows = 3 if epochs_rpf is not None else 2
    fig, ax = plt.subplots(rows, 1, figsize=(14, 3 * rows), sharex=True)
    to_uV = lambda x: x * 1e6

    ax[0].set_title(f"Brut (µV) – canaux: {', '.join(ch_names)}")
    ax[0].plot(to_uV(raw0.get_data(picks)[:, t]).T)

    ax[1].set_title("Après ICA (µV)")
    ax[1].plot(to_uV(raw_ica.get_data(picks)[:, t]).T)

    if epochs_rpf is not None:
        ax[2].set_title("Moyenne après RPF (µV)")
        # Reconstruit un Raw-like complet à partir des epochs nettoyées si besoin
        data_rpf = epochs_rpf.get_data(picks).mean(0)
        ax[2].plot(to_uV(data_rpf.T))

    ax[-1].set_xlabel("Samples")
    plt.tight_layout()
    plt.show()

    return picks, ch_names

# =================================================
# 6. EXEMPLE UTILISATION
# =================================================

if __name__ == "__main__":
    raw = mne.io.read_raw_edf("D:/EEG/raw/MN143/MN143_240115E-A.edf", preload=True)

    raw_p = preprocess_raw(raw)

    raw_ica, ica = run_ica(raw_p, method="picard", resample_hz=200, tstep=60.0)

    ep = segment_fixed_epochs(raw_ica, dur=4.0)
    ep_clean, d, thr = apply_rpf_batches(ep, batch_size=3000)

    quick_plot(raw, raw_ica, ep_clean, n_ch=5, dur=10)
