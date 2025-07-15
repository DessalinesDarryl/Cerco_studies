import os
import pathlib
import mne
import numpy as np
from mne.preprocessing import ICA
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann

# 1. FILTRAGE & RÉFÉRENCEMENT
def preprocess_raw(raw, l_freq=10, h_freq=100, notch=50, ref="A1"):
    raw_f = raw.copy().filter(l_freq, h_freq, fir_design="firwin")
    raw_f.notch_filter(notch)
    if ref in raw_f.ch_names:
        raw_f.set_eeg_reference([ref])
    else:
        raw_f.set_eeg_reference("average", projection=False)
    return raw_f

# 2. ICA
def run_ica(raw, method="picard", n_comp=0.999, random_state=42, tstep=30.0, resample_hz=150):
    if method not in {"fastica", "picard"}:
        raise ValueError("method doit être 'fastica' ou 'picard'")
    raw_fit = raw.copy()
    if resample_hz is not None and resample_hz < raw.info["sfreq"]:
        raw_fit = raw_fit.resample(resample_hz, npad="auto")
    fit_params = dict(ortho=False) if method == "picard" else {}
    ica = ICA(method=method, n_components=n_comp, random_state=random_state, fit_params=fit_params)
    ica.fit(raw_fit, tstep=tstep)
    eog_ch = next((c for c in raw.ch_names if "EOG" in c.upper()), None)
    ecg_ch = next((c for c in raw.ch_names if "ECG" in c.upper()), None)
    if eog_ch:
        bad_eog, _ = ica.find_bads_eog(raw_fit, ch_name=eog_ch)
        ica.exclude.extend(bad_eog)
    if ecg_ch:
        bad_ecg, _ = ica.find_bads_ecg(raw_fit, ch_name=ecg_ch)
        ica.exclude.extend(bad_ecg)
    return ica.apply(raw.copy()), ica

# 3. SEGMENTATION
def segment_fixed_epochs(raw, dur=4.0):
    events = mne.make_fixed_length_events(raw, 1, dur)
    ep = mne.Epochs(raw, events, 1, 0, dur, baseline=None, preload=True)
    ep._data = ep.get_data().astype("float32")
    return ep

# 4. RPF
def _reg_cov(c, lam):
    return c + lam * np.eye(c.shape[-1], dtype=c.dtype)

def _epoch_cov(epoch_data, lam):
    return _reg_cov(np.cov(epoch_data), lam)

def apply_rpf_batches(epochs, z=2.0, lam=1e-6, batch_size=5000):
    n_epochs = len(epochs)
    covs_list = []
    for start in range(0, n_epochs, batch_size):
        batch = epochs[start:start+batch_size].get_data()
        covs_list.extend(_epoch_cov(ep, lam) for ep in batch)
        del batch
    covs = np.stack(covs_list)
    mu = mean_riemann(covs)
    dist = np.fromiter((distance_riemann(c, mu) for c in covs_list), dtype=np.float32)
    thr = float(dist.mean() + z * dist.std())
    keep = dist < thr
    return epochs[keep], dist, thr

# 5. RECONSTRUCTION
def epochs_to_raw(epochs):
    data = np.concatenate(epochs.get_data(), axis=-1)
    return mne.io.RawArray(data, epochs.info.copy())

# 6. PIPELINE FICHIER UNIQUE
def process_file(edf_path, out_ica_dir, out_rpf_dir, new_name, z_rpf=3.0):
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        raw_p = preprocess_raw(raw)

        try:
            raw_ica, _ = run_ica(raw_p)
        except Exception as ica_err:
            return f"{edf_path.name}: erreur ICA → {ica_err}"

        # ICA
        out_ica = out_ica_dir / new_name
        out_ica.parent.mkdir(parents=True, exist_ok=True)
        raw_ica.save(out_ica, overwrite=True, verbose="ERROR")

        # RPF
        """"
        ep = segment_fixed_epochs(raw_ica)
        ep_clean, dist, thr = apply_rpf_batches(ep, z=z_rpf, batch_size=3000)

        if len(ep_clean) == 0:
            return f"{edf_path.name}: aucune époque propre (z={z_rpf}, thr={thr:.2f})"

        raw_rpf = epochs_to_raw(ep_clean)
        out_rpf = out_rpf_dir / new_name
        out_rpf.parent.mkdir(parents=True, exist_ok=True)
        raw_rpf.save(out_rpf, overwrite=True, verbose="ERROR")

        return f"{edf_path.name}: OK ({len(ep_clean)} époques propres)"
        """

    except Exception as e:
        return f"{edf_path.name}: erreur générale → {e}"


# 7. MAIN AUTOMATIQUE
if __name__ == "__main__":
    root_raw = pathlib.Path(r"C:\Users\Dessalines\Desktop\EEG\raw")
    out_ica_dir = pathlib.Path(r"D:/EEG/preprocessed_ica")
    out_rpf_dir = pathlib.Path(r"D:/EEG/preprocessed_rpf")

    edf_paths = list(root_raw.rglob("*.edf"))
    print(f"{len(edf_paths)} fichiers .edf trouvés.")

    for i, path in enumerate(edf_paths):
        parent_name = path.parent.name
        basename = path.stem.replace(" ", "").replace("-", "").upper()
        new_name = f"{parent_name}_{basename}_{i:03d}.fif"

        print(f"\nTraitement de {path.name} → sauvegarde sous {new_name}")

        # Chemin cible déjà traité ?
        out_fif = out_ica_dir / new_name
        if out_fif.exists():
            print(f"{new_name} déjà traité, ignoré.")
            continue

        res = process_file(
            edf_path=path,
            out_ica_dir=out_ica_dir,
            out_rpf_dir=out_rpf_dir,
            new_name=new_name
        )
        print(res)
