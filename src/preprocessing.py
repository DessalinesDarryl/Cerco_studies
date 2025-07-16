import os
import pathlib
import mne
import numpy as np
from mne.preprocessing import ICA

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
    if method not in {"fastica", "picard", "infomax"}:
        raise ValueError("method doit être 'fastica', 'picard' ou 'infomax'")

    # Copie pour entraînement ICA sur EEG uniquement
    raw_fit = raw.copy().pick_types(eeg=True)
    if resample_hz is not None and resample_hz < raw_fit.info["sfreq"]:
        raw_fit = raw_fit.resample(resample_hz, npad="auto")

    fit_params = dict(ortho=False) if method == "picard" else {}
    ica = ICA(method=method, n_components=n_comp, random_state=random_state, fit_params=fit_params)
    ica.fit(raw_fit, tstep=tstep)

    # Détection des composantes EOG/ECG dans le raw complet
    eog_ch = next((c for c in raw.ch_names if "EOG" in c.upper()), None)
    ecg_ch = next((c for c in raw.ch_names if "ECG" in c.upper()), None)
    if eog_ch:
        bad_eog, _ = ica.find_bads_eog(raw, ch_name=eog_ch)
        ica.exclude.extend(bad_eog)
    if ecg_ch:
        bad_ecg, _ = ica.find_bads_ecg(raw, ch_name=ecg_ch)
        ica.exclude.extend(bad_ecg)

    # Application à tout le signal 
    return ica.apply(raw.copy()), ica

# 3. PIPELINE FICHIER UNIQUE
def process_file(edf_path, out_ica_dir, new_name):
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        raw_p = preprocess_raw(raw)

        try:
            raw_ica, _ = run_ica(raw_p)
        except Exception as ica_err:
            return f"{edf_path.name}: erreur ICA -> {ica_err}"

        # ICA
        out_ica = out_ica_dir / new_name
        out_ica.parent.mkdir(parents=True, exist_ok=True)
        raw_ica.save(out_ica, overwrite=True, verbose="ERROR")

        return f"{edf_path.name}: OK"

    except Exception as e:
        return f"{edf_path.name}: erreur générale -> {e}"

# 4. MAIN AUTOMATIQUE
if __name__ == "__main__":
    root_raw = pathlib.Path(r"C:\\Users\\Dessalines\\Desktop\\EEG\\raw")
    out_ica_dir = pathlib.Path(r"D:/EEG/preprocessed_ica")

    edf_paths = list(root_raw.rglob("*.edf"))
    print(f"{len(edf_paths)} fichiers .edf trouvés.")

    for i, path in enumerate(edf_paths):
        parent_name = path.parent.name
        basename = path.stem.replace(" ", "").replace("-", "").upper()
        new_name = f"{parent_name}_{basename}_{i:03d}.fif"

        print(f"\nTraitement de {path.name} -> sauvegarde sous {new_name}")

        # Chemin cible déjà traité ?
        out_fif = out_ica_dir / new_name
        if out_fif.exists():
            print(f"{new_name} déjà traité, ignoré.")
            continue

        res = process_file(
            edf_path=path,
            out_ica_dir=out_ica_dir,
            new_name=new_name
        )
        print(res)
