# src/preprocessing.py

import os
import sys
from pathlib import Path
import platform
import mne
import numpy as np
from mne.preprocessing import ICA
from utils import apply_custom_bipolar_montage
from filters import apply_custom_filters

# 1. FILTRAGE & RÉFÉRENÇAGE
def preprocess_raw_A1(raw, l_freq=0.3, h_freq=100, notch=50, ref="A1"):
    raw_f = raw.copy().filter(l_freq, h_freq, fir_design="firwin")
    raw_f.notch_filter(notch)
    if ref in raw_f.ch_names:
        raw_f.set_eeg_reference([ref])
    else:
        raw_f.set_eeg_reference("average", projection=False)
    return raw_f

def preprocess_raw_bip(raw, l_freq=0.3, h_freq=100, notch=50):
    raw_f = raw.copy().filter(l_freq, h_freq, fir_design="firwin")
    raw_f.notch_filter(notch)
    print("Application du montage bipolaire personnalisé...")
    raw_f = apply_custom_bipolar_montage(raw)
    print("Application des filtres sur les canaux EOG, EMG et EEG...")
    raw_f = apply_custom_filters(raw)
    return raw_f

# 2. DÉTECTION ET INTERPOLATION DES ARTÉFACTS
def detect_and_interpolate_artifacts(raw, edf_path, win_sec=4, method='covar', threshold=3):
    import yasa
    import pandas as pd

    sfreq = raw.info['sfreq']
    data = raw.get_data()
    n_samples = data.shape[1]
    win_samples = int(win_sec * sfreq)

    art, zscores = yasa.art_detect(data, sf=sfreq, window=win_sec, method=method, threshold=threshold)

    valid_mask = np.full(n_samples, False)
    for i, is_art in enumerate(art):
        if not is_art:
            start = i * win_samples
            end = min(start + win_samples, n_samples)
            valid_mask[start:end] = True

    interp_data = data.copy()
    x = np.arange(n_samples)
    for ch in range(data.shape[0]):
        bad = ~valid_mask
        good = valid_mask
        interp_data[ch, bad] = np.interp(x[bad], x[good], data[ch, good])

    # Export CSV
    art_windows = [(i * win_sec, i * win_sec + win_sec) for i, a in enumerate(art) if a]
    df_art = pd.DataFrame(art_windows, columns=["start_time_s", "end_time_s"])

    out_dir = edf_path.parent
    export_path = out_dir / f"{edf_path.stem}_artifact_windows.csv"
    df_art.to_csv(export_path, index=False)
    print(f"Artéfacts exportés : {len(df_art)} fenêtres -> {export_path.name}")

    # Annotations MNE
    onset = [start for start, end in art_windows]
    duration = [win_sec] * len(onset)
    description = ["BAD_Artifact"] * len(onset)
    annotations = mne.Annotations(onset=onset, duration=duration, description=description)

    raw_interp = raw.copy()
    raw_interp._data = interp_data
    raw_interp.set_annotations(annotations)

    # Log TXT par patient
    pourcentage_conserve = valid_mask.sum() / len(valid_mask) * 100
    patient_name = edf_path.stem.split("_")[0]
    log_path = out_dir / f"{patient_name}_artifact_report.txt"
    with open(log_path, "a") as f:
        f.write(f"{edf_path.stem}.fif : {pourcentage_conserve:.2f}% du signal conservé après suppression des artéfacts.\n")

    return raw_interp, valid_mask

# 3. PIPELINE UNIQUE FICHIER
def process_file(edf_path, out_yasa_dir, new_name, montage):
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

        # Renommage
        print(f"Liste des canaux actuels : {raw.ch_names}")
        rename_dict = {ch: ch.replace("EEG ", "", 1) for ch in raw.ch_names if ch.startswith("EEG ")}
        if rename_dict:
            raw.rename_channels(rename_dict)
            print(f"Liste des canaux à jour : {raw.ch_names}")

        # Prétraitement
        if montage == "bipolaire":
            raw_p = preprocess_raw_bip(raw)
        else:
            raw_p = preprocess_raw_A1(raw)

        # Dossier patient
        patient_id = edf_path.parent.name
        patient_dir = out_yasa_dir / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        out_yasa = patient_dir / new_name

        # Nettoyage YASA
        raw_clean, valid_mask = detect_and_interpolate_artifacts(raw_p, out_yasa)

        # Sauvegarde .fif
        raw_clean.save(out_yasa, overwrite=True, verbose="ERROR")
        return f"{edf_path.name}: OK"

    except Exception as e:
        return f"{edf_path.name}: erreur -> {e}"

# 4. MAIN
if __name__ == "__main__":
    response = input("Le montage est-il bipolaire ? (y/n) : ").strip().lower()
    if response not in {"y", "n"}:
        print("Réponse invalide. Entrez 'y' pour oui ou 'n' pour non.")
        sys.exit(1)

    montage = "bipolaire" if response == "y" else "monopolaire"
    print(f"Montage sélectionné : {montage}")

    system = platform.system()
    if system == "Darwin":
        disque = "/Volumes/Crucial X6"
    elif system == "Windows":
        disque = "D:"
    else:
        raise RuntimeError("Système non supporté.")

    root_raw = Path(f"{disque}/EEG/raw")
    out_base = Path(f"{disque}/EEG/preprocessed")
    out_yasa_dir = out_base / montage / "full"

    edf_paths = list(root_raw.rglob("*.edf"))
    print(f"{len(edf_paths)} fichiers .edf trouvés.")

    for path in edf_paths:
        parent_name = path.parent.name
        suffix = "bip" if montage == "bipolaire" else "monop"
        new_name = f"{parent_name}_preprocessed_{suffix}.fif"
        out_fif = out_yasa_dir / parent_name / new_name

        if out_fif.exists():
            print(f"{new_name} déjà traité, ignoré.")
            continue

        print(f"\nTraitement de {path.name} -> {new_name}")
        result = process_file(path, out_yasa_dir, new_name, montage)
        print(result)
