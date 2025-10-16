# src/feature_extraction.py

import os
import numpy as np
import mne
import pywt
from pathlib import Path
import platform
from scipy.stats import skew
from scipy.signal import welch
from filters import filter_all_bands
from mne.preprocessing import ICA
from sklearn.decomposition import PCA


def extract_band_features(raw_band, band_name):
    data, _ = raw_band.get_data(return_times=True)
    sfreq = raw_band.info['sfreq']
    features = {}

    # Caractéristiques statistiques
    features[f'{band_name}_mean'] = np.mean(data, axis=1)
    features[f'{band_name}_var'] = np.var(data, axis=1)
    features[f'{band_name}_std'] = np.std(data, axis=1)
    features[f'{band_name}_skew'] = skew(data, axis=1)
    features[f'{band_name}_power'] = np.mean(data**2, axis=1)

    # PSD moyenne via Welch
    _, psd = welch(data, sfreq, axis=1)
    features[f'{band_name}_psd_mean'] = np.mean(psd, axis=1)

    # Wavelet
    wavelet_stats = []
    for ch in data:
        coeffs = pywt.wavedec(ch, 'db4', level=4)
        ch_stats = [np.mean(c) for c in coeffs] + [np.std(c) for c in coeffs]
        wavelet_stats.append(ch_stats)
    features[f'{band_name}_wavelet'] = np.array(wavelet_stats)

    # PCA
    pca_var = extract_pca_features(data)
    features[f'{band_name}_pca_var'] = pca_var

    # ICA
    ica_energy = extract_ica_features_infomax(raw_band)
    features[f'{band_name}_ica_infomax_energy'] = ica_energy

    return features

def extract_all_band_features(raw):
    band_raws = filter_all_bands(raw)
    all_features = {}
    for band_name, raw_band in band_raws.items():
        band_features = extract_band_features(raw_band, band_name.replace(" ", "_"))
        all_features.update(band_features)
    return all_features

def extract_pca_features(data, n_components=5):
    """
    Applique PCA et retourne la variance expliquée des premières composantes.
    """
    pca = PCA(n_components=n_components)
    pca.fit(data.T)  # shape = (time, channels)
    return pca.explained_variance_ratio_

def extract_ica_features_infomax(raw_band, n_components=5):
    """
    Applique ICA et retourne l'énergie moyenne des sources indépendantes.
    """
    try:
        ica = ICA(n_components=n_components, method='infomax', random_state=42, max_iter='auto')
        ica.fit(raw_band)

        sources = ica.get_sources(raw_band).get_data()
        energies = np.mean(sources**2, axis=1)
        return energies
    except Exception as e:
        print(f" ICA Infomax failed: {e}")
        return np.zeros(n_components)

def load_and_extract(fif_file):
    raw = mne.io.read_raw_fif(fif_file, preload=True)
    return extract_all_band_features(raw)

def batch_process(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    input_dir = Path(input_dir)

    for fif_file in input_dir.rglob("*.fif"):
        if not fif_file.name.endswith("REM_concat.fif") or fif_file.name.startswith("._"):
            continue

        print(f"Processing {fif_file}...")
        features = load_and_extract(fif_file)

        # Création d’un sous-dossier dans results/ avec le nom du sujet
        subject = fif_file.parent.name
        save_subdir = Path(output_dir) / subject
        save_subdir.mkdir(parents=True, exist_ok=True)

        save_path = save_subdir / fif_file.name.replace(".fif", "_features.npz")
        np.savez_compressed(save_path, **features)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    input_dir = Path(f"../documents/EEG/preprocessed/bipolaire/2_rem_only/gp2")
    output_dir = f"features/bipolaire/rem_only"
    batch_process(input_dir, output_dir)
