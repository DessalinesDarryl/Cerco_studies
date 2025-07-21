# src/feature_extraction.py

import os
import numpy as np
import mne
import pywt
from scipy.stats import skew
from scipy.signal import welch
from filters import filter_all_bands  

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

    # Ondelettes
    wavelet_stats = []
    for ch in data:
        coeffs = pywt.wavedec(ch, 'db4', level=4)
        ch_stats = [np.mean(c) for c in coeffs] + [np.std(c) for c in coeffs]
        wavelet_stats.append(ch_stats)
    features[f'{band_name}_wavelet'] = np.array(wavelet_stats)

    return features

def extract_all_band_features(raw):
    band_raws = filter_all_bands(raw)
    all_features = {}
    for band_name, raw_band in band_raws.items():
        band_features = extract_band_features(raw_band, band_name.replace(" ", "_"))
        all_features.update(band_features)
    return all_features

def load_and_extract(fif_file):
    raw = mne.io.read_raw_fif(fif_file, preload=True)
    return extract_all_band_features(raw)

def batch_process(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith('.fif'):
            print(f"Processing {file}...")
            features = load_and_extract(os.path.join(input_dir, file))
            save_path = os.path.join(output_dir, file.replace('.fif', '_features.npz'))
            np.savez_compressed(save_path, **features)
            print(f"Saved: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    batch_process(args.input_dir, args.output_dir)
