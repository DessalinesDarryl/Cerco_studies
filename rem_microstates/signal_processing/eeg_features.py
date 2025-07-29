from mne.time_frequency import psd_array_welch
import numpy as np

def extract_spectral_features(epoch):
    data = epoch.copy().pick_types(eeg=True).get_data()
    sfreq = epoch.info['sfreq']
    
    # Moyenne sur toutes les époques (n_epochs x n_channels x n_times)
    psd, freqs = psd_array_welch(data, sfreq=sfreq, fmin=1, fmax=45, n_fft=256)
    
    # Moyenne sur les canaux
    mean_psd = psd.mean(axis=0)  # shape (n_freqs,)
    
    beta_mask = (freqs >= 12) & (freqs <= 30)
    gamma_mask = (freqs >= 30) & (freqs <= 45)
    
    beta_power = mean_psd[beta_mask].sum()
    gamma_power = mean_psd[gamma_mask].sum()
    
    score = gamma_power / (beta_power + 1e-10)
    return min(score, 1.0)  # normalisé dans [0,1]
