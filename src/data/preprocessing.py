# src/data/preprocessing.py
import numpy as np, mne, os
from scipy.signal import iirnotch, filtfilt
def notch_inplace(raw, freq=50.0, q=30.0, picks=None):
    data = raw.get_data(picks=picks); fs = raw.info["sfreq"]
    b,a = iirnotch(w0=freq/(fs/2), Q=q); data = filtfilt(b,a,data,axis=1)
    raw._data[picks if picks is not None else slice(None)] = data
def basic_preprocess(raw, l_freq, h_freq, notch, ref="A1", resample_hz=250):
    if ref and ref in raw.ch_names: raw._data = raw._data - raw.get_data(picks=[raw.ch_names.index(ref)])
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    if notch: notch_inplace(raw, notch)
    if resample_hz: raw.resample(resample_hz)
    return raw
def save_epochs(epochs: mne.Epochs, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True); epochs.save(out_path, overwrite=True)
