# src/features/spectral.py
import numpy as np, pandas as pd, mne, os
from scipy.stats import skew, kurtosis
def bandpow(x, sfreq, fmin, fmax):
    psd, freqs = mne.time_frequency.psd_array_welch(x, sfreq, fmin=fmin, fmax=fmax, verbose=False)
    return np.mean(10*np.log10(psd+1e-12), axis=1)
def extract_epoch_features(epochs_fif: str, bands: dict, stats=("mean","var","skew","kurt")):
    epochs = mne.read_epochs(epochs_fif, preload=True, verbose=False)
    X = []; info=[]
    for i,ep in enumerate(epochs):
        row={}
        for ch_idx,ch_name in enumerate(epochs.ch_names):
            sig = ep[ch_idx]; sf=epochs.info["sfreq"]
            for bname,(f1,f2) in bands.items():
                p = bandpow(sig, sf, f1,f2).mean()
                row[f"{ch_name}_{bname}_db"]=float(p)
        # stats glob
        v = np.array(list(row.values()))
        if "mean" in stats: row["feat_mean"]=float(v.mean())
        if "var" in stats: row["feat_var"]=float(v.var())
        if "skew" in stats: row["feat_skew"]=float(skew(v))
        if "kurt" in stats: row["feat_kurt"]=float(kurtosis(v))
        X.append(row); info.append({"epoch_idx":i})
    df = pd.DataFrame(X); meta = pd.DataFrame(info)
    return df, meta
