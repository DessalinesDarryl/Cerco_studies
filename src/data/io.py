# src/data/io.py
import os, glob, mne
from typing import List
def find_edf_files(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "**", "*.edf"), recursive=True))
def read_raw(path: str, preload=True) -> mne.io.BaseRaw:
    return mne.io.read_raw_edf(path, preload=preload, verbose=False)
def save_fif(raw: "mne.io.BaseRaw", out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True); raw.save(out_path, overwrite=True)
