# src/annotations.py
import os
import pandas as pd
import mne
from pathlib import Path

def load_annotation_file(txt_path):
    df = pd.read_csv(txt_path, sep="\t", names=["start", "temps", "stage", "index"])
    df = df.dropna(subset=["start", "stage"])
    df["duration"] = df["start"].shift(-1) - df["start"]
    df = df[:-1]
    rem_df = df[df["stage"].str.upper().str.strip() == "REM"]
    return rem_df[["start", "duration"]].values

def get_rem_annotations(base_name, annot_dir):
    """
    Cherche automatiquement un fichier .txt d'annotations basé sur le dossier du code patient.
    Ex: base_name = AN166_240425CA -> cherche un .txt dans D:/EEG/raw/AN166/
    """
    patient_code = base_name.split("_")[0]
    txt_dir = Path(annot_dir) / patient_code
    txt_candidates = list(txt_dir.glob("*.txt"))

    for txt_path in txt_candidates:
        try:
            rem_intervals = load_annotation_file(txt_path)
            if len(rem_intervals) > 0:
                return mne.Annotations(
                    onset=[start for start, _ in rem_intervals],
                    duration=[dur for _, dur in rem_intervals],
                    description=["REM"] * len(rem_intervals)
                )
        except Exception:
            continue
    return None
