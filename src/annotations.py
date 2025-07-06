# src/annotations.py
import os
import glob
import pandas as pd
import mne

def load_annotation_file(txt_path):
    df = pd.read_csv(txt_path, sep="\t", names=["start", "temps", "stage", "index"])
    df = df.dropna(subset=["start", "stage"])
    df["duration"] = df["start"].shift(-1) - df["start"]
    df = df[:-1]
    rem_df = df[df["stage"].str.upper().str.strip() == "REM"]
    return rem_df[["start", "duration"]].values

def get_rem_annotations(base_name, annot_dir):
    txt_candidates = glob.glob(os.path.join(annot_dir, "**", f"{base_name}hypnoEXP.txt"), recursive=True)
    if not txt_candidates:
        return None
    txt_path = txt_candidates[0]
    rem_intervals = load_annotation_file(txt_path)
    if len(rem_intervals) == 0:
        return None
    return mne.Annotations(
        onset=[start for start, _ in rem_intervals],
        duration=[dur for _, dur in rem_intervals],
        description=["REM"] * len(rem_intervals)
    )
