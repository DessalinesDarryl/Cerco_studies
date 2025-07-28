# src/annotations.py
import os
import pandas as pd
import mne
import numpy as np
from pathlib import Path

def load_annotation_file(txt_path):
    """
    Charge un fichier d’annotations de sommeil au format .txt et extrait les périodes de sommeil REM.

    Le fichier doit être tabulé avec au moins les colonnes suivantes :
    - "start" : temps de début (en secondes),
    - "stage" : stade de sommeil (ex. : REM, N2, etc.).

    La durée de chaque période est calculée comme la différence entre deux onsets successifs.

    Paramètres
    ----------
    txt_path : str ou Path
        Chemin vers le fichier d’annotations .txt.

    Retour
    ------
    numpy.ndarray
        Tableau 2D de forme (n, 2) contenant les segments REM,
        où chaque ligne est (start, duration) en secondes.
    """

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


def load_all_annotations_from_file(txt_path, stage_map=None):
    """
    Charge les étiquettes de stades de sommeil à partir d'un fichier texte tabulé.

    Paramètres
    ----------
    txt_path : str
        Chemin vers le fichier contenant les étiquettes (une ligne par époque de 30s).
    stage_map : dict ou None
        Dictionnaire de mappage (ex: {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}).

    Retour
    ------
    y_true : np.ndarray
        Tableau d'entiers représentant les stades de sommeil, un par époque.
    """
    labels = []

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                label_raw = parts[2].strip().upper()
                if stage_map:
                    if label_raw not in stage_map:
                        raise ValueError(f"Étiquette inconnue dans le fichier : '{label_raw}'")
                    label = stage_map[label_raw]
                else:
                    label = label_raw
                labels.append(label)

    return np.array(labels, dtype=int if stage_map else str)

