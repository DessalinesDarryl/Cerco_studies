"""Entrées/sorties basiques pour fichiers EEG (EDF/FIF).

Ce module centralise la découverte des fichiers EDF, la lecture MNE
et la sauvegarde au format FIF pour le pipeline de prétraitement.
"""

import glob
import os

import mne

from typing import List


def find_edf_files(root: str) -> List[str]:
    """Retourne tous les fichiers EDF présents sous `root`."""
    return sorted(glob.glob(os.path.join(root, "**", "*.edf"), recursive=True))


def read_raw(path: str, preload=True) -> mne.io.BaseRaw:
    """Lit un enregistrement EDF via MNE avec un mode silencieux."""
    return mne.io.read_raw_edf(path, preload=preload, verbose=False)


def save_fif(raw: "mne.io.BaseRaw", out_path: str):
    """Sauvegarde un `Raw` MNE au format FIF en créant le dossier si besoin."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    raw.save(out_path, overwrite=True)
