# montages.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class Montage:
    name: str
    # Liste de triplets (anode, cathode, nouveau_nom)
    pairs: List[Tuple[str, str, str]]
    # Canaux “raw” à garder tels quels (EMG/ECG/ronflement, etc.)
    keep_raw: List[str]


# === Montage bipolaire "default" ===
MONTAGE_DEFAULT = Montage(
    name="default",
    pairs=[
        ("EOGD", "A1",  "EOGD-A1"),
        ("EOGG", "A1",  "EOGG-A1"),
        ("Fp2",  "C4",  "Fp2-C4"),
        ("C4",   "O2",  "C4-O2"),
        ("T4",   "O2",  "T4-O2"),
        ("Cz",   "Pz",  "Cz-Pz"),
        ("Fp1",  "C3",  "Fp1-C3"),
        ("C3",   "O1",  "C3-O1"),
        ("Fp1",  "T3",  "Fp1-T3"),
        ("T3",   "O1",  "T3-O1"),
    ],
    keep_raw=["Menton", "JAMBG", "JAMBD", "RONF", "EMG1", "EMG2", "ECG"],
)

# === Montage bipolaire pour le groupe 1 ===
MONTAGE_GP1 = Montage(
    name="gp1",
    pairs=[
        ("Fp1", "T3",  "Fp1-T3"), 
        ("Fp1", "C3",  "Fp1-C3"),
        ("T3",  "O1",  "T3-O1"),
        ("C3",  "O1",  "C3-O1"),
        ("Fp2", "T4",  "Fp2-T4"),
        ("Fp2", "C4",  "Fp2-C4"),
        ("T4",  "O2",  "T4-O2"),
        ("C4",  "O2",  "C4-O2"),
        ("Cz",  "Pz",  "Cz-Pz"),
    ],
    keep_raw=["Menton", "JAMBG", "JAMBD", "RONF", "EMG1", "EMG2", "ECG", "EOGD", "EOGG"],
)

# === Montage bipolaire pour le groupe 1 & 2 ===
MONTAGE_GP2 = Montage(
    name="gp2",
    pairs=[
        ("Fp1", "T3",  "Fp1-T3"),
        ("Fp1", "C3",  "Fp1-C3"),
        ("T3",  "O1",  "T3-O1"),
        ("Fp2", "T4",  "Fp2-T4"),
        ("Fp2", "C4",  "Fp2-C4"),
        ("T4",  "O2",  "T4-O2"),
        ("Fp1", "A1", "Fp1-A1"),
        ("Fp2", "A1", "Fp2-A1"),
        ("T3", "A1", "T3-A1"),
        ("C3", "A1", "C3-A1"),
        ("T4", "A1", "T4-A1"),
        ("C4", "A1", "C4-A1"),
    ],
    keep_raw=["Menton", "JAMBG", "JAMBD", "RONF", "EMG1", "EMG2", "ECG", "EOGD", "EOGG"],
)

# On enregistre tous les montages connus ici
MONTAGES: Dict[str, Montage] = {
    MONTAGE_DEFAULT.name: MONTAGE_DEFAULT,
    MONTAGE_GP1.name: MONTAGE_GP1,
    MONTAGE_GP2.name: MONTAGE_GP2,
}


def get_montage(name: str) -> Montage:
    key = name.lower().strip()
    if key not in MONTAGES:
        raise KeyError(
            f"Montage inconnu: '{name}'. Montages disponibles: {list(MONTAGES.keys())}"
        )
    return MONTAGES[key]
