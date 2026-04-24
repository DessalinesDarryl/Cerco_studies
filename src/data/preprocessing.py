#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocessing.py

Fonctions de prétraitement pour les enregistrements :
- Application du montage bipolaire gp2 (avec conservation EMG/ECG/EOG bruts)
- Filtres :
    * EEG : 0.5-80 Hz + notch 50 Hz
    * EMG : 30-100 Hz + notch 50 Hz
- Lecture et parsing des hypnogrammes (txt "EXP" ou csv)
- Upsampling de l'hypnogramme vers la longueur des données
- Détection d'artéfacts avec YASA (art_detect) sur les EEG
- Annotation des artéfacts ("ARTEFACT") dans raw.annotations
- Annotation des segments REM dans raw.annotations à partir de l'hypnogramme

À utiliser depuis le script "preprocess.py", par exemple :

    raw = mne.io.read_raw_edf(path, preload=True)
    raw = preprocess_record(
        raw=raw,
        base_name=base,
        hypno_root=Path("/home/.../EEG/raw"),
        eeg_params={"l_freq": 0.5, "h_freq": 80.0, "notch": 50.0},
        emg_params={"hp": 30.0, "lp": 100.0, "notch": 50.0},
        yasa_params={"win_sec": 4.0, "method": "covar", "threshold": 3.0, "include": "sleep"},
    )

"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import mne


# ======================================================================
# 1) Construction du montage bipolaire gp2
# ======================================================================

@dataclass
class Montage:
    name: str
    # Liste des triplets (anode, cathode, nouveau_nom)
    pairs: List[Tuple[str, str, str]]
    # Canaux conservés tels quels (EMG, ECG, EOG, ronflement, etc.)
    keep_raw: List[str]


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


def apply_montage_gp2(raw: mne.io.BaseRaw, base_name: str) -> mne.io.BaseRaw:
    """
    Crée un montage bipolaire gp2 à partir du Raw.
    - Les paires bipolaires sont calculées : anode - cathode.
    - Les canaux listés dans keep_raw sont conservés tels quels (si présents).
    """
    sf = raw.info["sfreq"]
    raw_in = raw.copy()

    data_bip = []
    ch_names_bip = []
    ch_types_bip = []

    # 1) Construction des dérivations bipolaires définies dans `MONTAGE_GP2`
    for anode, cathode, new_name in MONTAGE_GP2.pairs:
        if anode not in raw_in.ch_names or cathode not in raw_in.ch_names:
            print(f"[{base_name}] Skip {new_name} : canal manquant ({anode} ou {cathode})")
            continue

        try:
            a_idx = raw_in.ch_names.index(anode)
            c_idx = raw_in.ch_names.index(cathode)
            sig_anode = raw_in.get_data(picks=[a_idx])[0]
            sig_cath  = raw_in.get_data(picks=[c_idx])[0]
        except Exception as e:
            print(f"[{base_name}] Skip {new_name} : erreur lors de l'accès au signal ({e})")
            continue

        bip = sig_anode - sig_cath
        data_bip.append(bip)
        ch_names_bip.append(new_name)
        ch_types_bip.append("eeg")

    # 2) Conservation des canaux auxiliaires sous leur forme d'origine
    for ch in MONTAGE_GP2.keep_raw:
        if ch not in raw_in.ch_names:
            continue
        idx = raw_in.ch_names.index(ch)
        sig = raw_in.get_data(picks=[idx])[0]
        data_bip.append(sig)
        ch_names_bip.append(ch)
        ch_types_bip.append(raw_in.get_channel_types(picks=[idx])[0])

    if not data_bip:
        # Aucun canal exploitable pour le montage : on renvoie une copie du Raw initial.
        return raw_in

    data_bip = np.vstack(data_bip)
    info_new = mne.create_info(
        ch_names=ch_names_bip,
        sfreq=sf,
        ch_types=ch_types_bip,
        verbose=False,
    )
    raw_bip = mne.io.RawArray(data_bip, info_new, verbose=False)

    # 3) Restauration des annotations existantes sur le nouveau `RawArray`
    if raw_in.annotations is not None and len(raw_in.annotations) > 0:
        ann_old = raw_in.annotations
        ann_existing = mne.Annotations(
            onset=ann_old.onset.tolist(),
            duration=ann_old.duration.tolist(),
            description=ann_old.description.tolist(),
            orig_time=None,
        )
        raw_bip.set_annotations(ann_existing)

    return raw_bip


# ======================================================================
# 2) Filtres EEG et EMG
# ======================================================================

def filter_eeg(raw: mne.io.BaseRaw, l_freq=0.5, h_freq=80.0, notch=50.0):
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks):
        raw.filter(l_freq=l_freq, h_freq=h_freq, picks=eeg_picks, verbose=False)
        if notch:
            raw.notch_filter(freqs=[notch], picks=eeg_picks, verbose=False)
    return eeg_picks


def filter_emg(raw: mne.io.BaseRaw, hp=30.0, lp=100.0, notch=50.0):
    emg_picks = mne.pick_types(raw.info, emg=True, exclude=[])
    if len(emg_picks):
        raw.filter(l_freq=hp, h_freq=lp, picks=emg_picks, verbose=False)
        if notch:
            raw.notch_filter(freqs=[notch], picks=emg_picks, verbose=False)
    return emg_picks


# ======================================================================
# 3) Lecture de l'hypnogramme, conversion YASA et upsampling
# ======================================================================

YASA_CODE = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
# Certains exports `EXP` utilisent un codage propriétaire : 1=Wake, 2=REM, 3=N1, 4=N2, 5=N3.
EXP_NUM_TO_YASA = {1: 0, 2: 4, 3: 1, 4: 2, 5: 3}


def _infer_epoch_len_sec(seconds_col: List[float]) -> float:
    if len(seconds_col) < 2:
        return 30.0
    diffs = np.diff(seconds_col)
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if len(diffs) else 30.0


def _read_hypno_txt(path: Path):
    """
    Lit un .txt type EXP : colonnes ~ [seconds, HH:MM:SS, label, code]
    Retourne (codes_YASA_par_epoch, epoch_len_sec)
    """
    seconds = []
    labels = []
    codes_num = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 3:
                continue
            try:
                sec = float(parts[0])
            except Exception:
                continue
            seconds.append(sec)
            lab = parts[2].upper()
            labels.append(lab)
            if len(parts) >= 4:
                try:
                    codes_num.append(int(parts[3]))
                except Exception:
                    codes_num.append(None)
            else:
                codes_num.append(None)

    if not seconds:
        raise ValueError(f"Hypnogramme vide ou non lisible: {path}")

    epoch_len = _infer_epoch_len_sec(seconds)

    out = []
    for lab, cnum in zip(labels, codes_num):
        if lab in YASA_CODE:
            out.append(YASA_CODE[lab])
        elif cnum is not None and cnum in EXP_NUM_TO_YASA:
            out.append(EXP_NUM_TO_YASA[cnum])
        else:
            out.append(-2)  # unscored
    return np.asarray(out, dtype=int), epoch_len


def _read_hypno_csv(path: Path):
    """
    CSV générique. Essaie colonnes : "stage"/"label" texte, sinon "code" num.
    Optionnellement une colonne "seconds" pour inférer l'epoch.
    """
    import pandas as pd

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    epoch_len = 30.0
    if "seconds" in cols:
        sec_col = df[cols["seconds"]].values
        if len(sec_col) >= 2:
            epoch_len = _infer_epoch_len_sec(list(sec_col))

    if "stage" in cols:
        labels = df[cols["stage"]].astype(str).str.upper().tolist()
    elif "label" in cols:
        labels = df[cols["label"]].astype(str).str.upper().tolist()
    else:
        labels = None

    codes = None
    if "code" in cols:
        try:
            codes = df[cols["code"]].astype(float).astype(int).tolist()
        except Exception:
            codes = None

    out = []
    if labels is not None:
        for i, lab in enumerate(labels):
            if lab in YASA_CODE:
                out.append(YASA_CODE[lab])
            else:
                cnum = None if codes is None else codes[i]
                if cnum is not None and cnum in EXP_NUM_TO_YASA:
                    out.append(EXP_NUM_TO_YASA[cnum])
                else:
                    out.append(-2)
    elif codes is not None:
        for cnum in codes:
            out.append(EXP_NUM_TO_YASA.get(cnum, -2))
    else:
        raise ValueError(
            f"Colonnes de stade introuvables dans {path} (attendu 'stage'/'label' ou 'code')."
        )

    return np.asarray(out, dtype=int), float(epoch_len)


def _find_hypno_file(base: str, hypno_root: Path) -> Optional[Path]:
    """
    Cherche un fichier hypnogramme dans :
        hypno_root/base/
    ou   hypno_root/base_*/
    ou   hypno_root directement

    suffixes supportés :
        *_hypno*.txt|csv
        *hypnogram*.txt|csv
        *_hypnoEXP.txt|csv
    """
    if hypno_root is None or not hypno_root.exists():
        return None

    patterns = [
        f"{base}*hypno*.txt", f"{base}*hypno*.csv",
        f"{base}*.hypno.txt", f"{base}*.hypno.csv",
        f"{base}_hypnoEXP.txt", f"{base}_hypnoEXP.csv",
        f"{base}*hypnogram*.txt", f"{base}*hypnogram*.csv",
    ]

    candidates: List[Path] = []

    # Sous-dossier patient direct : hypno_root/base/
    subdir = hypno_root / base
    if subdir.exists():
        for pat in patterns:
            candidates.extend(subdir.glob(pat))

    # Sous-dossiers du type hypno_root/base_*
    for child in hypno_root.glob(f"{base}*"):
        if child.is_dir():
            for pat in patterns:
                candidates.extend(child.glob(pat))

    # Racine hypno_root (fallback)
    for pat in patterns:
        candidates.extend(hypno_root.glob(pat))

    if not candidates:
        return None

    # On prend le fichier le plus gros (souvent plus complet)
    return max(candidates, key=lambda p: p.stat().st_size)


def load_hypnogram(base_name: str, hypno_root: Optional[Path]) -> Tuple[Optional[np.ndarray], Optional[float]]:
    if hypno_root is None:
        return None, None
    hp = _find_hypno_file(base_name, hypno_root)
    if hp is None:
        return None, None
    try:
        if hp.suffix.lower() == ".txt":
            hypno_epochs, epoch_len = _read_hypno_txt(hp)
        else:
            hypno_epochs, epoch_len = _read_hypno_csv(hp)
        return hypno_epochs, epoch_len
    except Exception:
        return None, None


def upsample_hypno_to_data(hypno_epochs: np.ndarray, epoch_len_sec: float,
                           data_len: int, sf_data: float) -> np.ndarray:
    """
    Upsample l'hypnogramme (au pas epoch_len_sec) à un vecteur par échantillon (len = data_len).
    """
    import yasa

    sf_hypno = 1.0 / float(epoch_len_sec)
    hypno_up = yasa.hypno_upsample_to_data(
        hypno=hypno_epochs,
        sf_hypno=sf_hypno,
        data=np.zeros((1, data_len), dtype=float),
        sf_data=sf_data,
    )
    return hypno_up.astype(int)


def hypno_to_rem_intervals(hypno_epochs: np.ndarray, epoch_len_sec: float):
    """
    Convertit une séquence de codes YASA (0..4) en intervalles REM (code == 4),
    au format liste de (start_s, end_s).
    """
    intervals = []
    current_start = None
    for i, code in enumerate(hypno_epochs):
        if code == YASA_CODE["REM"]:
            if current_start is None:
                current_start = i * epoch_len_sec
        else:
            if current_start is not None:
                end_t = i * epoch_len_sec
                if end_t > current_start:
                    intervals.append((current_start, end_t))
                current_start = None
    # Fin de série
    if current_start is not None:
        end_t = len(hypno_epochs) * epoch_len_sec
        if end_t > current_start:
            intervals.append((current_start, end_t))
    return intervals


def add_rem_annotations(raw: mne.io.BaseRaw, hypno_epochs: np.ndarray, epoch_len_sec: float, desc: str = "REM"):
    rem_int = hypno_to_rem_intervals(hypno_epochs, epoch_len_sec)
    if not rem_int:
        return

    onset = [s for s, _ in rem_int]
    duration = [e - s for s, e in rem_int]

    ann = mne.Annotations(
        onset=onset,
        duration=duration,
        description=[desc] * len(onset),
        orig_time=None,
    )

    existing = raw.annotations if getattr(raw, "annotations", None) is not None else None

    if existing is not None and len(existing) > 0:
        existing_new = mne.Annotations(
            onset=existing.onset.tolist(),
            duration=existing.duration.tolist(),
            description=existing.description.tolist(),
            orig_time=None,
        )
        raw.set_annotations(existing_new + ann)
    else:
        raw.set_annotations(ann)


# ======================================================================
# 4) Détection d'artéfacts YASA (simplifiée à l'essentiel)
# ======================================================================

def detect_artifacts_windows(raw: mne.io.BaseRaw,
                             win_sec: float = 4.0,
                             method: str = "covar",
                             threshold: float = 3.0,
                             hypno_samples: Optional[np.ndarray] = None,
                             include_stages: str = "sleep"):
    """
    Applique yasa.art_detect sur les EEG uniquement, optionnellement contraint par hypnogramme.
    include_stages: 'sleep' -> (1,2,3,4) ; 'rem' -> (4,) ; 'all' -> (0,1,2,3,4)
    Retourne:
      - art : vecteur binaire par fenêtre
      - windows_s : liste de (start_s, end_s) pour les fenêtres artefactuées fusionnées
    """
    try:
        import yasa
    except Exception as e:
        raise RuntimeError("Le module 'yasa' est requis (`pip install yasa`).") from e

    picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, emg=False, misc=False)
    raw_eeg = raw.copy().pick(picks) if len(picks) > 0 else raw.copy()

    sf = float(raw_eeg.info["sfreq"])
    data = raw_eeg.get_data() * 1e6  # µV
    n_samples = data.shape[1]

    include_map = {
        "sleep": (1, 2, 3, 4),
        "rem": (4,),
        "all": (0, 1, 2, 3, 4),
    }
    include = include_map.get(include_stages, (1, 2, 3, 4))

    hypno_vec = None
    if hypno_samples is not None:
        if len(hypno_samples) != n_samples:
            raise ValueError(f"Hypnogramme upsamplé ({len(hypno_samples)}) != n_samples ({n_samples}).")
        hypno_vec = hypno_samples.astype(int)

    if method not in {"covar", "std"}:
        raise ValueError("`method` doit être 'covar' ou 'std'.")

    art, _zs = yasa.art_detect(
        data=data,
        sf=sf,
        window=win_sec,
        hypno=hypno_vec,
        include=include,
        method=method,
        threshold=threshold,
        verbose=False,
    )

    t_end = n_samples / sf
    windows_s = []
    for i, is_art in enumerate(art):
        if is_art:
            s = i * win_sec
            e = min((i + 1) * win_sec, t_end)
            if e > s:
                windows_s.append((float(s), float(e)))
    windows_s = merge_intervals(windows_s)
    return art, windows_s


def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    ints = sorted(intervals, key=lambda x: x[0])
    merged = [ints[0]]
    for s, e in ints[1:]:
        s0, e0 = merged[-1]
        if s <= e0:
            merged[-1] = (s0, max(e0, e))
        else:
            merged.append((s, e))
    return merged


def annotate_artifacts(raw: mne.io.BaseRaw, windows_s: List[Tuple[float, float]], desc: str = "ARTEFACT") -> mne.io.BaseRaw:
    if not windows_s:
        return raw

    onset = [s for s, _ in windows_s]
    duration = [e - s for _, e in windows_s]

    ann = mne.Annotations(
        onset=onset,
        duration=duration,
        description=[desc] * len(onset),
        orig_time=None,
    )

    existing = raw.annotations if getattr(raw, "annotations", None) is not None else None

    if existing is not None and len(existing) > 0:
        existing_new = mne.Annotations(
            onset=existing.onset.tolist(),
            duration=existing.duration.tolist(),
            description=existing.description.tolist(),
            orig_time=None,
        )
        raw.set_annotations(existing_new + ann)
    else:
        raw.set_annotations(ann)

    return raw

def _check_required_gp2_channels(raw: mne.io.BaseRaw, base_name: str):
    """
    Vérifie que tous les canaux nécessaires au montage gp2 sont présents.
    Si des canaux manquent, lève une RuntimeError marquée 'MISSING_CHANNELS'
    pour permettre au script appelant de skipper proprement le patient.
    """
    missing = set()
    for anode, cathode, _ in MONTAGE_GP2.pairs:
        if anode not in raw.ch_names:
            missing.add(anode)
        if cathode not in raw.ch_names:
            missing.add(cathode)

    if missing:
        # Message volontairement structuré pour être interprété côté `scripts/preprocess.py`.
        missing_str = ",".join(sorted(missing))
        raise RuntimeError(f"MISSING_CHANNELS:{base_name}:{missing_str}")


# ======================================================================
# 5) Orchestration complète du prétraitement
# ======================================================================

def preprocess_record(
    raw: mne.io.BaseRaw,
    base_name: str,
    hypno_root: Optional[Path] = None,
    eeg_params: Optional[dict] = None,
    emg_params: Optional[dict] = None,
    yasa_params: Optional[dict] = None,
) -> mne.io.BaseRaw:
    """
    Exécute le prétraitement complet d'un enregistrement.

    Étapes
    ------
    0. normalisation des noms de canaux et vérification des canaux requis
    1. application du montage bipolaire gp2
    2. filtrage EEG (0.5-80 Hz + notch) et EMG (30-100 Hz + notch)
    3. chargement optionnel de l'hypnogramme et conversion vers les codes YASA
    4. annotation des segments REM dans `raw.annotations`
    5. upsampling de l'hypnogramme au niveau échantillon
    6. détection puis annotation des fenêtres d'artéfacts EEG

    Paramètres:
      - base_name : identifiant patient/fichier, utilisé pour retrouver l'hypnogramme
      - hypno_root : dossier contenant les fichiers hypnogrammes ({base_name}_hypnoEXP.txt/csv, etc.)
      - eeg_params : dict {l_freq, h_freq, notch}
      - emg_params : dict {hp, lp, notch}
      - yasa_params : dict {win_sec, method, threshold, include}
    """
    eeg_params = eeg_params or dict(l_freq=0.5, h_freq=80.0, notch=50.0)
    emg_params = emg_params or dict(hp=30.0, lp=100.0, notch=50.0)
    yasa_params = yasa_params or dict(win_sec=4.0, method="covar", threshold=3.0, include="sleep")

    # 0) Normalisation des noms de canaux puis vérification du montage requis
    rename_dict = {ch: ch.replace("EEG ", "") for ch in raw.ch_names if ch.startswith("EEG ")}
    if rename_dict:
        raw.rename_channels(rename_dict)
        print(f"[{base_name}] Canaux renommés : {rename_dict}")

    _check_required_gp2_channels(raw, base_name)

    # 1) Application du montage bipolaire gp2
    raw = apply_montage_gp2(raw, base_name)


    # 2) Application des filtres EEG et EMG
    eeg_picks = filter_eeg(raw, **eeg_params)
    _ = filter_emg(raw, **emg_params)

    # 3) Chargement optionnel de l'hypnogramme
    hypno_epochs, epoch_len = load_hypnogram(base_name, hypno_root) if hypno_root is not None else (None, None)

    # 4) Ajout des annotations REM si l'hypnogramme est disponible
    if hypno_epochs is not None and epoch_len is not None:
        add_rem_annotations(raw, hypno_epochs, epoch_len)

    # 5) Upsampling de l'hypnogramme au niveau échantillon pour YASA
    hypno_samples = None
    if hypno_epochs is not None and epoch_len is not None:
        sf = float(raw.info["sfreq"])
        n_samples = raw.n_times
        hypno_samples = upsample_hypno_to_data(hypno_epochs, epoch_len, n_samples, sf)

    # 6) Détection et annotation des artéfacts EEG
    try:
        _, art_windows = detect_artifacts_windows(
            raw,
            win_sec=yasa_params.get("win_sec", 4.0),
            method=yasa_params.get("method", "covar"),
            threshold=yasa_params.get("threshold", 3.0),
            hypno_samples=hypno_samples,
            include_stages=yasa_params.get("include", "sleep"),
        )
        raw = annotate_artifacts(raw, art_windows, desc="ARTEFACT")
    except Exception:
        # En cas d'échec, le pipeline continue sans annotations d'artéfacts.
        print("Pas d'annotation ARTEFACT : YASA n'est probablement pas installé.")
        pass

    return raw
