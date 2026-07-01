from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
01_preprocess.py  -  Prétraitement batch des fichiers EDF
=============================================================================

Ce script est AUTONOME : il ne dépend d'aucun dossier src/, ni de fichier YAML.

Ce qu'il fait :
  - Recherche tous les fichiers EDF sous RAW_ROOT
  - Pour chaque patient :
      1. Lecture du fichier EDF
      2. Montage bipolaire GP2
      3. Filtrage EEG (0.5–80 Hz + notch 50 Hz) et EMG (30–100 Hz + notch 50 Hz)
      4. Chargement de l'hypnogramme (txt ou csv) et annotation des segments REM
      5. Détection des artefacts EEG avec YASA et annotation "ARTEFACT"
  - Sauvegarde au format .fif dans PREPROCESSED_ROOT

Comment utiliser ce script :
  1. Modifier les chemins dans la section "PARAMÈTRES" ci-dessous
  2. Lancer :  python scripts/scripts_fonctionnels/01_preprocess.py 2>&1 | tee logfiles/log_01-preprocessing.txt

Dépendances requises :
  pip install mne yasa numpy pandas

=============================================================================
 
INPUTS
------
RAW_ROOT        : dossier contenant les fichiers EDF bruts
                  Structure attendue : RAW_ROOT/**/*.edf (récursif)
                  Exemple            : data/raw/AN166/AN166_raw.edf
 
HYPNO_ROOT      : dossier contenant les hypnogrammes patients
                  Formats acceptés   : {patient}_hypnoEXP.txt  |  .csv
                  Exemple            : data/raw/AN166/AN166_hypnoEXP.txt
                  (None = utilise les annotations présentes dans le FIF)
 
OUTPUTS
-------
PREPROCESSED_ROOT : fichiers .fif prétraités, arborescence conservée
                    Exemple : data/preprocessed/AN166/AN166_raw.fif
                    Contenu : signal filtré (EEG 0.5–80 Hz, EMG 30–100 Hz)
                              + annotations REM et ARTEFACT dans raw.annotations
 
DÉPENDANCES PIPELINE
--------------------
Ce script est l'ÉTAPE 1. Sa sortie alimente :
  > 02_segment_rem.py   (PREPROCESSED_ROOT > rem_epo/)
  > 03_segment_rswa.py  (PREPROCESSED_ROOT > CSV EMG)
"""

# ============================================================================
# PARAMÈTRES  <<<  À MODIFIER SELON LA CONFIGURATION SOUHAITÉE
# ============================================================================

# Dossier contenant les fichiers EDF bruts (recherche récursive)
RAW_ROOT = r"c:\dev\raw\data_raw_edf"

# Dossier de sortie pour les fichiers .fif prétraités
PREPROCESSED_ROOT =  r"c:\dev\raw\data_raw_fif"

# Dossier contenant les hypnogrammes ({patient}_hypnoEXP.txt ou .csv)
# Mettre None si vous n'avez pas d'hypnogramme
HYPNO_ROOT = r"c:\dev\raw\hypnogrammes"

# Paramètres de filtrage EEG
EEG_L_FREQ  = 0.5    # Fréquence de coupure basse (Hz)
EEG_H_FREQ  = 80.0   # Fréquence de coupure haute (Hz)
EEG_NOTCH   = 50.0   # Filtre coupe-bande (Hz) - mettre None pour désactiver

# Paramètres de filtrage EMG
EMG_HP      = 30.0   # Fréquence de coupure basse (Hz)
EMG_LP      = 100.0  # Fréquence de coupure haute (Hz)
EMG_NOTCH   = 50.0   # Filtre coupe-bande (Hz) - mettre None pour désactiver

# Paramètres de détection d'artefacts YASA
YASA_WIN_SEC    = 4.0     # Durée des fenêtres d'analyse (secondes)
YASA_METHOD     = "covar" # Méthode : "covar" ou "std"
YASA_THRESHOLD  = 3.0     # Seuil (en écarts-types)
YASA_INCLUDE    = "sleep" # Stades analysés : "sleep", "rem" ou "all"

# Nombre de patients traités en parallèle
N_WORKERS = 4  # Réduire si votre machine manque de RAM

# ============================================================================
# IMPORTS
# ============================================================================

import glob
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np

# ============================================================================
# LOGGER
# ============================================================================

def _get_logger(name: str = "preprocess") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


# ============================================================================
# MONTAGE BIPOLAIRE GP2
# ============================================================================

@dataclass
class Montage:
    name: str
    pairs: List[Tuple[str, str, str]]   # (anode, cathode, nouveau_nom)
    keep_raw: List[str]                 # canaux conservés tels quels


MONTAGE_GP2 = Montage(
    name="gp2",
    pairs=[
        ("Fp1", "T3",  "Fp1-T3"),
        ("Fp1", "C3",  "Fp1-C3"),
        ("T3",  "O1",  "T3-O1"),
        ("Fp2", "T4",  "Fp2-T4"),
        ("Fp2", "C4",  "Fp2-C4"),
        ("T4",  "O2",  "T4-O2"),
        ("Fp1", "A1",  "Fp1-A1"),
        ("Fp2", "A1",  "Fp2-A1"),
        ("T3",  "A1",  "T3-A1"),
        ("C3",  "A1",  "C3-A1"),
        ("T4",  "A1",  "T4-A1"),
        ("C4",  "A1",  "C4-A1"),
    ],
    keep_raw=["Menton", "JAMBG", "JAMBD", "RONF", "EMG1", "EMG2", "ECG", "EOGD", "EOGG"],
)


def _check_required_channels(raw: mne.io.BaseRaw, base_name: str) -> None:
    """Vérifie que les canaux nécessaires au montage GP2 sont présents."""
    missing = set()
    for anode, cathode, _ in MONTAGE_GP2.pairs:
        if anode not in raw.ch_names:
            missing.add(anode)
        if cathode not in raw.ch_names:
            missing.add(cathode)
    if missing:
        missing_str = ",".join(sorted(missing))
        raise RuntimeError(f"MISSING_CHANNELS:{base_name}:{missing_str}")


def _apply_montage_gp2(raw: mne.io.BaseRaw, base_name: str) -> mne.io.BaseRaw:
    """Calcule les dérivations bipolaires et conserve les canaux auxiliaires."""
    sf = raw.info["sfreq"]
    raw_in = raw.copy()

    data_bip, ch_names_bip, ch_types_bip = [], [], []

    # Dérivations bipolaires : anode - cathode
    for anode, cathode, new_name in MONTAGE_GP2.pairs:
        if anode not in raw_in.ch_names or cathode not in raw_in.ch_names:
            print(f"[{base_name}] Skip {new_name} : canal manquant ({anode} ou {cathode})")
            continue
        try:
            a_idx = raw_in.ch_names.index(anode)
            c_idx = raw_in.ch_names.index(cathode)
            bip = raw_in.get_data(picks=[a_idx])[0] - raw_in.get_data(picks=[c_idx])[0]
        except Exception as e:
            print(f"[{base_name}] Skip {new_name} : {e}")
            continue
        data_bip.append(bip)
        ch_names_bip.append(new_name)
        ch_types_bip.append("eeg")

    # Canaux auxiliaires (EMG, EOG, ECG…) conservés tels quels
    for ch in MONTAGE_GP2.keep_raw:
        if ch not in raw_in.ch_names:
            continue
        idx = raw_in.ch_names.index(ch)
        data_bip.append(raw_in.get_data(picks=[idx])[0])
        ch_names_bip.append(ch)
        ch_types_bip.append(raw_in.get_channel_types(picks=[idx])[0])

    if not data_bip:
        return raw_in

    info_new = mne.create_info(ch_names=ch_names_bip, sfreq=sf,
                                ch_types=ch_types_bip, verbose=False)
    raw_bip = mne.io.RawArray(np.vstack(data_bip), info_new, verbose=False)

    # Restauration des annotations existantes
    if raw_in.annotations is not None and len(raw_in.annotations) > 0:
        ann = raw_in.annotations
        raw_bip.set_annotations(mne.Annotations(
            onset=ann.onset.tolist(),
            duration=ann.duration.tolist(),
            description=ann.description.tolist(),
            orig_time=None,
        ))
    return raw_bip


# ============================================================================
# FILTRES
# ============================================================================

def _filter_eeg(raw: mne.io.BaseRaw, l_freq: float, h_freq: float, notch: Optional[float]) -> None:
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(picks):
        raw.filter(l_freq=l_freq, h_freq=h_freq, picks=picks, verbose=False)
        if notch:
            raw.notch_filter(freqs=[notch], picks=picks, verbose=False)


def _filter_emg(raw: mne.io.BaseRaw, hp: float, lp: float, notch: Optional[float]) -> None:
    picks = mne.pick_types(raw.info, emg=True, exclude=[])
    if len(picks):
        raw.filter(l_freq=hp, h_freq=lp, picks=picks, verbose=False)
        if notch:
            raw.notch_filter(freqs=[notch], picks=picks, verbose=False)


# ============================================================================
# HYPNOGRAMME
# ============================================================================

YASA_CODE = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
EXP_NUM_TO_YASA = {1: 0, 2: 4, 3: 1, 4: 2, 5: 3}


def _infer_epoch_len(seconds_col: List[float]) -> float:
    if len(seconds_col) < 2:
        return 30.0
    diffs = np.diff(seconds_col)
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if len(diffs) else 30.0


def _read_hypno_txt(path: Path) -> Tuple[np.ndarray, float]:
    """Lit un fichier hypnogramme .txt au format EXP."""
    seconds, labels, codes_num = [], [], []
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
            labels.append(parts[2].upper())
            try:
                codes_num.append(int(parts[3]) if len(parts) >= 4 else None)
            except Exception:
                codes_num.append(None)

    if not seconds:
        raise ValueError(f"Hypnogramme vide ou non lisible : {path}")

    epoch_len = _infer_epoch_len(seconds)
    out = []
    for lab, cnum in zip(labels, codes_num):
        if lab in YASA_CODE:
            out.append(YASA_CODE[lab])
        elif cnum is not None and cnum in EXP_NUM_TO_YASA:
            out.append(EXP_NUM_TO_YASA[cnum])
        else:
            out.append(-2)
    return np.asarray(out, dtype=int), epoch_len


def _read_hypno_csv(path: Path) -> Tuple[np.ndarray, float]:
    """Lit un fichier hypnogramme .csv générique."""
    import pandas as pd

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    epoch_len = 30.0
    if "seconds" in cols:
        sec_col = df[cols["seconds"]].values
        if len(sec_col) >= 2:
            epoch_len = _infer_epoch_len(list(sec_col))

    labels = None
    if "stage" in cols:
        labels = df[cols["stage"]].astype(str).str.upper().tolist()
    elif "label" in cols:
        labels = df[cols["label"]].astype(str).str.upper().tolist()

    codes = None
    if "code" in cols:
        try:
            codes = df[cols["code"]].astype(float).astype(int).tolist()
        except Exception:
            pass

    out = []
    if labels is not None:
        for i, lab in enumerate(labels):
            if lab in YASA_CODE:
                out.append(YASA_CODE[lab])
            else:
                cnum = None if codes is None else codes[i]
                out.append(EXP_NUM_TO_YASA.get(cnum, -2) if cnum is not None else -2)
    elif codes is not None:
        out = [EXP_NUM_TO_YASA.get(c, -2) for c in codes]
    else:
        raise ValueError(f"Colonnes de stade introuvables dans {path}.")

    return np.asarray(out, dtype=int), float(epoch_len)


def _find_hypno_file(base: str, hypno_root: Path) -> Optional[Path]:
    """Cherche le fichier hypnogramme correspondant au patient."""
    if hypno_root is None or not hypno_root.exists():
        return None

    patterns = [
        f"{base}*hypno*.txt", f"{base}*hypno*.csv",
        f"{base}_hypnoEXP.txt", f"{base}_hypnoEXP.csv",
        f"{base}*hypnogram*.txt", f"{base}*hypnogram*.csv",
    ]
    candidates: List[Path] = []

    for search_dir in [hypno_root / base, hypno_root]:
        if search_dir.exists():
            for pat in patterns:
                candidates.extend(search_dir.glob(pat))

    for child in hypno_root.glob(f"{base}*"):
        if child.is_dir():
            for pat in patterns:
                candidates.extend(child.glob(pat))

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def _load_hypnogram(base_name: str, hypno_root: Optional[Path]) -> Tuple[Optional[np.ndarray], Optional[float]]:
    if hypno_root is None:
        return None, None
    hp = _find_hypno_file(base_name, Path(hypno_root))
    if hp is None:
        return None, None
    try:
        if hp.suffix.lower() == ".txt":
            return _read_hypno_txt(hp)
        else:
            return _read_hypno_csv(hp)
    except Exception:
        return None, None


def _upsample_hypno(hypno_epochs: np.ndarray, epoch_len_sec: float,
                    n_samples: int, sf: float) -> np.ndarray:
    import yasa
    hypno_up = yasa.hypno_upsample_to_data(
        hypno=hypno_epochs,
        sf_hypno=1.0 / float(epoch_len_sec),
        data=np.zeros((1, n_samples), dtype=float),
        sf_data=sf,
    )
    return hypno_up.astype(int)


def _hypno_to_rem_intervals(hypno_epochs: np.ndarray, epoch_len_sec: float) -> List[Tuple[float, float]]:
    """Convertit les codes YASA en liste d'intervalles REM (start_s, end_s)."""
    intervals, current_start = [], None
    for i, code in enumerate(hypno_epochs):
        if code == 4:  # REM
            if current_start is None:
                current_start = i * epoch_len_sec
        else:
            if current_start is not None:
                end_t = i * epoch_len_sec
                if end_t > current_start:
                    intervals.append((current_start, end_t))
                current_start = None
    if current_start is not None:
        end_t = len(hypno_epochs) * epoch_len_sec
        if end_t > current_start:
            intervals.append((current_start, end_t))
    return intervals


def _add_rem_annotations(raw: mne.io.BaseRaw, hypno_epochs: np.ndarray, epoch_len_sec: float) -> None:
    """Ajoute les annotations REM dans l'objet Raw."""
    rem_intervals = _hypno_to_rem_intervals(hypno_epochs, epoch_len_sec)
    if not rem_intervals:
        return
    ann = mne.Annotations(
        onset=[s for s, _ in rem_intervals],
        duration=[e - s for s, e in rem_intervals],
        description=["REM"] * len(rem_intervals),
        orig_time=None,
    )
    existing = raw.annotations
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


# ============================================================================
# DÉTECTION D'ARTEFACTS YASA
# ============================================================================

def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
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


def _detect_artifacts(raw: mne.io.BaseRaw,
                      win_sec: float,
                      method: str,
                      threshold: float,
                      hypno_samples: Optional[np.ndarray],
                      include_stages: str) -> List[Tuple[float, float]]:
    """Détecte les artefacts EEG avec YASA et retourne les intervalles artefactés."""
    import yasa

    picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, emg=False, misc=False)
    raw_eeg = raw.copy().pick(picks) if len(picks) > 0 else raw.copy()
    sf = float(raw_eeg.info["sfreq"])
    data = raw_eeg.get_data() * 1e6  # µV
    n_samples = data.shape[1]

    include_map = {"sleep": (1, 2, 3, 4), "rem": (4,), "all": (0, 1, 2, 3, 4)}
    include = include_map.get(include_stages, (1, 2, 3, 4))

    hypno_vec = None
    if hypno_samples is not None:
        if len(hypno_samples) != n_samples:
            raise ValueError(f"Hypno upsamplé ({len(hypno_samples)}) ≠ n_samples ({n_samples})")
        hypno_vec = hypno_samples.astype(int)

    art, _ = yasa.art_detect(
        data=data, sf=sf, window=win_sec,
        hypno=hypno_vec, include=include,
        method=method, threshold=threshold, verbose=False,
    )

    t_end = n_samples / sf
    windows_s = []
    for i, is_art in enumerate(art):
        if is_art:
            s = i * win_sec
            e = min((i + 1) * win_sec, t_end)
            if e > s:
                windows_s.append((float(s), float(e)))

    return _merge_intervals(windows_s)


def _annotate_artifacts(raw: mne.io.BaseRaw, windows_s: List[Tuple[float, float]]) -> mne.io.BaseRaw:
    """Ajoute les annotations ARTEFACT dans l'objet Raw."""
    if not windows_s:
        return raw
    ann = mne.Annotations(
        onset=[s for s, _ in windows_s],
        duration=[e - s for s, e in windows_s],
        description=["ARTEFACT"] * len(windows_s),
        orig_time=None,
    )
    existing = raw.annotations
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


# ============================================================================
# PIPELINE COMPLET (un patient)
# ============================================================================

def _preprocess_record(raw: mne.io.BaseRaw, base_name: str) -> mne.io.BaseRaw:
    """Applique le pipeline complet de prétraitement à un enregistrement."""

    # 0) Normalisation des noms de canaux (supprime le préfixe "EEG ")
    rename_dict = {ch: ch.replace("EEG ", "") for ch in raw.ch_names if ch.startswith("EEG ")}
    if rename_dict:
        raw.rename_channels(rename_dict)

    # 0b) Vérification des canaux requis pour le montage GP2
    _check_required_channels(raw, base_name)

    # 1) Montage bipolaire GP2
    raw = _apply_montage_gp2(raw, base_name)

    # 2) Filtrage EEG et EMG
    _filter_eeg(raw, l_freq=EEG_L_FREQ, h_freq=EEG_H_FREQ, notch=EEG_NOTCH)
    _filter_emg(raw, hp=EMG_HP, lp=EMG_LP, notch=EMG_NOTCH)

    # 3) Chargement de l'hypnogramme
    hypno_root = Path(HYPNO_ROOT) if HYPNO_ROOT is not None else None
    hypno_epochs, epoch_len = _load_hypnogram(base_name, hypno_root)

    # 4) Annotation des segments REM
    if hypno_epochs is not None and epoch_len is not None:
        _add_rem_annotations(raw, hypno_epochs, epoch_len)

    # 5) Upsampling de l'hypnogramme au niveau échantillon (pour YASA)
    hypno_samples = None
    if hypno_epochs is not None and epoch_len is not None:
        hypno_samples = _upsample_hypno(hypno_epochs, epoch_len,
                                        raw.n_times, float(raw.info["sfreq"]))

    # 6) Détection et annotation des artefacts EEG
    try:
        art_windows = _detect_artifacts(
            raw,
            win_sec=YASA_WIN_SEC,
            method=YASA_METHOD,
            threshold=YASA_THRESHOLD,
            hypno_samples=hypno_samples,
            include_stages=YASA_INCLUDE,
        )
        raw = _annotate_artifacts(raw, art_windows)
    except Exception as e:
        import traceback
        print(f"[{base_name}] YASA erreur : {e}")
        traceback.print_exc()

    return raw


# ============================================================================
# TRAITEMENT D'UN FICHIER EDF (worker)
# ============================================================================

def _process_one_edf(edf_path_str: str) -> None:
    """Worker : prétraite un fichier EDF et sauvegarde le résultat en .fif."""
    log = _get_logger("preprocess")
    edf_path = Path(edf_path_str)
    raw_root = Path(RAW_ROOT)
    out_root = Path(PREPROCESSED_ROOT)

    # Identifiant patient (ex : "AN166_raw.edf" > "AN166")
    base_name = edf_path.stem.split("_")[0]
    log.info(f"=== Prétraitement {base_name} ({edf_path.name}) ===")

    # Lecture du fichier EDF
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Pipeline de prétraitement
    raw_prep = _preprocess_record(raw, base_name)

    # Construction du chemin de sortie (en conservant l'arborescence relative)
    rel = edf_path.relative_to(raw_root).with_suffix(".fif")
    out_path = out_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sauvegarde
    raw_prep.save(out_path, overwrite=True)
    log.info(f"[{base_name}] Sauvegardé > {out_path}")


# ============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================================

def main() -> None:
    log = _get_logger("preprocess")

    raw_root = Path(RAW_ROOT)

    # Recherche de tous les fichiers EDF
    edfs = [Path(p) for p in glob.glob(str(raw_root / "**" / "*.edf"), recursive=True)]
    edfs += [Path(p) for p in glob.glob(str(raw_root / "**" / "*.EDF"), recursive=True)]

    if not edfs:
        log.warning(f"Aucun fichier .edf trouvé sous {raw_root}")
        return

    log.info(f"{len(edfs)} fichiers EDF trouvés. Lancement avec {N_WORKERS} worker(s).")

    skipped_missing: List[Tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_process_one_edf, str(p)): p for p in edfs}

        for fut in as_completed(futures):
            p = futures[fut]
            try:
                fut.result()
            except Exception as e:
                msg = str(e)
                log.error(f"[ERREUR] {p.name} : {msg}")
                if msg.startswith("MISSING_CHANNELS:"):
                    try:
                        _, base_name, missing_str = msg.split(":", 2)
                        skipped_missing.append((base_name, missing_str))
                    except ValueError:
                        skipped_missing.append((p.stem.split("_")[0], "???"))

    # Résumé final
    if skipped_missing:
        log.warning("Patients ignorés (canaux manquants) :")
        for base_name, missing_str in skipped_missing:
            log.warning(f"  - {base_name} : {missing_str}")
    else:
        log.info("Prétraitement terminé sans erreur.")


if __name__ == "__main__":
    main()