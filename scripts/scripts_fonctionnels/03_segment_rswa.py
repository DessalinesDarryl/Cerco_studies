from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
03_segment_rswa.py  -  Détection des marqueurs RSWA (phasique / tonique)
=============================================================================

Ce script est AUTONOME : il ne dépend d'aucun dossier src/, ni de fichier YAML.

Ce qu'il fait :
  Pour chaque patient (sous-dossier de PREPROCESSED_ROOT) :
    1. Lecture du fichier .fif/.edf
    2. Chargement de l'hypnogramme (fichier texte ou annotations Raw)
    3. Construction des paires NREM>REM
    4. Construction des enveloppes EMG (filtre 30-100 Hz + MA 1 s)
    5. Détection des événements PHASIQUES (seuil 95e percentile NREM)
    6. Détection de la tonicité EOG par epoch de 4 s
    7. Calcul du ratio tonique EMG (médiane REM / médiane NREM)
    8. Label RSWA par epoch (tonic_ratio > 1.3 OU tonic_eog == True)

Sorties :
  - OUTPUT_CSV : un CSV regroupant tous les patients avec :
      - lignes type "PHASIC"       : événements phasiques
      - lignes type "REM_EPOCH_4S" : résumé par epoch 4 s, par canal
      - lignes type "ERROR/EMPTY"  : traçabilité des erreurs

Comment utiliser ce script :
  1. Modifier les chemins dans la section "PARAMÈTRES" ci-dessous
  2. Lancer :  python scripts/scripts_fonctionnels/03_segment_rswa.py 2>&1 | tee logfiles/log_03_segment_rswa.txt

Dépendances requises :
  pip install mne numpy pandas

=============================================================================

INPUTS
------
PREPROCESSED_ROOT : sous-dossiers patients contenant les enregistrements
                    Formats acceptés : .fif  |  .edf  |  .bdf  |  .gdf
                    Exemple : data/preprocessed/AN166/AN166_raw.fif
 
HYPNO_ROOT        : hypnogrammes patients (identiques à 01_preprocess.py)
                    (None = fallback sur annotations raw)
 
OUTPUTS
-------
OUTPUT_CSV : CSV unique regroupant tous les patients
             Chemin    : results/emg/rbd_emg_events_and_summary_4s_per_channel.csv
             Colonnes principales :
               type           = "PHASIC" | "REM_EPOCH_4S" | "ERROR" | "EMPTY"
               patient_id     - identifiant patient
               channel        - canal EMG analysé
               phasic_ratio   - fraction de l'epoch occupée par activité phasique
               tonic_ratio    - médiane REM / médiane NREM (enveloppe EMG)
               tonic_eog      - bool : tonicité détectée via EOG
               rswa           - bool : RSWA epoch (tonic_ratio>1.3 OU tonic_eog)
 
En cas d'erreur patient > ligne type="ERROR" + colonne "error" explicative.
 
DÉPENDANCES PIPELINE
--------------------
Ce script est l'ÉTAPE 3 (branche EMG). Il est INDÉPENDANT de 02_segment_rem.py.
Sa sortie alimente :
  > 04_extract_features.py  (fusion EEG + EMG)
  > 05_build_dataset.py     (fusion finale)
  > B_rswa_emg.ipynb     (visualisation)
"""

# ============================================================================
# PARAMÈTRES  <<<  À MODIFIER SELON LA CONFIGURATION SOUHAITÉE
# ============================================================================

# Dossier contenant les sous-dossiers patients (chacun avec un .fif ou .edf)
PREPROCESSED_ROOT = r"c:\dev\raw\data_raw_fif\siwar"

# Dossier contenant les hypnogrammes ({patient}_hypnoEXP.txt ou .csv)
# Mettre None pour utiliser uniquement les annotations présentes dans le fichier .fif
HYPNO_ROOT = r"c:\dev\raw\hypnogrammes"

# Fichier CSV de sortie
OUTPUT_CSV = r"c:\dev\Cerco_studies\data\rbd_emg_events_and_summary_4s_per_channel.csv"

# Durée des époques REM en secondes
REM_EPOCH_LEN = 4.0

# Durée minimale (s) pour enregistrer un événement phasique
MIN_PHASIC_DUR = 0.5

# Canaux EMG à utiliser (liste de noms) - None pour détection automatique
EMG_CHANNELS = None  # Exemple : ["Menton", "JAMBG", "JAMBD"]

# Nombre de patients traités en parallèle
N_WORKERS = 4

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
from typing import List, Optional, Tuple

import mne
import numpy as np
import pandas as pd

# ============================================================================
# LOGGER
# ============================================================================

def _get_logger(name: str = "segment_rswa") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


# ============================================================================
# TYPES ET CONSTANTES
# ============================================================================

IntervalSamp = Tuple[int, int]  # (start_sample, end_sample)

STAGE_MAP = {
    "W": "W", "V": "W", "WAKE": "W",
    "N1": "NREM", "S1": "NREM", "1": "NREM",
    "N2": "NREM", "S2": "NREM", "2": "NREM",
    "N3": "NREM", "S3": "NREM", "S4": "NREM", "N4": "NREM", "3": "NREM", "4": "NREM",
    "NREM": "NREM", "REM": "REM", "R": "REM", "SP": "REM",
}

FIF_GLOB_PATTERNS = ["*_art_annotated.fif", "*.fif"]
HYPNO_CANDIDATES = ["{pid}_hypnoEXP.txt", "{pid}_hypnoEXP.csv",
                    "{pid}_hypno.txt", "{pid}_hypnogram.txt"]


# ============================================================================
# STRUCTURES DE DONNÉES
# ============================================================================

@dataclass
class Episode:
    """Épisode consolidé d'un stade (REM/NREM/W)."""
    stage: str
    onset: float
    duration: float

    @property
    def end(self) -> float:
        return self.onset + self.duration


# ============================================================================
# DÉCOUVERTE DES FICHIERS
# ============================================================================

def _find_patient_dirs(root: str) -> List[str]:
    subs = [p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)]
    subs.sort()
    return subs


def _find_recording_file(patient_dir: str) -> Optional[str]:
    for pat in FIF_GLOB_PATTERNS:
        files = glob.glob(os.path.join(patient_dir, pat))
        if files:
            return files[0]
    for pat in ["*.fif", "*.edf", "*.bdf", "*.gdf"]:
        files = glob.glob(os.path.join(patient_dir, pat))
        if files:
            return files[0]
    return None


def _find_hypnogram_file(patient_dir: str) -> Optional[str]:
    pid = os.path.basename(patient_dir.rstrip(os.sep))
    hypno_root = HYPNO_ROOT

    if hypno_root is not None:
        for search_dir in [os.path.join(hypno_root, pid), hypno_root]:
            for tmpl in HYPNO_CANDIDATES:
                cand = os.path.join(search_dir, tmpl.format(pid=pid))
                if os.path.isfile(cand):
                    return cand
            # motif générique si le nom exact ne correspond pas
            for ext in ("*.txt", "*.csv"):
                matches = glob.glob(os.path.join(search_dir, f"{pid}*hypno*{ext}"))
                if matches:
                    return matches[0]

    for ext in ("*.csv", "*.txt", "*.tsv"):
        files = (glob.glob(os.path.join(patient_dir, f"*hypno*{ext}"))
                 + glob.glob(os.path.join(patient_dir, ext)))
        files = sorted(files, key=lambda x: (0 if "hypno" in os.path.basename(x).lower() else 1, x))
        if files:
            return files[0]

    return None


# ============================================================================
# HYPNOGRAMME : PARSING ET CONSOLIDATION EN ÉPISODES
# ============================================================================

def _norm_stage(s: str) -> str:
    s = re.sub(r"[^A-Z0-9]+", "", str(s).upper())
    if s in ("N1", "N2", "N3", "S1", "S2", "S3", "S4", "N4", "1", "2", "3", "4"):
        return "NREM"
    if s in ("REM", "R", "SP"):
        return "REM"
    if s in ("W", "WAKE", "V"):
        return "W"
    return STAGE_MAP.get(s, s)


def _aggregate_to_episodes(rows: List[Tuple[float, float, str]]) -> List[Episode]:
    """Fusionne des époques (start,end,stage) en épisodes continus."""
    if not rows:
        return []

    rows = [(float(s), float(e), _norm_stage(stg)) for (s, e, stg) in rows]
    rows.sort(key=lambda x: (x[0], x[1]))

    episodes: List[Episode] = []
    cur_stage, cur_s, cur_e = None, None, None

    for s, e, stg in rows:
        if cur_stage is None:
            cur_stage, cur_s, cur_e = stg, s, e
            continue
        if stg == cur_stage and abs(s - cur_e) <= 1.0:
            cur_e = max(cur_e, e)
        else:
            episodes.append(Episode(stage=cur_stage, onset=float(cur_s),
                                    duration=max(0.0, float(cur_e - cur_s))))
            cur_stage, cur_s, cur_e = stg, s, e

    if cur_stage is not None:
        episodes.append(Episode(stage=cur_stage, onset=float(cur_s),
                                duration=max(0.0, float(cur_e - cur_s))))
    return episodes


def _read_hypno_txt(path: str) -> List[Episode]:
    """Lit un fichier *_hypnoEXP.txt (colonnes: end_sec, hhmmss, stage, code)."""
    df = pd.read_csv(path, sep=r"\s+", engine="python", header=None,
                     names=["end_sec", "hhmmss", "stage", "code"], usecols=[0, 1, 2, 3])
    df = df.dropna(subset=["end_sec", "stage"]).copy()
    df["end_sec"] = pd.to_numeric(df["end_sec"], errors="coerce")
    df = df.dropna(subset=["end_sec"]).sort_values("end_sec")

    te = df["end_sec"].to_numpy(float)
    diffs = np.diff(te) if te.size > 1 else np.array([])
    epoch_len = float(pd.Series(np.round(diffs, 1)).mode().iloc[0]) if diffs.size else 30.0

    rows = [(float(e - epoch_len), float(e), stg)
            for e, stg in zip(df["end_sec"], df["stage"])]
    return _aggregate_to_episodes(rows)


def _read_hypno_csv_3cols(path: str) -> List[Episode]:
    """Lit un CSV (epoch, abs_time, stage)."""
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 3:
        return []

    if "position" in str(df.iloc[0, 0]).lower() and "epoch" in str(df.iloc[0, 0]).lower():
        df = df.iloc[1:, :]

    df = df.iloc[:, :3].copy()
    df.columns = ["epoch", "abs_time", "stage"]
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch", "stage"]).sort_values("epoch")

    def _t2s(t):
        try:
            h, m, s = str(t).split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
        except Exception:
            return np.nan

    times = df["abs_time"].map(_t2s).to_numpy()

    if np.isfinite(times).all():
        unwrapped, shift, last = [], 0.0, None
        for v in times:
            vv = v + shift
            if last is not None and vv < last - 1e-6:
                shift += 86400.0
                vv = v + shift
            unwrapped.append(vv)
            last = vv
        unwrapped = np.asarray(unwrapped, float)
        diffs = np.diff(unwrapped) if unwrapped.size > 1 else np.array([])
        est = float(pd.Series(np.round(diffs, 1)).mode().iloc[0]) if diffs.size else 30.0
        epoch_len = est if 5.0 <= est <= 60.0 else 30.0
    else:
        epoch_len = 30.0

    rows = [(float(ep * epoch_len - epoch_len), float(ep * epoch_len), stg)
            for ep, stg in zip(df["epoch"].to_numpy(dtype=int), df["stage"].tolist())]
    return _aggregate_to_episodes(rows)


def _parse_hypnogram_file(path: str) -> List[Episode]:
    """Dispatcher : lit un fichier hypnogramme et retourne des épisodes."""
    if path is None or not os.path.isfile(path):
        return []

    base = os.path.basename(path).lower()
    ext = os.path.splitext(base)[1]

    try:
        if base.endswith("_hypnoexp.txt") or ext == ".txt":
            return _read_hypno_txt(path)

        if base.endswith("_hypnoexp.csv") or ext == ".csv":
            try:
                eps = _read_hypno_csv_3cols(path)
                if eps:
                    return eps
            except Exception:
                pass

        df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_table(path)
        cols = {c.lower(): c for c in df.columns}
        onset_col = cols.get("onset_sec") or cols.get("onset") or cols.get("start_sec") or cols.get("start")
        dur_col = cols.get("duration_sec") or cols.get("duration") or cols.get("dur_sec") or cols.get("dur")
        stage_col = cols.get("stage") or cols.get("label") or cols.get("stade") or cols.get("sleep_stage")

        if not all([onset_col, dur_col, stage_col]):
            return []

        rows = [(float(row[onset_col]), float(row[onset_col]) + float(row[dur_col]),
                 _norm_stage(str(row[stage_col])))
                for _, row in df.iterrows()]
        return _aggregate_to_episodes(rows)
    except Exception:
        return []


def _parse_hypnogram_from_raw(raw: mne.io.BaseRaw) -> List[Episode]:
    """Fallback : récupère les stades depuis raw.annotations."""
    if raw.annotations is None or len(raw.annotations) == 0:
        return []

    rows = []
    for ann in raw.annotations:
        desc = str(ann["description"]).strip().upper()
        stage = None
        if "REM" in desc or desc == "R":
            stage = "REM"
        elif any(x in desc for x in ["N1", "N2", "N3", "S1", "S2", "S3", "S4", "N4"]):
            stage = "NREM"
        elif desc == "W" or "WAKE" in desc:
            stage = "W"
        if stage and float(ann["duration"]) > 0:
            onset = float(ann["onset"])
            rows.append((onset, onset + float(ann["duration"]), stage))

    return _aggregate_to_episodes(rows)


# ============================================================================
# TRAITEMENT DU SIGNAL EMG / EOG
# ============================================================================

def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.copy()
    kernel = np.ones(win, dtype=float) / win
    pad = win // 2
    y = np.convolve(np.pad(x, (pad, pad), mode="edge"), kernel, mode="valid")
    return y[:len(x)]


def _load_raw(path: str) -> mne.io.BaseRaw:
    ext = os.path.splitext(path)[1].lower()
    readers = {".fif": mne.io.read_raw_fif, ".edf": mne.io.read_raw_edf,
               ".bdf": mne.io.read_raw_bdf, ".gdf": mne.io.read_raw_gdf}
    reader = readers.get(ext, mne.io.read_raw)
    return reader(path, preload=True, verbose=False)


KNOWN_EMG_NAMES = {"menton", "jambg", "jambd", "emg1", "emg2"}

def _load_raw_with_emg(path, emg_channels):
    raw = _load_raw(path)
    if emg_channels:
        picks = [ch for ch in emg_channels if ch in raw.ch_names]
    else:
        idx = mne.pick_types(raw.info, emg=True, eeg=False, eog=False, stim=False, misc=False)
        picks = [raw.ch_names[i] for i in idx]
        picks += [ch for ch in raw.ch_names
                  if ch not in picks and (
                      "emg" in ch.lower() or ch.lower() in KNOWN_EMG_NAMES
                  )]
    if not picks:
        raise RuntimeError(f"Aucun canal EMG trouvé dans {path}.")
    return raw, picks


def _build_emg_envelopes(raw: mne.io.BaseRaw, emg_chs: List[str]):
    emg = raw.copy().pick_channels(emg_chs)
    emg.filter(l_freq=30.0, h_freq=100.0, picks="all", method="fir", verbose=False)
    data = emg.get_data()
    sfreq = float(emg.info["sfreq"])
    win = max(1, int(round(sfreq * 1.0)))
    envs = np.stack([_moving_average(np.abs(data[i]), win) for i in range(data.shape[0])])
    return envs, emg.ch_names, sfreq


def _load_eog_signal(raw: mne.io.BaseRaw):
    eog_chs = [ch for ch in raw.ch_names if "EOGD" in ch.upper() or "EOGG" in ch.upper()]
    if eog_chs:
        raw.set_channel_types({ch: "eog" for ch in eog_chs})
    picks = mne.pick_types(raw.info, eog=True, eeg=False, emg=False, stim=False, misc=False)
    if len(picks) == 0:
        return None, None
    eog = raw.copy().pick(picks)
    # µV (cohérence avec seuil 25 µV dans detect_tonic_eog_epoch)
    return eog.get_data() * 1e6, float(eog.info["sfreq"])


# ============================================================================
# DÉTECTION PHASIQUE ET TONIQUE
# ============================================================================

def _detect_phasic_events(envelope: np.ndarray, sfreq: float,
                          rem_start: int, rem_end: int,
                          nrem_ref_start: int, nrem_ref_end: int) -> List[IntervalSamp]:
    """Détecte les bursts EMG phasiques (seuil = 95e percentile NREM)."""
    nrem_ref = envelope[nrem_ref_start:nrem_ref_end]
    if len(nrem_ref) == 0:
        return []

    thr = float(np.percentile(nrem_ref, 95.0))
    above = envelope[rem_start:rem_end] > thr
    min_len = int(np.ceil(3.0 * sfreq))  # 3 s minimum

    events: List[IntervalSamp] = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i + 1
            while j < len(above) and above[j]:
                j += 1
            if (j - i) >= min_len:
                events.append((rem_start + i, rem_start + j))
            i = j
        else:
            i += 1

    if not events:
        return events

    # Fusion des événements séparés par <= 1 s
    merged = [events[0]]
    max_gap = int(np.floor(1.0 * sfreq))
    for s, e in events[1:]:
        ps, pe = merged[-1]
        if (s - pe) <= max_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


def _clip_events(events: List[IntervalSamp], w0: int, w1: int) -> List[IntervalSamp]:
    return [(max(s, w0), min(e, w1)) for s, e in events if min(e, w1) > max(s, w0)]


def _phasic_time_ratio(events: List[IntervalSamp], w0: int, w1: int, sfreq: float):
    dur = sum(e - s for s, e in events) / float(sfreq)
    win_dur = max(1e-9, (w1 - w0) / float(sfreq))
    return float(dur), float(dur / win_dur)


def _detect_tonic_eog_epoch(eog: np.ndarray, sfreq: float, w0: int, w1: int,
                             phasic_in_win: List[IntervalSamp],
                             amp_thresh_uv: float = 25.0):
    """Détecte la tonicité EOG sur une epoch de 4 s (2 × 2 s)."""
    if eog is None:
        return np.nan

    sig = np.mean(eog[:, w0:w1], axis=0)
    mask = np.zeros(sig.shape[0], dtype=bool)
    for s, e in phasic_in_win:
        s_rel, e_rel = max(0, s - w0), min(sig.shape[0], e - w0)
        if s_rel < e_rel:
            mask[s_rel:e_rel] = True

    sig_clean = sig.copy()
    sig_clean[mask] = np.nan

    half = int(round(2.0 * sfreq))
    if sig_clean.size < 2 * half:
        return np.nan

    w1_, w2_ = sig_clean[:half], sig_clean[half:2 * half]
    if np.isnan(w1_).all() or np.isnan(w2_).all():
        return np.nan

    return bool(np.nanmax(np.abs(w1_)) <= amp_thresh_uv and
                np.nanmax(np.abs(w2_)) <= amp_thresh_uv)


def _compute_tonic_ratio(envelope: np.ndarray, sfreq: float, w0: int, w1: int,
                         phasic_in_win: List[IntervalSamp],
                         nrem_ref_start: int, nrem_ref_end: int,
                         phasic_ratio: float):
    """Calcule tonic_ratio = median(REM_clean) / median(NREM_clean)."""
    rem_env = envelope[w0:w1].copy()
    if rem_env.size == 0:
        return np.nan, False

    mask = np.zeros(rem_env.shape[0], dtype=bool)
    for s, e in phasic_in_win:
        s_rel, e_rel = max(0, s - w0), min(rem_env.shape[0], e - w0)
        if s_rel < e_rel:
            mask[s_rel:e_rel] = True

    rem_clean = rem_env[~mask] if (~mask).any() else rem_env
    nrem_ref = envelope[nrem_ref_start:nrem_ref_end]
    if nrem_ref.size == 0:
        return np.nan, phasic_ratio > 0.75

    thr_ref = float(np.percentile(nrem_ref, 95.0))
    nrem_clean = nrem_ref[nrem_ref <= thr_ref]
    if nrem_clean.size < int(round(0.5 * sfreq)):
        nrem_clean = nrem_ref

    rem_med = float(np.median(rem_clean)) if rem_clean.size else np.nan
    nrem_med = float(np.median(nrem_clean)) if nrem_clean.size else np.nan

    if not np.isfinite(rem_med) or not np.isfinite(nrem_med) or nrem_med <= 0:
        return np.nan, phasic_ratio > 0.75

    return float(rem_med / max(nrem_med, 1e-12)), phasic_ratio > 0.75


# ============================================================================
# PAIRES NREM>REM ET DÉCOUPAGE EN ÉPOQUES
# ============================================================================

def _collect_rem_nrem_pairs(episodes: List[Episode], min_len_sec: float = 10.0):
    pairs = []
    episodes = sorted(episodes, key=lambda e: e.onset)
    for i, ep in enumerate(episodes):
        if ep.stage != "REM" or ep.duration < min_len_sec:
            continue
        for j in range(i - 1, -1, -1):
            if (episodes[j].stage == "NREM" and episodes[j].duration >= min_len_sec
                    and (abs(episodes[j].end - ep.onset) <= 2.0 or episodes[j].end <= ep.onset)):
                pairs.append((episodes[j], ep))
                break
    return pairs


def _split_into_epochs(rem_start: int, rem_end: int, sfreq: float, epoch_len_s: float):
    win = max(1, int(round(epoch_len_s * sfreq)))
    epochs, s = [], rem_start
    while s + win <= rem_end:
        epochs.append((s, s + win))
        s += win
    return epochs


# ============================================================================
# TRAITEMENT D'UN PATIENT
# ============================================================================

def _process_patient(patient_dir: str) -> pd.DataFrame:
    patient_id = os.path.basename(patient_dir.rstrip(os.sep))
    rec = None
    raw = None
    eog_data = None

    try:
        rec = _find_recording_file(patient_dir)
        if rec is None:
            raise RuntimeError("Aucun fichier d'enregistrement trouvé.")

        raw, emg_chs = _load_raw_with_emg(rec, EMG_CHANNELS)

        eog_data, eog_sfreq = _load_eog_signal(raw)
        if eog_data is None:
            raise RuntimeError("EOG manquant.")

        if abs(float(eog_sfreq) - float(raw.info["sfreq"])) > 1e-6:
            raise RuntimeError(f"sfreq EOG ({eog_sfreq}) ≠ sfreq raw ({raw.info['sfreq']}).")

        envs, chs, sfreq = _build_emg_envelopes(raw, emg_chs)
        n_samples = envs.shape[1]

        hyp_path = _find_hypnogram_file(patient_dir)
        episodes = _parse_hypnogram_file(hyp_path) if hyp_path else _parse_hypnogram_from_raw(raw)
        if not episodes:
            raise RuntimeError("Hypnogramme introuvable ou vide.")

        pairs = _collect_rem_nrem_pairs(episodes)
        if not pairs:
            return pd.DataFrame([{"patient_id": patient_id, "type": "EMPTY_NO_PAIRS",
                                   "episode_index": -1,
                                   "error": f"Pas de paires NREM>REM ({len(episodes)} épisodes)"}])

        rows = []
        for idx_pair, (nrem_ep, rem_ep) in enumerate(pairs):
            rem_start = max(0, int(np.round((rem_ep.onset + 2.0) * sfreq)))
            rem_end = max(rem_start + 1, min(int(np.round((rem_ep.end - 2.0) * sfreq)), n_samples))
            nrem_ref_end = int(np.round((nrem_ep.end - 2.0) * sfreq))
            nrem_ref_start = max(0, int(np.round(max(nrem_ep.end - 30.0, nrem_ep.onset + 2.0) * sfreq)))
            nrem_ref_end = max(nrem_ref_start + 1, min(nrem_ref_end, n_samples))

            epoch_windows = _split_into_epochs(rem_start, rem_end, sfreq, REM_EPOCH_LEN)
            if not epoch_windows:
                continue

            for ch_i, ch_name in enumerate(chs):
                env = envs[ch_i, :]
                phasic_all = _detect_phasic_events(env, sfreq, rem_start, rem_end,
                                                   nrem_ref_start, nrem_ref_end)

                for s_idx, e_idx in phasic_all:
                    dur_sec = (e_idx - s_idx) / sfreq
                    if dur_sec >= MIN_PHASIC_DUR:
                        rows.append({"patient_id": patient_id, "channel": ch_name,
                                     "type": "PHASIC", "episode_index": idx_pair,
                                     "start_sec": s_idx / sfreq, "end_sec": e_idx / sfreq,
                                     "duration_sec": dur_sec})

                for ep_k, (w0, w1) in enumerate(epoch_windows):
                    phasic_in_win = _clip_events(phasic_all, w0, w1)
                    tonic_eog = _detect_tonic_eog_epoch(eog_data, sfreq, w0, w1, phasic_in_win)
                    dur_phasic, phasic_ratio = _phasic_time_ratio(phasic_in_win, w0, w1, sfreq)
                    tonic_ratio, tonic_excluded = _compute_tonic_ratio(
                        env, sfreq, w0, w1, phasic_in_win, nrem_ref_start, nrem_ref_end, phasic_ratio)

                    rswa = ((np.isfinite(tonic_ratio) and tonic_ratio > 1.3) or tonic_eog is True)

                    rows.append({"patient_id": patient_id, "channel": ch_name,
                                 "type": "REM_EPOCH_4S", "episode_index": idx_pair,
                                 "epoch_index": ep_k, "epoch_len_sec": REM_EPOCH_LEN,
                                 "epoch_start_sec": w0 / sfreq, "epoch_end_sec": w1 / sfreq,
                                 "phasic_time_sec": dur_phasic, "tonic_eog": tonic_eog,
                                 "phasic_ratio": phasic_ratio, "very_phasic": phasic_ratio > 0.75,
                                 "tonic_ratio": tonic_ratio, "tonic_excluded": tonic_excluded,
                                 "rswa": bool(rswa), "phasic_count": len(phasic_in_win)})

        if not rows:
            return pd.DataFrame([{"patient_id": patient_id, "type": "EMPTY",
                                   "episode_index": -1, "error": "Aucune epoch retenue."}])
        return pd.DataFrame(rows)

    except Exception as e:
        return pd.DataFrame([{"patient_id": patient_id, "type": "ERROR",
                               "episode_index": -1,
                               "error": f"{e} (rec={rec}, raw={raw is not None}, eog={eog_data is not None})"}])


# ============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================================

def main() -> None:
    log = _get_logger()

    if not os.path.isdir(PREPROCESSED_ROOT):
        log.error(f"PREPROCESSED_ROOT introuvable : {PREPROCESSED_ROOT}")
        sys.exit(1)

    patients = _find_patient_dirs(PREPROCESSED_ROOT)
    valid = [p for p in patients if _find_recording_file(p) is not None]

    if not valid:
        log.error(f"Aucun fichier .fif/.edf trouvé dans {PREPROCESSED_ROOT}")
        sys.exit(1)

    log.info(f"{len(valid)} patients valides. Lancement avec {N_WORKERS} worker(s).")

    dfs = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_process_patient, p): p for p in valid}
        for fut in as_completed(futures):
            dfs.append(fut.result())

    out_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"CSV écrit : {OUTPUT_CSV}")

    if "type" in out_df.columns and (out_df["type"] == "ERROR").any():
        err_path = os.path.splitext(OUTPUT_CSV)[0] + "_errors.csv"
        out_df[out_df["type"] == "ERROR"][["patient_id", "error"]].to_csv(err_path, index=False)
        log.warning(f"Erreurs détectées > {err_path}")


if __name__ == "__main__":
    main()
