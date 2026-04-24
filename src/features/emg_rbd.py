#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emg_rbd.py — Détection RSWA (PHASIC / TONIC) à partir de l’EMG durant le REM (par canal)

Objectif
--------
Ce script détecte des marqueurs de REM Sleep Without Atonia (RSWA) typiques du RBD,
en analysant principalement l’EMG (enveloppe par canal) au sein d’épisodes REM, et en
utilisant l’EOG comme critère “tonique” complémentaire.

Il produit un CSV unique contenant :
- des événements "PHASIC" (bursts) détectés dans les épisodes REM, par canal
- un résumé "REM_EPOCH_4S" par epoch REM de 4 secondes, par canal, incluant :
    - phasic_ratio, phasic_time_sec, phasic_count
    - tonic_eog (bool) : tonicité détectée via EOG sur 4 s
    - tonic_ratio (float) : ratio médiane(EMG_REM_clean) / médiane(EMG_NREM_clean)
    - rswa (bool) : règle heuristique RSWA au niveau epoch

Définitions et méthodologie (implémentation courante)
-----------------------------------------------------
1) Hypnogramme / stades
   - On harmonise les stades vers {REM, NREM, W}.
   - On regroupe les époques consécutives en épisodes (Episode).
   - On construit des paires NREM->REM : un épisode NREM “référence” juste avant un épisode REM.

2) Enveloppe EMG (par canal)
   - On charge les canaux EMG (détectés automatiquement ou fournis par l’utilisateur).
   - Filtre bande 30-100 Hz (FIR).
   - Enveloppe = lissage de |signal| par moyenne glissante 1 s.

3) PHASIC (par canal, dans un épisode REM)
   - Référence NREM : fenêtre de 30 s avant la fin de l’épisode NREM (avec marges).
   - Seuil = 95e percentile de l’enveloppe EMG sur la fenêtre NREM de référence.
   - Dans le REM : segments où enveloppe > seuil.
   - Un événement phasique est retenu si durée >= 3 s (paramètre interne actuellement).
   - Fusion d’événements si gap <= 1 s.

   !!! Remarque importante :
   - Un min_len=3 s est très strict pour du "phasique" (souvent plus court).
     C’est à valider / adapter selon vos critères (à discuter avant article).

4) TONIC via EOG (par epoch de 4 s)
   - On exige la présence d’EOG.
   - On masque les périodes phasiques (pour éviter que les bursts EMG/EOG perturbent l’évaluation).
   - On découpe l’epoch en 2 sous-fenêtres de 2 s :
        tonic=True si max(|EOG|) <= 25 µV sur chacune des 2 fenêtres.

   !!! Très important : unité EOG
   - MNE renvoie les signaux en Volts.
   - Pour µV, il faut multiplier par 1e6 (et non 1e9).
   - Dans ton code initial, tu avais *1e9 (nV). Ici on corrige pour cohérence avec 25 µV.

5) TONIC ratio EMG (par epoch, par canal)
   - On masque les échantillons phasiques dans l’epoch.
   - On prend la médiane sur les samples restants (REM_clean).
   - On prend la médiane d’une NREM_clean (NREM ref tronquée au 95e percentile).
   - tonic_ratio = median(REM_clean) / median(NREM_clean)

6) RSWA epoch (heuristique)
   - RSWA=True si (tonic_ratio > 1.3) OU (tonic_eog == True)
   - Ces seuils sont à documenter (et idéalement tester en sensibilité).

Entrées
-------
- --input_dir : dossier contenant un sous-dossier par patient (par défaut GP2_ROOT)
- Enregistrement patient : .fif/.edf/.bdf/.gdf (patterns prioritaires)
- Hypnogramme : recherché dans RAW_ROOT/pid/ (candidats) sinon fallback annotations raw

Sortie
------
- --output_csv : CSV consolidé pour tous les patients

Usage
-----
python scripts/emg_rbd.py \
    --input_dir /home/.../XAI/data \
    --output_csv data/processed/rbd/rbd_emg_events_and_summary_4s_per_channel.csv \
    --n_jobs 20 \
    --rem_epoch_len 4.0 \
    --min_phasic_dur 0.5

"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mne
import numpy as np
import pandas as pd


# =====================================================================
# Configuration des chemins et patterns de fichiers
# =====================================================================
GP2_ROOT = "/home/darryld/documents/EEG/preprocessed/XAI/data"
RAW_ROOT = "/home/darryld/documents/EEG/raw"

FIF_GLOB_PATTERNS = ["*_art_annotated.fif", "*.fif"]
HYPNO_CANDIDATES = ["{pid}_hypnoEXP.txt", "{pid}_hypnoEXP.csv", "{pid}_hypno.txt", "{pid}_hypnogram.txt"]

STAGE_MAP = {
    "W": "W", "V": "W", "WAKE": "W",
    "N1": "NREM", "S1": "NREM", "1": "NREM",
    "N2": "NREM", "S2": "NREM", "2": "NREM",
    "N3": "NREM", "S3": "NREM", "S4": "NREM", "N4": "NREM", "3": "NREM", "4": "NREM",
    "NREM": "NREM", "REM": "REM", "R": "REM", "SP": "REM",
}

IntervalSamp = Tuple[int, int]  # (start_sample, end_sample)


# =====================================================================
# Structures de données pour représenter le sleep scoring
# =====================================================================
@dataclass
class Episode:
    """Épisode consolidé d’un stade (REM/NREM/W) dans l’hypnogramme."""
    stage: str
    onset: float
    duration: float

    @property
    def end(self) -> float:
        """Fin de l’épisode (sec)."""
        return self.onset + self.duration


# =====================================================================
# Découverte des fichiers : patients, enregistrements, hypnogrammes
# =====================================================================
def find_patient_dirs(input_dir: str) -> List[str]:
    """Retourne les sous-dossiers patients présents dans input_dir (triés)."""
    subs = [p for p in glob.glob(os.path.join(input_dir, "*")) if os.path.isdir(p)]
    subs.sort()
    print(f"[SCAN] {len(subs)} dossiers patients trouvés dans: {input_dir}")
    return subs


def find_recording_file(patient_dir: str) -> Optional[str]:
    """Trouve un fichier d’enregistrement dans le dossier patient.

    Priorité :
    - patterns FIF_GLOB_PATTERNS (ex: *_art_annotated.fif)
    - puis extensions génériques (.fif/.edf/.bdf/.gdf)

    Args:
        patient_dir: Chemin du dossier patient.

    Returns:
        Chemin du fichier trouvé, ou None.
    """
    for pat in FIF_GLOB_PATTERNS:
        files = glob.glob(os.path.join(patient_dir, pat))
        if files:
            return files[0]

    for pat in ["*.fif", "*.edf", "*.bdf", "*.gdf"]:
        files = glob.glob(os.path.join(patient_dir, pat))
        if files:
            return files[0]

    return None


def find_hypnogram_file(patient_dir: str) -> Optional[str]:
    """Cherche un hypnogramme associé au patient.

    Stratégie :
    1) RAW_ROOT/pid/ + candidats HYPNO_CANDIDATES
    2) sinon, recherche *hypno* dans patient_dir
    3) sinon, fallback sur premier txt/csv/tsv trouvé

    Args:
        patient_dir: Dossier patient.

    Returns:
        Chemin du fichier hypnogramme ou None.
    """
    pid = os.path.basename(patient_dir.rstrip(os.sep))
    raw_pid_dir = os.path.join(RAW_ROOT, pid)

    for tmpl in HYPNO_CANDIDATES:
        cand = os.path.join(raw_pid_dir, tmpl.format(pid=pid))
        if os.path.isfile(cand):
            return cand

    for ext in ("*.csv", "*.txt", "*.tsv"):
        files = glob.glob(os.path.join(patient_dir, f"*hypno*{ext}")) + glob.glob(os.path.join(patient_dir, ext))
        files = sorted(files, key=lambda x: (0 if "hypno" in os.path.basename(x).lower() else 1, x))
        if files:
            return files[0]

    return None


# =====================================================================
# Hypnogram parsing: normalisation stades + agrégation en épisodes
# =====================================================================
def _norm_stage_to_rem_nrem_w(s: str) -> str:
    """Normalise un label de stade en 'REM', 'NREM' ou 'W'."""
    s = re.sub(r"[^A-Z0-9]+", "", str(s).upper())
    if s in ("N1", "N2", "N3", "S1", "S2", "S3", "S4", "N4", "1", "2", "3", "4"):
        return "NREM"
    if s in ("REM", "R", "SP"):
        return "REM"
    if s in ("W", "WAKE", "V"):
        return "W"
    return STAGE_MAP.get(s, s)


def _aggregate_epochs_to_episodes(rows: List[Tuple[float, float, str]]) -> List[Episode]:
    """Agrège des époques (start,end,stage) en épisodes continus.

    - Harmonise stage en REM/NREM/W
    - Trie par temps
    - Fusionne si même stage et contiguïté (tolérance <= 1 s)
    """
    if not rows:
        return []

    rows = [(float(s), float(e), _norm_stage_to_rem_nrem_w(stg)) for (s, e, stg) in rows]
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
            episodes.append(Episode(stage=cur_stage, onset=float(cur_s), duration=max(0.0, float(cur_e - cur_s))))
            cur_stage, cur_s, cur_e = stg, s, e

    if cur_stage is not None:
        episodes.append(Episode(stage=cur_stage, onset=float(cur_s), duration=max(0.0, float(cur_e - cur_s))))

    return episodes


def _read_txt_hypnoexp(path: str) -> List[Episode]:
    """Parse un hypnogramme *_hypnoEXP.txt (format end_sec + stage)."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        header=None,
        names=["end_sec", "hhmmss", "stage", "code"],
        usecols=[0, 1, 2, 3],
    )
    df = df.dropna(subset=["end_sec", "stage"]).copy()
    df["end_sec"] = pd.to_numeric(df["end_sec"], errors="coerce")
    df["stage"] = df["stage"].astype(str)
    df = df.dropna(subset=["end_sec"]).sort_values("end_sec")

    te = df["end_sec"].to_numpy(float)
    diffs = np.diff(te) if te.size > 1 else np.array([])
    epoch_len = float(pd.Series(np.round(diffs, 1)).mode().iloc[0]) if diffs.size else 30.0

    rows = []
    for end_sec, stg in zip(df["end_sec"], df["stage"]):
        rows.append((float(end_sec - epoch_len), float(end_sec), stg))

    return _aggregate_epochs_to_episodes(rows)


def _read_csv_3cols(path: str) -> List[Episode]:
    """Parse un CSV (epoch, abs_time, stage). Essayez d’estimer epoch_len via abs_time."""
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 3:
        return []

    header_like = str(df.iloc[0, 0]).strip().lower()
    if "position" in header_like and "epoch" in header_like:
        df = df.iloc[1:, :]

    df = df.iloc[:, :3].copy()
    df.columns = ["epoch", "abs_time", "stage"]
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["stage"] = df["stage"].astype(str)
    df = df.dropna(subset=["epoch", "stage"]).sort_values("epoch")

    def _time_to_sec(tstr: str):
        try:
            h, m, s = tstr.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
        except Exception:
            return np.nan

    times = df["abs_time"].astype(str).map(_time_to_sec).to_numpy()

    if np.isfinite(times).all():
        # unwrap si passage de minuit
        unwrapped, shift, last = [], 0.0, None
        for v in times:
            vv = v + shift
            if last is not None and vv < last - 1e-6:
                shift += 24 * 3600.0
                vv = v + shift
            unwrapped.append(vv)
            last = vv
        unwrapped = np.asarray(unwrapped, float)

        diffs = np.diff(unwrapped) if unwrapped.size > 1 else np.array([])
        est_epoch = float(pd.Series(np.round(diffs, 1)).mode().iloc[0]) if diffs.size else 30.0
        epoch_len = est_epoch if 5.0 <= est_epoch <= 60.0 else 30.0
    else:
        epoch_len = 30.0

    rows = []
    for ep, stg in zip(df["epoch"].to_numpy(dtype=int), df["stage"].tolist()):
        end_sec = ep * epoch_len
        rows.append((float(end_sec - epoch_len), float(end_sec), stg))

    return _aggregate_epochs_to_episodes(rows)


def parse_hypnogram_table(path: str) -> List[Episode]:
    """Parse un hypnogramme depuis un fichier (txt/csv/tsv) vers Episodes."""
    if path is None or not os.path.isfile(path):
        return []

    base = os.path.basename(path).lower()
    ext = os.path.splitext(base)[1]

    try:
        if base.endswith("_hypnoexp.txt") or ext == ".txt":
            return _read_txt_hypnoexp(path)

        if base.endswith("_hypnoexp.csv") or ext == ".csv":
            try:
                eps = _read_csv_3cols(path)
                if eps:
                    return eps
            except Exception:
                pass

        df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_table(path)
        cols = {c.lower(): c for c in df.columns}
        onset_col = cols.get("onset_sec") or cols.get("onset") or cols.get("start_sec") or cols.get("start")
        dur_col = cols.get("duration_sec") or cols.get("duration") or cols.get("dur_sec") or cols.get("dur")
        stage_col = cols.get("stage") or cols.get("label") or cols.get("stade") or cols.get("sleep_stage")

        if onset_col is None or dur_col is None or stage_col is None:
            return []

        rows = []
        for _, row in df.iterrows():
            stage_m = _norm_stage_to_rem_nrem_w(str(row[stage_col]))
            onset = float(row[onset_col])
            dur = float(row[dur_col])
            rows.append((onset, onset + dur, stage_m))

        return _aggregate_epochs_to_episodes(rows)

    except Exception:
        return []


def parse_hypnogram_from_raw(raw: mne.io.BaseRaw) -> List[Episode]:
    """Fallback : récupère REM/NREM/W à partir de raw.annotations si pas de fichier hypnogramme."""
    if raw.annotations is None or len(raw.annotations) == 0:
        return []

    rows = []
    for ann in raw.annotations:
        desc = str(ann["description"]).strip().upper()
        stage = None

        if "REM" in desc or desc == "R":
            stage = "REM"
        elif "N1" in desc or "S1" in desc:
            stage = "NREM"
        elif "N2" in desc or "S2" in desc:
            stage = "NREM"
        elif "N3" in desc or "S3" in desc or "S4" in desc or "N4" in desc:
            stage = "NREM"
        elif desc == "W" or "WAKE" in desc:
            stage = "W"

        if stage is None:
            continue

        onset = float(ann["onset"])
        dur = float(ann["duration"])
        if dur > 0:
            rows.append((onset, onset + dur, stage))

    return _aggregate_epochs_to_episodes(rows)


# =====================================================================
# Traitement signal : enveloppe EMG et EOG
# =====================================================================
def moving_average(x: np.ndarray, win_samps: int) -> np.ndarray:
    """Moyenne glissante centrée avec padding edge (même taille que x)."""
    if win_samps <= 1:
        return x.copy()
    kernel = np.ones(win_samps, dtype=float) / win_samps
    pad = win_samps // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    y = np.convolve(x_pad, kernel, mode="valid")
    return y[: len(x)]


def load_raw_with_emg(path: str, emg_channels: Optional[List[str]]):
    """Charge un raw et retourne les canaux EMG détectés.

    Détection :
    - si emg_channels fourni : intersection avec raw.ch_names
    - sinon : pick_types(emg=True) + fallback "emg" substring

    Raises:
        RuntimeError si aucun canal EMG.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".fif":
        raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    elif ext == ".edf":
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    elif ext == ".bdf":
        raw = mne.io.read_raw_bdf(path, preload=True, verbose=False)
    elif ext == ".gdf":
        raw = mne.io.read_raw_gdf(path, preload=True, verbose=False)
    else:
        raw = mne.io.read_raw(path, preload=True, verbose=False)

    if emg_channels:
        picks = [ch for ch in emg_channels if ch in raw.ch_names]
    else:
        idx_type = mne.pick_types(raw.info, emg=True, eeg=False, meg=False, eog=False, stim=False, misc=False)
        picks = [raw.ch_names[i] for i in (idx_type if isinstance(idx_type, np.ndarray) else [])]
        picks += [ch for ch in raw.ch_names if "emg" in ch.lower() and ch not in picks]

    if not picks:
        raise RuntimeError(f"No EMG channels found in {path}. Provide --emg_channels or set channel types to 'emg'.")

    return raw, picks


def build_emg_envelopes(raw: mne.io.BaseRaw, emg_chs: List[str]):
    """Construit l’enveloppe EMG par canal (30-100 Hz, |.| + MA 1s)."""
    emg = raw.copy().pick_channels(emg_chs)
    emg.filter(l_freq=30.0, h_freq=100.0, picks="all", method="fir", verbose=False)

    data = emg.get_data()  # volts
    sfreq = float(emg.info["sfreq"])

    win = max(1, int(round(sfreq * 1.0)))  # 1 seconde
    envs = np.empty_like(data)
    for i in range(data.shape[0]):
        envs[i, :] = moving_average(np.abs(data[i, :]), win)

    return envs, emg.ch_names, sfreq


def load_eog_signal(raw: mne.io.BaseRaw):
    """Charge EOG et retourne data en nV + sfreq.
    """
    # Force type eog si noms contiennent EOGD/EOGG
    eog_chs = [ch for ch in raw.ch_names if ("EOGD" in ch.upper()) or ("EOGG" in ch.upper())]
    if eog_chs:
        raw.set_channel_types({ch: "eog" for ch in eog_chs})

    picks = mne.pick_types(raw.info, eog=True, eeg=False, emg=False, meg=False, stim=False, misc=False)
    if len(picks) == 0:
        return None, None

    eog = raw.copy().pick(picks)
    data_nv = eog.get_data() * 1e9  # nV
    sfreq = float(eog.info["sfreq"])
    return data_nv, sfreq


# =====================================================================
# Détection tonic EOG sur epoch (4s) après masquage phasic
# =====================================================================
def detect_tonic_eog_epoch(
    eog: np.ndarray,
    sfreq: float,
    w0: int,
    w1: int,
    phasic_in_win: List[IntervalSamp],
    amp_thresh: float = 25000.0,
):
    """Retourne True/False si l’epoch est “tonic” au sens EOG, sinon np.nan.

    Critère :
    - On moyenne les canaux EOG
    - On masque les segments phasiques
    - On découpe 4s en 2 fenêtres 2s : max(|EOG|) <= 25 µV sur chacune
    """
    if eog is None:
        return np.nan

    sig = np.mean(eog[:, w0:w1], axis=0)

    mask = np.zeros(sig.shape[0], dtype=bool)
    for s, e in phasic_in_win:
        s_rel = max(0, s - w0)
        e_rel = min(sig.shape[0], e - w0)
        if s_rel < e_rel:
            mask[s_rel:e_rel] = True

    sig_clean = sig.copy()
    sig_clean[mask] = np.nan

    half = int(round(2.0 * sfreq))
    if sig_clean.size < 2 * half:
        return np.nan

    win1 = sig_clean[:half]
    win2 = sig_clean[half : 2 * half]

    if np.isnan(win1).all() or np.isnan(win2).all():
        return np.nan

    amp1 = np.nanmax(np.abs(win1))
    amp2 = np.nanmax(np.abs(win2))
    return bool((amp1 <= amp_thresh) and (amp2 <= amp_thresh))


# =====================================================================
# Détection phasic EMG par canal (seuil NREM 95e percentile)
# =====================================================================
def detect_phasic_events_ch(
    envelope: np.ndarray,
    sfreq: float,
    rem_start: int,
    rem_end: int,
    nrem_ref_start: int,
    nrem_ref_end: int,
) -> List[IntervalSamp]:
    """Détecte des événements phasiques en REM via seuil NREM (95e percentile).

    Méthode :
    - thr = percentile95(enveloppe sur fenêtre NREM de référence)
    - événement si enveloppe>thr pendant au moins 3 s (min_len=3s)
    - fusion si gap <= 1s

    !!! À discuter : min_len=3s est potentiellement trop élevé pour "phasique".
    """
    nrem_ref = envelope[nrem_ref_start:nrem_ref_end]
    if len(nrem_ref) == 0:
        return []

    thr = float(np.percentile(nrem_ref, 95.0))
    rem_env = envelope[rem_start:rem_end]
    above = rem_env > thr

    min_len = int(np.ceil(3.0 * sfreq))
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

    merged = [events[0]]
    max_gap = int(np.floor(1.0 * sfreq))
    for s, e in events[1:]:
        ps, pe = merged[-1]
        if (s - pe) <= max_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    return merged


def clip_events_to_window(events: List[IntervalSamp], w0: int, w1: int) -> List[IntervalSamp]:
    """Clippe des événements (samples) à la fenêtre [w0,w1]."""
    out: List[IntervalSamp] = []
    for s, e in events:
        ss, ee = max(s, w0), min(e, w1)
        if ee > ss:
            out.append((ss, ee))
    return out


def phasic_time_ratio(events_clipped: List[IntervalSamp], w0: int, w1: int, sfreq: float):
    """Temps phasique (s) et ratio phasique sur une fenêtre [w0,w1]."""
    dur = sum(e - s for s, e in events_clipped) / float(sfreq)
    win_dur = max(1e-9, (w1 - w0) / float(sfreq))
    return float(dur), float(dur / win_dur)


def compute_tonic_ratio_epoch_ch(
    envelope: np.ndarray,
    sfreq: float,
    w0: int,
    w1: int,
    phasic_in_win: List[IntervalSamp],
    nrem_ref_start: int,
    nrem_ref_end: int,
    phasic_ratio_epoch: float,
    *,
    nrem_clip_percentile: float = 95.0,
    min_nrem_seconds_for_clean: float = 0.5,
    tiny: float = 1e-12,
):
    """Calcule tonic_ratio = median(REM_clean)/median(NREM_clean) sur une epoch.

    - REM_clean = enveloppe sur [w0,w1] sans samples phasiques
    - NREM_clean = NREM_ref tronqué au percentile nrem_clip_percentile
    - tonic_excluded = True si epoch trop phasique (phasic_ratio > 0.75)

    Returns:
        (tonic_ratio, tonic_excluded)
    """
    rem_env = envelope[w0:w1].copy()
    if rem_env.size == 0:
        return (np.nan, False)

    mask = np.zeros(rem_env.shape[0], dtype=bool)
    for s, e in phasic_in_win:
        s_rel = max(0, s - w0)
        e_rel = min(rem_env.shape[0], e - w0)
        if s_rel < e_rel:
            mask[s_rel:e_rel] = True

    rem_clean = rem_env[~mask] if (~mask).any() else rem_env

    nrem_ref = envelope[nrem_ref_start:nrem_ref_end]
    if nrem_ref.size == 0:
        return (np.nan, phasic_ratio_epoch > 0.75)

    thr_ref = float(np.percentile(nrem_ref, float(nrem_clip_percentile)))
    nrem_clean = nrem_ref[nrem_ref <= thr_ref]

    if nrem_clean.size < int(round(float(min_nrem_seconds_for_clean) * float(sfreq))):
        nrem_clean = nrem_ref

    rem_med = float(np.median(rem_clean)) if rem_clean.size else np.nan
    nrem_med = float(np.median(nrem_clean)) if nrem_clean.size else np.nan

    if not np.isfinite(rem_med) or not np.isfinite(nrem_med) or nrem_med <= 0:
        return (np.nan, phasic_ratio_epoch > 0.75)

    tonic_ratio = rem_med / max(nrem_med, float(tiny))
    return (float(tonic_ratio), phasic_ratio_epoch > 0.75)


# =====================================================================
# Paires NREM->REM et découpage en epochs 4s
# =====================================================================
def collect_rem_nrem_pairs(episodes: List[Episode], min_len_sec: float = 10.0):
    """Construit des paires (NREM, REM) pour calibrer les seuils NREM.

    Pour chaque épisode REM assez long, on cherche l’épisode NREM immédiatement avant.
    """
    pairs = []
    episodes = sorted(episodes, key=lambda e: e.onset)

    for i, ep in enumerate(episodes):
        if ep.stage != "REM" or ep.duration < float(min_len_sec):
            continue

        prior = None
        for j in range(i - 1, -1, -1):
            if episodes[j].stage == "NREM" and episodes[j].duration >= float(min_len_sec):
                if abs(episodes[j].end - ep.onset) <= 2.0 or episodes[j].end <= ep.onset:
                    prior = episodes[j]
                    break
                else:
                    break

        if prior is not None:
            pairs.append((prior, ep))

    return pairs


def split_into_epochs(rem_start: int, rem_end: int, sfreq: float, epoch_len_s: float):
    """Découpe un intervalle REM en fenêtres non chevauchantes de epoch_len_s."""
    win = int(round(float(epoch_len_s) * float(sfreq)))
    win = max(win, 1)
    epochs = []
    s = rem_start
    while s + win <= rem_end:
        epochs.append((s, s + win))
        s += win
    return epochs


# =====================================================================
# Traitement patient
# =====================================================================
def process_patient(patient_dir: str, args) -> pd.DataFrame:
    """Pipeline patient complet : charge raw, hypnogramme, calcule phasic/tonic/rswa.

    Retourne un DataFrame (événements + résumés), ou une ligne ERROR/EMPTY.
    """
    patient_id = os.path.basename(patient_dir.rstrip(os.sep))

    rec = None
    raw = None
    eog_data = None

    try:
        rec = find_recording_file(patient_dir)
        if rec is None:
            raise RuntimeError("No recording file (edf/fif) found.")

        raw, emg_chs = load_raw_with_emg(rec, args.emg_channels)

        eog_data, eog_sfreq = load_eog_signal(raw)
        if eog_data is None or eog_sfreq is None:
            raise RuntimeError("EOG manquant - tonic EOG impossible.")

        # Assure indices cohérents (sinon w0/w1 ne correspond pas)
        if abs(float(eog_sfreq) - float(raw.info["sfreq"])) > 1e-6:
            raise RuntimeError(
                f"sfreq EOG ({eog_sfreq}) != sfreq raw/EMG ({raw.info['sfreq']}) - resampling requis."
            )

        envs, chs, sfreq = build_emg_envelopes(raw, emg_chs)
        n_samples = envs.shape[1]

        hyp_path = find_hypnogram_file(patient_dir)
        episodes = parse_hypnogram_table(hyp_path) if hyp_path else parse_hypnogram_from_raw(raw)
        if not episodes:
            raise RuntimeError("No hypnogram found or parsed.")

        pairs = collect_rem_nrem_pairs(episodes, min_len_sec=10.0)
        if not pairs:
            print(f"[{patient_id}] NO_PAIRS: episodes={len(episodes)} hyp={hyp_path}")
            return pd.DataFrame([{
                "patient_id": patient_id,
                "type": "EMPTY_NO_PAIRS",
                "episode_index": -1,
                "error": f"No NREM->REM pairs (episodes={len(episodes)} hyp={hyp_path})",
            }])

        rows = []

        for idx_pair, (nrem_ep, rem_ep) in enumerate(pairs):
            # Marges 2 s
            rem_start = int(np.round((rem_ep.onset + 2.0) * sfreq))
            rem_end = int(np.round((rem_ep.end - 2.0) * sfreq))

            # NREM ref = ~30s avant fin NREM (marges 2s)
            nrem_ref_end = int(np.round((nrem_ep.end - 2.0) * sfreq))
            nrem_ref_start = int(np.round(max(nrem_ep.end - 30.0, nrem_ep.onset + 2.0) * sfreq))

            # Bornage
            rem_start = max(0, min(rem_start, n_samples - 1))
            rem_end = max(rem_start + 1, min(rem_end, n_samples))

            nrem_ref_start = max(0, min(nrem_ref_start, n_samples - 1))
            nrem_ref_end = max(nrem_ref_start + 1, min(nrem_ref_end, n_samples))

            epochs = split_into_epochs(rem_start, rem_end, sfreq, args.rem_epoch_len)
            if not epochs:
                print(f"[{patient_id}] NO_EPOCHS: rem_dur={(rem_end-rem_start)/sfreq:.1f}s after margins")
                continue

            for ch_i, ch_name in enumerate(chs):
                env = envs[ch_i, :]

                # PHASIC (épisode)
                phasic_all = detect_phasic_events_ch(env, sfreq, rem_start, rem_end, nrem_ref_start, nrem_ref_end)

                # Enregistre événements phasiques “longs” (>= min_phasic_dur)
                for s_idx, e_idx in phasic_all:
                    dur_sec = (e_idx - s_idx) / sfreq
                    if dur_sec >= float(args.min_phasic_dur):
                        rows.append({
                            "patient_id": patient_id,
                            "channel": ch_name,
                            "type": "PHASIC",
                            "episode_index": idx_pair,
                            "start_sec": s_idx / sfreq,
                            "end_sec": e_idx / sfreq,
                            "duration_sec": dur_sec,
                        })

                # Résumé par epoch 4s
                for ep_k, (w0, w1) in enumerate(epochs):
                    phasic_in_win = clip_events_to_window(phasic_all, w0, w1)

                    tonic_eog_epoch = detect_tonic_eog_epoch(
                        eog=eog_data,
                        sfreq=eog_sfreq,
                        w0=w0,
                        w1=w1,
                        phasic_in_win=phasic_in_win,
                    )

                    dur_phasic, phasic_ratio_epoch = phasic_time_ratio(phasic_in_win, w0, w1, sfreq)
                    very_phasic_epoch = (phasic_ratio_epoch > 0.75)

                    tonic_ratio_epoch, tonic_excluded_epoch = compute_tonic_ratio_epoch_ch(
                        env, sfreq, w0, w1, phasic_in_win,
                        nrem_ref_start, nrem_ref_end, phasic_ratio_epoch,
                    )

                    # RSWA heuristique
                    rswa_epoch = (
                        (np.isfinite(tonic_ratio_epoch) and (tonic_ratio_epoch > 1.3))
                        or (tonic_eog_epoch is True)
                    )

                    rows.append({
                        "patient_id": patient_id,
                        "channel": ch_name,
                        "type": "REM_EPOCH_4S",
                        "episode_index": idx_pair,
                        "epoch_index": ep_k,
                        "epoch_len_sec": float(args.rem_epoch_len),
                        "epoch_start_sec": w0 / sfreq,
                        "epoch_end_sec": w1 / sfreq,
                        "phasic_time_sec": float(dur_phasic),
                        "tonic_eog": tonic_eog_epoch,
                        "phasic_ratio": float(phasic_ratio_epoch),
                        "very_phasic": bool(very_phasic_epoch),
                        "tonic_ratio": tonic_ratio_epoch,
                        "tonic_excluded": bool(tonic_excluded_epoch),
                        "rswa": bool(rswa_epoch),
                        "phasic_count": int(len(phasic_in_win)),
                    })

        if len(rows) == 0:
            return pd.DataFrame([{
                "patient_id": patient_id,
                "type": "EMPTY",
                "episode_index": -1,
                "error": "Aucun résultat: pas de paires NREM->REM ou REM trop court ou aucune epoch retenue",
            }])

        return pd.DataFrame(rows)

    except Exception as e:
        return pd.DataFrame([{
            "patient_id": patient_id,
            "type": "ERROR",
            "episode_index": -1,
            "error": f"{e} (rec={rec}, raw_loaded={raw is not None}, eog_loaded={eog_data is not None})",
        }])


# =====================================================================
# Main
# =====================================================================
def main():
    """Parse CLI, checks I/O, exécute le batch patient, écrit les CSV."""
    parser = argparse.ArgumentParser(
        description="Detect phasic bursts & RSWA during REM on 4s epochs - per channel."
    )
    parser.add_argument("--input_dir", default=GP2_ROOT, help="Directory containing patient subfolders.")
    parser.add_argument(
        "--output_csv",
        default="data/processed/rbd/rbd_emg_events_and_summary_4s_per_channel.csv",
        help="Path to write the CSV (events + per-epoch summaries, per channel).",
    )
    parser.add_argument(
        "--emg_channels",
        nargs="*",
        default=None,
        help="Optional list of EMG channel names to use.",
    )
    parser.add_argument("--n_jobs", type=int, default=20, help="Number of parallel workers.")
    parser.add_argument("--rem_epoch_len", type=float, default=4.0, help="Longueur des époques REM (s).")
    parser.add_argument("--min_phasic_dur", type=float, default=0.5, help="Durée minimale PHASIC retenue (s).")
    args = parser.parse_args()

    # --- Checks ---
    if not os.path.isdir(args.input_dir):
        print(f"[ERREUR] Le dossier input_dir n'existe pas : {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(RAW_ROOT):
        print(f"[ERREUR] Le dossier RAW_ROOT (hypnogrammes) n'existe pas : {RAW_ROOT}", file=sys.stderr)
        sys.exit(1)

    patients = find_patient_dirs(args.input_dir)
    if not patients:
        print(f"[ERREUR] Aucun dossier patient trouvé dans : {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    valid_patients = []
    for p in patients:
        rec = find_recording_file(p)
        if rec is not None:
            valid_patients.append(p)

    if not valid_patients:
        print(f"[ERREUR] Aucun fichier .fif/.edf trouvé dans : {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    patients = valid_patients
    print(f"[CHECK] {len(patients)} patients valides trouvés.")

    # --- Run batch ---
    dfs = []
    if args.n_jobs and args.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            fut2p = {ex.submit(process_patient, p, args): p for p in patients}
            for fut in as_completed(fut2p):
                dfs.append(fut.result())
    else:
        for p in patients:
            dfs.append(process_patient(p, args))

    out_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"[OK] CSV écrit: {args.output_csv}")

    if "type" in out_df.columns and (out_df["type"] == "ERROR").any():
        err_path = os.path.splitext(args.output_csv)[0] + "_errors.csv"
        out_df[out_df["type"] == "ERROR"][["patient_id", "error"]].to_csv(err_path, index=False)
        print(f"[INFO] Des patients ont échoué. Détails: {err_path}")


if __name__ == "__main__":
    main()
