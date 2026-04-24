#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
segment_rswa.py — Détection RSWA (phasic/tonic) par canal sur epochs REM de 4 s

Résumé
-------
Ce script détecte des marqueurs de REM Sleep Without Atonia (RSWA) chez des patients, en
analysant l’EMG (enveloppe) et l’EOG durant le REM.

Il produit :
1) des événements **PHASIC** (bursts EMG) détectés dans des épisodes REM
2) un résumé **par epoch REM de 4 secondes** (par canal), incluant :
   - temps phasique / ratio phasique
   - indicateur tonic (via EOG)
   - ratio tonique EMG (médiane REM / médiane NREM)
   - label RSWA au niveau epoch (règle heuristique)

Contexte et définitions (dans ce script)
---------------------------------------
- Les stades sont harmonisés en {REM, NREM, W}.
- On construit des paires NREM->REM (référence NREM “proche” avant un épisode REM).
- PHASIC (par canal) :
    - enveloppe EMG filtrée 30-100 Hz puis amplitude lissée (MA 1 s)
    - seuil = 95e percentile de l’enveloppe EMG sur une fenêtre NREM de référence
    - événement phasique si > seuil pendant au moins 3 s (paramétrable dans la fonction)
    - fusion d’événements séparés par <= 1 s
- TONIC (par epoch, via EOG) :
    - on considère l’EOG (supposé déjà filtré ou exploitable)
    - on masque les périodes phasiques pour éviter de “polluer” la tonicité
    - critère tonic = sur 2 fenêtres consécutives de 2 s : max(|EOG|) <= 25 µV
- TONIC ratio (par canal et epoch) :
    - on masque la partie phasique dans l’epoch
    - on compare la médiane de l’enveloppe EMG en REM vs médiane NREM (référence)
    - tonic_ratio = median(REM_clean) / median(NREM_clean)
- RSWA epoch :
    - RSWA=True si (tonic_ratio > 1.3) OU tonic_eog_epoch == True
    - ces seuils sont des heuristiques : à valider / calibrer sur ta cohorte.

Entrées
-------
- Un dossier de patients (par défaut GP2_ROOT), contenant pour chaque patient un enregistrement
  (.fif / .edf / .bdf / .gdf). Le script privilégie certains patterns de fichiers.
- Un hypnogramme par patient (dans RAW_ROOT/patient_id/), sinon fallback sur les annotations Raw.
  Formats pris en charge :
    - *_hypnoEXP.txt (format spécifique)
    - *_hypnoEXP.csv ou csv “3 colonnes”
    - csv/tsv/table avec colonnes onset/duration/stage (flexible)
    - sinon, annotations MNE dans raw.annotations

Sortie
------
- Un CSV unique (args.output_csv) regroupant tous les patients :
    - lignes de type "PHASIC" : événements phasiques (start/end/durée)
    - lignes de type "REM_EPOCH_4S" : résumé par epoch REM de 4 s, par canal
    - lignes "ERROR" / "EMPTY" : traçabilité des cas problématiques

Usage
-----
    python scripts/detect_rswa_markers_emg_eog.py \
        --input_dir /path/to/XAI/data \
        --output_csv data/processed/rbd/rbd_emg_events_and_summary_4s_per_channel.csv \
        --n_jobs 20 \
        --rem_epoch_len 4.0 \
        --min_phasic_dur 0.5

Remarques importantes
--------------------
- Ce script suppose que l’EOG et l’EMG ont la même fréquence d’échantillonnage.
  Sinon, il faut resampler (refus explicite).
- Les règles (95e percentile NREM, durées min, seuils tonic_ratio/EOG) doivent être documentées
  dans ton papier et, idéalement, testées en sensibilité (ablation).
- Plusieurs sections “à revoir” sont commentées, notamment la corrélation EOG-EMG.

"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Initialisation du chemin projet pour permettre les imports internes.
# ---------------------------------------------------------------------
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
# Constantes et paramètres de chemins
# =====================================================================
GP2_ROOT = "/home/darryld/documents/EEG/preprocessed/XAI/data"
RAW_ROOT = "/home/darryld/documents/EEG/raw"

FIF_GLOB_PATTERNS = ["*_art_annotated.fif", "*.fif"]
HYPNO_CANDIDATES = ["{pid}_hypnoEXP.txt", "{pid}_hypnoEXP.csv", "{pid}_hypno.txt", "{pid}_hypnogram.txt"]

# Harmonisation des libellés de stades
STAGE_MAP = {
    "W": "W", "V": "W", "WAKE": "W",
    "N1": "NREM", "S1": "NREM", "1": "NREM",
    "N2": "NREM", "S2": "NREM", "2": "NREM",
    "N3": "NREM", "S3": "NREM", "S4": "NREM", "N4": "NREM", "3": "NREM", "4": "NREM",
    "NREM": "NREM", "REM": "REM", "R": "REM", "SP": "REM",
}

# Types utilitaires
IntervalSamp = Tuple[int, int]   # (start_sample, end_sample)
IntervalSec = Tuple[float, float]  # (start_sec, end_sec)


# =====================================================================
# Structure de données pour les épisodes de sommeil
# =====================================================================
@dataclass
class Episode:
    """Représente un épisode de sommeil consolidé (stade, onset, durée)."""
    stage: str
    onset: float
    duration: float

    @property
    def end(self) -> float:
        """Temps de fin en secondes."""
        return self.onset + self.duration


# =====================================================================
# Découverte des patients, enregistrements et hypnogrammes
# =====================================================================
def find_patient_dirs(input_dir: str) -> List[str]:
    """Liste les sous-dossiers patients dans un dossier racine (triés)."""
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
        patient_dir: Dossier patient.

    Returns:
        Chemin vers le premier fichier trouvé, ou None.
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
    """Cherche un hypnogramme (priorité RAW_ROOT/pid/ + candidats).

    Stratégie :
    1) Chercher dans RAW_ROOT/pid/ avec des patterns connus (HYPNO_CANDIDATES)
    2) Sinon, chercher un fichier *hypno* dans le dossier patient
    3) Sinon, fallback vers n'importe quel csv/txt/tsv du dossier patient

    Args:
        patient_dir: Dossier patient.

    Returns:
        Chemin du fichier hypnogramme ou None.
    """
    pid = os.path.basename(patient_dir.rstrip(os.sep))
    raw_pid_dir = os.path.join(RAW_ROOT, pid)

    # 1) Candidats standard
    for tmpl in HYPNO_CANDIDATES:
        cand = os.path.join(raw_pid_dir, tmpl.format(pid=pid))
        if os.path.isfile(cand):
            return cand

    # 2) Recherche “heuristique”
    for ext in ("*.csv", "*.txt", "*.tsv"):
        files = glob.glob(os.path.join(patient_dir, f"*hypno*{ext}")) + glob.glob(os.path.join(patient_dir, ext))
        files = sorted(files, key=lambda x: (0 if "hypno" in os.path.basename(x).lower() else 1, x))
        if files:
            return files[0]

    return None


# =====================================================================
# Hypnogram parsing: harmonisation + consolidation
# =====================================================================
def _norm_stage_to_rem_nrem_w(s: str) -> str:
    """Normalise un label de stade en {REM, NREM, W}.

    Args:
        s: Label brut (N2, S3, REM, W, etc.)

    Returns:
        'REM', 'NREM', 'W' ou un fallback STAGE_MAP.
    """
    s = re.sub(r"[^A-Z0-9]+", "", str(s).upper())
    if s in ("N1", "N2", "N3", "S1", "S2", "S3", "S4", "N4", "1", "2", "3", "4"):
        return "NREM"
    if s in ("REM", "R", "SP"):
        return "REM"
    if s in ("W", "WAKE", "V"):
        return "W"
    return STAGE_MAP.get(s, s)


def _aggregate_epochs_to_episodes(rows: List[Tuple[float, float, str]]) -> List[Episode]:
    """Agrège une liste d’époques (start,end,stage) en épisodes continus.

    On fusionne des époques successives de même stade si elles se touchent
    (tolérance <= 1 s).

    Args:
        rows: Liste (start_sec, end_sec, stage_raw).

    Returns:
        Liste d'objets Episode consolidés.
    """
    if not rows:
        return []

    # Harmonisation + tri
    rows = [(float(s), float(e), _norm_stage_to_rem_nrem_w(stg)) for (s, e, stg) in rows]
    rows.sort(key=lambda x: (x[0], x[1]))

    episodes: List[Episode] = []
    cur_stage: Optional[str] = None
    cur_s: Optional[float] = None
    cur_e: Optional[float] = None

    for s, e, stg in rows:
        if cur_stage is None:
            cur_stage, cur_s, cur_e = stg, s, e
            continue

        # Fusion si même stade et continuité (tolérance 1 s)
        if stg == cur_stage and abs(s - cur_e) <= 1.0:
            cur_e = max(cur_e, e)
        else:
            episodes.append(Episode(stage=cur_stage, onset=float(cur_s), duration=max(0.0, float(cur_e - cur_s))))
            cur_stage, cur_s, cur_e = stg, s, e

    # Ajout du dernier
    if cur_stage is not None:
        episodes.append(Episode(stage=cur_stage, onset=float(cur_s), duration=max(0.0, float(cur_e - cur_s))))

    return episodes


def _read_txt_hypnoexp(path: str) -> List[Episode]:
    """Lit un fichier hypnogramme type *_hypnoEXP.txt et renvoie des Episodes.

    Format supposé : colonnes (end_sec, hhmmss, stage, code) séparées par espaces.

    Stratégie :
    - on estime epoch_len via la modalité des diffs de end_sec
    - on reconstruit (start,end,stage) pour chaque ligne

    Args:
        path: Chemin du fichier txt.

    Returns:
        Episodes harmonisés.
    """
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

    # epoch_len estimé (par défaut 30 s)
    epoch_len = float(pd.Series(np.round(diffs, 1)).mode().iloc[0]) if diffs.size else 30.0

    rows = []
    for end_sec, stg in zip(df["end_sec"], df["stage"]):
        rows.append((float(end_sec - epoch_len), float(end_sec), stg))

    return _aggregate_epochs_to_episodes(rows)


def _read_csv_3cols(path: str) -> List[Episode]:
    """Lit un CSV “3 colonnes” (epoch, abs_time, stage) et renvoie des Episodes.

    Le fichier peut contenir une 1ère ligne de header “like”.
    On essaie d’estimer epoch_len à partir de abs_time si possible,
    sinon on prend 30 s.

    Args:
        path: Chemin du CSV.

    Returns:
        Episodes harmonisés (possiblement vide si parsing impossible).
    """
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

    def _time_to_sec(tstr: str) -> float:
        try:
            h, m, s = tstr.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
        except Exception:
            return np.nan

    times = df["abs_time"].astype(str).map(_time_to_sec).to_numpy()

    if np.isfinite(times).all():
        # Unwrap sur 24h si minuit passé
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
    """Parse un hypnogramme fichier vers une liste d'Episodes harmonisés.

    Formats supportés :
    - *_hypnoEXP.txt : parsing dédié (espaces, end_sec, stage)
    - *_hypnoEXP.csv ou csv 3 colonnes : parsing dédié
    - csv/tsv générique : colonnes onset/duration/stage (noms flexibles)

    Args:
        path: Chemin vers l’hypnogramme.

    Returns:
        Liste d’épisodes harmonisés, possiblement vide si échec.
    """
    if path is None or not os.path.isfile(path):
        return []

    base = os.path.basename(path).lower()
    ext = os.path.splitext(base)[1]

    try:
        # TXT hypnoEXP
        if base.endswith("_hypnoexp.txt") or ext == ".txt":
            return _read_txt_hypnoexp(path)

        # CSV hypnoEXP / 3 cols
        if base.endswith("_hypnoexp.csv") or ext == ".csv":
            try:
                eps = _read_csv_3cols(path)
                if eps:
                    return eps
            except Exception:
                pass

        # Fallback : table avec colonnes onset/duration/stage
        df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_table(path)
        cols = {c.lower(): c for c in df.columns}

        onset_col = cols.get("onset_sec") or cols.get("onset") or cols.get("start_sec") or cols.get("start")
        dur_col = cols.get("duration_sec") or cols.get("duration") or cols.get("dur_sec") or cols.get("dur")
        stage_col = cols.get("stage") or cols.get("label") or cols.get("stade") or cols.get("sleep_stage")

        if onset_col is None or dur_col is None or stage_col is None:
            return []

        rows = []
        for _, row in df.iterrows():
            stg = _norm_stage_to_rem_nrem_w(str(row[stage_col]))
            onset = float(row[onset_col])
            dur = float(row[dur_col])
            rows.append((onset, onset + dur, stg))

        return _aggregate_epochs_to_episodes(rows)

    except Exception:
        return []


def parse_hypnogram_from_raw(raw: mne.io.BaseRaw) -> List[Episode]:
    """Fallback : récupère les stades à partir de raw.annotations (si pas de fichier).

    Args:
        raw: Objet Raw MNE.

    Returns:
        Episodes consolidés, possiblement vide si pas d’annotations.
    """
    if raw.annotations is None or len(raw.annotations) == 0:
        return []

    rows: List[Tuple[float, float, str]] = []
    for ann in raw.annotations:
        desc = str(ann["description"]).strip().upper()
        stage: Optional[str] = None

        # Heuristique de mapping
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
# Signal processing helpers
# =====================================================================
def moving_average(x: np.ndarray, win_samps: int) -> np.ndarray:
    """Moyenne glissante centrée.

    Args:
        x: Signal 1D.
        win_samps: Taille de fenêtre en échantillons.

    Returns:
        Signal lissé, même longueur que x.

    Notes:
        - Padding "edge" pour limiter les effets de bord.
    """
    if win_samps <= 1:
        return x.copy()

    kernel = np.ones(win_samps, dtype=float) / win_samps
    pad = win_samps // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    y = np.convolve(x_pad, kernel, mode="valid")
    return y[: len(x)]


def load_raw_with_emg(path: str, emg_channels: Optional[List[str]]):
    """Charge un enregistrement (fif/edf/bdf/gdf) et détecte les canaux EMG.

    Args:
        path: Chemin fichier.
        emg_channels: Liste optionnelle de noms EMG à forcer.

    Returns:
        (raw, emg_channel_names)

    Raises:
        RuntimeError: si aucun canal EMG n’est détecté.
    """
    ext = os.path.splitext(path)[1].lower()

    # Lecture selon extension
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

    # Sélection EMG
    if emg_channels:
        picks = [ch for ch in emg_channels if ch in raw.ch_names]
    else:
        idx_type = mne.pick_types(raw.info, emg=True, eeg=False, meg=False, eog=False, stim=False, misc=False)
        picks = [raw.ch_names[i] for i in (idx_type if isinstance(idx_type, np.ndarray) else [])]
        picks += [ch for ch in raw.ch_names if "emg" in ch.lower() and ch not in picks]

    if not picks:
        raise RuntimeError(
            f"No EMG channels found in {path}. Provide --emg_channels or set channel types to 'emg'."
        )

    return raw, picks


def build_emg_envelopes(raw: mne.io.BaseRaw, emg_chs: List[str]):
    """Construit l’enveloppe EMG par canal.

    Pipeline :
    1) Pick EMG channels
    2) Filtre bande 30-100 Hz (FIR)
    3) Enveloppe = moving_average(|signal|) avec fenêtre 1 s

    Args:
        raw: Raw MNE.
        emg_chs: Noms de canaux EMG.

    Returns:
        envs: ndarray shape (n_ch, n_samples)
        ch_names: liste des canaux
        sfreq: fréquence d’échantillonnage
    """
    emg = raw.copy().pick_channels(emg_chs)

    # Filtrage EMG standard pour RSWA (à valider selon ton protocole)
    emg.filter(l_freq=30.0, h_freq=100.0, picks="all", method="fir", verbose=False)

    data = emg.get_data()  # volts
    sfreq = float(emg.info["sfreq"])

    win = max(1, int(round(sfreq * 1.0)))  # 1 seconde
    envs = np.empty_like(data)

    for i in range(data.shape[0]):
        envs[i, :] = moving_average(np.abs(data[i, :]), win)

    return envs, emg.ch_names, sfreq


def load_eog_signal(raw: mne.io.BaseRaw):
    """Charge les signaux EOG (et force le type sur certains noms).

    Règle projet :
    - force type 'eog' pour les canaux contenant 'EOGD' ou 'EOGG'
    - retourne la donnée convertie en µV (approx) et la sfreq

    Args:
        raw: Raw MNE.

    Returns:
        (data, sfreq) où data shape = (n_eog_ch, n_samples)
        ou (None, None) si aucun EOG.
    """
    # 1) Forcer le type EOG si besoin
    eog_chs = [ch for ch in raw.ch_names if ("EOGD" in ch.upper()) or ("EOGG" in ch.upper())]
    if eog_chs:
        raw.set_channel_types({ch: "eog" for ch in eog_chs})

    # 2) Picks EOG
    picks = mne.pick_types(raw.info, eog=True, eeg=False, emg=False, meg=False, stim=False, misc=False)
    if len(picks) == 0:
        return None, None

    eog = raw.copy().pick(picks)

    # ⚠️ Unité : MNE retourne en volts.
    # Pour µV => *1e6. Ton code précédent utilisait *1e9 (nV).
    # Ici on fait µV explicitement pour être cohérent avec seuil 25 µV.
    data_uv = eog.get_data() * 1e6
    sfreq = float(eog.info["sfreq"])
    return data_uv, sfreq


# =====================================================================
# Tonic EOG (par epoch) — masquage phasique
# =====================================================================
def detect_tonic_eog_epoch(
    eog: np.ndarray,
    sfreq: float,
    w0: int,
    w1: int,
    phasic_in_win: List[IntervalSamp],
    amp_thresh_uv: float = 25.0,
) -> float:
    """Détecte la tonicité EOG sur une fenêtre (epoch) de 4 s.

    Critère (dans ce script) :
    - on moyenne les canaux EOG si plusieurs
    - on masque les échantillons appartenant à des événements phasiques
    - on découpe l’epoch en 2 sous-fenêtres consécutives de 2 s
    - tonic=True si max(|EOG|) <= amp_thresh_uv sur chacune des 2 sous-fenêtres

    Args:
        eog: Données EOG en µV, shape (n_ch, n_samples).
        sfreq: Fréquence d’échantillonnage (Hz).
        w0: Début de l’epoch (en samples).
        w1: Fin de l’epoch (en samples).
        phasic_in_win: Liste d’événements phasiques (en samples) recoupés dans [w0,w1].
        amp_thresh_uv: Seuil amplitude en µV (défaut 25 µV).

    Returns:
        bool ou np.nan si non calculable (ex: taille insuffisante, tout masqué).

    Notes:
        - Ce critère est une heuristique “tonic via EOG”. À justifier dans ton papier.
        - Il est crucial que l’unité soit bien µV.
    """
    if eog is None:
        return np.nan

    # Moyenne des canaux EOG (si plusieurs)
    sig = np.mean(eog[:, w0:w1], axis=0)

    # Masquage des périodes phasiques
    mask = np.zeros(sig.shape[0], dtype=bool)
    for s, e in phasic_in_win:
        s_rel = max(0, s - w0)
        e_rel = min(sig.shape[0], e - w0)
        if s_rel < e_rel:
            mask[s_rel:e_rel] = True

    sig_clean = sig.copy()
    sig_clean[mask] = np.nan

    # Split 2 × 2 s
    half = int(round(2.0 * sfreq))
    if sig_clean.size < 2 * half:
        return np.nan

    win1 = sig_clean[:half]
    win2 = sig_clean[half : 2 * half]

    if np.isnan(win1).all() or np.isnan(win2).all():
        return np.nan

    amp1 = np.nanmax(np.abs(win1))
    amp2 = np.nanmax(np.abs(win2))

    tonic = (amp1 <= amp_thresh_uv) and (amp2 <= amp_thresh_uv)
    return bool(tonic)


# =====================================================================
# Phasic EMG (par canal) — seuil = 95e percentile NREM de référence
# =====================================================================
def detect_phasic_events_ch(
    envelope: np.ndarray,
    sfreq: float,
    rem_start: int,
    rem_end: int,
    nrem_ref_start: int,
    nrem_ref_end: int,
) -> List[IntervalSamp]:
    """Détecte des événements phasiques EMG dans un épisode REM, pour un canal.

    Méthode (ici) :
    1) On définit une fenêtre NREM de référence (juste avant l’épisode REM)
    2) Seuil = 95e percentile de l’enveloppe EMG dans cette fenêtre NREM
    3) Dans la portion REM, on détecte les segments où enveloppe > seuil
    4) On conserve un événement si sa durée >= 3 s (min_len)
    5) On fusionne les événements séparés par un gap <= 1 s

    Args:
        envelope: Enveloppe EMG (1D), shape (n_samples,).
        sfreq: Fréquence d’échantillonnage.
        rem_start/rem_end: bornes REM (samples).
        nrem_ref_start/nrem_ref_end: bornes NREM référence (samples).

    Returns:
        Liste d’événements (start_sample, end_sample) dans l’espace global du signal.

    Notes importantes :
    - Le choix du 95e percentile NREM est une heuristique : il faut la justifier,
      ou comparer à d’autres seuils (90/95/97.5) en ablation.
    - min_len=3 s ici est très strict : à valider (souvent phasic bursts plus courts).
      Tu peux ajuster dans le code si tes critères sont différents.
    """
    nrem_ref = envelope[nrem_ref_start:nrem_ref_end]
    if nrem_ref.size == 0:
        return []

    thr = float(np.percentile(nrem_ref, 95.0))

    rem_env = envelope[rem_start:rem_end]
    above = rem_env > thr

    # Durée minimale au-dessus du seuil (ici 3 s)
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

    # Fusion d’événements proches
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
    """Clippe des événements (samples) à l’intérieur d’une fenêtre [w0,w1]."""
    out: List[IntervalSamp] = []
    for s, e in events:
        ss, ee = max(s, w0), min(e, w1)
        if ee > ss:
            out.append((ss, ee))
    return out


def phasic_time_ratio(events_clipped: List[IntervalSamp], w0: int, w1: int, sfreq: float) -> Tuple[float, float]:
    """Calcule le temps phasique (s) et le ratio phasique (0-1) sur une fenêtre.

    Args:
        events_clipped: événements phasiques dans l’epoch.
        w0/w1: bornes (samples) de l’epoch.
        sfreq: Hz.

    Returns:
        (dur_phasic_sec, ratio_phasic) où ratio = dur_phasic / durée_epoch.
    """
    dur = sum(e - s for s, e in events_clipped) / float(sfreq)
    win_dur = max(1e-9, (w1 - w0) / float(sfreq))
    return float(dur), float(dur / win_dur)


# =====================================================================
# Tonic EMG ratio (médiane REM / médiane NREM) — avec masquage phasique
# =====================================================================
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
) -> Tuple[float, bool]:
    """Calcule un ratio tonique EMG sur une epoch, et un flag d’exclusion.

    Idée :
    - tonic_ratio = median(REM_clean) / median(NREM_clean)
    - REM_clean = enveloppe dans l’epoch après masquage des samples phasiques
    - NREM_clean = NREM_ref où on retire les valeurs extrêmes > 95e percentile

    Args:
        envelope: enveloppe EMG 1D.
        sfreq: Hz.
        w0/w1: bornes epoch (samples).
        phasic_in_win: événements phasiques dans l’epoch.
        nrem_ref_start/nrem_ref_end: fenêtre NREM de référence.
        phasic_ratio_epoch: ratio phasique (utilisé pour exclusion si trop phasique).
        nrem_clip_percentile: percentile au-delà duquel on exclut les valeurs NREM (défaut 95).
        min_nrem_seconds_for_clean: minimum de points “clean” NREM exigé.
        tiny: évite division par 0.

    Returns:
        (tonic_ratio, tonic_excluded)
        - tonic_ratio peut être np.nan si non calculable
        - tonic_excluded True si epoch trop phasique (ici phasic_ratio > 0.75)

    Notes:
        - L’exclusion "très phasique" est une règle heuristique.
        - Les médianes sont robustes aux outliers.
    """
    rem_env = envelope[w0:w1].copy()
    if rem_env.size == 0:
        return (np.nan, False)

    # Masquage phasique dans l’epoch
    mask = np.zeros(rem_env.shape[0], dtype=bool)
    for s, e in phasic_in_win:
        s_rel = max(0, s - w0)
        e_rel = min(rem_env.shape[0], e - w0)
        if s_rel < e_rel:
            mask[s_rel:e_rel] = True

    rem_clean = rem_env[~mask] if (~mask).any() else rem_env

    # NREM référence
    nrem_ref = envelope[nrem_ref_start:nrem_ref_end]
    if nrem_ref.size == 0:
        return (np.nan, phasic_ratio_epoch > 0.75)

    # “Nettoyage” NREM : enlever les plus grosses valeurs (artefacts/tonic)
    thr_ref = float(np.percentile(nrem_ref, float(nrem_clip_percentile)))
    nrem_clean = nrem_ref[nrem_ref <= thr_ref]

    # Si trop peu de points, on garde nrem_ref brut
    if nrem_clean.size < int(round(float(min_nrem_seconds_for_clean) * float(sfreq))):
        nrem_clean = nrem_ref

    rem_med = float(np.median(rem_clean)) if rem_clean.size else np.nan
    nrem_med = float(np.median(nrem_clean)) if nrem_clean.size else np.nan

    if not np.isfinite(rem_med) or not np.isfinite(nrem_med) or nrem_med <= 0:
        return (np.nan, phasic_ratio_epoch > 0.75)

    tonic_ratio = rem_med / max(nrem_med, float(tiny))
    return (float(tonic_ratio), phasic_ratio_epoch > 0.75)


# =====================================================================
# Construction des paires NREM->REM + epochs 4s
# =====================================================================
def collect_rem_nrem_pairs(episodes: List[Episode], min_len_sec: float = 10.0):
    """Construit des paires (NREM, REM) pour calibrer les seuils sur NREM.

    Règle :
    - on prend chaque épisode REM suffisamment long
    - on cherche le dernier épisode NREM suffisamment long juste avant
    - tolérance : NREM end proche de REM onset (<=2 s) ou end <= onset

    Args:
        episodes: liste d’épisodes (triés ou non).
        min_len_sec: durée minimale (sec) pour retenir un épisode.

    Returns:
        Liste de tuples (episode_nrem, episode_rem).
    """
    pairs = []
    episodes = sorted(episodes, key=lambda e: e.onset)

    for i, ep in enumerate(episodes):
        if ep.stage != "REM" or ep.duration < float(min_len_sec):
            continue

        prior = None
        for j in range(i - 1, -1, -1):
            if episodes[j].stage == "NREM" and episodes[j].duration >= float(min_len_sec):
                # si contigu ou antérieur
                if abs(episodes[j].end - ep.onset) <= 2.0 or episodes[j].end <= ep.onset:
                    prior = episodes[j]
                    break
                else:
                    break

        if prior is not None:
            pairs.append((prior, ep))

    return pairs


def split_into_epochs(rem_start: int, rem_end: int, sfreq: float, epoch_len_s: float):
    """Découpe [rem_start, rem_end] en fenêtres non chevauchantes de epoch_len_s.

    Args:
        rem_start/rem_end: bornes en samples.
        sfreq: Hz.
        epoch_len_s: durée fenêtre en secondes.

    Returns:
        Liste de (w0,w1) en samples.
    """
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
    """Traite un patient : lecture raw, hypnogramme, détection phasic/tonic/rswa.

    Args:
        patient_dir: dossier patient sous input_dir.
        args: namespace argparse.

    Returns:
        DataFrame contenant :
          - événements phasiques (type="PHASIC")
          - résumé par epoch 4s (type="REM_EPOCH_4S")
        ou une ligne de type ERROR/EMPTY en cas de problème.
    """
    patient_id = os.path.basename(patient_dir.rstrip(os.sep))

    rec: Optional[str] = None
    raw: Optional[mne.io.BaseRaw] = None
    eog_data: Optional[np.ndarray] = None

    try:
        # -----------------------------------------------------------------
        # 1) Trouver et charger l’enregistrement + canaux EMG
        # -----------------------------------------------------------------
        rec = find_recording_file(patient_dir)
        if rec is None:
            raise RuntimeError("No recording file (edf/fif) found.")

        raw, emg_chs = load_raw_with_emg(rec, args.emg_channels)

        # -----------------------------------------------------------------
        # 2) Charger EOG (obligatoire ici : tonic basé sur EOG)
        # -----------------------------------------------------------------
        eog_data, eog_sfreq = load_eog_signal(raw)
        if eog_data is None or eog_sfreq is None:
            raise RuntimeError("EOG manquant - tonic EOG impossible.")

        # Consistance des indices temporels : même sfreq
        if abs(float(eog_sfreq) - float(raw.info["sfreq"])) > 1e-6:
            raise RuntimeError(
                f"sfreq EOG ({eog_sfreq}) != sfreq raw/EMG ({raw.info['sfreq']}) - resampling requis."
            )

        # -----------------------------------------------------------------
        # 3) Construire enveloppes EMG
        # -----------------------------------------------------------------
        envs, chs, sfreq = build_emg_envelopes(raw, emg_chs)
        n_samples = envs.shape[1]

        # -----------------------------------------------------------------
        # 4) Hypnogramme -> épisodes -> paires NREM->REM
        # -----------------------------------------------------------------
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

        # -----------------------------------------------------------------
        # 5) Analyse sur chaque paire NREM->REM, par canal et par epoch
        # -----------------------------------------------------------------
        rows = []

        for idx_pair, (nrem_ep, rem_ep) in enumerate(pairs):
            # Marges (2 s) pour éviter bords (transition stade, artefacts)
            rem_start = int(np.round((rem_ep.onset + 2.0) * sfreq))
            rem_end = int(np.round((rem_ep.end - 2.0) * sfreq))

            # NREM référence : dernier 30 s avant fin NREM (en évitant 2 s bord)
            nrem_ref_end = int(np.round((nrem_ep.end - 2.0) * sfreq))
            nrem_ref_start = int(np.round(max(nrem_ep.end - 30.0, nrem_ep.onset + 2.0) * sfreq))

            # Bornage safe
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

                # 5a) Détection des événements phasiques dans l’épisode REM (global)
                phasic_all = detect_phasic_events_ch(
                    env, sfreq,
                    rem_start, rem_end,
                    nrem_ref_start, nrem_ref_end,
                )

                # On log tous les événements phasiques “longs” (>= min_phasic_dur)
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

                # 5b) Résumé par epoch 4s
                for ep_k, (w0, w1) in enumerate(epochs):
                    phasic_in_win = clip_events_to_window(phasic_all, w0, w1)

                    # Tonic EOG sur l’epoch (avec masquage phasic)
                    tonic_eog_epoch = detect_tonic_eog_epoch(
                        eog=eog_data,
                        sfreq=eog_sfreq,
                        w0=w0,
                        w1=w1,
                        phasic_in_win=phasic_in_win,
                    )

                    # Phasic ratio
                    dur_phasic, phasic_ratio_epoch = phasic_time_ratio(phasic_in_win, w0, w1, sfreq)
                    very_phasic_epoch = (phasic_ratio_epoch > 0.75)

                    # Tonic ratio EMG
                    tonic_ratio_epoch, tonic_excluded_epoch = compute_tonic_ratio_epoch_ch(
                        env, sfreq, w0, w1, phasic_in_win,
                        nrem_ref_start, nrem_ref_end, phasic_ratio_epoch,
                    )

                    # Règle RSWA au niveau epoch
                    rswa_epoch = (
                        (np.isfinite(tonic_ratio_epoch) and (tonic_ratio_epoch > 1.3))
                        or (tonic_eog_epoch is True)
                    )

                    # TODO futur : corrélation eye_emg_corr (désactivée)
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
def main() -> None:
    """Entrée principale : parse args, checks, parallélisation, export CSV."""
    parser = argparse.ArgumentParser(
        description="Detect phasic bursts & RSWA during REM on 4s epochs - per channel."
    )
    parser.add_argument("--input_dir", default=GP2_ROOT, help="Directory containing patient subfolders.")
    parser.add_argument(
        "--output_csv",
        default="data/processed/rbd/rbd_emg_events_and_summary_4s_per_channel.csv",
        help="Chemin CSV de sortie (événements + résumé per-epoch, per channel).",
    )
    parser.add_argument(
        "--emg_channels",
        nargs="*",
        default=None,
        help="Liste optionnelle de canaux EMG à utiliser (sinon détection automatique).",
    )
    parser.add_argument("--n_jobs", type=int, default=20, help="Nombre de workers en parallèle.")
    parser.add_argument("--rem_epoch_len", type=float, default=4.0, help="Longueur des époques REM en secondes.")
    parser.add_argument("--min_phasic_dur", type=float, default=0.5, help="Durée minimale (s) d’un événement PHASIC.")
    args = parser.parse_args()

    # -------------------------
    # Checks (robustes, explicites)
    # -------------------------
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

    # Conserver uniquement les patients avec un enregistrement détecté
    valid_patients = []
    for p in patients:
        rec = find_recording_file(p)
        if rec is not None:
            valid_patients.append(p)

    if not valid_patients:
        print(f"[ERREUR] Aucun fichier .fif/.edf trouvé dans les dossiers patients de : {args.input_dir}", file=sys.stderr)
        print("[ERREUR] Vérifie que GP2_ROOT pointe vers les bons fichiers prétraités.")
        sys.exit(1)

    patients = valid_patients
    print(f"[CHECK] {len(patients)} patients valides trouvés.")

    # -------------------------
    # Parallélisation patients
    # -------------------------
    dfs = []
    if args.n_jobs and args.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            fut2p = {ex.submit(process_patient, p, args): p for p in patients}
            for fut in as_completed(fut2p):
                dfs.append(fut.result())
    else:
        for p in patients:
            dfs.append(process_patient(p, args))

    # -------------------------
    # Export CSV + erreurs
    # -------------------------
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
