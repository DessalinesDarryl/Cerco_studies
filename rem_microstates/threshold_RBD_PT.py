#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect RBD markers (PHASIC & TONIC RSWA) from EMG during REM sleep
— CALCULS PAR CANAL, sur ÉPOQUES REM non chevauchantes de 4 s (par défaut).

Adapté de Grenot et al., SLEEP 2024 (logique de seuil P95 NREM, >=0.5 s, fusion <0.5 s).

Sortie CSV (par patient ET par canal) :
- "PHASIC" : bouffées fusionnées par canal (start_sec, end_sec, duration_sec, channel)
- "REM_EPOCH_4S" : une ligne par époque 4 s et par canal
  (epoch_start_sec, epoch_end_sec, phasic_ratio, tonic_ratio, rswa, very_phasic, phasic_count, channel)
"""

# =========================
# 0) IMPORTS & CONSTANTES
# =========================
import argparse, os, sys, glob
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import mne

# --- Emplacements (adapte si besoin)
GP2_ROOT  = "/home/darryld/documents/EEG/preprocessed/bipolaire/1_noArtefacts/gp2/"
RAW_ROOT  = "/home/darryld/documents/EEG/raw/"
FIF_GLOB_PATTERNS = ["*_art_annotated.fif", "*.fif"]
HYPNO_CANDIDATES  = ["{pid}_hypnoEXP.txt", "{pid}_hypnoEXP.csv", "{pid}_hypno.txt", "{pid}_hypnogram.txt"]

# --- Normalisation de stades
STAGE_MAP = {
    "W": "W", "V":"W", "WAKE":"W",
    "N1": "NREM", "S1": "NREM", "1": "NREM",
    "N2": "NREM", "S2": "NREM", "2": "NREM",
    "N3": "NREM", "S3": "NREM", "S4":"NREM", "N4":"NREM", "3":"NREM", "4":"NREM",
    "NREM":"NREM", "REM":"REM", "R":"REM"
}

# ============================
# 1) STRUCTURES & UTILITAIRES
# ============================
@dataclass
class Episode:
    stage: str
    onset: float  # seconds
    duration: float  # seconds
    @property
    def end(self) -> float:
        return self.onset + self.duration

def find_patient_dirs(input_dir: str) -> List[str]:
    subs = [p for p in glob.glob(os.path.join(input_dir, "*")) if os.path.isdir(p)]
    subs.sort()
    print(f"[SCAN] {len(subs)} dossiers patients trouvés dans: {input_dir}")
    return subs

def find_recording_file(patient_dir: str) -> Optional[str]:
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

# =========================
# 2) PARSING HYPNOGRAMME
# =========================
def _norm_stage_to_rem_nrem_w(s: str) -> str:
    s = str(s).strip().upper()
    return STAGE_MAP.get(s, s)

def _aggregate_epochs_to_episodes(rows: List[Tuple[float, float, str]]) -> List[Episode]:
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
    df = pd.read_csv(path, sep=r"\s+", engine="python", header=None,
                     names=["end_sec", "hhmmss", "stage", "code"], usecols=[0,1,2,3])
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
            h, m, s = tstr.split(":"); return int(h)*3600 + int(m)*60 + float(s)
        except Exception: return np.nan
    times = df["abs_time"].astype(str).map(_time_to_sec).to_numpy()
    if np.isfinite(times).all():
        unwrapped, shift, last = [], 0.0, None
        for v in times:
            vv = v + shift
            if last is not None and vv < last - 1e-6:
                shift += 24*3600.0; vv = v + shift
            unwrapped.append(vv); last = vv
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
        # Fallback: CSV/TXT avec onset/duration/stage
        df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_table(path)
        cols = {c.lower(): c for c in df.columns}
        onset_col = cols.get("onset_sec") or cols.get("onset") or cols.get("start_sec") or cols.get("start")
        dur_col   = cols.get("duration_sec") or cols.get("duration") or cols.get("dur_sec") or cols.get("dur")
        stage_col = cols.get("stage") or cols.get("label") or cols.get("stade") or cols.get("sleep_stage")
        if onset_col is None or dur_col is None or stage_col is None:
            return []
        rows = []
        for _, row in df.iterrows():
            stage_m = _norm_stage_to_rem_nrem_w(str(row[stage_col]))
            rows.append((float(row[onset_col]), float(row[onset_col]) + float(row[dur_col]), stage_m))
        return _aggregate_epochs_to_episodes(rows)
    except Exception:
        return []

def parse_hypnogram_from_raw(raw: mne.io.BaseRaw) -> List[Episode]:
    episodes: List[Episode] = []
    if raw.annotations is None or len(raw.annotations) == 0:
        return episodes
    for ann in raw.annotations:
        desc = str(ann["description"]).strip().upper()
        stage = None
        if "REM" in desc or desc == "R": stage = "REM"
        elif "N1" in desc or "S1" in desc: stage = "NREM"
        elif "N2" in desc or "S2" in desc: stage = "NREM"
        elif "N3" in desc or "S3" in desc or "S4" in desc or "N4" in desc: stage = "NREM"
        elif "W" == desc or "WAKE" in desc: stage = "W"
        if stage is None: continue
        episodes.append(Episode(stage=stage, onset=float(ann["onset"]), duration=float(ann["duration"])))
    return _aggregate_epochs_to_episodes([(ep.onset, ep.end, ep.stage) for ep in episodes])

# =========================
# 3) ENVELOPPES EMG PAR CANAL
# =========================
def moving_average(x: np.ndarray, win_samps: int) -> np.ndarray:
    if win_samps <= 1: return x.copy()
    kernel = np.ones(win_samps, dtype=float) / win_samps
    pad = win_samps // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    y = np.convolve(x_pad, kernel, mode="valid")
    return y[:len(x)]

def load_raw_with_emg(path: str, emg_channels: Optional[List[str]]):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".fif":   raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    elif ext == ".edf": raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    elif ext == ".bdf": raw = mne.io.read_raw_bdf(path, preload=True, verbose=False)
    elif ext == ".gdf": raw = mne.io.read_raw_gdf(path, preload=True, verbose=False)
    else:               raw = mne.io.read_raw(path, preload=True, verbose=False)

    if emg_channels:
        picks = [ch for ch in emg_channels if ch in raw.ch_names]
    else:
        idx_type = mne.pick_types(raw.info, emg=True, eeg=False, meg=False, eog=False, stim=False, misc=False)
        picks = [raw.ch_names[i] for i in (idx_type if isinstance(idx_type, np.ndarray) else [])]
        picks += [ch for ch in raw.ch_names if "emg" in ch.lower() and ch not in picks]
    if not picks:
        raise RuntimeError(f"No EMG channels found in {path}. Provide --emg_channels or set channel types to 'emg'.")
    return raw, picks

def build_emg_envelopes(raw: mne.io.BaseRaw, emg_chs: List[str]) -> Tuple[np.ndarray, List[str], float]:
    """
    Retourne:
      - envs: array (n_ch, n_samples) = enveloppe par canal (PB 30–100 Hz, rectif, lissage 1 s)
      - chs:  liste des noms de canaux dans le même ordre
      - sfreq: fréquence d'échantillonnage
    """
    emg = raw.copy().pick_channels(emg_chs)
    emg.filter(l_freq=30.0, h_freq=100.0, picks="all", method="fir", verbose=False)
    data = emg.get_data()  # (n_ch, n_samples)
    sfreq = emg.info["sfreq"]
    win = max(1, int(round(sfreq * 1.0)))
    envs = np.empty_like(data)
    for i in range(data.shape[0]):
        envs[i, :] = moving_average(np.abs(data[i, :]), win)
    return envs, emg.ch_names, sfreq

# =========================
# 4) OUTILS PHASIC/TONIC — PAR CANAL
# =========================
def detect_phasic_events_ch(envelope: np.ndarray, sfreq: float,
                            rem_start: int, rem_end: int,
                            nrem_ref_start: int, nrem_ref_end: int) -> List[Tuple[int, int]]:
    """Bouffées phasiques sur UN canal: >=3 s au-dessus du P95 NREM précédent, fusion gap<0.5 s."""
    nrem_ref = envelope[nrem_ref_start:nrem_ref_end]
    if len(nrem_ref) == 0:
        return []
    thr = np.percentile(nrem_ref, 95.0)
    rem_env = envelope[rem_start:rem_end]
    above = rem_env > thr
    min_len = int(np.ceil(3.0 * sfreq))
    events = []
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
    max_gap = int(np.floor(0.5 * sfreq))
    for s, e in events[1:]:
        ps, pe = merged[-1]
        if (s - pe) <= max_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged

def clip_events_to_window(events: List[Tuple[int,int]], w0: int, w1: int) -> List[Tuple[int,int]]:
    out = []
    for s, e in events:
        ss, ee = max(s, w0), min(e, w1)
        if ee > ss:
            out.append((ss, ee))
    return out

def phasic_time_ratio(events_clipped: List[Tuple[int,int]], w0: int, w1: int, sfreq: float) -> Tuple[float, float]:
    dur = sum(e - s for s, e in events_clipped) / sfreq
    win_dur = max(1e-9, (w1 - w0) / sfreq)
    return dur, float(dur / win_dur)

def compute_tonic_ratio_epoch_ch(
    envelope: np.ndarray,
    sfreq: float,
    w0: int,
    w1: int,
    phasic_in_win: List[Tuple[int, int]],
    nrem_ref_start: int,
    nrem_ref_end: int,
    phasic_ratio_epoch: float,
    *,
    nrem_clip_percentile: float = 95.0,
    min_nrem_seconds_for_clean: float = 0.5,
    tiny: float = 1e-12
) -> Tuple[float, bool]:
    """
    Calcule le TONIC ratio d'une époque REM (par canal), en suivant la logique Grenot et al. :
        tonic_ratio = median(REM_epoch_sans_phasic) / median(NREM_nettoyé_<=P95)

    Paramètres
    ----------
    envelope : np.ndarray
        Enveloppe EMG du canal (après filtrage 30–100 Hz, rectification, lissage 1 s).
        Forme (n_samples,).
    sfreq : float
        Fréquence d'échantillonnage en Hz (ex. 500.0).
    w0, w1 : int
        Indices échantillons délimitant l’époque REM courante (non chevauchante, p.ex. 4 s).
        Périmètre d’analyse : envelope[w0:w1].
    phasic_in_win : List[Tuple[int, int]]
        Bouffées phasiques (indices absolus) **intersectionnées** à l’époque [w0, w1).
        Chaque tuple (s, e) respecte 0 <= w0 <= s < e <= w1.
    nrem_ref_start, nrem_ref_end : int
        Indices échantillons de la fenêtre NREM de référence (strictement avant le REM), typiquement 20–30 s.
        Le nettoyage se fait en tronquant au <= P95 sur cette fenêtre.
    phasic_ratio_epoch : float
        Rapport (durée phasic dans l’époque) / (durée époque), utilisé pour marquer les époques "very phasic".
        NB : on **ne** exclut pas automatiquement ces époques, on retourne juste le flag.

    Options (kw-only)
    -----------------
    nrem_clip_percentile : float, par défaut 95.0
        Percentile appliqué à la fenêtre NREM de référence pour supprimer les pointes (winsorisation dure).
    min_nrem_seconds_for_clean : float, par défaut 0.5
        Si le nombre d’échantillons NREM restant après tronquage est < 0.5 s,
        on retombe sur la fenêtre NREM brute (sans tronquage).
    tiny : float, par défaut 1e-12
        Petite constante pour éviter les divisions instables si la médiane NREM est ~0.

    Retour
    ------
    (tonic_ratio, very_phasic_flag) : Tuple[float, bool]
        tonic_ratio : float
            median(REM_sans_phasic) / median(NREM_nettoyé_<=P95). NaN si fenêtres invalides.
        very_phasic_flag : bool
            True si phasic_ratio_epoch > 0.75 (marqueur d’exclusion **théorique**).
            À toi de décider en amont/aval si tu exclus ces époques de l’agrégat.

    Détails de l’algorithme
    -----------------------
    1) On extrait l’enveloppe de l’époque REM courante : rem_env = envelope[w0:w1].
    2) On masque tous les échantillons appartenant à des REM phasiques dans cette fenêtre.
       Cela donne rem_clean (si tout est phasic, on garde rem_env en fallback).
    3) On extrait la fenêtre NREM de référence nrem_ref = envelope[nrem_ref_start:nrem_ref_end].
       - On calcule le seuil P95 et on garde nrem_clean = nrem_ref[nrem_ref <= P95].
       - Si nrem_clean est trop courte (< min_nrem_seconds_for_clean), on retombe sur nrem_ref brute.
    4) On calcule les médianes rem_med, nrem_med. Si nrem_med ~ 0, on renvoie (NaN, flag).
    5) tonic_ratio = rem_med / max(nrem_med, tiny).
    6) very_phasic_flag = (phasic_ratio_epoch > 0.75), sans forcer l’exclusion.

    Cas limites gérés
    -----------------
    - Époque vide (w1 <= w0) ou NREM vide => (NaN, False)
    - Tout REM masqué par phasic => fallback à rem_env (on évite un tableau vide)
    - NREM nettoyé trop court => fallback à NREM brut
    - nrem_med ≈ 0 => NaN (ou ratio très grand si tu choisis un `tiny` plus élevé)

    Notes pratiques
    ---------------
    - Si tu veux vraiment **exclure** les époques very_phasic du calcul TONIC global,
      fais-le à l’étape d’agrégation (p.ex. filtrer `very_phasic_flag` avant toute moyenne/stat).
    - Tu peux ajuster `nrem_clip_percentile` (p.ex. 90–98) selon le niveau de bruit du NREM.
    """
    # ---------- 1) REM : extraction & masque phasic ----------
    rem_env = envelope[w0:w1].copy()
    if rem_env.size == 0:
        return (np.nan, False)

    # masque des échantillons phasiques à exclure de l'estimation TONIC
    mask = np.zeros(rem_env.shape[0], dtype=bool)
    for s, e in phasic_in_win:
        s_rel = max(0, s - w0)
        e_rel = min(rem_env.shape[0], e - w0)
        if s_rel < e_rel:
            mask[s_rel:e_rel] = True

    # si tout est masqué (rare mais possible sur 4 s), fallback: on garde tout
    rem_clean = rem_env[~mask] if (~mask).any() else rem_env

    # ---------- 2) NREM ref : extraction & tronquage <= Pxx ----------
    nrem_ref = envelope[nrem_ref_start:nrem_ref_end]
    if nrem_ref.size == 0:
        return (np.nan, phasic_ratio_epoch > 0.75)

    # seuil percentile 
    thr_ref = np.percentile(nrem_ref, nrem_clip_percentile)
    nrem_clean = nrem_ref[nrem_ref <= thr_ref]

    # longueur minimale après nettoyage (sinon on garde la fenêtre brute)
    if nrem_clean.size < int(round(min_nrem_seconds_for_clean * sfreq)):
        nrem_clean = nrem_ref

    # ---------- 3) Médians & ratio ----------
    rem_med = float(np.median(rem_clean)) if rem_clean.size else np.nan
    nrem_med = float(np.median(nrem_clean)) if nrem_clean.size else np.nan

    if not np.isfinite(rem_med) or not np.isfinite(nrem_med) or nrem_med <= 0:
        return (np.nan, phasic_ratio_epoch > 0.75)

    tonic_ratio = rem_med / max(nrem_med, tiny)
    return (tonic_ratio, phasic_ratio_epoch > 0.75)


# =========================
# 5) PAIRS NREM->REM & ÉPOQUES 4 s
# =========================
def collect_rem_nrem_pairs(episodes: List[Episode], min_len_sec: float = 10.0) -> List[Tuple[Episode, Episode]]:
    pairs = []
    episodes = sorted(episodes, key=lambda e: e.onset)
    for i, ep in enumerate(episodes):
        if ep.stage != "REM" or ep.duration < min_len_sec:
            continue
        prior = None
        for j in range(i - 1, -1, -1):
            if episodes[j].stage == "NREM" and episodes[j].duration >= min_len_sec:
                if abs(episodes[j].end - ep.onset) <= 2.0 or episodes[j].end <= ep.onset:
                    prior = episodes[j]; break
                else:
                    break
        if prior is not None:
            pairs.append((prior, ep))
    return pairs

def split_into_epochs(rem_start: int, rem_end: int, sfreq: float, epoch_len_s: float) -> List[Tuple[int,int]]:
    win = int(round(epoch_len_s * sfreq))
    if win < 1: win = 1
    epochs = []
    s = rem_start
    while s + win <= rem_end:
        epochs.append((s, s + win))
        s += win
    return epochs

# =========================
# 6) TRAITEMENT PAR PATIENT (PAR CANAL)
# =========================
def process_patient(patient_dir: str, args) -> pd.DataFrame:
    patient_id = os.path.basename(patient_dir.rstrip(os.sep))
    rec = None
    try:
        # (A) Lecture Raw + sélection canaux EMG
        rec = find_recording_file(patient_dir)
        if rec is None:
            raise RuntimeError("No recording file (edf/fif) found.")
        raw, emg_chs = load_raw_with_emg(rec, args.emg_channels)

        # (B) Enveloppes par canal
        envs, chs, sfreq = build_emg_envelopes(raw, emg_chs)
        n_samples = envs.shape[1]

        # (C) Hypnogramme > paires NREM->REM
        hyp_path = find_hypnogram_file(patient_dir)
        episodes = parse_hypnogram_table(hyp_path) if hyp_path else parse_hypnogram_from_raw(raw)
        if not episodes:
            raise RuntimeError("No hypnogram found or parsed.")
        pairs = collect_rem_nrem_pairs(episodes, min_len_sec=10.0)

        rows = []
        # (D) Boucle sur paires NREM->REM
        for idx_pair, (nrem_ep, rem_ep) in enumerate(pairs):
            # (D1) Marges transitions 
            rem_start = int(np.round((rem_ep.onset + 2.0) * sfreq)) 
            rem_end = int(np.round((rem_ep.end - 2.0) * sfreq)) 
            nrem_end_for_ref = int(np.round((nrem_ep.end - 2.0) * sfreq)) 
            nrem_start_for_ref = int(np.round(max(nrem_ep.end - 30.0, nrem_ep.onset + 2.0) * sfreq)) 
            rem_start = max(0, min(rem_start, n_samples - 1)) 
            rem_end = max(rem_start + 1, min(rem_end, n_samples)) 
            nrem_ref_start = max(0, min(nrem_start_for_ref, n_samples - 1)) 
            nrem_ref_end = max(nrem_ref_start + 1, min(nrem_end_for_ref, n_samples))


            # (D2) Époques 4 s dans le REM
            epochs = split_into_epochs(rem_start, rem_end, sfreq, args.rem_epoch_len)

            # (D3) Boucle PAR CANAL
            for ch_i, ch_name in enumerate(chs):
                env = envs[ch_i, :]

                # (D3.1) Bouffées phasiques (sur tout l'épisode REM) pour ce canal
                phasic_all = detect_phasic_events_ch(env, sfreq, rem_start, rem_end, nrem_ref_start, nrem_ref_end)

                # (D3.2) Export événements PHASIC (par canal)
                for s_idx, e_idx in phasic_all:
                    dur_sec = (e_idx - s_idx) / sfreq
                    if dur_sec >= args.min_phasic_dur:
                        rows.append({
                            "patient_id": patient_id,
                            "channel": ch_name,
                            "type": "PHASIC",
                            "episode_index": idx_pair,
                            "start_sec": s_idx / sfreq,
                            "end_sec": e_idx / sfreq,
                            "duration_sec": dur_sec
                        })

                # (D3.3) Par époque 4 s : ratios PHASIC & TONIC, RSWA
                for ep_k, (w0, w1) in enumerate(epochs):
                    phasic_in_win = clip_events_to_window(phasic_all, w0, w1)
                    dur_phasic, phasic_ratio_epoch = phasic_time_ratio(phasic_in_win, w0, w1, sfreq)
                    very_phasic_epoch = (phasic_ratio_epoch > 0.75)

                    tonic_ratio_epoch, tonic_excluded_epoch = compute_tonic_ratio_epoch_ch(
                        env, sfreq, w0, w1, phasic_in_win, nrem_ref_start, nrem_ref_end, phasic_ratio_epoch
                    )
                    rswa_epoch = (not np.isnan(tonic_ratio_epoch)) and (tonic_ratio_epoch > 1.0)

                    rows.append({
                        "patient_id": patient_id,
                        "channel": ch_name,
                        "type": "REM_EPOCH_4S",
                        "episode_index": idx_pair,
                        "epoch_index": ep_k,
                        "epoch_len_sec": args.rem_epoch_len,
                        "epoch_start_sec": w0 / sfreq,
                        "epoch_end_sec":   w1 / sfreq,
                        "phasic_time_sec": dur_phasic,
                        "phasic_ratio": phasic_ratio_epoch,
                        "very_phasic": bool(very_phasic_epoch),
                        "tonic_ratio": tonic_ratio_epoch,
                        "tonic_excluded": bool(tonic_excluded_epoch),
                        "rswa": bool(rswa_epoch),
                        "phasic_count": len(phasic_in_win)
                    })

        return pd.DataFrame(rows)

    except Exception as e:
        return pd.DataFrame([{
            "patient_id": os.path.basename(patient_dir.rstrip(os.sep)),
            "type": "ERROR",
            "episode_index": -1,
            "error": f"{e} (rec={rec})"
        }])

# =========================
# 7) CLI & MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Detect phasic bursts & RSWA during REM on 4s epochs — per channel.")
    parser.add_argument("--input_dir", default=GP2_ROOT, help="Directory containing patient subfolders.")
    parser.add_argument("--output_csv", default="/home/darryld/Cerco_studies/data/rbd_emg_events_and_summary_4s_per_channel.csv",
                        help="Path to write the CSV (events + per-epoch summaries, per channel).")
    parser.add_argument("--emg_channels", nargs="*", default=None,
                        help="Optional list of EMG channel names to use (e.g., JAMBG JAMBD Menton EMG1 EMG2).")
    parser.add_argument("--n_jobs", type=int, default=20, help="Number of parallel workers.")
    parser.add_argument("--rem_epoch_len", type=float, default=4.0,
                        help="Longueur des époques REM en secondes (par défaut 4 s).")
    parser.add_argument("--min_phasic_dur", type=float, default=0.5,
                        help="Durée minimale (s) pour conserver un événement PHASIC (>= 0.5 s).")
    args = parser.parse_args()

    patients = find_patient_dirs(args.input_dir)
    if not patients:
        print("[ERREUR] Aucun dossier patient trouvé.", file=sys.stderr)
        sys.exit(1)

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
        out_df[out_df["type"]=="ERROR"][["patient_id","error"]].to_csv(err_path, index=False)
        print(f"[INFO] Des patients ont échoué. Détails: {err_path}")

if __name__ == "__main__":
    main()
