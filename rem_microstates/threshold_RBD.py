# === À coller dans UNE cellule notebook ===
from __future__ import annotations
import os, re, unicodedata, math, glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import mne
from scipy.signal import iirnotch, filtfilt, welch
from scipy.stats import shapiro
from collections import defaultdict
from pathlib import Path

# ===================== PARAMÈTRES GLOBAUX =====================
GP2_ROOT  = "/home/darryld/documents/EEG/preprocessed/bipolaire/1_noArtefacts/gp2/"
RAW_ROOT  = "/home/darryld/documents/EEG/raw/"
FIF_GLOB_PATTERNS = ["*_art_annotated.fif", "*.fif"]  # ordre d'essai
HYPNO_CANDIDATES  = ["{pid}_hypnoEXP.txt", "{pid}_hypnoEXP.csv", "{pid}_hypno.txt", "{pid}_hypnogram.txt"]

# Fenêtrage MAAV
WIN_LEN_S = 0.1
OVERLAP   = 0.5

# Tests de normalité (W de Shapiro)
W1_THR = 0.95  # sur signal (30 s)
W2_THR = 0.90  # sur distribution des MAAV (30 s)

# Seuil MAAV (RMS > FACTEUR * thr_MAAV)
MAAV_PCTL = 55.0
RMS_FACTOR = 2.0

# Critères menton
FREQ_MED_MAX_CHIN = 115.0   # Hz
ZCR_MIN_PER_S_CHIN = 140.0  # /s

# Heuristiques EMG membres
FREQ_MED_MAX_LIMB = 95.0    # Hz
ZCR_MIN_PER_S_LIMB = 120.0  # /s
USE_ADAPTIVE_ZCR = True
ZCR_PCTL = 60.0
ZCR_FLOOR = 110.0

# Hypnogramme & granularité
REM_EPOCH_SEC = 30.0
REM_KEYS = ("rem", "sleep stage r", "stage r", " r ", " r", "r ")

# Clustering
MAX_GAP_S = 0.50
DUTY_MIN  = 0.60
GLOBAL_MERGE_GAP_S = 30.0
MIN_SEG_DUR_S = 3.0

# Détection robuste 50 Hz
FIFTY_HZ_MAX_FRAC = 0.08
FIFTY_HZ_PROM_DB  = 5.0
FIFTY_HZ_BW = 1.0

DEBUG = False
N_WORKERS = 20

# ===================== UTILITAIRES COMMUNS =====================

# --- Chargement windows_by_patient depuis ../data/windows_by_patient.csv ---
def load_windows_by_patient_from_csv(base_file: str) -> dict:
    """
    base_file: __file__ du script .py (sert à résoudre le chemin relatif ../data)
    Retour: dict {patient_id: [ {start, end, duration(s), description, extrait/entier}, ... ]}
    """
    csv_path = (Path(base_file).resolve().parent / "../data/windows_by_patient.csv").resolve()
    d = defaultdict(list)
    try:
        if csv_path.is_file():
            df_wp = pd.read_csv(csv_path)
            # Colonnes attendues: patient_id, start, end, duration(s), description, extrait/entier
            for _, r in df_wp.iterrows():
                pid = str(r.get("patient_id", "")).strip()
                if not pid:
                    continue
                d[pid].append({
                    "start": float(r.get("start", float("nan"))),
                    "end": float(r.get("end", float("nan"))),
                    "duration(s)": float(r.get("duration(s)", float("nan"))),
                    "description": r.get("description", None),
                    "extrait/entier": r.get("extrait/entier", None),
                })
        else:
            print(f"windows_by_patient.csv introuvable: {csv_path}")
    except Exception as e:
        print(f"Erreur lecture windows_by_patient.csv: {e}")
    return d

WINDOWS_BY_PATIENT = load_windows_by_patient_from_csv(__file__)

def _strip_accents_lower(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c)).lower()

def _as_1d(x):
    x = np.asarray(x)
    return x.ravel()

def zero_crossing_rate(x, sfreq):
    x = _as_1d(x)
    s = np.sign(x); s[s == 0] = 1
    zc = np.sum(np.diff(s) != 0)
    dur = len(x)/sfreq
    return (zc/dur) if dur > 0 else 0.0

def median_frequency(x, sfreq):
    x = _as_1d(x)
    nper = min(max(int(sfreq), 256), len(x))  # ~1 s
    if len(x) < 2*nper:
        nper = max(128, len(x)//2)
    f, Pxx = welch(x, fs=sfreq, nperseg=max(64, nper))
    cs = np.cumsum(Pxx)
    if cs.size == 0 or cs[-1] <= 0:
        return np.nan
    half = cs[-1] / 2.0
    idx = int(np.searchsorted(cs, half))
    return float(f[min(idx, len(f)-1)])

def _bandpower(f, Pxx, fmin, fmax):
    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(Pxx[mask], f[mask]))

def has_50hz_harmonic(x, sfreq, base_band=(30.0, 100.0),
                      center=50.0, bw=FIFTY_HZ_BW,
                      harmonics=(1, 2),
                      max_frac=FIFTY_HZ_MAX_FRAC,
                      prom_db=FIFTY_HZ_PROM_DB):
    x = _as_1d(x)
    nper = min(max(int(sfreq), 512), len(x))
    f, Pxx = welch(x, fs=sfreq, nperseg=max(256, nper))
    baseP = _bandpower(f, Pxx, base_band[0], base_band[1]) + 1e-12
    for h in harmonics:
        fc = center * h
        p_band = _bandpower(f, Pxx, fc - bw, fc + bw)
        neigh_w = 3.0 * bw
        p_neigh = _bandpower(f, Pxx, fc - neigh_w, fc + neigh_w) - p_band
        hz_band = 2.0 * bw
        hz_neigh = max(1e-6, 2.0 * neigh_w - 2.0 * bw)
        p_neigh_per_hz = (p_neigh / hz_neigh)
        p_band_per_hz  = (p_band  / hz_band)
        prom = 10.0 * np.log10((p_band_per_hz + 1e-12) / (p_neigh_per_hz + 1e-12))
        frac = p_band / baseP
        if (frac > max_frac) and (prom > prom_db):
            return True
    return False

def rms(x):
    x = _as_1d(x).astype(float)
    return float(np.sqrt(np.mean(x**2))) if x.size else 0.0

# ===================== HYPNOGRAMME / REM =====================
def read_hypnogram_intervals_rem(path: str):
    """
    Lecture robuste des hypnogrammes hypnoEXP (.txt ou .csv) :
    - Supporte fichiers sans header (txt) et CSV (avec/sans header)
    - Détecte automatiquement la colonne temporelle (fin en secondes t_end_s
      ou début onset_sec/start_sec) et calcule la longueur d'epoch (mode des diffs)
    - Renvoie des intervalles REM fusionnés [(start_s, end_s), ...] et l'epoch_len
    """
    if path is None or not os.path.isfile(path):
        return [], 30.0

    # 1) Lecture robuste
    ext = Path(path).suffix.lower()
    df = None
    try:
        if ext == ".csv":
            # Essai 1: CSV "classique"
            df = pd.read_csv(path)
        if df is None or df.empty:
            # Essai 2: parser générique (séparateurs espaces ou virgules)
            df = pd.read_table(path, sep=r"[,\s]+", engine="python", header=None, comment="#")
    except Exception:
        # Dernier recours: lecture très permissive
        df = pd.read_table(path, sep=None, engine="python", header=None, comment="#")

    # 2) Normalisation colonnes -> tenter détection automatique
    cols_lower = {str(c).lower(): c for c in df.columns}

    # Stage column
    stage_col = None
    for key in ("stage", "stade", "sleep_stage", "label", "etat"):
        if key in cols_lower:
            stage_col = cols_lower[key]
            break
    if stage_col is None:
        # Sans header : souvent la 3e colonne (index 2) est le stage textuel, sinon dernière
        stage_col = df.columns[2] if len(df.columns) >= 3 else df.columns[-1]

    # Time column(s)
    end_pref = None
    for key in ("t_end_s", "end_sec", "end_s", "fin_s", "tfin_s"):
        if key in cols_lower:
            end_pref = cols_lower[key]
            break
    start_pref = None
    for key in ("onset_sec", "start_sec", "debut_sec", "t_start_s", "start_s"):
        if key in cols_lower:
            start_pref = cols_lower[key]
            break

    # Si pas de colonnes nommées, on suppose : col0 = temps (fin), col2 = stage
    if end_pref is None and start_pref is None:
        # Choisir la première colonne numérique comme temps de référence
        cand = df.columns[0]
        end_pref = cand

    # 3) Construire une table normalisée (t_start, t_end, stage)
    tmp = df[[end_pref] if end_pref is not None else [start_pref]].copy()
    tmp.columns = ["t_ref"]
    tmp["stage"] = df[stage_col].astype(str).str.strip().str.upper()
    tmp = tmp.dropna(subset=["t_ref", "stage"]).sort_values("t_ref")

    # 4) Déterminer epoch_len via la mode des diffs arrondies (en s)
    te = tmp["t_ref"].to_numpy(dtype=float)
    diffs = np.diff(te) if len(te) > 1 else np.array([])
    if diffs.size:
        # robustifier : arrondir à 0.1 s, puis mode
        epoch_len = float(pd.Series(np.round(diffs, 1)).mode().iloc[0])
        if not np.isfinite(epoch_len) or epoch_len <= 0:
            epoch_len = 30.0
    else:
        epoch_len = 30.0

    # 5) Si t_ref est une "fin", calculer t_start = t_end - epoch_len
    #    Si t_ref est un "début", calculer t_end = t_start + epoch_len
    #    Heuristique : si on a explicitement une colonne "start" détectée, considérer t_ref comme début
    is_start_ref = (start_pref is not None) and (end_pref is None)
    if is_start_ref:
        t_start = te
        t_end = te + epoch_len
    else:
        t_end = te
        t_start = te - epoch_len

    # 6) Construire et fusionner les runs REM
    is_rem = tmp["stage"].isin(["R", "REM"])
    rem_intervals, run_start, prev_end = [], None, None
    for s0, e1, flag in zip(t_start, t_end, is_rem):
        if flag:
            if run_start is None:
                run_start = float(s0)
            prev_end = float(e1)
        else:
            if run_start is not None:
                rem_intervals.append((float(run_start), float(prev_end)))
                run_start, prev_end = None, None
    if run_start is not None and prev_end is not None:
        rem_intervals.append((float(run_start), float(prev_end)))

    return rem_intervals, epoch_len

def get_rem_intervals(raw: mne.io.BaseRaw,
                      hypno_path: str | None = None,
                      hypno_offset_s: float = 0.0,
                      rem_keys=REM_KEYS):
    # 1) Hypnogramme
    rem_intervals, epoch_len = read_hypnogram_intervals_rem(hypno_path)
    if rem_intervals:
        if hypno_offset_s:
            rem_intervals = [(a + hypno_offset_s, b + hypno_offset_s) for (a, b) in rem_intervals]
        return rem_intervals, epoch_len, "hypnogram"
    # 2) Fallback: annotations MNE
    rems = []
    if raw.annotations is not None and len(raw.annotations) > 0:
        for desc, onset, dur in zip(raw.annotations.description,
                                    raw.annotations.onset,
                                    raw.annotations.duration):
            d = _strip_accents_lower(desc)
            if any(k in d for k in rem_keys):
                rems.append((float(onset), float(onset + dur)))
        if rems:
            rems.sort()
            merged, cur_s, cur_e = [], rems[0][0], rems[0][1]
            for s, e in rems[1:]:
                if s <= cur_e: cur_e = max(cur_e, e)
                else:
                    merged.append((cur_s, cur_e))
                    cur_s, cur_e = s, e
            merged.append((cur_s, cur_e))
            return merged, 30.0, "annotations"
    # 3) Full record
    return [(0.0, raw.times[-1])], 30.0, "full_record"

def split_into_epochs(intervals, epoch_len=REM_EPOCH_SEC):
    out = []
    for (s, e) in intervals:
        t = s
        while t + epoch_len <= e + 1e-9:
            out.append((t, t + epoch_len))
            t += epoch_len
    return out

def is_chin_channel(name: str) -> bool:
    n = _strip_accents_lower(name)
    return any(k in n for k in ("menton", "subment", "chin", "mentalis", "masseter"))

# ===================== TYPAGE & PRÉTRAITEMENT EMG =====================
def _guess_set_eog_types(raw: mne.io.BaseRaw):
    current_eog = mne.pick_types(raw.info, eog=True)
    if len(current_eog) > 0: return
    eog_like = []
    for ch in raw.ch_names:
        n = _strip_accents_lower(ch)
        if any(k in n for k in ("eog", "heog", "veog", "loc", "roc", "eo", "oe")):
            eog_like.append(ch)
    if eog_like:
        raw.set_channel_types({ch: 'eog' for ch in eog_like})

def _guess_set_emg_types(raw: mne.io.BaseRaw):
    current_emg = mne.pick_types(raw.info, emg=True)
    if len(current_emg) > 0: return
    emg_like = []
    for ch in raw.ch_names:
        n = _strip_accents_lower(ch)
        if any(k in n for k in ("emg", "menton", "subment", "chin", "masseter",
                                "tib", "jambe", "jamb", "ant", "fcr", "fds", "ext", "flex")):
            emg_like.append(ch)
    if emg_like:
        raw.set_channel_types({ch: 'emg' for ch in emg_like})

def is_chin_channel(ch_name: str) -> bool:
    n = _strip_accents_lower(ch_name)
    return any(k in n for k in ("menton", "subment", "chin", "mentalis", "masseter"))

def bandpass_eeg_eog(raw: mne.io.BaseRaw, eog_band=(0.3, 15.0), eeg_band=(0.3, 100.0)):
    if not raw.preload:
        raw.load_data()
    _guess_set_eog_types(raw)
    _guess_set_emg_types(raw)
    picks_eog = mne.pick_types(raw.info, eog=True, meg=False, eeg=False, stim=False, misc=False)
    if len(picks_eog) > 0:
        raw.filter(l_freq=eog_band[0], h_freq=eog_band[1], picks=picks_eog, method="iir", verbose=False)
    picks_eeg = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, stim=False, misc=False)
    if len(picks_eeg) > 0:
        raw.filter(l_freq=eeg_band[0], h_freq=eeg_band[1], picks=picks_eeg, method="iir", verbose=False)

def preprocess_emg_channel(raw: mne.io.BaseRaw, pick_idx: int,
                           hp=30.0, lp=100.0, notch_freq=50.0, notch_q=30.0,
                           resample_hz=500.0):
    # garde-fou
    if not (0 <= pick_idx < len(raw.ch_names)):
        raise IndexError(f"pick_idx hors bornes: {pick_idx} / {len(raw.ch_names)}")

    raw1 = raw.copy()
    if not raw1.preload:
        raw1.load_data()

    ch_name = raw1.ch_names[pick_idx]

    # force le type EMG si besoin
    if raw1.get_channel_types(picks=[pick_idx])[0] != 'emg':
        raw1.set_channel_types({ch_name: 'emg'})

    # extraction mono-canal
    raw_m = raw1.copy().pick(picks=[pick_idx])

    # 30–100 Hz (IIR)
    raw_m.filter(l_freq=hp, h_freq=lp, method="iir", picks=[0], verbose=False)

    # Notch 50 Hz (IIR + filtfilt)
    fs = raw_m.info["sfreq"]
    b, a = iirnotch(w0=notch_freq/(fs/2), Q=notch_q)
    data = raw_m.get_data()
    data = filtfilt(b, a, data, axis=1)
    raw_m._data[:] = data

    # Resample à 500 Hz uniquement pour le menton (match robuste)
    if resample_hz is not None and abs(raw_m.info["sfreq"] - resample_hz) > 1e-6 and is_chin_channel(ch_name):
        raw_m.resample(resample_hz)

    return raw_m, ch_name

# ===================== MAAV / TESTS / CLUSTERS =====================
def compute_maav(x, sfreq, win_len_s=WIN_LEN_S, overlap=OVERLAP):
    x = _as_1d(x); n = len(x)
    L = int(round(win_len_s * sfreq))
    H = int(round(L * (1 - overlap)))
    if L <= 0 or H <= 0:
        raise ValueError("Fenêtrage invalide.")
    starts = np.arange(0, max(n - L + 1, 0), H, dtype=int)
    mav, tcent, idxs = [], [], []
    for s0 in starts:
        s1 = s0 + L
        if s1 > n: break
        w = x[s0:s1]
        mav.append(np.mean(np.abs(w)))
        tcent.append((s0 + s1)/(2*sfreq))
        idxs.append((s0, s1))
    return np.asarray(mav), np.asarray(tcent), idxs, L, H

def shapiro_tests_epoch(x_filt, maav_values, W1_thr=W1_THR, W2_thr=W2_THR):
    W1 = shapiro(_as_1d(x_filt))[0] if len(x_filt) >= 3 else 0.0
    W2 = shapiro(_as_1d(maav_values))[0] if len(maav_values) >= 3 else 0.0
    # poursuivre seulement si W1 < seuil ET W2 < seuil
    active = (W1 < W1_thr) and (W2 < W2_thr)
    return active, float(W1), float(W2)

def zcr_series_from_windows(x, sfreq, idx_windows):
    vals = []
    for (s0, s1) in idx_windows:
        vals.append(zero_crossing_rate(x[s0:s1], sfreq))
    return np.array(vals, float)

def _clusters_with_allowed_gaps(above_mask: np.ndarray, H_sec: float, max_gap_s: float):
    idx_active = np.where(above_mask > 0)[0]
    if idx_active.size == 0: return []
    clusters = []
    c_start = idx_active[0]
    c_actives = [idx_active[0]]
    last_active = idx_active[0]
    for a in idx_active[1:]:
        gap_points = a - last_active - 1
        gap_time = gap_points * H_sec
        if gap_time <= max_gap_s:
            last_active = a
            c_actives.append(a)
        else:
            clusters.append((c_start, last_active, c_actives.copy()))
            c_start = a
            last_active = a
            c_actives = [a]
    clusters.append((c_start, last_active, c_actives.copy()))
    return clusters

def detect_validate_segments(x_filt, sfreq,
                             maav_values, idx_windows,
                             freq_median_max,
                             zcr_min_per_s,
                             rms_factor_vs_thr,
                             active_strategy="percentile",
                             maav_pct=MAAV_PCTL,
                             fft_harmonics=True,
                             apply_feature_rules=True):
    """
    active_strategy:
        - "percentile": thr = percentile(maav, maav_pct); above = maav > thr  (menton)
        - "twice_min":  thr = 2 * min(maav);              above = maav > thr  (membres)
    """
    mv = _as_1d(maav_values)
    debug_rows = []
    valid_segments = []
    if mv.size == 0:
        return valid_segments, 0.0, debug_rows

    L_s = WIN_LEN_S
    H_s = WIN_LEN_S * (1.0 - OVERLAP)

    thr = (3.0 * float(np.min(mv))) if (active_strategy == "twice_min") else float(np.percentile(mv, maav_pct))
    thr_src = "3x_min" if (active_strategy == "twice_min") else f"pct{maav_pct:g}"

    above = (mv > thr).astype(int)
    clusters = _clusters_with_allowed_gaps(above, H_s, MAX_GAP_S)

    for (i_start, i_end, active_idx) in clusters:
        span = (i_end - i_start) * H_s + L_s
        n_active = len(active_idx)
        coverage = n_active * H_s + (L_s if n_active > 0 else 0.0)
        duty = coverage / span if span > 0 else 0.0

        if n_active <= 1:
            max_gap = span - L_s
        else:
            diffs = np.diff(sorted(active_idx))
            max_gap = float(np.max((diffs - 1) * H_s))

        s0 = idx_windows[i_start][0]
        s1 = idx_windows[i_end][1]
        seg = x_filt[s0:s1]
        dur = (s1 - s0) / sfreq if s1 > s0 else 0.0

        mf  = median_frequency(seg, sfreq)
        r   = rms(seg)
        zcr = zero_crossing_rate(seg, sfreq)
        fft_bad = has_50hz_harmonic(seg, sfreq) if fft_harmonics else False

        span_ok  = (span >= MIN_SEG_DUR_S)
        duty_ok  = (duty >= DUTY_MIN)
        gap_ok   = (max_gap <= MAX_GAP_S)

        if apply_feature_rules:
            mf_ok    = (mf < freq_median_max)
            rms_ok   = (r  > rms_factor_vs_thr * thr)
            zcr_ok   = (zcr > zcr_min_per_s)
            notch_ok = (not fft_bad)
            accepted = (span_ok and duty_ok and gap_ok and mf_ok and rms_ok and zcr_ok and notch_ok and (dur > 0))
        else:
            mf_ok = rms_ok = zcr_ok = True
            notch_ok = True
            accepted = (span_ok and duty_ok and gap_ok and (dur > 0))

        debug_rows.append({
            "i_start": i_start, "i_end": i_end,
            "span_s": round(span, 3), "coverage_s": round(coverage, 3),
            "duty": round(duty, 3), "max_gap_s": round(max_gap, 3), "n_active": int(n_active),
            "maav_thr_uV": round(thr*1e6, 3), "thr_source": thr_src,
            "dur_s": round(dur, 3),
            "mf_Hz": round(float(mf), 3) if not np.isnan(mf) else np.nan,
            "rms_uV": round(r*1e6, 3), "zcr_per_s": round(float(zcr), 2),
            "notch_50Hz_fail": bool(fft_bad),
            "span_ok": bool(span_ok), "duty_ok": bool(duty_ok), "gap_ok": bool(gap_ok),
            "mf_ok": bool(mf_ok), "rms_ok": bool(rms_ok), "zcr_ok": bool(zcr_ok), "notch_ok": bool(notch_ok),
            "accepted": bool(accepted)
        })

        if accepted:
            valid_segments.append({
                "t0_rel": s0/sfreq, "t1_rel": s1/sfreq, "dur": dur,
                "median_freq": mf, "rms": r, "zcr": zcr, "thr_maav": thr, "passes": True
            })

    return valid_segments, float(thr), debug_rows

def merge_events_global(all_seg_rows, gap_s=GLOBAL_MERGE_GAP_S):
    if not all_seg_rows: return []
    rows = sorted(all_seg_rows, key=lambda r: r["onset_s"])
    merged, cur = [], None
    for r in rows:
        if cur is None:
            cur = {
                "onset_s": r["onset_s"], "end_s": r["end_s"],
                "duration_s": r["end_s"] - r["onset_s"],
                "epoch_start_s": r["epoch_start_s"], "epoch_end_s": r["epoch_end_s"],
                "channels_involved": {r["channel"]},
                "median_freq_max": r["median_freq"], "rms_uV_max": r["rms_uV"], "zcr_per_s_max": r["zcr_per_s"],
                "W1_epoch_mean": r["W1_epoch"], "W2_epoch_mean": r["W2_epoch"],
                "thr_maav_epoch_uV_mean": r["thr_maav_epoch_uV"], "_count": 1,
            }
            continue
        gap = r["onset_s"] - cur["end_s"]
        if gap < gap_s:
            cur["end_s"] = max(cur["end_s"], r["end_s"])
            cur["duration_s"] = cur["end_s"] - cur["onset_s"]
            cur["epoch_start_s"] = min(cur["epoch_start_s"], r["epoch_start_s"])
            cur["epoch_end_s"] = max(cur["epoch_end_s"], r["epoch_end_s"])
            cur["channels_involved"].add(r["channel"])
            cur["median_freq_max"] = max(cur["median_freq_max"], r["median_freq"])
            cur["rms_uV_max"] = max(cur["rms_uV_max"], r["rms_uV"])
            cur["zcr_per_s_max"] = max(cur["zcr_per_s_max"], r["zcr_per_s"])
            cur["W1_epoch_mean"] += r["W1_epoch"]
            cur["W2_epoch_mean"] += r["W2_epoch"]
            cur["thr_maav_epoch_uV_mean"] += r["thr_maav_epoch_uV"]
            cur["_count"] += 1
        else:
            merged.append(cur)
            cur = {
                "onset_s": r["onset_s"], "end_s": r["end_s"],
                "duration_s": r["end_s"] - r["onset_s"],
                "epoch_start_s": r["epoch_start_s"], "epoch_end_s": r["epoch_end_s"],
                "channels_involved": {r["channel"]},
                "median_freq_max": r["median_freq"], "rms_uV_max": r["rms_uV"], "zcr_per_s_max": r["zcr_per_s"],
                "W1_epoch_mean": r["W1_epoch"], "W2_epoch_mean": r["W2_epoch"],
                "thr_maav_epoch_uV_mean": r["thr_maav_epoch_uV"], "_count": 1,
            }
    if cur is not None: merged.append(cur)

    out = []
    for i, m in enumerate(merged, 1):
        cnt = max(1, m.pop("_count"))
        m["W1_epoch_mean"] = round(m["W1_epoch_mean"] / cnt, 4)
        m["W2_epoch_mean"] = round(m["W2_epoch_mean"] / cnt, 4)
        m["thr_maav_epoch_uV_mean"] = round(m["thr_maav_epoch_uV_mean"] / cnt, 3)
        m["median_freq_max"] = round(m["median_freq_max"], 3)
        m["rms_uV_max"] = round(m["rms_uV_max"], 3)
        m["zcr_per_s_max"] = round(m["zcr_per_s_max"], 2)
        m["channels_involved"] = ",".join(sorted(m["channels_involved"]))
        m["name"] = f"RBD_{i}"
        out.append(m)
    return out

# ===================== PROCESSUS PAR PATIENT =====================
def find_patient_fif(patient_dir: str) -> str | None:
    for pat in FIF_GLOB_PATTERNS:
        matches = glob.glob(str(Path(patient_dir) / pat))
        if matches:
            # On préfère un fichier qui matche {pid}_*.fif si possible
            return sorted(matches)[0]
    return None

def find_hypnogram_path(pid: str) -> str | None:
    raw_dir = Path(RAW_ROOT) / pid
    for tpl in HYPNO_CANDIDATES:
        cand = raw_dir / tpl.format(pid=pid)
        if cand.exists():
            return str(cand)
    # dernier recours : 1er *.txt du dossier raw
    txts = sorted(raw_dir.glob("*.txt"))
    return str(txts[0]) if txts else None

def analyze_patient(patient_dir: str) -> dict:
    pid = Path(patient_dir).name
    fif_path = find_patient_fif(patient_dir)
    if not fif_path:
        return {"patient": pid, "status": "no_fif", "csv": None, "n_events": 0, "mean_dur_s": 0.0}

    out_csv = str(Path(patient_dir) / f"{pid}_RBD_windows.csv")
    dbg_csv = str(Path(patient_dir) / f"{pid}_RBD_windows_debug.csv")

    raw0 = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    bandpass_eeg_eog(raw0, eog_band=(0.3, 15.0), eeg_band=(0.3, 100.0))
    _guess_set_emg_types(raw0)

    emg_picks = mne.pick_types(raw0.info, emg=True, meg=False, eeg=False, eog=False, stim=False, misc=False)
    if len(emg_picks) == 0:
        return {"patient": pid, "status": "no_emg", "csv": None, "n_events": 0, "mean_dur_s": 0.0}

    hypno = find_hypnogram_path(pid)
    rem_intervals, hyp_epoch_len, source = get_rem_intervals(raw0, hypno_path=hypno, hypno_offset_s=0.0, rem_keys=REM_KEYS)

    rem_epochs = split_into_epochs(rem_intervals, epoch_len=REM_EPOCH_SEC)
    if len(rem_epochs) == 0:
        return {"patient": pid, "status": "no_rem_epochs", "csv": None, "n_events": 0, "mean_dur_s": 0.0}

    all_rows = []
    debug_rows_all = []

    for p in emg_picks:
        raw_emg, ch_name = preprocess_emg_channel(raw0, p, hp=30.0, lp=100.0, notch_freq=50.0, notch_q=30.0, resample_hz=500.0)
        sf = raw_emg.info["sfreq"]
        chin_mode = is_chin_channel(ch_name)
        active_strategy = "percentile" if chin_mode else "twice_min"
        freq_limit = FREQ_MED_MAX_CHIN if chin_mode else FREQ_MED_MAX_LIMB

        for (t0, t1) in rem_epochs:
            seg = raw_emg.copy().crop(tmin=t0, tmax=t1, include_tmax=False)
            x = seg.get_data()[0]

            maav, tcent, idx_windows, L, H = compute_maav(x, sf, WIN_LEN_S, OVERLAP)

            proceed, W1, W2 = shapiro_tests_epoch(x, maav, W1_THR, W2_THR)
            if not proceed:
                if DEBUG: print(f"[{pid}/{ch_name}] {t0:.1f}-{t1:.1f}s: W1={W1:.3f}, W2={W2:.3f} -> skip")
                continue

            zcr_thr_epoch = ZCR_MIN_PER_S_CHIN if chin_mode else ZCR_MIN_PER_S_LIMB
            if USE_ADAPTIVE_ZCR:
                zcr_win = [zero_crossing_rate(x[s0:s1], sf) for (s0, s1) in idx_windows]
                if len(zcr_win):
                    zcr_thr_epoch = max(ZCR_FLOOR, float(np.percentile(zcr_win, ZCR_PCTL)))

            segments, thr, debug_rows = detect_validate_segments(
                x_filt=x, sfreq=sf,
                maav_values=maav, idx_windows=idx_windows,
                freq_median_max=freq_limit, zcr_min_per_s=zcr_thr_epoch,
                rms_factor_vs_thr=RMS_FACTOR,
                active_strategy=active_strategy, maav_pct=MAAV_PCTL,
                fft_harmonics=True, apply_feature_rules=chin_mode
            )

            for row in debug_rows:
                row.update({
                    "patient": pid, "channel": ch_name,
                    "epoch_start_s": round(t0, 3), "epoch_end_s": round(t1, 3),
                    "W1_epoch": round(W1, 4), "W2_epoch": round(W2, 4)
                })
            debug_rows_all.extend(debug_rows)

            for s in segments:
                onset_abs = t0 + s["t0_rel"]
                dur = s["dur"]
                rms_uV = 1e6 * float(s["rms"])
                thr_uV = 1e6 * float(s["thr_maav"])
                all_rows.append({
                    "patient": pid,
                    "onset_s": round(onset_abs, 3),
                    "end_s": round(onset_abs + dur, 3),
                    "duration_s": round(dur, 3),
                    "epoch_start_s": round(t0, 3),
                    "epoch_end_s": round(t1, 3),
                    "channel": ch_name,
                    "W1_epoch": round(W1, 4),
                    "W2_epoch": round(W2, 4),
                    "thr_maav_epoch_uV": round(thr_uV, 3),
                    "median_freq": round(float(s["median_freq"]), 3),
                    "rms_uV": round(rms_uV, 3),
                    "zcr_per_s": round(float(s["zcr"]), 2),
                })

    merged_events = merge_events_global(all_rows, gap_s=GLOBAL_MERGE_GAP_S)

    # Export principal
    df = pd.DataFrame(merged_events, columns=[
        "name","onset_s","end_s","duration_s",
        "epoch_start_s","epoch_end_s",
        "channels_involved",
        "W1_epoch_mean","W2_epoch_mean","thr_maav_epoch_uV_mean",
        "median_freq_max","rms_uV_max","zcr_per_s_max"
    ])
    df.to_csv(out_csv, index=False)

    # Export debug (clusters)
    if len(debug_rows_all) > 0:
        df_dbg = pd.DataFrame(debug_rows_all, columns=[
            "patient","channel","epoch_start_s","epoch_end_s","W1_epoch","W2_epoch",
            "i_start","i_end","span_s","coverage_s","duty","max_gap_s","n_active",
            "maav_thr_uV","thr_source","dur_s","mf_Hz","rms_uV","zcr_per_s",
            "notch_50Hz_fail","span_ok","duty_ok","gap_ok","mf_ok","rms_ok","zcr_ok","notch_ok","accepted"
        ])
        df_dbg.to_csv(dbg_csv, index=False)

    n_events = len(df)
    mean_dur = float(df["duration_s"].mean()) if n_events > 0 else 0.0

    # Info depuis le CSV ../data/windows_by_patient.csv
    n_manual = len(WINDOWS_BY_PATIENT.get(pid, []))
    info = f" >>> manual_windows={n_manual}"

    print(f"[{pid}] events={n_events} mean_dur={mean_dur:.2f}s -> {out_csv}{info}")
    return {"patient": pid, "status": "ok", "csv": out_csv, "n_events": n_events, "mean_dur_s": mean_dur}

# ===================== LANCEUR PARALLÉLISÉ =====================
def run_all_patients(gp2_root=GP2_ROOT):
    if not os.path.isdir(gp2_root):
        raise FileNotFoundError(f"GP2 root introuvable: {gp2_root}")

    patient_dirs = [str(p) for p in Path(gp2_root).iterdir() if p.is_dir()]
    patient_dirs.sort()
    if not patient_dirs:
        raise RuntimeError("Aucun dossier patient trouvé.")

    print(f"Patients à traiter: {len(patient_dirs)} | workers={N_WORKERS}")

    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        fut2pid = {ex.submit(analyze_patient, d): Path(d).name for d in patient_dirs}
        for fut in as_completed(fut2pid):
            pid = fut2pid[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"[{pid}] ERREUR: {e}")
                res = {"patient": pid, "status": f"error:{e}", "csv": None, "n_events": 0, "mean_dur_s": 0.0}
            results.append(res)

    # Index global
    df_idx = pd.DataFrame(results, columns=["patient","status","csv","n_events","mean_dur_s"])
    idx_path = str(Path(gp2_root) / "RBD_windows_INDEX.csv")
    df_idx.sort_values("patient").to_csv(idx_path, index=False)
    print(f"\n Index global écrit: {idx_path}")
    print(df_idx["status"].value_counts(dropna=False))
    return idx_path, df_idx

# ===================== EXÉCUTION =====================
index_path, df_index = run_all_patients()
index_path
