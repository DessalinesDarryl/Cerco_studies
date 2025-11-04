#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect RBD markers (PHASIC bursts & TONIC RSWA) from EMG during REM sleep,
suivant Grenot et al., SLEEP 2024 (zsae178).

- Filtre passe-bande EMG 30–100 Hz
- Rectification + enveloppe lissée (moyenne glissante 1 s)
- Phasic : >95e percentile des 30 s de NREM précédents, durée >= 0,5 s
- Tonic : ratio médian (REM sans phasic) / (NREM nettoyé) ; RSWA si ratio > 1
- Sortie CSV : une ligne par bouffée phasique et par épisode REM avec RSWA.

Usage:
  python detect_rbd_emg.py --input_dir /path/patients --output_csv /path/out.csv --n_jobs 4
  (optionnel) --emg_channels EMG1 EMG2

Structure d'entrée (exemple):
  /patients/PATIENT_ID/recording.edf|.fif + RAW_ROOT/PATIENT_ID/{pid}_hypnoEXP.txt|.csv
"""
import argparse
import os
import sys
import glob
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

import mne

# --- Emplacements & motifs de fichiers
GP2_ROOT  = "/home/darryld/documents/EEG/preprocessed/bipolaire/1_noArtefacts/gp2/"
RAW_ROOT  = "/home/darryld/documents/EEG/raw/"
FIF_GLOB_PATTERNS = ["*_art_annotated.fif", "*.fif"]  # ordre d'essai
HYPNO_CANDIDATES  = ["{pid}_hypnoEXP.txt", "{pid}_hypnoEXP.csv", "{pid}_hypno.txt", "{pid}_hypnogram.txt"]

STAGE_MAP = {
    "W": "W",
    "N1": "NREM",
    "N2": "NREM",
    "N3": "NREM",
    "NREM": "NREM",
    "REM": "REM",
    "R": "REM",
    "S1": "NREM",
    "S2": "NREM",
    "S3": "NREM",
    "S4": "NREM",
    "V": "W",   # "Vigilance" -> Wake
    "1": "NREM",
    "2": "NREM",
    "3": "NREM",
    "4": "NREM",
}

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
    print(f"Found {len(subs)} patient directories in {input_dir}.")
    return subs

def find_recording_file(patient_dir: str) -> Optional[str]:
    """
    Cherche d'abord dans le dossier patient les FIF selon FIF_GLOB_PATTERNS (ordre).
    Retombe ensuite sur les extensions standard si rien n'est trouvé.
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
    """
    Hypnogrammes stockés dans RAW_ROOT/{pid}/ avec HYPNO_CANDIDATES.
    En fallback, cherche localement dans patient_dir.
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

# ---------- PARSE des hypnogrammes (TES 2 FORMATS) ----------
def _norm_stage_to_rem_nrem_w(s: str) -> str:
    s = str(s).strip().upper()
    return STAGE_MAP.get(s, s)

def _aggregate_epochs_to_episodes(rows: List[Tuple[float, float, str]]) -> List[Episode]:
    """
    rows: liste de tuples (start_sec, end_sec, stage_code_original)
    Agrège les epochs consécutifs de même stage en Episodes(stage=W/NREM/REM).
    """
    if not rows:
        return []
    # normaliser et trier
    rows = [(float(s), float(e), _norm_stage_to_rem_nrem_w(stg)) for (s, e, stg) in rows]
    rows.sort(key=lambda x: (x[0], x[1]))
    episodes: List[Episode] = []
    cur_stage, cur_s, cur_e = None, None, None
    for s, e, stg in rows:
        if cur_stage is None:
            cur_stage, cur_s, cur_e = stg, s, e
            continue
        # Continuité temporelle (tolérance 1 s) et même stage
        if stg == cur_stage and abs(s - cur_e) <= 1.0:
            cur_e = max(cur_e, e)
        else:
            episodes.append(Episode(stage=cur_stage, onset=float(cur_s), duration=max(0.0, float(cur_e - cur_s))))
            cur_stage, cur_s, cur_e = stg, s, e
    if cur_stage is not None:
        episodes.append(Episode(stage=cur_stage, onset=float(cur_s), duration=max(0.0, float(cur_e - cur_s))))
    return episodes

def _read_txt_hypnoexp(path: str) -> List[Episode]:
    """
    TXT hypnoEXP: lignes "end_sec\tHH:MM:SS\tSTAGE\tCODE"
    On utilise end_sec, on dérive start_sec = end_sec - epoch_len, epoch_len par mode(diff(end_sec)).
    """
    df = pd.read_csv(
        path, sep=r"\s+", engine="python", header=None,
        names=["end_sec", "hhmmss", "stage", "code"],
        usecols=[0, 1, 2, 3]
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
    """
    CSV à 3 colonnes :
      0: "position (epoch)"
      1: "absolute position (hh:mm:ss.ms)"
      2: 'Default Staging Set ("stage")'
    """
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 3:
        return []
    # retirer l'en-tête si présent
    header_like = str(df.iloc[0, 0]).strip().lower()
    if "position" in header_like and "epoch" in header_like:
        df = df.iloc[1:, :]
    df = df.iloc[:, :3].copy()
    df.columns = ["epoch", "abs_time", "stage"]
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["stage"] = df["stage"].astype(str)
    df = df.dropna(subset=["epoch", "stage"]).sort_values("epoch")
    # estimer epoch_len depuis l'horloge (unwrap 24h), sinon 30 s
    def _time_to_sec(tstr: str):
        try:
            h, m, s = tstr.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
        except Exception:
            return np.nan
    times = df["abs_time"].astype(str).map(_time_to_sec).to_numpy()
    if np.isfinite(times).all():
        unwrapped = []
        shift = 0.0
        last = None
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
    # construire lignes (start,end,stage) à partir de epoch index
    rows = []
    for ep, stg in zip(df["epoch"].to_numpy(dtype=int), df["stage"].tolist()):
        end_sec = ep * epoch_len
        rows.append((float(end_sec - epoch_len), float(end_sec), stg))
    return _aggregate_epochs_to_episodes(rows)

def parse_hypnogram_table(path: str) -> List[Episode]:
    """
    Routeur de parsing :
      - *_hypnoEXP.txt  -> format 4 colonnes (end_sec ... stage ...)
      - *_hypnoEXP.csv  -> format 3 colonnes (epoch, abs_time, stage)
      - sinon           -> fallback: attend onset_sec/duration_sec/stage
    Retour: liste d'Episode(stage in {W,NREM,REM})
    """
    if path is None or not os.path.isfile(path):
        return []
    base = os.path.basename(path).lower()
    ext = os.path.splitext(base)[1]
    try:
        if base.endswith("_hypnoexp.txt") or ext == ".txt":
            return _read_txt_hypnoexp(path)
        if base.endswith("_hypnoexp.csv") or ext == ".csv":
            # vérifier si c'est un CSV 3 colonnes (ton format). Si échec -> fallback générique
            try:
                eps = _read_csv_3cols(path)
                if eps:
                    return eps
            except Exception:
                pass
        # --- Fallback générique (onset/duration/stage) ---
        df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_table(path)
        cols = {c.lower(): c for c in df.columns}
        onset_col = cols.get("onset_sec") or cols.get("onset") or cols.get("start_sec") or cols.get("start")
        dur_col = cols.get("duration_sec") or cols.get("duration") or cols.get("dur_sec") or cols.get("dur")
        stage_col = cols.get("stage") or cols.get("label") or cols.get("stade") or cols.get("sleep_stage")
        if onset_col is None or dur_col is None or stage_col is None:
            # Rien de conforme trouvé
            return []
        rows = []
        for _, row in df.iterrows():
            stage_m = _norm_stage_to_rem_nrem_w(str(row[stage_col]))
            rows.append((float(row[onset_col]), float(row[onset_col]) + float(row[dur_col]), stage_m))
        return _aggregate_epochs_to_episodes(rows)
    except Exception:
        return []

# ---------- Fallback: annotations MNE ----------
def parse_hypnogram_from_raw(raw: mne.io.BaseRaw) -> List[Episode]:
    episodes: List[Episode] = []
    if raw.annotations is None or len(raw.annotations) == 0:
        return episodes
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
        elif "W" == desc or "WAKE" in desc:
            stage = "W"
        if stage is None:
            continue
        episodes.append(Episode(stage=stage, onset=float(ann["onset"]), duration=float(ann["duration"])))
    return _aggregate_epochs_to_episodes([(ep.onset, ep.end, ep.stage) for ep in episodes])

# ---------- Traitement EMG ----------
def moving_average(x: np.ndarray, win_samps: int) -> np.ndarray:
    if win_samps <= 1:
        return x.copy()
    kernel = np.ones(win_samps, dtype=float) / win_samps
    pad = win_samps // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    y = np.convolve(x_pad, kernel, mode="valid")
    return y[:len(x)]

def detect_phasic_events(envelope: np.ndarray, sfreq: float, rem_start: int, rem_end: int, nrem_ref_start: int, nrem_ref_end: int) -> List[Tuple[int, int]]:
    """Bouffées phasiques dans le REM: segments >=0.5 s au-dessus du 95e percentile (NREM précédent)."""
    nrem_ref = envelope[nrem_ref_start:nrem_ref_end]
    if len(nrem_ref) == 0:
        return []
    thr = np.percentile(nrem_ref, 95.0)
    rem_env = envelope[rem_start:rem_end]
    above = rem_env > thr
    events = []
    i = 0
    min_len = int(np.ceil(0.5 * sfreq))
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
    return events

def compute_tonic_ratio(envelope: np.ndarray, sfreq: float, rem_start: int, rem_end: int, phasic_events: List[Tuple[int, int]], nrem_ref_start: int, nrem_ref_end: int) -> float:
    """Ratio médian REM_sans_phasic / NREM_nettoyé(<=95e). RSWA si > 1."""
    rem_env = envelope[rem_start:rem_end].copy()
    mask = np.zeros(len(rem_env), dtype=bool)
    for s, e in phasic_events:
        s_rel = max(0, s - rem_start)
        e_rel = min(len(rem_env), e - rem_start)
        if s_rel < e_rel:
            mask[s_rel:e_rel] = True
    rem_clean = rem_env[~mask]
    if len(rem_clean) == 0:
        rem_clean = rem_env

    nrem_ref = envelope[nrem_ref_start:nrem_ref_end]
    if len(nrem_ref) == 0:
        return np.nan
    thr_ref = np.percentile(nrem_ref, 95.0)
    nrem_clean = nrem_ref[nrem_ref <= thr_ref]
    if len(nrem_clean) < int(0.5 * sfreq):
        nrem_clean = nrem_ref

    rem_med = np.median(rem_clean)
    nrem_med = np.median(nrem_clean)
    if nrem_med == 0:
        return np.nan
    return float(rem_med / nrem_med)

def load_raw_with_emg(path: str, emg_channels: Optional[List[str]]):
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
        picks = [ch for ch in raw.ch_names if "emg" in ch.lower()]
    if not picks:
        picks_idx = mne.pick_types(raw.info, emg=True, misc=False, ref_meg=False, eeg=False, meg=False, stim=False, eog=False)
        if isinstance(picks_idx, np.ndarray) and len(picks_idx):
            picks = [raw.ch_names[i] for i in picks_idx]
    if not picks:
        raise RuntimeError(f"No EMG channels found in {path}. Please provide with --emg_channels.")
    return raw, picks

def build_emg_envelope(raw: mne.io.BaseRaw, emg_chs: List[str], l_freq: float = 30.0, h_freq: float = 100.0) -> np.ndarray:
    """Filtrage 30–100 Hz, rectification, enveloppe lissée (1 s). Si plusieurs EMG, RMS inter-canaux."""
    emg = raw.copy().pick_channels(emg_chs)
    emg.filter(l_freq=l_freq, h_freq=h_freq, picks="all", method="fir", verbose=False)
    data = emg.get_data()
    if data.shape[0] > 1:
        x = np.sqrt(np.mean(data**2, axis=0))  # RMS across channels
    else:
        x = data[0]
    x_rect = np.abs(x)
    sfreq = emg.info["sfreq"]
    win = int(round(sfreq * 1.0))
    if win < 1:
        win = 1
    x_env = moving_average(x_rect, win)
    return x_env

def collect_rem_nrem_pairs(episodes: List[Episode], min_len_sec: float = 10.0) -> List[Tuple[Episode, Episode]]:
    """Pairs (NREM, REM) : REM ≥10 s immédiatement précédé d’un NREM ≥10 s (tolérance 2 s de gap)."""
    pairs = []
    episodes = sorted(episodes, key=lambda e: e.onset)
    for i, ep in enumerate(episodes):
        if ep.stage != "REM" or ep.duration < min_len_sec:
            continue
        prior = None
        for j in range(i - 1, -1, -1):
            if episodes[j].stage == "NREM" and episodes[j].duration >= min_len_sec:
                if abs(episodes[j].end - ep.onset) <= 2.0 or episodes[j].end <= ep.onset:
                    prior = episodes[j]
                    break
                else:
                    break
        if prior is not None:
            pairs.append((prior, ep))
    return pairs

def process_patient(patient_dir: str, args) -> pd.DataFrame:
    patient_id = os.path.basename(patient_dir.rstrip(os.sep))
    rec = None
    try:
        rec = find_recording_file(patient_dir)
        if rec is None:
            raise RuntimeError("No recording file (edf/fif) found.")
        raw, emg_chs = load_raw_with_emg(rec, args.emg_channels)
        sfreq = raw.info["sfreq"]
        n_samples = raw.n_times
        env = build_emg_envelope(raw, emg_chs, l_freq=30.0, h_freq=100.0)

        hyp_path = find_hypnogram_file(patient_dir)
        if hyp_path:
            episodes = parse_hypnogram_table(hyp_path)
        else:
            episodes = parse_hypnogram_from_raw(raw)
        if not episodes:
            raise RuntimeError("No hypnogram found or parsed.")

        pairs = collect_rem_nrem_pairs(episodes, min_len_sec=10.0)

        rows = []
        for idx, (nrem_ep, rem_ep) in enumerate(pairs):
            # retirer 2 s au début/fin (transitions)
            rem_start = int(np.round((rem_ep.onset + 2.0) * sfreq))
            rem_end = int(np.round((rem_ep.end - 2.0) * sfreq))
            nrem_end_for_ref = int(np.round((nrem_ep.end - 2.0) * sfreq))
            nrem_start_for_ref = int(np.round(max(nrem_ep.end - 30.0, nrem_ep.onset + 2.0) * sfreq))

            # bornes
            rem_start = max(0, min(rem_start, n_samples - 1))
            rem_end = max(rem_start + 1, min(rem_end, n_samples))
            nrem_ref_start = max(0, min(nrem_start_for_ref, n_samples - 1))
            nrem_ref_end = max(nrem_ref_start + 1, min(nrem_end_for_ref, n_samples))

            # phasic
            phasic = detect_phasic_events(env, sfreq, rem_start, rem_end, nrem_ref_start, nrem_ref_end)

            # tonic
            tonic_ratio = compute_tonic_ratio(env, sfreq, rem_start, rem_end, phasic, nrem_ref_start, nrem_ref_end)
            is_rswa = (not np.isnan(tonic_ratio)) and (tonic_ratio > 1.0)

            # RSWA : on logge l'épisode REM entier (tronqué)
            if is_rswa:
                rows.append({
                    "patient_id": patient_id,
                    "type": "TONIC_RSWA",
                    "start_sec": rem_start / sfreq,
                    "end_sec": rem_end / sfreq,
                    "duration_sec": (rem_end - rem_start) / sfreq,
                    "episode_index": idx,
                    "tonic_ratio": tonic_ratio,
                    "num_phasic_events": len(phasic)
                })
            # PHASIC : une ligne par bouffée
            for s_idx, e_idx in phasic:
                rows.append({
                    "patient_id": patient_id,
                    "type": "PHASIC",
                    "start_sec": s_idx / sfreq,
                    "end_sec": e_idx / sfreq,
                    "duration_sec": (e_idx - s_idx) / sfreq,
                    "episode_index": idx,
                    "tonic_ratio": tonic_ratio,
                    "num_phasic_events": len(phasic)
                })
        df = pd.DataFrame(rows)

        # Filtre: ne garder que les événements d'au moins args.min_rbd_dur secondes
        if not df.empty and "duration_sec" in df.columns:
            df = df[df["duration_sec"] >= args.min_rbd_dur].reset_index(drop=True)

        return df


    except Exception as e:
        return pd.DataFrame([{
            "patient_id": os.path.basename(patient_dir.rstrip(os.sep)),
            "type": "ERROR",
            "start_sec": np.nan,
            "end_sec": np.nan,
            "duration_sec": np.nan,
            "episode_index": -1,
            "tonic_ratio": np.nan,
            "num_phasic_events": -1,
            "error": f"{e} (rec={rec})"
        }])

def main():
    parser = argparse.ArgumentParser(description="Detect RBD markers (phasic events & RSWA) from EMG in REM sleep.")
    parser.add_argument("--input_dir", default="/home/darryld/documents/EEG/preprocessed/bipolaire/1_noArtefacts/gp2/", help="Directory containing patient subfolders.")
    parser.add_argument("--output_csv", default="/home/darryld/Cerco_studies/data/out_thresholds_PT.csv", help="Path to write the aggregated CSV.")
    parser.add_argument("--emg_channels", nargs="*", default=None, help="Optional list of EMG channel names to use.")
    parser.add_argument("--n_jobs", type=int, default=20, help="Number of parallel workers.")
    parser.add_argument("--min_rbd_dur", type=float, default=3.0,
                    help="Durée minimale (s) pour conserver un événement RBD (PHASIC ou TONIC).")
    args = parser.parse_args()

    patients = find_patient_dirs(args.input_dir)
    if not patients:
        print("No patient directories found.", file=sys.stderr)
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

    out_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(
        columns=["patient_id","type","start_sec","end_sec","duration_sec","episode_index","tonic_ratio","num_phasic_events","error"]
    )
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Wrote: {args.output_csv}")

    if "error" in out_df.columns and out_df["error"].notna().any():
        err_path = os.path.splitext(args.output_csv)[0] + "_errors.csv"
        out_df[out_df["type"]=="ERROR"][["patient_id","error"]].to_csv(err_path, index=False)
        print(f"Some patients failed. See: {err_path}")

if __name__ == "__main__":
    main()
