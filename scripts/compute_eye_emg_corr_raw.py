#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calcule la corrélation EOG–EMG sur les signaux RAW (non rectifiés, non lissés)
pour chaque epoch REM de 4 secondes.

Principe
--------
- EOG : filtrage passe-bande 0.3–10 Hz
- EMG : filtrage passe-bande 30–100 Hz
- corrélation de Pearson sur signaux centrés-réduits à l'échelle de l'epoch
- optionnel : corrélation maximale avec décalage temporel et estimation du lag (ms)

Le script réutilise la logique de parcours patient et de lecture d'hypnogrammes
déjà présente dans le pipeline principal.
"""

import os, sys, glob, argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import mne
from concurrent.futures import ProcessPoolExecutor, as_completed

# -------------------------
# Configuration des chemins et constantes par défaut
# -------------------------
GP2_ROOT  = "data/processed/preprocessed"
RAW_ROOT  = "data/raw"
FIF_GLOB_PATTERNS = ["*_art_annotated.fif", "*.fif"]
HYPNO_CANDIDATES  = ["{pid}_hypnoEXP.txt", "{pid}_hypnoEXP.csv", "{pid}_hypno.txt", "{pid}_hypnogram.txt"]

STAGE_MAP = {
    "W": "W", "V": "W", "WAKE": "W",
    "N1": "NREM", "S1": "NREM", "1": "NREM",
    "N2": "NREM", "S2": "NREM", "2": "NREM",
    "N3": "NREM", "S3": "NREM", "S4": "NREM", "N4": "NREM", "3": "NREM", "4": "NREM",
    "NREM": "NREM", "REM": "REM", "R": "REM", "SP": "REM"
}

# -------------------------
# Structures de données
# -------------------------
@dataclass
class Episode:
    stage: str
    onset: float
    duration: float

    @property
    def end(self) -> float:
        return self.onset + self.duration


# -------------------------
# Fonctions utilitaires de découverte des fichiers
# -------------------------
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


# -------------------------
# Fonctions utilitaires de parsing des hypnogrammes
# -------------------------
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
                     names=["end_sec", "hhmmss", "stage", "code"], usecols=[0, 1, 2, 3])
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

def parse_hypnogram_table(path: str) -> List[Episode]:
    if path is None or not os.path.isfile(path):
        return []
    base = os.path.basename(path).lower()
    ext = os.path.splitext(base)[1]
    try:
        if base.endswith("_hypnoexp.txt") or ext == ".txt":
            return _read_txt_hypnoexp(path)

        # fallback: try generic onset/duration/stage tables
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

def collect_rem_nrem_pairs(episodes: List[Episode], min_len_sec: float = 10.0):
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

def split_into_epochs(rem_start: int, rem_end: int, sfreq: float, epoch_len_s: float):
    win = int(round(epoch_len_s * sfreq))
    win = max(1, win)
    epochs = []
    s = rem_start
    while s + win <= rem_end:
        epochs.append((s, s + win))
        s += win
    return epochs


# -------------------------
# Chargement des signaux RAW filtrés (sans rectification ni lissage)
# -------------------------
def load_raw(path: str) -> mne.io.BaseRaw:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".fif":
        return mne.io.read_raw_fif(path, preload=True, verbose=False)
    if ext == ".edf":
        return mne.io.read_raw_edf(path, preload=True, verbose=False)
    if ext == ".bdf":
        return mne.io.read_raw_bdf(path, preload=True, verbose=False)
    if ext == ".gdf":
        return mne.io.read_raw_gdf(path, preload=True, verbose=False)
    return mne.io.read_raw(path, preload=True, verbose=False)

def pick_emg_channels(raw: mne.io.BaseRaw, emg_channels: Optional[List[str]]) -> List[str]:
    if emg_channels:
        picks = [ch for ch in emg_channels if ch in raw.ch_names]
    else:
        idx_type = mne.pick_types(raw.info, emg=True, eeg=False, meg=False, eog=False, stim=False, misc=False)
        picks = [raw.ch_names[i] for i in (idx_type if isinstance(idx_type, np.ndarray) else [])]
        picks += [ch for ch in raw.ch_names if "emg" in ch.lower() and ch not in picks]
    if not picks:
        raise RuntimeError("No EMG channels found. Provide --emg_channels or set channel types to 'emg'.")
    return picks

def pick_eog_channels(raw: mne.io.BaseRaw) -> List[str]:
    # force type for common names
    eog_chs = [ch for ch in raw.ch_names if ("EOGD" in ch.upper()) or ("EOGG" in ch.upper()) or ("EOG" == ch.upper())]
    if eog_chs:
        raw.set_channel_types({ch: "eog" for ch in eog_chs})
    picks = mne.pick_types(raw.info, eog=True, eeg=False, emg=False, meg=False, stim=False, misc=False)
    if len(picks) == 0:
        raise RuntimeError("EOG manquant — impossible de calculer la corrélation EOG–EMG.")
    return [raw.ch_names[i] for i in picks]

def zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, float)
    m = np.mean(x)
    s = np.std(x)
    if not np.isfinite(s) or s < eps:
        return np.full_like(x, np.nan, dtype=float)
    return (x - m) / s

def corr_pearson(x: np.ndarray, y: np.ndarray) -> float:
    xz = zscore(x)
    yz = zscore(y)
    if np.isnan(xz).any() or np.isnan(yz).any() or xz.size != yz.size or xz.size < 3:
        return np.nan
    return float(np.corrcoef(xz, yz)[0, 1])

def corr_maxlag(x: np.ndarray, y: np.ndarray, sfreq: float, max_lag_ms: float = 200.0) -> Tuple[float, float]:
    """
    Returns (r_max, lag_ms) where lag_ms is positive if y is shifted forward relative to x.
    """
    xz = zscore(x)
    yz = zscore(y)
    if np.isnan(xz).any() or np.isnan(yz).any() or xz.size != yz.size or xz.size < 10:
        return (np.nan, np.nan)

    max_lag_samp = int(round((max_lag_ms / 1000.0) * sfreq))
    max_lag_samp = max(1, max_lag_samp)

    # normalized cross-correlation via dot products on z-scored signals
    # r(lag) = mean(x[t] * y[t+lag])
    best_r = -np.inf
    best_lag = 0
    n = xz.size

    for lag in range(-max_lag_samp, max_lag_samp + 1):
        if lag < 0:
            xx = xz[-lag:]
            yy = yz[:n + lag]
        elif lag > 0:
            xx = xz[:n - lag]
            yy = yz[lag:]
        else:
            xx = xz
            yy = yz

        if xx.size < 10:
            continue
        r = float(np.mean(xx * yy))
        if r > best_r:
            best_r = r
            best_lag = lag

    return (float(best_r), float(1000.0 * best_lag / sfreq))


# -------------------------
# Traitement principal patient par patient
# -------------------------
def process_patient(patient_dir: str, args) -> pd.DataFrame:
    patient_id = os.path.basename(patient_dir.rstrip(os.sep))
    rec = None
    raw = None
    try:
        rec = find_recording_file(patient_dir)
        if rec is None:
            raise RuntimeError("No recording file found.")

        raw = load_raw(rec)

        # 1) Sélection des canaux EOG et EMG utilisés pour le calcul
        emg_chs = pick_emg_channels(raw, args.emg_channels)
        eog_chs = pick_eog_channels(raw)

        sfreq = float(raw.info["sfreq"])

        # 2) Préparation des copies filtrées sans rectification ni lissage
        #    EOG filtré 0.3-10 Hz
        eog_raw = raw.copy().pick_channels(eog_chs)
        eog_raw.filter(l_freq=args.eog_hp, h_freq=args.eog_lp, picks="all", method="fir", verbose=False)

        #    EMG filtré 30-100 Hz (ou bornes fournies en argument)
        emg_raw = raw.copy().pick_channels(emg_chs)
        emg_raw.filter(l_freq=args.emg_hp, h_freq=args.emg_lp, picks="all", method="fir", verbose=False)

        eog_data = eog_raw.get_data()  # volts
        emg_data = emg_raw.get_data()  # volts

        # 3) Lecture de l'hypnogramme puis construction des paires NREM -> REM
        hyp_path = find_hypnogram_file(patient_dir)
        episodes = parse_hypnogram_table(hyp_path) if hyp_path else parse_hypnogram_from_raw(raw)
        if not episodes:
            raise RuntimeError("No hypnogram found or parsed.")

        pairs = collect_rem_nrem_pairs(episodes, min_len_sec=10.0)
        if not pairs:
            raise RuntimeError("No REM/NREM pairs found.")

        n_samples = raw.n_times
        rows = []

        for idx_pair, (nrem_ep, rem_ep) in enumerate(pairs):
            rem_start = int(np.round((rem_ep.onset + 2.0) * sfreq))
            rem_end   = int(np.round((rem_ep.end   - 2.0) * sfreq))

            rem_start = max(0, min(rem_start, n_samples - 1))
            rem_end   = max(rem_start + 1, min(rem_end, n_samples))

            epochs = split_into_epochs(rem_start, rem_end, sfreq, args.rem_epoch_len)

            for ep_k, (w0, w1) in enumerate(epochs):
                # 4) Agrégation EOG : moyenne si plusieurs canaux sont disponibles
                eog_seg = np.mean(eog_data[:, w0:w1], axis=0)

                # 5) Calcul des corrélations par canal EMG puis agrégation robuste
                corr_per_ch = []
                corrmax_per_ch = []
                lagms_per_ch = []

                for ch_i, ch_name in enumerate(emg_chs):
                    emg_seg = emg_data[ch_i, w0:w1]

                    r = corr_pearson(emg_seg, eog_seg)
                    rmax, lagms = corr_maxlag(emg_seg, eog_seg, sfreq, max_lag_ms=args.max_lag_ms)

                    corr_per_ch.append(r)
                    corrmax_per_ch.append(rmax)
                    lagms_per_ch.append(lagms)

                    rows.append({
                        "patient_id": patient_id,
                        "episode_index": idx_pair,
                        "epoch_index": ep_k,
                        "epoch_len_sec": args.rem_epoch_len,
                        "epoch_start_sec": w0 / sfreq,
                        "epoch_end_sec": w1 / sfreq,

                        "emg_channel": ch_name,
                        "eog_channels": ",".join(eog_chs),

                        # Corrélation RAW (sans rectification, sans lissage)
                        "eye_emg_corr_raw": r,
                        "eye_emg_corrmax_raw": rmax,
                        "eye_emg_lag_ms_raw": lagms,

                        # Paramètres de filtrage rappelés pour la traçabilité
                        "emg_hp": args.emg_hp, "emg_lp": args.emg_lp,
                        "eog_hp": args.eog_hp, "eog_lp": args.eog_lp,
                    })

                # 6) Sauvegarde d'une version agrégée à l'échelle de l'epoch
                rows.append({
                    "patient_id": patient_id,
                    "episode_index": idx_pair,
                    "epoch_index": ep_k,
                    "epoch_len_sec": args.rem_epoch_len,
                    "epoch_start_sec": w0 / sfreq,
                    "epoch_end_sec": w1 / sfreq,
                    "emg_channel": "EMG_AGG",
                    "eog_channels": ",".join(eog_chs),
                    "eye_emg_corr_raw": float(np.nanmedian(corr_per_ch)) if len(corr_per_ch) else np.nan,
                    "eye_emg_corrmax_raw": float(np.nanmedian(corrmax_per_ch)) if len(corrmax_per_ch) else np.nan,
                    "eye_emg_lag_ms_raw": float(np.nanmedian(lagms_per_ch)) if len(lagms_per_ch) else np.nan,
                    "emg_hp": args.emg_hp, "emg_lp": args.emg_lp,
                    "eog_hp": args.eog_hp, "eog_lp": args.eog_lp,
                })

        return pd.DataFrame(rows)

    except Exception as e:
        return pd.DataFrame([{
            "patient_id": patient_id,
            "type": "ERROR",
            "error": f"{e} (rec={rec}, raw_loaded={raw is not None})"
        }])


def main():
    parser = argparse.ArgumentParser(
        description="Compute EOG–EMG correlation on non-rectified/non-smoothed signals per REM 4s epoch."
    )
    parser.add_argument("--input_dir", default=GP2_ROOT, help="Directory containing patient subfolders.")
    parser.add_argument("--output_csv", default="data/processed/rbd/eye_emg_corr_raw_4s.csv",
                        help="CSV output path.")

    parser.add_argument("--emg_channels", nargs="*", default=None,
                        help="Optional list of EMG channel names to use.")

    parser.add_argument("--rem_epoch_len", type=float, default=4.0, help="REM epoch length in seconds.")
    parser.add_argument("--n_jobs", type=int, default=20, help="Parallel workers.")

    # Paramètres de filtrage appliqués aux signaux RAW avant calcul des corrélations
    parser.add_argument("--eog_hp", type=float, default=0.3)
    parser.add_argument("--eog_lp", type=float, default=10.0)
    parser.add_argument("--emg_hp", type=float, default=30.0)
    parser.add_argument("--emg_lp", type=float, default=100.0)

    # Paramètre de recherche pour la corrélation maximale avec décalage
    parser.add_argument("--max_lag_ms", type=float, default=200.0,
                        help="Max lag (ms) for corrmax (cross-correlation).")

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"[ERREUR] input_dir n'existe pas : {args.input_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(RAW_ROOT):
        print(f"[ERREUR] RAW_ROOT n'existe pas : {RAW_ROOT}", file=sys.stderr)
        sys.exit(1)

    patients = find_patient_dirs(args.input_dir)
    if not patients:
        print("[ERREUR] Aucun dossier patient trouvé.", file=sys.stderr)
        sys.exit(1)

    # 1) Filtrage des dossiers patients pour ne garder que ceux avec enregistrement
    valid_patients = []
    for p in patients:
        if find_recording_file(p) is not None:
            valid_patients.append(p)
    patients = valid_patients
    print(f"[CHECK] {len(patients)} patients valides trouvés.")

    # 2) Traitement parallèle ou séquentiel selon le nombre de workers demandé
    dfs = []
    if args.n_jobs and args.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            fut2p = {ex.submit(process_patient, p, args): p for p in patients}
            for fut in as_completed(fut2p):
                dfs.append(fut.result())
    else:
        for p in patients:
            dfs.append(process_patient(p, args))

    # 3) Concaténation finale et écriture du CSV de sortie
    out_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"[OK] CSV écrit: {args.output_csv}")

    # 4) Export optionnel d'un fichier séparé pour les patients en erreur
    if "type" in out_df.columns and (out_df["type"] == "ERROR").any():
        err_path = os.path.splitext(args.output_csv)[0] + "_errors.csv"
        out_df[out_df["type"] == "ERROR"][["patient_id", "error"]].to_csv(err_path, index=False)
        print(f"[INFO] Des patients ont échoué. Détails: {err_path}")


if __name__ == "__main__":
    main()
