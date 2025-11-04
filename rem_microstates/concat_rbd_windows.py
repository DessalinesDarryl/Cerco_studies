#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import mne
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------- Paramètres chemins (tu peux adapter si besoin)
GP2_ROOT  = "/home/darryld/documents/EEG/preprocessed/bipolaire/1_noArtefacts/gp2/"
OUT_95P   = "/home/darryld/documents/EEG/preprocessed/bipolaire/1bis_RBD/method_95percentile"
OUT_MAAV  = "/home/darryld/documents/EEG/preprocessed/bipolaire/1bis_RBD/method_3MAAV"

FIF_GLOB_PATTERNS = ["*_art_annotated.fif", "*.fif"]  # ordre d'essai

# ---------- Utils

def find_patient_fif(patient_dir: str) -> Optional[str]:
    """Trouve un fichier FIF dans un dossier patient selon des motifs usuels."""
    for pat in FIF_GLOB_PATTERNS:
        matches = glob.glob(str(Path(patient_dir) / pat))
        if matches:
            return sorted(matches)[0]
    return None

def merge_intervals(intervals: List[Tuple[float, float]], merge_gap_s: float = 0.1) -> List[Tuple[float, float]]:
    """
    Fusionne des intervalles [start, end] (secondes) triés ou non, en fusionnant
    ceux qui se chevauchent ou sont séparés par un gap < merge_gap_s.
    """
    if not intervals:
        return []
    xs = sorted([(float(s), float(e)) for (s, e) in intervals if e > s], key=lambda t: (t[0], t[1]))
    merged = []
    cur_s, cur_e = xs[0]
    for s, e in xs[1:]:
        if s <= cur_e + merge_gap_s:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged

def load_events_from_out95(csv_path: str, min_dur_s: float = 3.0) -> Dict[str, List[Tuple[float, float]]]:
    """
    Lit out_thresholds_PT.csv (colonnes attendues: patient_id, start_sec, end_sec, duration_sec, type, ...)
    Retour: {pid: [(start_sec, end_sec), ...]}
    """
    df = pd.read_csv(csv_path)
    # colonnes robustes
    pid_col = "patient_id" if "patient_id" in df.columns else "patient"
    start_col = "start_sec" if "start_sec" in df.columns else "onset_s"
    end_col = "end_sec" if "end_sec" in df.columns else "end_s"
    dur_col = "duration_sec" if "duration_sec" in df.columns else "duration_s"

    # filtre durée
    if dur_col in df.columns:
        df = df[df[dur_col] >= float(min_dur_s)]

    out: Dict[str, List[Tuple[float, float]]] = {}
    for pid, g in df.groupby(pid_col):
        win = [(float(r[start_col]), float(r[end_col])) for _, r in g.iterrows() if (r[end_col] > r[start_col])]
        out[str(pid)] = merge_intervals(win, merge_gap_s=0.1)
    return out

def load_events_from_index(csv_index_path: str, min_dur_s: float = 3.0) -> Dict[str, List[Tuple[float, float]]]:
    """
    Lit un index MAAV (ex: RBD_windows_INDEX.csv) qui contient au moins les colonnes:
      - 'patient'
      - 'csv' (chemin vers un CSV par patient, issu de la méthode MAAV)
    Chaque CSV patient doit contenir 'onset_s' et 'end_s' (et 'duration_s' idéalement).
    Retour: {pid: [(start_sec, end_sec), ...]}
    """
    idx = pd.read_csv(csv_index_path)
    # colonnes robustes
    if "patient" not in idx.columns:
        raise ValueError("Index MAAV: colonne 'patient' manquante.")
    if "csv" not in idx.columns:
        # parfois 'CSV' en majuscule
        csv_col = "CSV" if "CSV" in idx.columns else None
        if csv_col is None:
            raise ValueError("Index MAAV: colonne 'csv' (ou 'CSV') manquante.")
        idx["csv"] = idx[csv_col]

    out: Dict[str, List[Tuple[float, float]]] = {}
    for _, row in idx.iterrows():
        pid = str(row["patient"])
        path = str(row["csv"]) if not (pd.isna(row["csv"])) else None
        if not path or not os.path.isfile(path):
            # tente dans le dossier patient
            cand = list(Path(GP2_ROOT, pid).glob(f"{pid}_RBD_windows.csv"))
            path = str(cand[0]) if cand else None
        if not path or not os.path.isfile(path):
            out[pid] = []
            continue

        dfp = pd.read_csv(path)
        # colonnes robustes
        start_col = "onset_s" if "onset_s" in dfp.columns else "start_sec"
        end_col = "end_s" if "end_s" in dfp.columns else "end_sec"
        dur_col = "duration_s" if "duration_s" in dfp.columns else ("duration_sec" if "duration_sec" in dfp.columns else None)

        if dur_col and dur_col in dfp.columns:
            dfp = dfp[dfp[dur_col] >= float(min_dur_s)]

        wins = [(float(r[start_col]), float(r[end_col])) for _, r in dfp.iterrows() if (r[end_col] > r[start_col])]
        out[pid] = merge_intervals(wins, merge_gap_s=0.1)
    return out

def cut_and_concat_patient(pid: str,
                           windows: List[Tuple[float, float]],
                           gp2_root: str,
                           save_root: str,
                           picks_emg_only: bool = False,
                           include_tmax: bool = False) -> Tuple[str, str, int, float]:
    """
    Charge l'enregistrement .fif du patient, découpe les fenêtres, concatène, sauvegarde.
    Retour: (pid, out_path, n_segments, total_duration_s)
    """
    patient_dir = str(Path(gp2_root) / pid)
    fif_path = find_patient_fif(patient_dir)
    if not fif_path:
        raise FileNotFoundError(f"[{pid}] pas de FIF trouvé dans {patient_dir}")

    if not windows:
        raise RuntimeError(f"[{pid}] aucune fenêtre RBD à concaténer.")

    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

    # Détermination des picks
    if picks_emg_only:
        picks = mne.pick_types(raw.info, emg=True, meg=False, eeg=False, eog=False, stim=False, misc=False)
        if len(picks) == 0:
            raise RuntimeError(f"[{pid}] aucun canal EMG trouvé (picks_emg_only=True).")
        raw = raw.pick(picks)
    # sinon: on garde tous les canaux

    # Découpes (crop) puis concat
    segments = []
    annots = []
    for i, (t0, t1) in enumerate(windows, 1):
        # bornes sûres
        t0c = max(0.0, min(t0, raw.times[-1]))
        t1c = max(t0c + 1e-6, min(t1, raw.times[-1]))
        seg = raw.copy().crop(tmin=t0c, tmax=t1c, include_tmax=include_tmax)
        segments.append(seg)
        annots.append({"onset_in_source_s": float(t0c), "end_in_source_s": float(t1c), "dur_s": float(t1c - t0c)})

    if len(segments) == 1:
        cat = segments[0]
    else:
        cat = mne.concatenate_raws(segments, on_mismatch="ignore", verbose=False)

    # Ajout d'annotations pour tracer les segments d'origine
    # On place une annotation par segment, aux onsets (dans le fichier concaténé)
    onset = 0.0
    new_annot = []
    for i, info in enumerate(annots, 1):
        dur = info["dur_s"]
        desc = f"RBD_SEG#{i} src_onset={info['onset_in_source_s']:.3f}s"
        new_annot.append(mne.Annotations(onset=[onset], duration=[dur], description=[desc]))
        onset += dur
    if new_annot:
        cat.set_annotations(sum(new_annot[1:], new_annot[0]))

    # Sauvegarde
    out_dir = Path(save_root) / pid
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"{pid}_RBD_concat.fif")
    cat.save(out_path, overwrite=True)
    total_dur = float(cat.times[-1]) if cat.n_times > 0 else 0.0
    return pid, out_path, len(segments), total_dur

# ---------- Main

def main():
    p = argparse.ArgumentParser(description="Concatène les fenêtres RBD par patient en un unique .fif")
    p.add_argument("--events_csv", required=True,
                   help="Chemin vers out_thresholds_PT.csv (95e) ou un index MAAV (ex: RBD_windows_INDEX.csv).")
    p.add_argument("--gp2_root", default=GP2_ROOT, help="Dossier racine contenant les sous-dossiers patients avec .fif.")
    p.add_argument("--out_95p", default=OUT_95P, help="Dossier de sortie pour la méthode 95e percentile.")
    p.add_argument("--out_maav", default=OUT_MAAV, help="Dossier de sortie pour la méthode 3MAAV.")
    p.add_argument("--min_dur_s", type=float, default=3.0, help="Durée minimale (s) d'une fenêtre conservée.")
    p.add_argument("--merge_gap_s", type=float, default=0.1, help="Fusionne les fenêtres séparées par moins que ce gap (s).")
    p.add_argument("--picks_emg_only", action="store_true", help="Ne garder que les canaux EMG.")
    p.add_argument("--n_jobs", type=int, default=8, help="Nombre de workers parallèles.")
    args = p.parse_args()

    in_base = os.path.basename(args.events_csv)
    is_95 = (in_base == "out_thresholds_PT.csv")
    save_root = args.out_95p if is_95 else args.out_maav

    # Charger les fenêtres
    if is_95:
        pid2wins = load_events_from_out95(args.events_csv, min_dur_s=args.min_dur_s)
    else:
        # suppose un index pointant vers les CSV par patient
        pid2wins = load_events_from_index(args.events_csv, min_dur_s=args.min_dur_s)

    # Appliquer fusion de gaps souhaitée
    if args.merge_gap_s is not None and args.merge_gap_s > 0:
        pid2wins = {pid: merge_intervals(wins, merge_gap_s=args.merge_gap_s) for pid, wins in pid2wins.items()}

    # Préparer liste patients à traiter (ceux qui ont des fenêtres)
    jobs = [(pid, wins) for pid, wins in pid2wins.items() if wins]

    if not jobs:
        print("Aucune fenêtre à traiter (vérifie le CSV et les durées minimales).")
        sys.exit(0)

    print(f"Patients à traiter: {len(jobs)} | sortie: {save_root}")

    results = []
    errors = []

    if args.n_jobs and args.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            fut2pid = {ex.submit(cut_and_concat_patient, pid, wins, args.gp2_root, save_root,
                                 args.picks_emg_only, False): pid
                       for (pid, wins) in jobs}
            for fut in as_completed(fut2pid):
                pid = fut2pid[fut]
                try:
                    res = fut.result()
                    results.append(res)
                    print(f"[{pid}] OK -> {res[1]} (segments={res[2]}, dur={res[3]:.1f}s)")
                except Exception as e:
                    errors.append((pid, str(e)))
                    print(f"[{pid}] ERREUR: {e}")
    else:
        for (pid, wins) in jobs:
            try:
                res = cut_and_concat_patient(pid, wins, args.gp2_root, save_root,
                                             args.picks_emg_only, False)
                results.append(res)
                print(f"[{pid}] OK -> {res[1]} (segments={res[2]}, dur={res[3]:.1f}s)")
            except Exception as e:
                errors.append((pid, str(e)))
                print(f"[{pid}] ERREUR: {e}")

    # Index de sortie
    out_idx = pd.DataFrame(results, columns=["patient", "out_fif", "n_segments", "duration_s"])
    out_idx_path = str(Path(save_root) / "RBD_concat_INDEX.csv")
    out_idx.sort_values("patient").to_csv(out_idx_path, index=False)
    print(f"\nIndex écrit: {out_idx_path}")

    if errors:
        err_df = pd.DataFrame(errors, columns=["patient", "error"])
        err_path = str(Path(save_root) / "RBD_concat_ERRORS.csv")
        err_df.to_csv(err_path, index=False)
        print(f"Des erreurs sont survenues, voir: {err_path}")

if __name__ == "__main__":
    main()
