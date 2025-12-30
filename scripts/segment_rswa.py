#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
segment_rswa_from_csv.py - OPTION A : PAS DE CONCATÉNATION (RAPIDE)

Objectif :
    Extraire uniquement les epochs RSWA (4 s) à partir d'un fichier .fif prétraité,
    en s'appuyant sur les fenêtres (start/end sec) d'un CSV RSWA.
    Optionnellement : ne garder que les fenêtres incluses dans REM et exclure ARTEFACT,
    exactement comme segment_rem.py.

Entrée :
    - {patient}.fif (prétraité, contient des annotations "REM" et "ARTEFACT")
    - CSV RSWA avec colonnes: patient_id (ou base) + epoch_start_sec + epoch_end_sec
      (optionnel: rswa bool)

Sorties :
    - {patient}_RSWA-epo.fif          (Epochs 4 s, RSWA-only, filtres REM/ARTEFACT optionnels)
    - {patient}_RSWA-epo_segments.csv (fenêtres réellement gardées)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import mne


# -------------------------
# Defaults (adapte si besoin)
# -------------------------
RSWA_CSV_DEFAULT = Path(
    "/home/darryld/Cerco_studies/data/processed/rbd/_rswa_only/"
    "rbd_emg_epochs_4s_rswa_only_aggregated.csv"
)
XAI_ROOT_DEFAULT = Path("/home/darryld/documents/EEG/preprocessed/XAI/data")
OUT_DIR_DEFAULT  = Path("/home/darryld/documents/EEG/results")


# -------------------------
# Utils
# -------------------------
def normalize_id(x: str) -> str:
    return "".join(ch for ch in str(x).strip().upper() if ch.isalnum())


def get_intervals_from_annotations(raw: mne.io.BaseRaw, label: str) -> List[Tuple[float, float]]:
    """Retourne une liste [(start, end)] pour les annotations d’un label donné."""
    intervals = []
    for ann in raw.annotations:
        if ann["description"] == label:
            s = float(ann["onset"])
            e = float(ann["onset"] + ann["duration"])
            if e > s:
                intervals.append((s, e))
    return intervals


def subtract_intervals(base_intervals: List[Tuple[float, float]],
                       remove_intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Soustrait remove_intervals de base_intervals.
    """
    if not base_intervals:
        return []

    out = []
    for b_start, b_end in base_intervals:
        seg = [(b_start, b_end)]
        for r_start, r_end in remove_intervals:
            new_seg = []
            for s, e in seg:
                if r_end <= s or r_start >= e:
                    new_seg.append((s, e))
                    continue
                if r_start > s:
                    new_seg.append((s, r_start))
                if r_end < e:
                    new_seg.append((r_end, e))
            seg = new_seg
        out.extend(seg)

    return [(s, e) for s, e in out if e > s]


def merge_close_intervals(intervals: List[Tuple[float, float]], gap_sec: float = 0.0) -> List[Tuple[float, float]]:
    """Fusionne des intervalles séparés par <= gap_sec."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e + gap_sec:
            cur_e = max(cur_e, e)
        else:
            out.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    out.append((cur_s, cur_e))
    return out


def intervals_from_rswa_df(seg_df: pd.DataFrame) -> List[Tuple[float, float]]:
    return [(float(s), float(e)) for s, e in zip(seg_df["epoch_start_sec"], seg_df["epoch_end_sec"]) if float(e) > float(s)]


def filter_windows_fully_inside(windows: List[Tuple[float, float]],
                               allowed: List[Tuple[float, float]],
                               epoch_len: float) -> List[Tuple[float, float]]:
    """
    Garde uniquement les fenêtres [s,e] (souvent 4s) entièrement incluses
    dans l'un des intervalles 'allowed' (ex: REM_clean).
    """
    if not windows or not allowed:
        return []

    allowed = sorted(allowed)
    starts = np.array([a[0] for a in allowed], dtype=float)
    ends   = np.array([a[1] for a in allowed], dtype=float)

    kept = []
    for s, e in windows:
        # sécurité: force epoch_len
        e = s + float(epoch_len)
        # cherche le dernier intervalle allowed dont start <= s
        idx = np.searchsorted(starts, s, side="right") - 1
        if idx < 0:
            continue
        if e <= ends[idx]:
            kept.append((s, e))
    return kept


def build_epochs_from_windows(raw: mne.io.BaseRaw,
                              windows: List[Tuple[float, float]],
                              epoch_len: float = 4.0) -> Optional[mne.Epochs]:
    """
    Crée des Epochs directement via events (rapide).
    Chaque fenêtre devient un event au sample correspondant à t0.
    """
    if not windows:
        return None

    sf = float(raw.info["sfreq"])
    events = []
    for t0, _ in windows:
        sample = int(round(float(t0) * sf))
        if 0 <= sample < raw.n_times:
            events.append([sample, 0, 1])

    if not events:
        return None

    events = np.asarray(events, dtype=int)
    event_id = {"RSWA": 1}

    picks = mne.pick_types(raw.info, eeg=True, eog=True, emg=True, ecg=True, misc=True)

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=0.0,
        tmax=float(epoch_len),
        baseline=None,
        picks=picks,
        preload=True,      # on veut sauver les données -> preload
        verbose=False,
    )
    return epochs


# -------------------------
# RSWA CSV loading
# -------------------------
def load_rswa_table(rswa_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(rswa_csv)

    if "patient_id" in df.columns:
        df["base"] = df["patient_id"].astype(str).map(normalize_id)
    elif "base" in df.columns:
        df["base"] = df["base"].astype(str).map(normalize_id)
    else:
        raise RuntimeError("RSWA CSV doit contenir 'patient_id' ou 'base'.")

    for c in ["epoch_start_sec", "epoch_end_sec"]:
        if c not in df.columns:
            raise RuntimeError(f"RSWA CSV doit contenir '{c}'.")

    df["epoch_start_sec"] = pd.to_numeric(df["epoch_start_sec"], errors="coerce")
    df["epoch_end_sec"]   = pd.to_numeric(df["epoch_end_sec"], errors="coerce")
    df = df.dropna(subset=["epoch_start_sec", "epoch_end_sec"]).copy()

    # keep rswa==True if present
    if "rswa" in df.columns:
        rs = df["rswa"]
        if rs.dtype != bool:
            rs = rs.astype(str).str.strip().str.lower().isin(["1","true","t","yes","y"])
        df = df[rs].copy()

    # --- DÉDUPLICATION RSWA (CSV per-channel) ---
    df = df.drop_duplicates(
        subset=["base", "epoch_start_sec", "epoch_end_sec"]
    )

    df = df.sort_values(["base", "epoch_start_sec", "epoch_end_sec"])
    return df



# -------------------------
# File finding
# -------------------------
FIF_GLOB_PATTERNS = ["*_art_annotated.fif", "*.fif"]

def find_patient_dirs(xai_root: Path) -> List[Path]:
    subs = [p for p in xai_root.glob("*") if p.is_dir()]
    subs.sort()
    return subs

def find_recording_file(patient_dir: Path) -> Optional[Path]:
    for pat in FIF_GLOB_PATTERNS:
        files = sorted(patient_dir.glob(pat))
        if files:
            return files[0]
    return None


# -------------------------
# Worker
# -------------------------
def process_one_base(base: str,
                     base_to_dir: dict,
                     rswa_df: pd.DataFrame,
                     out_dir: Path,
                     epoch_len: float,
                     keep_only_rem: bool,
                     exclude_artefact: bool,
                     merge_gap_sec: float,
                     overwrite: bool) -> Tuple[str, str, str]:

    try:
        pdir = base_to_dir.get(base)
        if pdir is None:
            return base, "SKIP", "patient dir introuvable"

        fif = find_recording_file(pdir)
        if fif is None:
            return base, "SKIP", f"aucun .fif trouvé dans {pdir}"

        segs = rswa_df[rswa_df["base"] == base][["epoch_start_sec", "epoch_end_sec"]].copy()
        if segs.empty:
            return base, "SKIP", "aucun segment RSWA dans le CSV"

        # fenêtres RSWA -> list
        windows = intervals_from_rswa_df(segs)
        # on force la durée à epoch_len (au cas où end_sec varie)
        windows = [(s, s + float(epoch_len)) for s, _ in windows]

        out_pdir = out_dir / base
        out_pdir.mkdir(parents=True, exist_ok=True)
        out_fif = out_pdir / f"{base}_RSWA-epo.fif"
        out_seg = out_pdir / f"{base}_RSWA-epo_segments.csv"

        if out_fif.exists() and not overwrite:
            return base, "OK", f"déjà présent: {out_fif}"

        raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
        tmax = float(raw.times[-1])

        # borne les fenêtres dans [0, tmax]
        windows = [(max(0.0, s), min(tmax, s + float(epoch_len))) for s, _ in windows]
        windows = [(s, e) for s, e in windows if e > s and (e - s) >= float(epoch_len) - 1e-6]

        if not windows:
            return base, "SKIP", "fenêtres RSWA vides après bornage"

        # --- Filtrage REM / ARTEFACT (même logique que segment_rem.py) ---
        if keep_only_rem:
            rem_int = get_intervals_from_annotations(raw, "REM")
            if not rem_int:
                return base, "SKIP", "aucun segment REM trouvé (keep-only-rem)"
        else:
            rem_int = []

        art_int = get_intervals_from_annotations(raw, "ARTEFACT") if exclude_artefact else []

        # allowed = REM minus ARTEFACT (si keep_only_rem), sinon juste "tout" minus ARTEFACT
        if keep_only_rem:
            allowed = subtract_intervals(rem_int, art_int) if exclude_artefact else rem_int
        else:
            # si on ne force pas REM, on retire juste ARTEFACT des RSWA via un subtract direct sur windows
            allowed = None

        if merge_gap_sec and merge_gap_sec > 0 and keep_only_rem:
            allowed = merge_close_intervals(allowed, gap_sec=float(merge_gap_sec))

        if keep_only_rem:
            windows_kept = filter_windows_fully_inside(windows, allowed, epoch_len=float(epoch_len))
        else:
            # si on ne force pas REM, on enlève les recouvrements ARTEFACT en soustrayant sur les fenêtres elles-mêmes
            # (ici windows sont déjà des 4s; si elles touchent un artefact, elles seront “cassées” -> on garde seulement celles intactes)
            if exclude_artefact and art_int:
                w_clean = subtract_intervals(windows, art_int)
                # subtract peut fragmenter; on garde seulement des segments de longueur epoch_len
                windows_kept = [(s, e) for s, e in w_clean if (e - s) >= float(epoch_len) - 1e-6]
                # re-force exactement 4s (au besoin)
                windows_kept = [(s, s + float(epoch_len)) for s, e in windows_kept if s + float(epoch_len) <= e + 1e-6]
            else:
                windows_kept = windows

        print(f"[{base}] windows_in={len(windows)}")
        print(f"[{base}] REM intervals={len(rem_int)} | ARTEFACT intervals={len(art_int)}")
        if keep_only_rem:
            print(f"[{base}] allowed(REM_clean) intervals={len(allowed)}")
        print(f"[{base}] windows_kept={len(windows_kept)}")
        print(f"[{base}] first_windows={windows[:5]}")
        if keep_only_rem and allowed:
            print(f"[{base}] first_allowed={allowed[:5]}")


        if not windows_kept:
            return base, "SKIP", "aucune fenêtre RSWA après filtres REM/ARTEFACT"

        # --- Epochs (rapide) ---
        epochs = build_epochs_from_windows(raw, windows_kept, epoch_len=float(epoch_len))
        if epochs is None or len(epochs) == 0:
            return base, "SKIP", "aucun epoch créé"

        epochs.save(out_fif, overwrite=True)

        # sauve segments utilisés
        pd.DataFrame(windows_kept, columns=["epoch_start_sec", "epoch_end_sec"]).to_csv(out_seg, index=False)

        msg = f"saved={out_fif.name} | epochs={len(epochs)} | windows_in={len(windows)} | windows_kept={len(windows_kept)}"
        return base, "OK", msg

    except Exception as e:
        return base, "ERROR", str(e)


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xai-root", type=str, default=str(XAI_ROOT_DEFAULT))
    p.add_argument("--rswa-csv", type=str, default=str(RSWA_CSV_DEFAULT))
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))
    p.add_argument("--patients", nargs="*", default=None)

    p.add_argument("--epoch-len", type=float, default=4.0, help="Durée des epochs (s).")
    p.add_argument("--keep-only-rem", action="store_true", help="Garder seulement les fenêtres RSWA incluses dans REM.")
    p.add_argument("--exclude-artefact", action="store_true", help="Exclure les fenêtres recouvrant ARTEFACT.")
    p.add_argument("--merge-gap-sec", type=float, default=0.0, help="Fusion d'intervalles REM_clean (utile si keep-only-rem).")

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--n-jobs", type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()

    xai_root = Path(args.xai_root)
    rswa_csv = Path(args.rswa_csv)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not xai_root.exists():
        raise SystemExit(f"[ERR] xai-root introuvable: {xai_root}")
    if not rswa_csv.exists():
        raise SystemExit(f"[ERR] rswa-csv introuvable: {rswa_csv}")

    rswa_df = load_rswa_table(rswa_csv)
    bases_csv = sorted(rswa_df["base"].unique().tolist())
    print("[INFO] RSWA CSV rows:", rswa_df.shape, "| patients:", len(bases_csv))

    patient_dirs = find_patient_dirs(xai_root)
    base_to_dir = {normalize_id(d.name): d for d in patient_dirs}
    bases_xai = sorted(base_to_dir.keys())
    print("[INFO] XAI patients dirs:", len(bases_xai))

    if args.patients:
        bases = sorted({normalize_id(b) for b in args.patients if str(b).strip()})
    else:
        bases = sorted(set(bases_xai) & set(bases_csv))

    print("[INFO] Patients traités:", len(bases))
    if not bases:
        print("[ERR] Intersection vide XAI_ROOT vs RSWA_CSV. Vérifie les IDs.")
        return

    n_ok = n_skip = n_err = 0

    if args.n_jobs and args.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            futs = [
                ex.submit(
                    process_one_base, base, base_to_dir, rswa_df, out_dir,
                    float(args.epoch_len), bool(args.keep_only_rem),
                    bool(args.exclude_artefact), float(args.merge_gap_sec),
                    bool(args.overwrite)
                )
                for base in bases
            ]
            for fut in as_completed(futs):
                base, status, msg = fut.result()
                print(f"[{base}] {status}: {msg}")
                if status == "OK":
                    n_ok += 1
                elif status == "SKIP":
                    n_skip += 1
                else:
                    n_err += 1
    else:
        for base in bases:
            base, status, msg = process_one_base(
                base, base_to_dir, rswa_df, out_dir,
                float(args.epoch_len), bool(args.keep_only_rem),
                bool(args.exclude_artefact), float(args.merge_gap_sec),
                bool(args.overwrite)
            )
            print(f"[{base}] {status}: {msg}")
            if status == "OK":
                n_ok += 1
            elif status == "SKIP":
                n_skip += 1
            else:
                n_err += 1

    print(f"\n[DONE] ok={n_ok} | skip={n_skip} | error={n_err}")


if __name__ == "__main__":
    main()
