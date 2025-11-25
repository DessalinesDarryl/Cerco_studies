#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
segment_rem.py - OPTION A : PAS DE CONCATÉNATION

Objectif :
    Extraire uniquement les époques REM sans artefacts à partir
    du fichier .fif prétraité (montage gp2 + REM + ARTEFACT).

Entrée :
    {patient}.fif  (prétraité)

Sortie :
    {patient}_REM-epo.fif   (Epochs 4 s, REM-only, artefacts exclus)

Principe :
    - On lit les annotations REM et ARTEFACT.
    - On crée un masque temporel REM.
    - On enlève les fenêtres ARTEFACT de ce masque.
    - On époche par fenêtres fixes de 4 s.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import os
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import mne
import numpy as np

from src.utils.config import load_yaml, add_common_args
from src.utils.logging import get_logger


def get_intervals_from_annotations(raw, label):
    """Retourne une liste [(start, end)] pour les annotations d’un label donné."""
    intervals = []
    for ann in raw.annotations:
        if ann["description"] == label:
            s = float(ann["onset"])
            e = float(ann["onset"] + ann["duration"])
            if e > s:
                intervals.append((s, e))
    return intervals


def subtract_intervals(base_intervals, remove_intervals):
    """
    Soustrait remove_intervals de base_intervals.
    Exemple :
        base  = [(10,40)]
        remove= [(20,25)]
    Résultat = [(10,20),(25,40)]
    """
    if not base_intervals:
        return []

    out = []

    for b_start, b_end in base_intervals:
        seg = [(b_start, b_end)]
        for r_start, r_end in remove_intervals:
            new_seg = []
            for s, e in seg:
                # pas de recouvrement
                if r_end <= s or r_start >= e:
                    new_seg.append((s, e))
                    continue
                # recouvre à gauche
                if r_start > s:
                    new_seg.append((s, r_start))
                # recouvre à droite
                if r_end < e:
                    new_seg.append((r_end, e))
            seg = new_seg
        out.extend(seg)

    # nettoyage
    out = [(s, e) for s, e in out if e > s]
    return out


def make_4s_epochs(raw, intervals, epoch_len=4.0):
    """
    Découpe des Epochs de "epoch_len" secondes à l’intérieur des intervals.
    Exemple :
        interval = (10, 22)
        -> epochs = [10–14], [14–18], [18–22]
    """
    events = []
    event_id = {"REM": 1}

    sf = raw.info["sfreq"]

    for (s, e) in intervals:
        t = s
        while t + epoch_len <= e:
            sample = int(t * sf)
            events.append([sample, 0, 1])
            t += epoch_len

    if not events:
        return None

    events = np.array(events)
    picks = mne.pick_types(
        raw.info, eeg=True, eog=True, emg=True, ecg=True, misc=True
    )

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=0.0,
        tmax=epoch_len,  # 4 s nominales
        baseline=None,
        picks=picks,
        preload=True,
        verbose=False,
    )

    return epochs


def _process_one_fif(fif_path: Path, out_root: Path, epoch_len: float):
    """Traitement REM/ARTEFACT + époquage pour un patient (un FIF)."""
    log = get_logger("segment_rem")

    base = fif_path.stem
    log.info(f"---- {base} ----")

    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

    # --- 1) Récupération REM ---
    rem_int = get_intervals_from_annotations(raw, "REM")
    if not rem_int:
        log.warning(f"[{base}] Aucun segment REM trouvé → skip")
        return

    # --- 2) Récupération ARTEFACT ---
    art_int = get_intervals_from_annotations(raw, "ARTEFACT")

    # --- 3) REM sans artefacts ---
    rem_clean = subtract_intervals(rem_int, art_int)
    if not rem_clean:
        log.warning(f"[{base}] Tous les REM sont artefactués → skip")
        return

    # --- 4) Époquage 4 s ---
    epochs = make_4s_epochs(raw, rem_clean, epoch_len=epoch_len)
    if epochs is None:
        log.warning(f"[{base}] Aucun epoch extrait → skip")
        return

    # --- 5) Sauvegarde ---
    out_path = out_root / f"{base}_REM-epo.fif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs.save(out_path, overwrite=True)
    log.info(f"[{base}] OK → {out_path}")


def main(cfg):
    log = get_logger("segment_rem")

    in_root = Path(cfg["in_root"])
    out_root = Path(cfg["out_root"])
    epoch_len = float(cfg.get("epoch_len_s", 4.0))
    n_workers = int(cfg.get("n_workers", 15))

    fif_files = sorted(in_root.glob("**/*.fif"))
    if not fif_files:
        log.warning(f"Aucun fichier FIF trouvé dans {in_root}")
        return

    log.info(f"{len(fif_files)} fichiers FIF trouvés. Lancement avec n_workers={n_workers}.")

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(_process_one_fif, fif_path, out_root, epoch_len): fif_path
            for fif_path in fif_files
        }

        for fut in as_completed(futures):
            f = futures[fut]
            try:
                fut.result()
            except Exception as e:
                log.error(f"[ERREUR] sur {f}: {e}")


if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument(
        "--n_workers",
        type=int,
        default=15,
        help="Nombre de workers en parallèle (par défaut: 15).",
    )
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    cfg["n_workers"] = args.n_workers
    main(cfg)
