#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

import mne

# === Assurer que le repo racine est dans sys.path ===
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # .../Cerco_studies
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml, add_common_args
from src.utils.logging import get_logger
from src.data.preprocessing import preprocess_record


def _process_one_edf(
    p: Path,
    raw_root: Path,
    out_root: Path,
    hypno_root: Path,
    eeg_params: dict,
    emg_params: dict,
    eog_params: dict,
    yasa_params: dict,
):
    """Traitement d’un patient (un fichier EDF) dans un worker."""
    log = get_logger("preprocess")  # logger côté worker
    p = Path(p)
    base_name = p.stem.split("_")[0]  # ex: =AN166 à partir de AN166_H0_raw.edf

    log.info(f"=== Prétraitement {base_name} ({p.name}) ===")

    raw = mne.io.read_raw_edf(p, preload=True, verbose=False)

    raw_prep = preprocess_record(
        raw=raw,
        base_name=base_name,
        hypno_root=hypno_root,
        eeg_params=eeg_params,
        emg_params=emg_params,
        eog_params=eog_params,
        yasa_params=yasa_params,
    )

    # On garde la même arbo relative mais en .fif
    rel = p.relative_to(raw_root).with_suffix(".fif")
    out_path = out_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_prep.save(out_path, overwrite=True)
    log.info(f"[{base_name}] Sauvegardé >>> {out_path}")


def main(cfg):
    log = get_logger("preprocess")

    raw_root   = Path(cfg["raw_root"])
    out_root   = Path(cfg["out_root"])
    hypno_root = Path(cfg["hypno_root"]) if cfg.get("hypno_root") else raw_root

    eeg_params  = cfg.get("eeg",  {"l_freq": 0.5, "h_freq": 80.0, "notch": 50.0})
    emg_params  = cfg.get("emg",  {"hp": 30.0, "lp": 100.0, "notch": 50.0})
    eog_params  = cfg.get("eog",  {"hp": 0.3, "lp": 10.0, "notch": None})
    yasa_params = cfg.get("yasa", {"win_sec": 4.0, "method": "covar", "threshold": 3.0, "include": "sleep"})

    n_workers = int(cfg.get("n_workers", 15))

    # Tous les EDF sous raw_root (minuscules et majuscules)
    edfs = glob.glob(str(raw_root / "**" / "*.edf"), recursive=True)
    edfs += glob.glob(str(raw_root / "**" / "*.EDF"), recursive=True)
    if not edfs:
        log.warning(f"Aucun fichier .edf/.EDF trouvé sous {raw_root}")
        return

    log.info(f"{len(edfs)} fichiers EDF trouvés. Lancement avec n_workers={n_workers}.")

    skipped_missing = []  # (base_name, missing_channels)

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(
                _process_one_edf,
                p,
                raw_root,
                out_root,
                hypno_root,
                eeg_params,
                emg_params,
                eog_params,
                yasa_params,
            ): p
            for p in edfs
        }

        for fut in as_completed(futures):
            p = futures[fut]
            try:
                fut.result()
            except Exception as e:
                msg = str(e)
                log.error(f"[ERREUR] lors du prétraitement de {p}: {msg}")

                # Détection de l'erreur "canaux manquants"
                if msg.startswith("MISSING_CHANNELS:"):
                    try:
                        _, base_name, missing_str = msg.split(":", 2)
                    except ValueError:
                        base_name = p.stem.split("_")[0]
                        missing_str = "???"
                    skipped_missing.append((base_name, missing_str))

    # Récap des patients skip pour canaux manquants
    if skipped_missing:
        log.warning("Patients SKIP pour canaux manquants (montage gp2 impossible) :")
        for base_name, missing_str in skipped_missing:
            log.warning(f"  - {base_name} : manquants = {missing_str}")



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
