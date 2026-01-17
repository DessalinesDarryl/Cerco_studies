#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de prétraitement batch de fichiers EDF.

Fonctionnement général :
- Recherche récursive de tous les fichiers EDF (.edf / .EDF) sous un dossier racine.
- Un fichier EDF = un enregistrement de patient.
- Lancement du prétraitement en parallèle (multi-process).
- Sauvegarde des données prétraitées au format .fif.
- Conservation de l’arborescence relative entre données brutes et prétraitées.
- Gestion contrôlée des erreurs (ex. canaux manquants).

Le traitement EEG/EMG/EOG réel est implémenté dans :
    src.data.preprocessing.preprocess_record
"""

import glob
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from typing import Dict, List, Tuple, Optional

import mne

# ============================================================================
# Ajout du répertoire racine du projet au PYTHONPATH
# Permet d’importer les modules internes via "src.*"
# ============================================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # ex : .../Cerco_studies
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml, add_common_args
from src.utils.logging import get_logger
from src.data.preprocessing import preprocess_record


def _extract_base_name(edf_path: Path) -> str:
    """
    Extrait l’identifiant patient (base_name) à partir du nom du fichier EDF.

    Convention attendue :
        AN166_raw.edf  →  base_name = "AN166"

    Paramètres
    ----------
    edf_path : Path
        Chemin vers le fichier EDF.

    Retour
    ------
    str
        Identifiant patient utilisé pour :
        - les logs
        - la recherche des fichiers d’annotations / hypnogrammes

    Remarque
    --------
    Si la convention de nommage change, c’est CETTE fonction
    qu’il faut modifier (point centralisé).
    """
    return edf_path.stem.split("_")[0]


def _find_edf_files(raw_root: Path) -> List[Path]:
    """
    Recherche récursive de tous les fichiers EDF sous un dossier racine.

    Paramètres
    ----------
    raw_root : Path
        Dossier racine contenant les données brutes.

    Retour
    ------
    List[Path]
        Liste des chemins vers tous les fichiers .edf et .EDF trouvés.
    """
    edfs = glob.glob(str(raw_root / "**" / "*.edf"), recursive=True)
    edfs += glob.glob(str(raw_root / "**" / "*.EDF"), recursive=True)
    return [Path(p) for p in edfs]


def _build_output_path(edf_path: Path, raw_root: Path, out_root: Path) -> Path:
    """
    Construit le chemin de sortie (.fif) en conservant l’arborescence relative.

    Exemple
    -------
    raw_root/A/B/file.edf  →  out_root/A/B/file.fif

    Paramètres
    ----------
    edf_path : Path
        Chemin du fichier EDF d’entrée.
    raw_root : Path
        Dossier racine des données brutes.
    out_root : Path
        Dossier racine des données prétraitées.

    Retour
    ------
    Path
        Chemin complet du fichier .fif de sortie.
    """
    rel = edf_path.relative_to(raw_root).with_suffix(".fif")
    return out_root / rel


def _process_one_edf(
    p: Path,
    raw_root: Path,
    out_root: Path,
    hypno_root: Path,
    eeg_params: Dict,
    emg_params: Dict,
    eog_params: Dict,
    yasa_params: Dict,
) -> None:
    

    """
    Fonction exécutée dans un worker : prétraitement d’un fichier EDF unique.

    Étapes :
    1) Lecture du fichier EDF avec MNE (preload=True).
    2) Application du pipeline de prétraitement (preprocess_record).
    3) Sauvegarde du résultat au format .fif.

    Paramètres
    ----------
    p : Path
        Chemin du fichier EDF à traiter.
    raw_root : Path
        Dossier racine des EDF.
    out_root : Path
        Dossier racine de sortie (.fif).
    hypno_root : Path
        Dossier racine pour localiser les hypnogrammes / annotations.
    eeg_params, emg_params, eog_params, yasa_params : dict
        Paramètres de traitement transmis à preprocess_record.
        Ces paramètres sont définis dans le fichier de configuration YAML > configs/preproc.

    Exceptions
    ----------
    Toute exception est propagée au processus principal.
    Les erreurs commençant par "MISSING_CHANNELS:" sont interprétées
    comme des échecs contrôlés (canaux requis absents).
    """


    # Début du code 

    log = get_logger("preprocess")
    p = Path(p)

    base_name = _extract_base_name(p)
    log.info(f"=== Prétraitement {base_name} ({p.name}) ===")

    # Lecture du fichier EDF (chargement complet en mémoire)
    raw = mne.io.read_raw_edf(p, preload=True, verbose=False)

    # Application du prétraitement EEG / EMG / EOG
    raw_prep = preprocess_record(
        raw=raw,
        base_name=base_name,
        hypno_root=hypno_root,
        eeg_params=eeg_params,
        emg_params=emg_params,
        eog_params=eog_params,
        yasa_params=yasa_params,
    )

    # Construction du chemin de sortie et création des dossiers si besoin
    out_path = _build_output_path(p, raw_root, out_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sauvegarde au format .fif
    raw_prep.save(out_path, overwrite=True)
    log.info(f"[{base_name}] Sauvegardé >>> {out_path}")


def _parse_missing_channels_error(
    msg: str,
    fallback_base_name: str
) -> Optional[Tuple[str, str]]:
    """
    Analyse une erreur indiquant des canaux manquants.

    Format attendu :
        MISSING_CHANNELS:<base_name>:<liste_canaux>

    Exemple :
        MISSING_CHANNELS:AN166:F3,F4,C3,C4

    Paramètres
    ----------
    msg : str
        Message de l’exception.
    fallback_base_name : str
        Identifiant patient utilisé si le parsing échoue.

    Retour
    ------
    (base_name, canaux_manquants) ou None
    """
    if not msg.startswith("MISSING_CHANNELS:"):
        return None

    try:
        _, base_name, missing_str = msg.split(":", 2)
        return base_name, missing_str
    except ValueError:
        return fallback_base_name, "???"


def main(cfg: Dict) -> None:
    """
    Point d’entrée principal du script.

    Responsabilités :
    - Chargement des chemins et paramètres depuis la config YAML.
    - Découverte des fichiers EDF.
    - Lancement du prétraitement en parallèle.
    - Collecte et résumé des patients ignorés pour canaux manquants.
    """
    log = get_logger("preprocess")

    # --- Résolution des chemins ---
    raw_root = Path(cfg["raw_root"])
    out_root = Path(cfg["out_root"])
    hypno_root = Path(cfg["hypno_root"]) if cfg.get("hypno_root") else raw_root

    # --- Paramètres par défaut (surchargés via YAML si présents) ---
    eeg_params = cfg.get("eeg", {"l_freq": 0.5, "h_freq": 80.0, "notch": 50.0})
    emg_params = cfg.get("emg", {"hp": 30.0, "lp": 100.0, "notch": 50.0})
    eog_params = cfg.get("eog", {"hp": 0.3, "lp": 10.0, "notch": None})
    yasa_params = cfg.get(
        "yasa",
        {"win_sec": 4.0, "method": "covar", "threshold": 3.0, "include": "sleep"},
    )

    n_workers = int(cfg.get("n_workers", 15)) # 15 processus en parallèle

    # --- Recherche des EDF ---
    edfs = _find_edf_files(raw_root)
    if not edfs:
        log.warning(f"Aucun fichier .edf/.EDF trouvé sous {raw_root}")
        return

    log.info(f"{len(edfs)} fichiers EDF trouvés. Lancement avec n_workers={n_workers}.")

    skipped_missing: List[Tuple[str, str]] = []

    # --- Exécution parallèle ---
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

                base_name = _extract_base_name(Path(p))
                parsed = _parse_missing_channels_error(msg, base_name)
                if parsed is not None:
                    skipped_missing.append(parsed)

    # --- Résumé final ---
    if skipped_missing:
        log.warning("Patients ignorés (canaux manquants, montage impossible) :")
        for base_name, missing_str in skipped_missing:
            log.warning(f"  - {base_name} : manquants = {missing_str}")


if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument(
        "--n_workers",
        type=int,
        default=15,
        help="Nombre de workers en parallèle (par défaut : 15).",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg["n_workers"] = args.n_workers

    main(cfg)
