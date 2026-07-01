#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
02_segment_rem.py  -  Extraction des époques REM
=============================================================================

Ce script est AUTONOME : il ne dépend d'aucun dossier src/, ni de fichier YAML.

Ce qu'il fait :
  - Recherche tous les fichiers .fif prétraités sous PREPROCESSED_ROOT
  - Pour chaque patient :
      1. Lecture du fichier .fif
      2. Extraction des intervalles REM (annotation "REM")
      3. Exclusion des intervalles artefactés (annotation "ARTEFACT")
      4. Découpage en fenêtres fixes de EPOCH_LEN_S secondes
      5. Sauvegarde au format _REM-epo.fif dans REM_EPO_ROOT

Comment utiliser ce script :
  1. Lancer 01_preprocess.py en premier
  2. Modifier les chemins dans la section "PARAMÈTRES" ci-dessous
  3. Lancer :  python scripts/scripts_fonctionnels/02_segment-rem.py 2>&1 | tee logfiles/log_02-segment-rem.txt

Dépendances requises :
  pip install mne numpy

=============================================================================

INPUTS
------
PREPROCESSED_ROOT : fichiers .fif prétraités (sortie de 01_preprocess.py)
                    Les fichiers _REM-epo.fif existants sont automatiquement exclus.
                    Exemple : data/preprocessed/AN166/AN166_raw.fif
 
OUTPUTS
-------
REM_EPO_ROOT : fichiers d'époques REM au format MNE Epochs
               Nommage   : {patient}_REM-epo.fif
               Exemple   : data/preprocessed/rem_epo/AN166_raw_REM-epo.fif
               Contenu   : N époques de EPOCH_LEN_S secondes, extraites des
                           segments REM nettoyés des artefacts
 
DÉPENDANCES PIPELINE
--------------------
Ce script est l'ÉTAPE 2. Il requiert la sortie de 01_preprocess.py.
Sa sortie alimente :
  > 04_extract_features.py  (rem_epo/ > features EEG)
"""

# ============================================================================
# PARAMÈTRES  <<<  À MODIFIER SELON VOTRE LA CONFIGURATION SOUHAITÉE
# ============================================================================

# Dossier contenant les fichiers .fif prétraités (sortie de 01_preprocess.py)
PREPROCESSED_ROOT = r"c:\dev\raw\data_raw_fif"

# Dossier de sortie pour les époques REM
REM_EPO_ROOT = r"c:\dev\Cerco_studies\data\rem_epo"

# Durée de chaque epoch en secondes (fenêtre fixe, non chevauchante)
EPOCH_LEN_S = 4.0

# Nombre de patients traités en parallèle
N_WORKERS = 4  # Réduire si votre machine manque de RAM

# ============================================================================
# IMPORTS
# ============================================================================

import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import mne
import numpy as np

# ============================================================================
# LOGGER
# ============================================================================

def _get_logger(name: str = "segment_rem") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


# ============================================================================
# TYPE UTILITAIRE
# ============================================================================

Interval = Tuple[float, float]  # (start_sec, end_sec)


# ============================================================================
# MANIPULATION DES INTERVALLES
# ============================================================================

def _get_intervals_from_annotations(raw: mne.io.BaseRaw, label: str) -> List[Interval]:
    """Extrait les intervalles (start_s, end_s) correspondant à un label d'annotation."""
    intervals: List[Interval] = []
    for ann in raw.annotations:
        if ann["description"] == label:
            start = float(ann["onset"])
            end = float(ann["onset"] + ann["duration"])
            if end > start:
                intervals.append((start, end))
    return intervals


def _subtract_intervals(base: List[Interval], remove: List[Interval]) -> List[Interval]:
    """
    Soustrait des intervalles à retirer d'une liste d'intervalles de base.

    Exemple :
        base   = [(10, 40)]
        remove = [(20, 25)]
        > [(10, 20), (25, 40)]
    """
    if not base:
        return []

    out: List[Interval] = []
    for b_start, b_end in base:
        segments = [(b_start, b_end)]

        for r_start, r_end in remove:
            new_segments: List[Interval] = []
            for s, e in segments:
                if r_end <= s or r_start >= e:
                    # Pas de recouvrement
                    new_segments.append((s, e))
                else:
                    # Partie gauche (si elle existe)
                    if r_start > s:
                        new_segments.append((s, r_start))
                    # Partie droite (si elle existe)
                    if r_end < e:
                        new_segments.append((r_end, e))
            segments = new_segments

        out.extend(segments)

    return [(s, e) for s, e in out if e > s]


# ============================================================================
# CRÉATION DES ÉPOQUES
# ============================================================================

def _make_fixed_length_epochs(
    raw: mne.io.BaseRaw,
    intervals: List[Interval],
    epoch_len: float,
) -> Optional[mne.Epochs]:
    """
    Crée des époques de durée fixe (non chevauchantes) à l'intérieur des intervalles.

    Seules les fenêtres entièrement contenues dans un intervalle sont gardées.
    Le tmax MNE est inclusif : on utilise tmax = epoch_len - 1/sfreq pour
    obtenir exactement epoch_len secondes.
    """
    if not intervals:
        return None

    sfreq = float(raw.info["sfreq"])
    if sfreq <= 0:
        raise ValueError("sfreq invalide.")

    events: List[List[int]] = []
    for start_sec, end_sec in intervals:
        t = start_sec
        while t + epoch_len <= end_sec:
            sample = int(round(t * sfreq))
            events.append([sample, 0, 1])
            t += epoch_len

    if not events:
        return None

    events_arr = np.asarray(events, dtype=int)
    picks = mne.pick_types(raw.info, eeg=True, eog=True, emg=True,
                           ecg=True, misc=True, exclude=[])
    tmax = epoch_len - (1.0 / sfreq)

    epochs = mne.Epochs(
        raw,
        events=events_arr,
        event_id={"REM": 1},
        tmin=0.0,
        tmax=tmax,
        baseline=None,
        picks=picks,
        preload=True,
        verbose=False,
    )
    return epochs


# ============================================================================
# TRAITEMENT D'UN FICHIER .FIF (worker)
# ============================================================================

def _process_one_fif(fif_path_str: str) -> None:
    """
    Worker : pour un fichier .fif, extrait les époques REM propres et sauvegarde.

    Pipeline :
      REM annotations \ ARTEFACT annotations > intervalles propres
      > fenêtres fixes de EPOCH_LEN_S secondes
      > sauvegarde _REM-epo.fif
    """
    log = _get_logger("segment_rem")
    fif_path = Path(fif_path_str)
    out_root = Path(REM_EPO_ROOT)

    base = fif_path.stem
    log.info(f"---- {base} ----")

    # Lecture du fichier .fif prétraité
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

    # 1) Intervalles REM
    rem_intervals = _get_intervals_from_annotations(raw, "REM")
    if not rem_intervals:
        log.warning(f"[{base}] Aucun segment REM trouvé > ignoré")
        return

    # 2) Intervalles artefacts (peut être vide)
    artefact_intervals = _get_intervals_from_annotations(raw, "ARTEFACT")

    # 3) REM "propre" = REM sans les artefacts
    rem_clean = _subtract_intervals(rem_intervals, artefact_intervals)
    if not rem_clean:
        log.warning(f"[{base}] Tous les segments REM sont artefactés > ignoré")
        return

    # 4) Découpage en époques fixes
    epochs = _make_fixed_length_epochs(raw, rem_clean, epoch_len=EPOCH_LEN_S)
    if epochs is None or len(epochs) == 0:
        log.warning(f"[{base}] Aucune epoch extraite > ignoré")
        return

    # 5) Sauvegarde
    out_path = out_root / f"{base}_REM-epo.fif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs.save(out_path, overwrite=True)
    log.info(f"[{base}] OK > {out_path} ({len(epochs)} époques)")


# ============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================================

def main() -> None:
    log = _get_logger("segment_rem")

    in_root = Path(PREPROCESSED_ROOT)

    # Recherche des fichiers .fif (exclut les fichiers _REM-epo.fif déjà générés)
    fif_files = [
        p for p in sorted(in_root.glob("**/*.fif"))
        if "_REM-epo" not in p.name
    ]

    if not fif_files:
        log.warning(f"Aucun fichier .fif trouvé dans {in_root}")
        return

    log.info(f"{len(fif_files)} fichiers .fif trouvés. Lancement avec {N_WORKERS} worker(s).")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_process_one_fif, str(p)): p for p in fif_files}

        for fut in as_completed(futures):
            fif = futures[fut]
            try:
                fut.result()
            except Exception as e:
                log.error(f"[ERREUR] {fif.name} : {e}", exc_info=True)

    log.info("Segmentation REM terminée.")


if __name__ == "__main__":
    main()
