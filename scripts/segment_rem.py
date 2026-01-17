#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
segment_rem.py — Option A : pas de concaténation (REM-only en epochs fixes)

Résumé
-------
Ce script extrait des **époques REM** (par défaut 4 secondes, non chevauchantes)
à partir de fichiers `.fif` **prétraités**, en **excluant** les segments annotés
comme artefacts.

L’idée est de produire un fichier Epochs par patient, prêt à être utilisé
ensuite pour l’extraction de features / RSWA / classification, sans recourir à
une concaténation de segments.

Entrées attendues
-----------------
- Fichiers MNE `.fif` prétraités (un par patient), contenant des annotations MNE :
    - "REM"       : segments REM (onset/duration en secondes)
    - "ARTEFACT"  : segments à exclure

Sorties
-------
- Pour chaque fichier `{patient}.fif` :
    - `{patient}_REM-epo.fif` : objet `mne.Epochs` contenant uniquement
      des epochs de longueur fixe (par défaut 4 s) entièrement incluses dans
      des intervalles REM **nettoyés** des artefacts.

Principe / Pipeline
-------------------
1) Lecture du Raw (.fif) prétraité
2) Extraction des intervalles REM via annotations "REM"
3) Extraction des intervalles artefacts via annotations "ARTEFACT"
4) Soustraction (REM \ ARTEFACT) => intervalles REM “propres”
5) Découpage en epochs fixes à l’intérieur des intervalles REM propres
6) Sauvegarde au format `.fif` (Epochs)

Remarques importantes
---------------------
- Les epochs sont créées uniquement si elles rentrent entièrement dans les
  intervalles REM propres (pas d’epoch partielle à la fin d’un intervalle).
- Les annotations doivent être présentes dans `raw.annotations`.
- Le `tmax` MNE est **inclusif** ; on utilise donc `tmax = epoch_len - 1/sfreq`
  pour obtenir exactement `epoch_len` secondes (évite un off-by-one).

Exemple CLI
-----------
    python scripts/segment_rem.py --config path/to/config.yaml --n_workers 15

Le fichier YAML doit contenir au minimum :
    in_root:  /chemin/vers/les/fif
    out_root: /chemin/vers/les/outputs
Optionnel :
    epoch_len_s: 4.0
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Ajout du dossier racine du projet au PYTHONPATH pour permettre
# l'import des modules internes (src.utils.*).
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np

from src.utils.config import load_yaml, add_common_args
from src.utils.logging import get_logger


# =====================================================================
# Types utiles
# =====================================================================
Interval = Tuple[float, float]  # (start_sec, end_sec)


# =====================================================================
# Fonctions utilitaires
# =====================================================================
def get_intervals_from_annotations(raw: mne.io.BaseRaw, label: str) -> List[Interval]:
    """Extrait les intervalles (start, end) d’un label d’annotation MNE.

    Cette fonction parcourt `raw.annotations` et sélectionne les annotations
    dont `description == label`. Elle retourne une liste d’intervalles en secondes.

    Args:
        raw: Objet MNE Raw contenant des annotations (`raw.annotations`).
        label: Libellé exact des annotations à extraire (ex: "REM", "ARTEFACT").

    Returns:
        Liste d’intervalles (start_sec, end_sec), avec end_sec > start_sec.

    Notes:
        - `onset` est en secondes (float)
        - `duration` est en secondes
        - Les annotations à durée nulle ou négative sont ignorées.
    """
    intervals: List[Interval] = []
    for ann in raw.annotations:
        if ann["description"] == label:
            start = float(ann["onset"])
            end = float(ann["onset"] + ann["duration"])
            if end > start:
                intervals.append((start, end))
    return intervals


def subtract_intervals(base_intervals: List[Interval], remove_intervals: List[Interval]) -> List[Interval]:
    """Soustrait des intervalles à supprimer d’une liste d’intervalles de base.

    Exemple:
        base   = [(10, 40)]
        remove = [(20, 25)]
        => [(10, 20), (25, 40)]

    Args:
        base_intervals: Intervalles de départ (ex: segments REM).
        remove_intervals: Intervalles à retirer (ex: segments ARTEFACT).

    Returns:
        Intervalles résultants après soustraction, nettoyés (end > start).

    Notes:
        - Cette implémentation gère plusieurs intervalles et recouvrements
          potentiels.
        - Les remove_intervals ne sont pas fusionnés : ce n’est pas obligatoire
          ici, mais possible si un jour tu veux optimiser.
    """
    if not base_intervals:
        return []

    out: List[Interval] = []

    # Pour chaque segment de base, on retire tous les segments à enlever
    for b_start, b_end in base_intervals:
        segments = [(b_start, b_end)]

        for r_start, r_end in remove_intervals:
            new_segments: List[Interval] = []

            for s, e in segments:
                # Cas 1 : aucun recouvrement
                if r_end <= s or r_start >= e:
                    new_segments.append((s, e))
                    continue

                # Cas 2 : recouvrement : on garde éventuellement la partie gauche
                if r_start > s:
                    new_segments.append((s, r_start))

                # Cas 3 : recouvrement : on garde éventuellement la partie droite
                if r_end < e:
                    new_segments.append((r_end, e))

            segments = new_segments

        out.extend(segments)

    # Nettoyage final : on enlève les segments vides / inversés
    out = [(s, e) for s, e in out if e > s]
    return out


def make_fixed_length_epochs(
    raw: mne.io.BaseRaw,
    intervals: List[Interval],
    epoch_len: float = 4.0,
    event_name: str = "REM",
) -> Optional[mne.Epochs]:
    """Crée des epochs fixes (non chevauchantes) dans une liste d’intervalles.

    On parcourt chaque intervalle (s, e) et on génère des fenêtres :
        [s, s+epoch_len], [s+epoch_len, s+2*epoch_len], ...,
    uniquement si la fenêtre est entièrement incluse dans l’intervalle.

    Args:
        raw: Objet Raw MNE (préchargé conseillé).
        intervals: Liste d’intervalles en secondes (start, end).
        epoch_len: Durée de chaque epoch en secondes (défaut: 4.0).
        event_name: Nom de l’événement MNE (défaut: "REM").

    Returns:
        Objet mne.Epochs si au moins un epoch est créé, sinon None.

    Notes techniques (MNE):
        - `events` est en indices d’échantillons (sample).
        - `tmax` est inclusif dans MNE. Pour obtenir une durée exacte de
          `epoch_len` secondes, on utilise:
              tmax = epoch_len - 1/sfreq
    """
    if not intervals:
        return None

    sfreq = float(raw.info["sfreq"])
    if sfreq <= 0:
        raise ValueError("sfreq invalide dans raw.info['sfreq'].")

    # Construction de la matrice events [sample, 0, event_id]
    events: List[List[int]] = []
    event_id = {event_name: 1}

    # Commentaire EEG:
    # - On convertit les temps en secondes -> indices d’échantillons.
    # - On crée un événement au début de chaque epoch.
    for start_sec, end_sec in intervals:
        t = start_sec
        while t + epoch_len <= end_sec:
            sample = int(round(t * sfreq))
            events.append([sample, 0, 1])
            t += epoch_len

    if not events:
        return None

    events_arr = np.asarray(events, dtype=int)

    # Picks : on récupère tout ce qui est utile selon ton pipeline
    # (EEG/EOG/EMG/ECG/MISC). Adapter si tu veux exclure certains types.
    picks = mne.pick_types(
        raw.info,
        eeg=True,
        eog=True,
        emg=True,
        ecg=True,
        misc=True,
        exclude=[],
    )

    # Point important :
    # MNE a un tmax inclusif => pour une fenêtre de N secondes exactes,
    # il faut retirer un pas d'échantillonnage.
    tmax = epoch_len - (1.0 / sfreq)

    epochs = mne.Epochs(
        raw,
        events=events_arr,
        event_id=event_id,
        tmin=0.0,
        tmax=tmax,
        baseline=None,
        picks=picks,
        preload=True,
        verbose=False,
    )

    return epochs


# =====================================================================
# Traitement "un fichier"
# =====================================================================
def _process_one_fif(fif_path: Path, out_root: Path, epoch_len: float) -> None:
    """Traite un fichier .fif : REM \ ARTEFACT -> epochs fixes -> sauvegarde.

    Args:
        fif_path: Chemin vers le fichier `.fif` prétraité.
        out_root: Dossier de sortie.
        epoch_len: Durée des epochs en secondes.

    Returns:
        None. (Sauvegarde sur disque + logs.)
    """
    log = get_logger("segment_rem")

    base = fif_path.stem
    log.info(f"---- {base} ----")

    # Lecture du Raw : preload=True pour pouvoir épocher rapidement ensuite
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

    # 1) Intervalles REM
    rem_intervals = get_intervals_from_annotations(raw, "REM")
    if not rem_intervals:
        log.warning(f"[{base}] Aucun segment REM trouvé -> skip")
        return

    # 2) Intervalles artefacts (optionnel : peut être vide)
    artefact_intervals = get_intervals_from_annotations(raw, "ARTEFACT")

    # 3) Soustraction REM \ ARTEFACT
    # Commentaire méthodo :
    # On retire les fenêtres artefactées des segments REM, afin que les epochs
    # extraites soient "propres".
    rem_clean = subtract_intervals(rem_intervals, artefact_intervals)
    if not rem_clean:
        log.warning(f"[{base}] Tous les REM sont artefactués -> skip")
        return

    # 4) Époquage
    epochs = make_fixed_length_epochs(raw, rem_clean, epoch_len=epoch_len, event_name="REM")
    if epochs is None or len(epochs) == 0:
        log.warning(f"[{base}] Aucun epoch extrait -> skip")
        return

    # 5) Sauvegarde
    out_path = out_root / f"{base}_REM-epo.fif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # overwrite=True : le comportement est volontaire pour reruns / debug
    epochs.save(out_path, overwrite=True)
    log.info(f"[{base}] OK -> {out_path} ({len(epochs)} epochs)")


# =====================================================================
# Main
# =====================================================================
def main(cfg: Dict) -> None:
    """Point d’entrée principal (config YAML + parallélisation).

    Args:
        cfg: Dictionnaire de configuration issu du YAML.

    Returns:
        None.
    """
    log = get_logger("segment_rem")

    in_root = Path(cfg["in_root"])
    out_root = Path(cfg["out_root"])
    epoch_len = float(cfg.get("epoch_len_s", 4.0))
    n_workers = int(cfg.get("n_workers", 15))

    # Recherche récursive
    fif_files = sorted(in_root.glob("**/*.fif"))
    if not fif_files:
        log.warning(f"Aucun fichier FIF trouvé dans {in_root}")
        return

    log.info(f"{len(fif_files)} fichiers FIF trouvés. Lancement avec n_workers={n_workers}.")

    # Parallélisation par patient
    # Commentaire perf :
    # - Chaque worker lit un Raw et génère des epochs : CPU + I/O.
    # - Ajuste n_workers selon ta RAM (preload=True) et la taille des fichiers.
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(_process_one_fif, fif_path, out_root, epoch_len): fif_path
            for fif_path in fif_files
        }

        for fut in as_completed(futures):
            fif = futures[fut]
            try:
                fut.result()
            except Exception as e:
                log.error(f"[ERREUR] sur {fif}: {e}", exc_info=True)


if __name__ == "__main__":
    # Arguments communs via ton utilitaire (probablement --config)
    ap = add_common_args(argparse.ArgumentParser(description="Extract clean REM fixed-length epochs from preprocessed FIFs."))
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
