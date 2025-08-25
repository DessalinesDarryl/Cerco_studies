#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segment REM (parallélisé par patient)
-------------------------------------

Ce script parcourt un dossier de fichiers EEG prétraités (.fif) et
extrait les segments de sommeil REM pour chaque patient **en parallèle**
(processus indépendants, un par patient).

Principales améliorations vs. version séquentielle :
- Exécution parallélisée par patient via `ProcessPoolExecutor`.
- Gestion robuste de Windows/macOS (start method "spawn"), `freeze_support`.
- Limitation du nombre de workers (par défaut: min(4, nb_coeurs)).
- Journalisation claire par patient + récapitulatif final.
- Saut automatique des segments déjà extraits (idempotent).
- Options CLI propres (pas d'`input()` interactif).
- Contrôle mémoire : `preload` paramétrable (par défaut True). Réduisez `--n-jobs` si besoin.

Dépendances : mne
Entrées :
- Dossier des .fif prétraités (par défaut détecté selon OS et montage)
- Dossier racine des annotations .txt (hypnogrammes)
- Dossier racine de sortie pour `rem_only/<PATIENT>/`

Exemples d'usage :
- Windows (montage bipolaire par défaut) :
  python segment_rem_parallel.py --n-jobs 4

- macOS (modifiez éventuellement le disque) :
  python segment_rem_parallel.py --montage monopolaire --n-jobs 3

"""
from __future__ import annotations

import argparse
import os
import sys
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import mne

# Import local: doit fournir une fonction get_rem_annotations(base_name, annot_dir)
# qui retourne un objet de type mne.Annotations-like avec attributs onset, duration, description
from annotations import get_rem_annotations


# -------------------- Utilitaires --------------------

def detect_disk_root() -> str:
    system = platform.system()
    if system == "Darwin":
        return "/Volumes/Crucial X6"
    elif system == "Windows":
        return "D:"
    else:
        raise RuntimeError("Système non supporté (seuls Windows et macOS sont gérés).")


def list_fif_files(fif_dir: Path, suffix: str) -> List[Path]:
    """Recherche **récursive** des .fif dans `fif_dir` et ses sous-dossiers."""
    files = [
        p for p in fif_dir.rglob(f"*_preprocessed_{suffix}.fif")
        if p.suffix == ".fif" and not p.name.startswith("._")
    ]
    return sorted(files)


@dataclass
class PatientResult:
    base_name: str
    n_segments_total: int = 0
    n_segments_saved: int = 0
    n_segments_skipped: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


# -------------------- Coeur de traitement --------------------

def extract_rem_segments(
    fif_path: Path,
    annot_dir: Path,
    output_root: Path,
    preload: bool = True,
    overwrite: bool = False,
    verbose: bool = True,
) -> PatientResult:
    """
    Extrait et sauvegarde les segments REM pour un patient (un .fif).
    Fonction **appelée dans un process dédié**.
    """
    base_name = Path(fif_path).stem.split('_')[0]
    output_dir = Path(output_root) / base_name
    output_dir.mkdir(parents=True, exist_ok=True)

    res = PatientResult(base_name=base_name)

    try:
        if verbose:
            print(f"\n[{base_name}] Chargement : {fif_path}")
        # Chargement du Raw
        raw = mne.io.read_raw_fif(fif_path, preload=preload, verbose="ERROR")
        raw.pick_types(eeg=True, eog=True, misc=True, stim=True, ecg=True, exclude=())

        # Récupération des annotations REM
        rem_annotations = get_rem_annotations(base_name, annot_dir)
        if rem_annotations is None or len(rem_annotations) == 0:
            if verbose:
                print(f"[{base_name}] Aucune période REM trouvée.")
            return res

        res.n_segments_total = len(rem_annotations.onset)
        if verbose:
            print(f"[{base_name}] {res.n_segments_total} segments REM détectés.")

        # Limites temporelles du signal
        sig_tmax = float(raw.times[-1])

        # Boucle segments
        for i, (onset, duration) in enumerate(zip(rem_annotations.onset, rem_annotations.duration), 1):
            try:
                tmin = float(onset)
                tmax = float(onset + duration)
                # Clamp aux limites du signal
                if tmin >= sig_tmax:
                    res.errors.append(f"Segment {i} hors limite (tmin >= sig_tmax).")
                    continue
                if tmax > sig_tmax:
                    tmax = sig_tmax
                if tmax <= tmin:
                    res.errors.append(f"Segment {i} invalide (tmax <= tmin).")
                    continue

                segment_path = output_dir / f"{base_name}_REM_{i}.fif"
                if segment_path.exists() and not overwrite:
                    if verbose:
                        print(f"[{base_name}] {segment_path.name} existe déjà → ignoré.")
                    res.n_segments_skipped += 1
                    continue

                if verbose:
                    print(f"[{base_name}] >>> Segment {i}: {tmin:.1f}s à {tmax:.1f}s")

                # Extraire et sauvegarder
                seg = raw.copy().crop(tmin=tmin, tmax=tmax)
                seg.save(segment_path, overwrite=True)
                res.n_segments_saved += 1

            except Exception as e:
                res.errors.append(f"Erreur segment {i}: {e}")

    except Exception as e:
        res.errors.append(f"Erreur patient: {e}")

    return res


# -------------------- Main (parallélisé) --------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extraction de segments REM en parallèle par patient.")
    parser.add_argument("--disk", type=str, default=None, help="Lettre/chemin du disque racine (détection auto si non fourni).")
    parser.add_argument("--montage", type=str, choices=["bipolaire", "monopolaire"], default="bipolaire", help="Type de montage.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Nombre de processus parallèles (par défaut: min(4, nb_coeurs)).")
    parser.add_argument("--preload", action="store_true", help="Charge les signaux en mémoire (plus rapide mais plus gourmand).")
    parser.add_argument("--no-preload", dest="preload", action="store_false", help="Ne pas précharger (économie mémoire).")
    parser.set_defaults(preload=True)
    parser.add_argument("--overwrite", action="store_true", help="Ré-écrire les segments existants.")
    parser.add_argument("--fif-dir", type=str, default=None, help="Dossier des fichiers .fif prétraités.")
    parser.add_argument("--annot-dir", type=str, default=None, help="Dossier racine contenant les annotations .txt (par patient).")
    parser.add_argument("--output-root", type=str, default=None, help="Dossier racine de sortie pour rem_only/.")
    parser.add_argument("--suffix", type=str, default=None, help="Suffixe des fichiers prétraités (bip | monop). Par défaut lié à --montage.")
    parser.add_argument("--dry-run", action="store_true", help="Lister les patients sans lancer le traitement.")
    return parser.parse_args()


def build_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, str]:
    # Déterminer disque racine
    disk = args.disk or detect_disk_root()

    # Déterminer suffix à partir du montage
    suffix = args.suffix
    if suffix is None:
        suffix = "bip" if args.montage == "bipolaire" else "monop"

    # Dossiers par défaut si non fournis
    if args.fif_dir is None:
        fif_dir = Path(f"{disk}/EEG/preprocessed/{args.montage}/full")
    else:
        fif_dir = Path(args.fif_dir)

    if args.annot_dir is None:
        annot_dir = Path(f"{disk}/EEG/raw")
    else:
        annot_dir = Path(args.annot_dir)

    if args.output_root is None:
        output_root = Path(f"{disk}/EEG/preprocessed/{args.montage}/rem_only")
    else:
        output_root = Path(args.output_root)

    return fif_dir, annot_dir, output_root, suffix


def main():
    # Windows: nécessaire pour le multiprocessing avec spawn
    try:
        import multiprocessing as mp
        # Sur Windows, 'spawn' est déjà le défaut; on explicite pour clarté
        if platform.system() == "Windows":
            mp.freeze_support()
    except Exception:
        pass

    mne.set_log_level("WARNING")

    args = parse_args()
    fif_dir, annot_dir, output_root, suffix = build_paths(args)

    # Listing des fichiers patients
    fif_files = list_fif_files(fif_dir, suffix)
    if len(fif_files) == 0:
        print(f"Aucun fichier .fif trouvé avec suffix '{suffix}' dans {fif_dir}")
        sys.exit(0)

    print(f"Trouvé {len(fif_files)} fichier(s) .fif à traiter (recherche récursive) dans {fif_dir}")
    if args.dry_run:
        for p in fif_files:
            try:
                rel = p.relative_to(fif_dir)
            except Exception:
                rel = p
            print("-", rel)
        sys.exit(0)

    # Nombre de workers
    max_default = max(1, min(4, (os.cpu_count() or 1)))
    n_jobs = args.n_jobs or max_default
    n_jobs = max(1, n_jobs)
    print(f"Lancement en parallèle avec n_jobs={n_jobs} | preload={'True' if args.preload else 'False'} | overwrite={'True' if args.overwrite else 'False'}")

    results: List[PatientResult] = []

    # Exécution parallélisée par patient
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        fut2path = {
            ex.submit(
                extract_rem_segments,
                fif_path=p,
                annot_dir=annot_dir,
                output_root=output_root,
                preload=args.preload,
                overwrite=args.overwrite,
                verbose=True,
            ): p
            for p in fif_files
        }

        for fut in as_completed(fut2path):
            p = fut2path[fut]
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:
                base = p.stem.split('_')[0]
                results.append(PatientResult(base_name=base, errors=[f"Exception non gérée: {e}"]))

    # Récapitulatif
    print("\n===== RÉCAPITULATIF =====")
    total_patients = len(results)
    total_segments = sum(r.n_segments_total for r in results)
    total_saved = sum(r.n_segments_saved for r in results)
    total_skipped = sum(r.n_segments_skipped for r in results)
    total_errors = sum(len(r.errors) for r in results)

    print(f"Patients traités : {total_patients}")
    print(f"Segments détectés : {total_segments}")
    print(f"Segments sauvegardés : {total_saved}")
    print(f"Segments ignorés : {total_skipped}")
    print(f"Erreurs (total) : {total_errors}")

    if total_errors:
        print("\n--- Détails des erreurs ---")
        for r in results:
            if r.errors:
                print(f"[{r.base_name}] ")
                for e in r.errors:
                    print("  -", e)


if __name__ == "__main__":
    main()
