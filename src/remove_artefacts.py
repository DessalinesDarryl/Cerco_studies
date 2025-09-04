#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remove_artefacts.py — Détection + annotation des artéfacts, puis export:
  - fichier .fif annoté (ARTEFACT),
  - fichier .fif “clean” (segments artefactués retirés).

Entrées : .edf/.EDF ou .fif
Sorties (par fichier traité), dans --out_dir/{patient}/ :
  - {base}_art_annotated.fif
  - {base}_art_removed.fif
  - {base}_artifact_windows.csv
  - {patient}_artifact_report.txt (append: % signal conservé)

Parallélisation calquée sur bipolar_from_edf_batch.py (Pool spawn + initializer).
"""

from __future__ import annotations
import os
import multiprocessing as mp
import tempfile
import argparse
from pathlib import Path
import numpy as np
import mne


# ===================== Détection par YASA =====================

def detect_artifacts_windows(raw: mne.io.BaseRaw, win_sec: float = 4.0,
                             method: str = "covar", threshold: float = 3.0):
    try:
        import yasa
    except Exception as e:
        raise RuntimeError("Le module 'yasa' est requis (`pip install yasa`).") from e

    sf = float(raw.info["sfreq"])
    data = raw.get_data()
    n_samples = data.shape[1]

    art, _ = yasa.art_detect(data, sf=sf, window=win_sec, method=method, threshold=threshold)

    t_end = n_samples / sf
    windows_s = []
    for i, is_art in enumerate(art):
        if is_art:
            s = i * win_sec
            e = min((i + 1) * win_sec, t_end)
            if e > s:
                windows_s.append((float(s), float(e)))
    return art, windows_s


# ===================== Outils d'intervalles =====================

def merge_intervals(intervals):
    if not intervals:
        return []
    ints = sorted(intervals, key=lambda x: x[0])
    merged = [ints[0]]
    for s, e in ints[1:]:
        s0, e0 = merged[-1]
        if s <= e0:
            merged[-1] = (s0, max(e0, e))
        else:
            merged.append((s, e))
    return merged

def complement_intervals(intervals, t_end):
    if not intervals:
        return [(0.0, float(t_end))]
    merged = merge_intervals(intervals)
    good = []
    last = 0.0
    for s, e in merged:
        if s > last:
            good.append((last, s))
        last = max(last, e)
    if last < t_end:
        good.append((last, t_end))
    return [(float(s), float(e)) for s, e in good if e > s]


# ===================== Sauvegardes =====================

def save_artifacts_csv(windows_s, out_csv):
    import csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_time_s", "end_time_s"])
        for s, e in windows_s:
            w.writerow([f"{s:.6f}", f"{e:.6f}"])

def annotate_raw(raw, windows_s):
    """Ajoute des annotations 'ARTEFACT' en respectant l'orig_time existant."""
    if not windows_s:
        return raw.copy()

    onset = [s for s, _ in windows_s]
    duration = [e - s for s, e in windows_s]
    desc = ["ARTEFACT"] * len(onset)

    # IMPORTANT : aligner orig_time sur celle des annotations déjà présentes
    existing = raw.annotations if getattr(raw, "annotations", None) is not None else None
    orig_time = existing.orig_time if existing is not None else None

    ann = mne.Annotations(onset=onset, duration=duration, description=desc, orig_time=orig_time)

    out = raw.copy()
    if existing is None or len(existing) == 0:
        out.set_annotations(ann)
    else:
        out.set_annotations(existing + ann) 
    return out


def cut_out_artifacts(raw, windows_s):
    if not windows_s:
        return raw.copy()
    good = complement_intervals(windows_s, t_end=float(raw.times[-1]))
    parts = []
    sf = float(raw.info["sfreq"])
    eps = 1.0 / sf
    for s, e in good:
        if e - s <= 0:
            continue
        seg = raw.copy().crop(tmin=float(s), tmax=float(e - eps), verbose="ERROR")
        parts.append(seg)
    if not parts:
        raise RuntimeError("Tous les segments sont artefactués, rien à conserver.")
    return mne.concatenate_raws(parts, verbose="ERROR")


# ===================== I/O helpers =====================

def load_raw_any(path: Path) -> mne.io.BaseRaw:
    if path.suffix.lower() == ".fif":
        return mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    elif path.suffix.lower() == ".edf":
        return mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    else:
        raise ValueError(f"Extension non supportée: {path.suffix}")

def derive_patient_and_base(in_path: Path):
    patient = in_path.parent.name
    base = in_path.stem
    return patient, base


# ===================== Découverte (style patients) =====================

def discover_items(in_root: Path) -> list[tuple[str, Path]]:
    """Retourne [(base, path)] en parcourant in_root :
       - Priorité aux fichiers *_bipolar.fif dans chaque sous-dossier patient
       - Sinon meilleur .fif
       - Sinon *_raw.edf / .EDF le plus gros
       - Fallback : fichiers à la racine (base = préfixe avant '_' si présent)
    """
    mapping = {}

    # 1) sous-dossiers par patient
    for child in sorted(in_root.iterdir()):
        if not child.is_dir():
            continue
        cands = []
        cands += list(child.glob("*_bipolar.fif"))
        if not cands:
            cands += list(child.glob("*.fif"))
        if not cands:
            cands += list(child.glob("*_raw.edf"))
            cands += list(child.glob("*_raw.EDF"))
        if cands:
            mapping[child.name] = max(cands, key=lambda p: p.stat().st_size)

    # 2) fichiers à la racine
    root_fifs = list(in_root.glob("*.fif"))
    root_edfs = list(in_root.glob("*_raw.edf")) + list(in_root.glob("*_raw.EDF"))
    for p in root_fifs + root_edfs:
        base = p.stem.split("_")[0]
        cur = mapping.get(base)
        if cur is None or p.stat().st_size > cur.stat().st_size:
            mapping[base] = p

    return sorted(mapping.items())


# ===================== Worker (style bipolar) =====================

SKIPPED = None  # liste partagée entre workers

def process_one(item, out_root: Path, win_sec: float, method: str, threshold: float, drop_prefix: bool):
    base, in_path = item
    out_dir = out_root / base
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw = load_raw_any(in_path)

        # Renommage optionnel "EEG " -> "" (utile si entrée = EDF bruts)
        if drop_prefix:
            ren = {ch: ch.replace("EEG ", "", 1) for ch in raw.ch_names if ch.startswith("EEG ")}
            if ren:
                raw.rename_channels(ren)

        # === Filtrage en amont ===
        raw.filter(l_freq=0.3, h_freq=100.0, fir_design="firwin", verbose="ERROR")
        raw.notch_filter(50.0, verbose="ERROR")

        # Détection artefacts
        _, art_windows = detect_artifacts_windows(raw, win_sec=win_sec, method=method, threshold=threshold)
        art_windows = merge_intervals(art_windows)

        # Sauvegardes
        out_csv = out_dir / f"{base}_artifact_windows.csv"
        save_artifacts_csv(art_windows, out_csv)

        raw_ann = annotate_raw(raw, art_windows)
        out_annot = out_dir / f"{base}_art_annotated.fif"
        raw_ann.save(out_annot, overwrite=True, verbose="ERROR")

        raw_clean = cut_out_artifacts(raw, art_windows)
        out_clean = out_dir / f"{base}_art_removed.fif"
        raw_clean.save(out_clean, overwrite=True, verbose="ERROR")

        # Rapport
        t_total = float(raw.times[-1])
        bad_total = float(sum((e - s) for s, e in art_windows))
        good_total = max(0.0, t_total - bad_total)
        pct_keep = (good_total / t_total * 100.0) if t_total > 0 else 0.0
        report = out_dir / f"{base}_artifact_report.txt"
        with report.open("a") as f:
            f.write(f"{base}.fif : {pct_keep:.2f}% du signal conservé après suppression des artéfacts.\n")

        print(f"[{base}] OK ({len(art_windows)} fenêtres, {pct_keep:.2f}% conservé)")
    except Exception as e:
        print(f"[{base}] ERREUR: {e}")
        SKIPPED.append((base, str(e)))

def _init_worker(gl_out_root: str, gl_win_sec: float, gl_method: str, gl_threshold: float,
                 gl_drop_prefix: bool, skipped):
    global OUT_ROOT, WIN_SEC, METHOD, THRESHOLD, DROP_PREFIX, SKIPPED
    OUT_ROOT = Path(gl_out_root)
    WIN_SEC = float(gl_win_sec)
    METHOD = str(gl_method)
    THRESHOLD = float(gl_threshold)
    DROP_PREFIX = bool(gl_drop_prefix)
    SKIPPED = skipped
    # Isoler le cache Matplotlib si jamais MNE l'utilise
    try:
        mpl_cache = os.path.join(tempfile.gettempdir(), f"mplcache_{os.getpid()}")
        os.environ["MPLCONFIGDIR"] = mpl_cache
        os.makedirs(mpl_cache, exist_ok=True)
    except Exception:
        pass

def _worker_wrapper(item):
    return process_one(item, OUT_ROOT, WIN_SEC, METHOD, THRESHOLD, DROP_PREFIX)


# ===================== Main (args conservés + ajout --workers) =====================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Détecter + annoter les artéfacts, exporter annoté et segments retirés.")
    ap.add_argument("--in_path", required=False, help="Fichier .edf/.fif OU dossier à parcourir récursivement")
    ap.add_argument("--out_dir", required=False, help="Dossier de sortie")
    ap.add_argument("--win_sec", type=float, default=4.0, help="Taille de fenêtre (s) pour YASA (défaut: 4)")
    ap.add_argument("--method", type=str, default="covar", choices=["covar", "zscore"], help="Méthode YASA")
    ap.add_argument("--threshold", type=float, default=3.0, help="Seuil YASA")
    ap.add_argument("--keep_prefix", action="store_true", help="Ne pas supprimer le préfixe 'EEG ' des noms de canaux")
    ap.add_argument("--workers", type=int, default=8, help="Nb de workers (0=CPU-1)")
    args = ap.parse_args()

    in_path = Path("/home/darryld/documents/EEG/preprocessed/bipolaire/raw_bip") if args.in_path is None else Path(args.in_path)
    out_root = Path("/home/darryld/documents/EEG/preprocessed/bipolaire/noArtefacts") if args.out_dir is None else Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Préparation des items à traiter (style "patients")
    if in_path.is_file():
        base = in_path.stem.split("_")[0]
        patients = [(base, in_path)]
        bases_preview = [base]
    else:
        patients = discover_items(in_path)
        bases_preview = [b for b, _ in patients]

    if not patients:
        raise SystemExit("Aucun fichier .edf/.EDF ou .fif trouvé")

    print(f"Patients détectés ({len(patients)}): {bases_preview}")

    cpu = os.cpu_count() or 1
    n_workers = (cpu - 1 if args.workers == 0 else args.workers)
    n_workers = min(max(1, n_workers), len(patients))
    print(f"Lancement en parallèle avec {n_workers} worker(s)")

    manager = mp.Manager()
    skipped = manager.list()

    ctx = mp.get_context("spawn")
    try:
        with ctx.Pool(
            processes=n_workers,
            maxtasksperchild=1,
            initializer=_init_worker,
            initargs=(str(out_root), float(args.win_sec), str(args.method), float(args.threshold),
                      (not args.keep_prefix), skipped),
        ) as pool:
            for _ in pool.imap_unordered(_worker_wrapper, patients, chunksize=1):
                pass
    except Exception as e:
        import traceback
        print(f"[POOL ERROR] {e}\n{traceback.format_exc()}")

    # Récapitulatif erreurs
    if skipped:
        print("\n=== RÉCAP FICHIERS NON TRAITÉS ===")
        for base, err in skipped:
            print(f"- {base}: {err}")
        recap_file = out_root / "artefact_skipped.txt"
        with open(recap_file, "w") as f:
            for base, err in skipped:
                f.write(f"{base}: {err}\n")
        print(f"\nRécap sauvegardé dans {recap_file}")
    else:
        print("\nTous les fichiers ont été traités avec succès ✅")
