#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
segment_rem.py - Extrait un fichier concat "REM sans artéfacts" par patient, en parallèle.

Entrées (par patient):
  - clean_root/{base}/{base}_art_annotated.fif  (timeline ORIGINE + annotations, dont 'ARTEFACT')
  - hypnogrammes dans annot_root (via get_rem_annotations)

Sortie:
  - out_root/{base}/{base}_REM_concat.fif

Parallélisation:
  - multiprocessing.Pool (spawn) + initializer, imap_unordered, maxtasksperchild=1
  - fragments REM écrits en FIF temporaires avant concat finale
"""

from __future__ import annotations
import os
import shutil
import tempfile
import argparse
import multiprocessing as mp
from pathlib import Path
import mne
import numpy as np

# ===== import projet: annotations REM =====
try:
    from annotations import get_rem_annotations
except Exception as e:
    raise ImportError("Impossible d'importer 'get_rem_annotations' depuis annotations.py") from e


# ========== Utils intervalles ==========
def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ints = sorted(((float(s), float(e)) for s, e in intervals), key=lambda x: x[0])
    merged = [ints[0]]
    for s, e in ints[1:]:
        s0, e0 = merged[-1]
        if s <= e0:
            merged[-1] = (s0, max(e0, e))
        else:
            merged.append((s, e))
    return merged

def _subtract_intervals(good: list[tuple[float, float]],
                        bad: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Retourne good \ bad (toutes timelines en secondes)."""
    A = _merge_intervals(good)
    B = _merge_intervals(bad)
    res: list[tuple[float, float]] = []
    for s, e in A:
        cur = [(s, e)]
        for bs, be in B:
            nxt = []
            for cs, ce in cur:
                if be <= cs or bs >= ce:  # pas d'intersection
                    nxt.append((cs, ce))
                else:
                    if cs < bs:
                        nxt.append((cs, bs))
                    if be < ce:
                        nxt.append((be, ce))
            cur = nxt
        res.extend(cur)
    return _merge_intervals([(s, e) for s, e in res if e > s])


# ========== Recherche fichiers / patients ==========
def _find_annotated_file(clean_root: Path, base: str) -> Path:
    pdir = Path(clean_root) / base
    ann = pdir / f"{base}_art_annotated.fif"
    if not ann.exists():
        raise FileNotFoundError(f"Fichier annoté non trouvé: {ann}")
    return ann

def discover_patients(clean_root: Path) -> list[str]:
    bases = []
    for child in sorted(clean_root.iterdir()):
        if not child.is_dir():
            continue
        base = child.name
        try:
            ann = _find_annotated_file(clean_root, base)
            if ann.exists():
                bases.append(base)
        except Exception:
            continue
    return bases


# ========== Worker ==========
SKIPPED = None  # liste partagée (base, raison)

def _process_one_base(base: str, annot_root: Path, clean_root: Path, out_root: Path,
                      overwrite: bool, tmp_root: Path | None) -> None:
    out_dir = out_root / base
    out_dir.mkdir(parents=True, exist_ok=True)
    out_concat = out_dir / f"{base}_REM_concat.fif"

    if out_concat.exists() and not overwrite:
        print(f"[{base}] déjà présent -> skip (utiliser --overwrite pour forcer)")
        return

    try:
        # 1) Récupère les intervalles REM sur la timeline ORIGINE (depuis hypnogrammes via get_rem_annotations)
        rem_ann = get_rem_annotations(base, annot_dir=str(annot_root))
        if rem_ann is None or len(rem_ann) == 0:
            raise RuntimeError(f"Aucune annotation REM --> {rem_ann} - {annot_root} - {base}")

        rem_intervals = []
        for onset, dur, desc in zip(rem_ann.onset, rem_ann.duration, rem_ann.description):
            if str(desc).upper() == "REM" and float(dur) > 0:
                s = float(onset)
                e = s + float(dur)
                rem_intervals.append((s, e))
        rem_intervals = _merge_intervals(rem_intervals)
        if not rem_intervals:
            raise RuntimeError("Aucun intervalle REM valide")

        # 2) Charge le fichier *annoté* (contient ARTEFACT)
        ann_fif = _find_annotated_file(clean_root, base)
        raw_ann = mne.io.read_raw_fif(ann_fif, preload=False, verbose="ERROR")
        sf = float(raw_ann.info["sfreq"])
        eps = 1.0 / sf
        t_end = float(raw_ann.times[-1])

        # 3) Récupère les fenêtres ARTEFACT sur la timeline ORIGINE
        bad = []
        if raw_ann.annotations is not None:
            for onset, dur, desc in zip(raw_ann.annotations.onset,
                                        raw_ann.annotations.duration,
                                        raw_ann.annotations.description):
                tag = (str(desc) or "").upper()
                if "ARTEFACT" in tag:
                    s = max(0.0, float(onset))
                    e = min(s + float(dur), t_end)
                    if e > s:
                        bad.append((s, e))

        # 4) REM nettoyés = REM - ARTEFACT (toujours timeline ORIGINE)
        rem_clean_orig = _subtract_intervals(rem_intervals, bad)
        if not rem_clean_orig:
            raise RuntimeError("Aucune intersection REM avec zones non-artefact.")

        # 5) Crop dans le *Raw annoté* : les autres annotations sont conservées 
        tmp_parent = Path(tmp_root) if tmp_root else Path(tempfile.gettempdir())
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"rem_{base}_", dir=str(tmp_parent)))
        tmp_files = []
        try:
            for k, (cs, ce) in enumerate(rem_clean_orig, 1):
                if ce - cs <= 0:
                    continue
                seg = raw_ann.copy().crop(tmin=float(cs), tmax=float(ce - eps), verbose="ERROR")
                tmp_f = tmp_dir / f"{base}_frag_{k:04d}.fif"
                seg.save(tmp_f, overwrite=True, verbose="ERROR")
                tmp_files.append(tmp_f)

            if not tmp_files:
                raise RuntimeError("Aucun fragment valide à concaténer")

            # Concat finale 
            raws = [mne.io.read_raw_fif(p, preload=False, verbose="ERROR") for p in tmp_files]
            rem_all = mne.concatenate_raws(raws, verbose="ERROR")
            rem_all.save(out_concat, overwrite=True, verbose="ERROR")
            print(f"[{base}] -> {out_concat.name} ({len(tmp_files)} fragments)")
        finally:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    except Exception as e:
        print(f"[{base}] ERREUR: {e}")
        SKIPPED.append((base, str(e)))


def _init_worker(gl_annot_root: str, gl_clean_root: str, gl_out_root: str,
                 gl_overwrite: bool, gl_tmp_root: str | None, skipped):
    global ANNOT_ROOT, CLEAN_ROOT, OUT_ROOT, OVERWRITE, TMP_ROOT, SKIPPED
    ANNOT_ROOT = Path(gl_annot_root)
    CLEAN_ROOT = Path(gl_clean_root)
    OUT_ROOT = Path(gl_out_root)
    OVERWRITE = bool(gl_overwrite)
    TMP_ROOT = Path(gl_tmp_root) if gl_tmp_root else None
    SKIPPED = skipped

    # isole le cache Matplotlib si nécessaire
    try:
        mpl_cache = os.path.join(tempfile.gettempdir(), f"mplcache_{os.getpid()}")
        os.environ["MPLCONFIGDIR"] = mpl_cache
        os.makedirs(mpl_cache, exist_ok=True)
    except Exception:
        pass

def _worker_wrapper(base: str):
    return _process_one_base(base, ANNOT_ROOT, CLEAN_ROOT, OUT_ROOT, OVERWRITE, TMP_ROOT)


# ========== Main (CLI) ==========
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extraction des segments REM sans artéfacts (concat unique par patient).")
    ap.add_argument("--clean_root", required=False, help="Racine contenant {base}/{base}_art_annotated.fif")
    ap.add_argument("--annot_root", required=False, help="Racine des hypnogrammes (txt/csv) pour get_rem_annotations")
    ap.add_argument("--out_root",   required=False, help="Dossier de sortie pour {base}_REM_concat.fif")
    ap.add_argument("--tmp_root",   required=False, help="Dossier pour fichiers temporaires (défaut: système)")
    ap.add_argument("--workers", type=int, default=8, help="Nb de workers (0=CPU-1)")
    ap.add_argument("--overwrite", action="store_true", help="Écraser les fichiers concat déjà présents")
    args = ap.parse_args()

    # Defaults (adaptés à tes chemins "gp1")
    clean_root = Path("/home/darryld/documents/EEG/preprocessed/bipolaire/1_noArtefacts/gp2") if args.clean_root is None else Path(args.clean_root)
    annot_root = Path("/home/darryld/documents/EEG/raw") if args.annot_root is None else Path(args.annot_root)
    out_root   = Path("/home/darryld/documents/EEG/preprocessed/bipolaire/2_rem_only/gp2") if args.out_root is None else Path(args.out_root)
    tmp_root   = Path(args.tmp_root) if args.tmp_root else None

    out_root.mkdir(parents=True, exist_ok=True)
    if tmp_root:
        tmp_root.mkdir(parents=True, exist_ok=True)

    # Découverte des patients 
    patients = discover_patients(clean_root)
    if not patients:
        raise SystemExit("Aucun patient trouvé avec *_art_annotated.fif")

    print(f"Patients détectés ({len(patients)}): {patients}")

    # Parallélisation
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
            initargs=(str(annot_root), str(clean_root), str(out_root),
                      bool(args.overwrite), str(tmp_root) if tmp_root else None, skipped),
        ) as pool:
            for _ in pool.imap_unordered(_worker_wrapper, patients, chunksize=1):
                pass
    except Exception as e:
        import traceback
        print(f"[POOL ERROR] {e}\n{traceback.format_exc()}")

    # Récap
    if skipped:
        print("\n=== RÉCAP PATIENTS NON TRAITÉS ===")
        for base, reason in skipped:
            print(f"- {base}: {reason}")
        recap_file = out_root / "rem_concat_skipped.txt"
        with open(recap_file, "w") as f:
            for base, reason in skipped:
                f.write(f"{base}: {reason}\n")
        print(f"\nRécap sauvegardé : {recap_file}")
    else:
        print("\nTous les patients ont été traités avec succès ")
