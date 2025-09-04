#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
segment_rem.py — Extrait UNIQUEMENT un fichier concat total REM sans artéfacts, en parallèle.

Entrées (par patient):
  - clean_root/{base}/{base}_art_annotated.fif  (timeline ORIGINE + annotations artéfacts)
  - clean_root/{base}/{base}_art_removed.fif    (timeline NETTOYÉE, artéfacts retirés)
  - hypnogrammes dans annot_root (via get_rem_annotations)

Sortie:
  - out_root/{base}/{base}_REM_concat.fif

Parallélisation:
  - multiprocessing.Pool (spawn) + initializer, imap_unordered, maxtasksperchild=1
  - fragments REM écrits en fichiers FIF temporaires (disk) avant concat finale
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


# ========== Utils intervalles / mapping ==========

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

def _complement_intervals(bad: list[tuple[float, float]], t_end: float) -> list[tuple[float, float]]:
    bad = _merge_intervals(bad)
    if not bad:
        return [(0.0, float(t_end))]
    good, last = [], 0.0
    for s, e in bad:
        if s > last:
            good.append((last, s))
        last = max(last, e)
    if last < t_end:
        good.append((last, t_end))
    return [(float(s), float(e)) for s, e in good if e > s]

def _build_mapping_from_art_annotations(annotated_fif: Path) -> list[tuple[float, float, float]]:
    """Construit la table ORIGINE->NETTOYÉE sous forme [(gs, ge, c0)]."""
    raw_ann = mne.io.read_raw_fif(annotated_fif, preload=False, verbose="ERROR")
    sf = float(raw_ann.info["sfreq"])
    t_end = float(raw_ann.times[-1])
    eps = 1.0 / sf

    bad = []
    if raw_ann.annotations is not None:
        for onset, dur, desc in zip(raw_ann.annotations.onset,
                                    raw_ann.annotations.duration,
                                    raw_ann.annotations.description):
            tag = (str(desc) or "").upper()
            if "BAD" in tag or "ARTEFACT" in tag or "ARTIFACT" in tag:
                s = max(0.0, float(onset))
                e = min(s + float(dur), t_end)
                if e > s:
                    bad.append((s, e))

    good = _complement_intervals(bad, t_end)
    mapping, t_clean = [], 0.0
    for s, e in good:
        if e - s <= 0:
            continue
        mapping.append((s, e - eps, t_clean))  # -eps pour éviter crop inclusif
        t_clean += (e - s)
    return mapping

def _map_intervals_to_clean(intervals_orig: list[tuple[float, float]],
                            mapping: list[tuple[float, float, float]]) -> list[tuple[float, float]]:
    """Mappe des intervalles ORIGINE -> NETTOYÉE avec la table [(gs, ge, c0)]."""
    out = []
    for s, e in intervals_orig:
        s, e = float(s), float(e)
        if e <= s:
            continue
        for gs, ge, c0 in mapping:
            a = max(s, gs)
            b = min(e, ge)
            if b > a:
                out.append((c0 + (a - gs), c0 + (b - gs)))
    return _merge_intervals(out)


# ========== Recherche fichiers / patients ==========

def _find_pair_files(clean_root: Path, base: str) -> tuple[Path, Path]:
    pdir = Path(clean_root) / base
    ann = pdir / f"{base}_art_annotated.fif"
    rem = pdir / f"{base}_art_removed.fif"
    if not ann.exists():
        raise FileNotFoundError(f"Annotations non trouvées: {ann}")
    if not rem.exists():
        raise FileNotFoundError(f"Clean non trouvé: {rem}")
    return ann, rem

def discover_patients(clean_root: Path) -> list[str]:
    bases = []
    for child in sorted(clean_root.iterdir()):
        if not child.is_dir():
            continue
        base = child.name
        try:
            ann, rem = _find_pair_files(clean_root, base)
            if ann.exists() and rem.exists():
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
        # 1) REM (timeline ORIGINE)
        rem_ann = get_rem_annotations(base, annot_dir=str(annot_root))
        if rem_ann is None or len(rem_ann) == 0:
            raise RuntimeError("Aucune annotation REM")

        rem_intervals_orig = []
        for onset, dur, desc in zip(rem_ann.onset, rem_ann.duration, rem_ann.description):
            if str(desc).upper() == "REM" and float(dur) > 0:
                s = float(onset)
                e = s + float(dur)
                rem_intervals_orig.append((s, e))
        if not rem_intervals_orig:
            raise RuntimeError("Aucun intervalle REM valide")

        # 2) Mapping ORIGINE -> NETTOYÉE
        ann_fif, clean_fif = _find_pair_files(clean_root, base)
        mapping = _build_mapping_from_art_annotations(ann_fif)
        if not mapping:
            raise RuntimeError("Mapping vide (tout artefactué ?)")

        # 3) Map des REM vers timeline clean
        rem_clean = _map_intervals_to_clean(rem_intervals_orig, mapping)
        if not rem_clean:
            raise RuntimeError("Aucune intersection REM avec les zones propres")

        # 4) Crop sur le fichier clean, en écrivant chaque fragment en FIF temporaire
        tmp_parent = Path(tmp_root) if tmp_root else Path(tempfile.gettempdir())
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"rem_{base}_", dir=str(tmp_parent)))
        tmp_files = []
        try:
            raw_clean = mne.io.read_raw_fif(clean_fif, preload=False, verbose="ERROR")
            sf = float(raw_clean.info["sfreq"])
            eps = 1.0 / sf

            for k, (cs, ce) in enumerate(rem_clean, 1):
                if ce - cs <= 0:
                    continue
                seg = raw_clean.copy().crop(tmin=float(cs), tmax=float(ce - eps), verbose="ERROR")
                tmp_f = tmp_dir / f"{base}_frag_{k:04d}.fif"
                seg.save(tmp_f, overwrite=True, verbose="ERROR")
                tmp_files.append(tmp_f)

            if not tmp_files:
                raise RuntimeError("Aucun fragment valide à concaténer")

            # 5) Concat finale (lecture lazy) -> save, puis cleanup
            raws = [mne.io.read_raw_fif(p, preload=False, verbose="ERROR") for p in tmp_files]
            rem_all = mne.concatenate_raws(raws, verbose="ERROR")
            rem_all.save(out_concat, overwrite=True, verbose="ERROR")
            print(f"[{base}] -> {out_concat.name} ({len(tmp_files)} fragments)")
        finally:
            # nettoyage des temporaires
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
    ap.add_argument("--clean_root", required=False, help="Racine contenant {base}/{base}_art_annotated.fif et _art_removed.fif")
    ap.add_argument("--annot_root", required=False, help="Racine des hypnogrammes (txt/csv) pour get_rem_annotations")
    ap.add_argument("--out_root",   required=False, help="Dossier de sortie pour {base}_REM_concat.fif")
    ap.add_argument("--tmp_root",   required=False, help="Dossier pour fichiers temporaires (défaut: système)")
    ap.add_argument("--workers", type=int, default=8, help="Nb de workers (0=CPU-1)")
    ap.add_argument("--overwrite", action="store_true", help="Écraser les fichiers concat déjà présents")
    args = ap.parse_args()

    # Defaults style "si None, chemins personnels"
    clean_root = Path("/home/darryld/documents/EEG/preprocessed/bipolaire/1_noArtefacts") if args.clean_root is None else Path(args.clean_root)
    annot_root = Path("/home/darryld/documents/EEG/raw") if args.annot_root is None else Path(args.annot_root)
    out_root   = Path("/home/darryld/documents/EEG/preprocessed/bipolaire/2_rem_only") if args.out_root is None else Path(args.out_root)
    tmp_root   = Path(args.tmp_root) if args.tmp_root else None

    out_root.mkdir(parents=True, exist_ok=True)
    if tmp_root:
        tmp_root.mkdir(parents=True, exist_ok=True)

    # Découverte des patients
    patients = discover_patients(clean_root)
    if not patients:
        raise SystemExit("Aucun patient trouvé avec *_art_annotated.fif et *_art_removed.fif")

    print(f"Patients détectés ({len(patients)}): {patients}")

    # Parallélisation (même logique que tes autres scripts)
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
