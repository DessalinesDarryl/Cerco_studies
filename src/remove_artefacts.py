#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remove_artefacts.py - Détection + annotation des artéfacts avec YASA (hypnogramme intégré).
Sorties (par patient) dans --out_dir/{patient}/ :
  - {base}_art_annotated.fif
  - {base}_art_removed.fif
  - {base}_artifact_windows.csv
  - {patient}_artifact_report.txt (append)

Points clés :
- Ne garde que les canaux EEG pour la détection (YASA recommande d'exclure EOG/EMG/ECG).
- Si hypnogramme trouvé, il est converti (Wake=0, N1=1, N2=2, N3=3, REM=4) et upsamplé à la longueur des données.
- --include: 'sleep' (N1/N2/N3/REM), 'rem' (REM uniquement) ou 'all' (inclut aussi Wake).

"""

from __future__ import annotations
import os
import csv
import re
import argparse
import tempfile
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import mne


# ===================== Hypnogramme: parsing & upsample =====================

YASA_CODE = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
# Certains exports "EXP" utilisent un code perso : 1=Wake, 2=REM, 3=N1, 4=N2, 5=N3
EXP_NUM_TO_YASA = {1: 0, 2: 4, 3: 1, 4: 2, 5: 3}

def _infer_epoch_len_sec(seconds_col: List[float]) -> float:
    if len(seconds_col) < 2:
        return 30.0
    diffs = np.diff(seconds_col)
    # mode approx des diffs
    return float(np.median(diffs)) if len(diffs) else 30.0

def _read_hypno_txt(path: Path) -> Tuple[np.ndarray, float]:
    """
    Lit un .txt type EXP : colonnes ~ [seconds, HH:MM:SS, label, code]
    Retourne (codes_YASA_par_epoch, epoch_len_sec)
    """
    seconds = []
    labels = []
    codes_num = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # split robuste (tab/espaces multiples)
            parts = re.split(r"\s+", line)
            # On attend au moins 3 colonnes (sec, time, label), parfois 4 (code)
            if len(parts) < 3:
                continue
            try:
                sec = float(parts[0])
            except Exception:
                continue
            seconds.append(sec)
            # label texte (W/N1/N2/N3/REM) supposé en 3e col
            lab = parts[2].upper()
            labels.append(lab)
            # code num éventuellement en 4e col
            if len(parts) >= 4:
                try:
                    codes_num.append(int(parts[3]))
                except Exception:
                    codes_num.append(None)
            else:
                codes_num.append(None)

    if not seconds:
        raise ValueError(f"Hypnogramme vide ou non lisible: {path}")

    epoch_len = _infer_epoch_len_sec(seconds)

    # Convertit en codes YASA (priorité au label texte; fallback sur code num EXP)
    out = []
    for lab, cnum in zip(labels, codes_num):
        if lab in YASA_CODE:
            out.append(YASA_CODE[lab])
        elif cnum is not None and cnum in EXP_NUM_TO_YASA:
            out.append(EXP_NUM_TO_YASA[cnum])
        else:
            # Unscored / inconnu -> -2 (YASA convention)
            out.append(-2)
    return np.asarray(out, dtype=int), epoch_len

def _read_hypno_csv(path: Path) -> Tuple[np.ndarray, float]:
    """
    CSV générique. Essaie colonnes : "stage"/"label" texte, sinon "code" num.
    Optionnellement une colonne "seconds" pour inférer l'epoch.
    """
    import pandas as pd
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    # Epoch length
    epoch_len = 30.0
    if "seconds" in cols:
        sec_col = df[cols["seconds"]].values
        if len(sec_col) >= 2:
            epoch_len = _infer_epoch_len_sec(list(sec_col))

    # Stage/label prioritaire
    if "stage" in cols:
        labels = df[cols["stage"]].astype(str).str.upper().tolist()
    elif "label" in cols:
        labels = df[cols["label"]].astype(str).str.upper().tolist()
    else:
        labels = None

    # Code numérique éventuel
    codes = None
    if "code" in cols:
        try:
            codes = df[cols["code"]].astype(float).astype(int).tolist()
        except Exception:
            codes = None

    out = []
    if labels is not None:
        for i, lab in enumerate(labels):
            if lab in YASA_CODE:
                out.append(YASA_CODE[lab])
            else:
                # fallback sur code numérique si présent
                cnum = None if codes is None else codes[i]
                if cnum is not None and cnum in EXP_NUM_TO_YASA:
                    out.append(EXP_NUM_TO_YASA[cnum])
                else:
                    out.append(-2)
    elif codes is not None:
        for cnum in codes:
            out.append(EXP_NUM_TO_YASA.get(cnum, -2))
    else:
        raise ValueError(f"Colonnes de stade introuvables dans {path} (attendu 'stage'/'label' ou 'code').")

    return np.asarray(out, dtype=int), float(epoch_len)

def _find_hypno_file(base: str, in_path: Path, hypno_root: Optional[Path]) -> Optional[Path]:
    """
    Cherche un fichier hypnogramme pour 'base' :
    - dans hypno_root/{base}/
    - ou à côté du fichier d'entrée (dossier parent ou patient)
    - suffixes: *_hypno*.txt|csv, *hypnogram*.txt|csv, *.hypno.txt|csv
    """
    candidates = []
    search_dirs = []
    if hypno_root:
        search_dirs += [hypno_root / base, hypno_root]
        print(f"Recherche hypnogramme dans {hypno_root}...")
    # patient dir
    search_dirs += [in_path.parent, in_path.parent.parent / base, in_path.parent.parent]

    patterns = [
        f"{base}*hypno*.txt", f"{base}*hypno*.csv",
        f"{base}*.hypno.txt", f"{base}*.hypno.csv",
        f"{base}_hypnoEXP.txt", f"{base}_hypnoEXP.csv",
    ]
    for d in search_dirs:
        if d and d.exists():
            for pat in patterns:
                candidates += list(d.glob(pat))

    if not candidates:
        return None
    # prend le plus gros fichier (souvent le plus complet)
    return max(candidates, key=lambda p: p.stat().st_size)

def _load_hypnogram(base: str, in_path: Path, hypno_root: Optional[Path]) -> Tuple[Optional[np.ndarray], Optional[float]]:
    hp = _find_hypno_file(base, in_path, hypno_root)
    if hp is None:
        print(f"[{base}] Hypnogramme introuvable -> détection sans hypnogramme.")
        return None, None
    try:
        if hp.suffix.lower() == ".txt":
            hypno_epochs, epoch_len = _read_hypno_txt(hp)
        elif hp.suffix.lower() == ".csv":
            hypno_epochs, epoch_len = _read_hypno_csv(hp)
        else:
            print(f"[{base}] Hypnogramme non supporté: {hp}")
            return None, None
        print(f"[{base}] Hypnogramme détecté: {hp.name} (epochs={len(hypno_epochs)}, epoch_len={epoch_len:.1f}s)")
        return hypno_epochs, epoch_len
    except Exception as e:
        print(f"[{base}] Erreur lecture hypnogramme: {e}")
        return None, None

def _upsample_hypno_to_data(hypno_epochs: np.ndarray, epoch_len_sec: float, data_len: int, sf_data: float) -> np.ndarray:
    """
    Upsample l'hypnogramme (au pas epoch_len_sec) à un vecteur par échantillon (len = data_len).
    """
    import yasa
    # Fréquence de l'hypnogramme (Hz), ex: 1/30 ≈ 0.0333 Hz
    sf_hypno = 1.0 / float(epoch_len_sec)
    hypno_up = yasa.hypno_upsample_to_data(
        hypno=hypno_epochs,
        sf_hypno=sf_hypno,
        data=np.zeros((1, data_len), dtype=float),  # shape (n_chan, n_samples) dummy
        sf_data=sf_data,
    )
    # hypno_up a la longueur des données (n_samples)
    return hypno_up


# ===================== Détection par YASA =====================

def detect_artifacts_windows(raw: mne.io.BaseRaw,
                             win_sec: float = 4.0,
                             method: str = "covar",
                             threshold: float = 3.0,
                             hypno_samples: Optional[np.ndarray] = None,
                             include_stages: str = "sleep"):
    """
    Applique yasa.art_detect sur les EEG uniquement, optionnellement contraint par hypnogramme.
    include_stages: 'sleep' -> (1,2,3,4) ; 'rem' -> (4,) ; 'all' -> (0,1,2,3,4)
    """
    try:
        import yasa
    except Exception as e:
        raise RuntimeError("Le module 'yasa' est requis (`pip install yasa`).") from e

    # Ne garder que les EEG (YASA recommande d'exclure EOG/EMG/ECG)
    picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, emg=False, misc=False)
    raw_eeg = raw.copy().pick(picks) if len(picks) > 0 else raw.copy()

    sf = float(raw_eeg.info["sfreq"])
    data = raw_eeg.get_data() * 1e6  # YASA attend des µV si array_like
    n_samples = data.shape[1]

    # include selon argument
    include_map = {
        "sleep": (1, 2, 3, 4),
        "rem": (4,),
        "all": (0, 1, 2, 3, 4),
    }
    include = include_map.get(include_stages, (1, 2, 3, 4))

    hypno_vec = None
    if hypno_samples is not None:
        if len(hypno_samples) != n_samples:
            raise ValueError(f"Hypnogramme upsamplé ({len(hypno_samples)}) != n_samples ({n_samples}).")
        hypno_vec = hypno_samples.astype(int)

    # YASA: method 'covar' (>=4 canaux) ou 'std' (1+ canaux)
    if method not in {"covar", "std"}:
        raise ValueError("`method` doit être 'covar' ou 'std' (pas 'zscore').")

    art, _zs = yasa.art_detect(
        data=data,
        sf=sf,
        window=win_sec,
        hypno=hypno_vec,
        include=include,
        method=method,
        threshold=threshold,
        verbose=False,
    )

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
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_time_s", "end_time_s"])
        for s, e in windows_s:
            w.writerow([f"{s:.6f}", f"{e:.6f}"])

def annotate_raw(raw, windows_s):
    if not windows_s:
        return raw.copy()
    onset = [s for s, _ in windows_s]
    duration = [e - s for s, e in windows_s]
    desc = ["ARTEFACT"] * len(onset)
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
    mapping: Dict[str, Path] = {}

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

    root_fifs = list(in_root.glob("*.fif"))
    root_edfs = list(in_root.glob("*_raw.edf")) + list(in_root.glob("*_raw.EDF"))
    for p in root_fifs + root_edfs:
        base = p.stem.split("_")[0]
        cur = mapping.get(base)
        if cur is None or p.stat().st_size > cur.stat().st_size:
            mapping[base] = p

    return sorted(mapping.items())


# ===================== Worker =====================

SKIPPED = None  # liste partagée entre workers

def process_one(item,
                out_root: Path,
                win_sec: float,
                method: str,
                threshold: float,
                drop_prefix: bool,
                include_stages: str,
                hypno_root: Optional[Path]):
    global OVERWRITE
    base, in_path = item
    base_short = in_path.stem.split("_")[0]
    out_dir = out_root / base_short
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv   = out_dir / f"{base_short}_artifact_windows.csv"
    out_annot = out_dir / f"{base_short}_art_annotated.fif"
    out_clean = out_dir / f"{base_short}_art_removed.fif"
    out_report = out_dir / f"{base_short}_artifact_report.txt"

    if (out_csv.exists() and out_annot.exists() and out_clean.exists() and out_report.exists()) and not OVERWRITE:
        print(f"[{base_short}] Déjà traité -> skip (utiliser --overwrite pour refaire)")
        return

    try:
        raw = load_raw_any(in_path)

        # Renommage optionnel "EEG " -> "" (utile pour EDF bruts)
        if drop_prefix:
            ren = {ch: ch.replace("EEG ", "", 1) for ch in raw.ch_names if ch.startswith("EEG ")}
            if ren:
                raw.rename_channels(ren)

        # Filtrage léger
        raw.filter(l_freq=0.3, h_freq=100.0, fir_design="firwin", verbose="ERROR")
        raw.notch_filter(50.0, verbose="ERROR")

        # Hypnogramme (optionnel)
        hypno_epochs, epoch_len = _load_hypnogram(base_short, in_path, hypno_root)
        hypno_samples = None
        if hypno_epochs is not None and epoch_len is not None:
            sf = float(raw.info["sfreq"])
            picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, emg=False, misc=False)

            if len(picks_eeg) == 0:
                # Pas de canaux EEG détectés : on retombe sur toute la longueur du raw
                n_samples = raw.n_times
            else:
                n_samples = raw.get_data(picks=picks_eeg).shape[-1]

            hypno_samples = _upsample_hypno_to_data(hypno_epochs, epoch_len, n_samples, sf)

        # Détection
        _, art_windows = detect_artifacts_windows(
            raw, win_sec=win_sec, method=method, threshold=threshold,
            hypno_samples=hypno_samples, include_stages=include_stages
        )
        art_windows = merge_intervals(art_windows)

        # Sauvegardes
        save_artifacts_csv(art_windows, out_csv)

        raw_ann = annotate_raw(raw, art_windows)
        raw_ann.save(out_annot, overwrite=True, verbose="ERROR")

        raw_clean = cut_out_artifacts(raw, art_windows)
        raw_clean.save(out_clean, overwrite=True, verbose="ERROR")

        # Rapport
        t_total = float(raw.times[-1])
        bad_total = float(sum((e - s) for s, e in art_windows))
        good_total = max(0.0, t_total - bad_total)
        pct_keep = (good_total / t_total * 100.0) if t_total > 0 else 0.0
        with out_report.open("a") as f:
            f.write(f"{base_short}.fif : {pct_keep:.2f}% du signal conservé après suppression des artéfacts.\n")

        print(f"[{base_short}] OK ({len(art_windows)} fenêtres, {pct_keep:.2f}% conservé)")
    except Exception as e:
        print(f"[{base_short}] ERREUR: {e}")
        SKIPPED.append((base_short, str(e)))

def _init_worker(gl_out_root: str, gl_win_sec: float, gl_method: str, gl_threshold: float,
                 gl_drop_prefix: bool, gl_overwrite: bool, gl_include: str, gl_hypno_root: Optional[str], skipped):
    global OUT_ROOT, WIN_SEC, METHOD, THRESHOLD, DROP_PREFIX, OVERWRITE, INCLUDE, HYPNO_ROOT, SKIPPED
    OUT_ROOT = Path(gl_out_root)
    OVERWRITE = bool(gl_overwrite)
    WIN_SEC = float(gl_win_sec)
    METHOD = str(gl_method)
    THRESHOLD = float(gl_threshold)
    DROP_PREFIX = bool(gl_drop_prefix)
    INCLUDE = str(gl_include)
    HYPNO_ROOT = Path(gl_hypno_root) if gl_hypno_root else None
    SKIPPED = skipped
    try:
        mpl_cache = os.path.join(tempfile.gettempdir(), f"mplcache_{os.getpid()}")
        os.environ["MPLCONFIGDIR"] = mpl_cache
        os.makedirs(mpl_cache, exist_ok=True)
    except Exception:
        pass

def _worker_wrapper(item):
    return process_one(item, OUT_ROOT, WIN_SEC, METHOD, THRESHOLD, DROP_PREFIX, INCLUDE, HYPNO_ROOT)


# ===================== Main =====================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Détecter + annoter les artéfacts (YASA) avec hypnogramme.")
    ap.add_argument("--in_path", required=False, help="Fichier .edf/.fif OU dossier à parcourir")
    ap.add_argument("--out_dir", required=False, help="Dossier de sortie")
    ap.add_argument("--overwrite", action="store_true", help="Recalculer même si sorties déjà présentes")
    ap.add_argument("--win_sec", type=float, default=4.0, help="Taille de fenêtre (s) pour YASA (défaut: 4)")
    ap.add_argument("--method", type=str, default="covar", choices=["covar", "std"], help="Méthode YASA ('covar' ou 'std')")
    ap.add_argument("--threshold", type=float, default=3.0, help="Seuil (z) YASA")
    ap.add_argument("--keep_prefix", action="store_true", help="Ne pas supprimer le préfixe 'EEG '")
    ap.add_argument("--include", type=str, default="sleep", choices=["sleep", "rem", "all"],
                    help="Stades à inclure pour la détection (sleep=N1/N2/N3/REM, rem=REM seul, all=inclut Wake)")
    ap.add_argument("--workers", type=int, default=8, help="Nb de workers (0=CPU-1)")
    args = ap.parse_args()

    in_path = Path("/home/darryld/documents/EEG/preprocessed/bipolaire/0_raw_bip_gp2") if args.in_path is None else Path(args.in_path)
    out_root = Path("/home/darryld/documents/EEG/preprocessed/bipolaire/1_noArtefacts/gp2") if args.out_dir is None else Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    hypno_root = Path("/home/darryld/documents/EEG/raw/")

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
                      (not args.keep_prefix), bool(args.overwrite), str(args.include),
                      (str(hypno_root) if hypno_root else None), skipped),
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
        print("\nTous les fichiers ont été traités avec succès")
