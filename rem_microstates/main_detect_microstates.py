# main_detect_microstates.py

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # macOS/Accelerate

import multiprocessing as mp
import faulthandler; faulthandler.enable()  # log des crashes natifs

# ==========================
#   Imports 1
# ==========================
import platform
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import shutil
import tempfile
import traceback
import gc

# ==========================
#   Imports 2
# ==========================
from signal_processing.loader import load_signals_and_annotations
from signal_processing.windowing import segment_rem_in_windows
from signal_processing.eog_analysis import detect_eog_microstate
from signal_processing.annotation import annotate_microstates
from signal_processing.filters import apply_custom_filters  # noqa: F401 (si non utilisé)
from utils.preprocessing_bip import apply_custom_bipolar_montage  # noqa: F401 (si non utilisé)

import mne
mne.set_config('MNE_MEMMAP_MIN_SIZE', '1M', set_env=True)  # favorise memmap


# -------------------------------------------------------------------
# Utilitaires
# -------------------------------------------------------------------

def _free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def _free_big(*objs):
    """Ferme/supprime des objets lourds et force un GC."""
    for o in objs:
        try:
            if hasattr(o, "close"):
                o.close()
        except Exception:
            pass
        try:
            del o
        except Exception:
            pass
    gc.collect()


def _load_microstates_excel(xlsx_path: Path) -> pd.DataFrame:
    """
    Lit un *_microstates_*.xlsx potentiellement hétérogène et renvoie un DataFrame
    standardisé: colonnes ['tmin','tmax','label'] en numériques.
    Lignes non convertibles supprimées avec log.
    """
    # Étape: Lecture et normalisation de l'Excel existant
    try:
        df = pd.read_excel(xlsx_path, engine="openpyxl")
    except Exception:
        df = pd.read_excel(xlsx_path)

    def norm(s): return str(s).strip().lower()
    cols = {norm(c): c for c in df.columns}

    candidates_tmin = ["tmin", "start", "t_start", "debut", "onset", "begin"]
    candidates_tmax = ["tmax", "end", "t_end", "fin", "offset", "stop"]
    c_label = cols.get("label") or cols.get("classe") or cols.get("class") or cols.get("etat") or cols.get("state")

    def pick(cands):
        for k in cands:
            if k in cols: return cols[k]
        return None

    c_tmin = pick(candidates_tmin) or ("tmin" if "tmin" in df.columns else None)
    c_tmax = pick(candidates_tmax) or ("tmax" if "tmax" in df.columns else None)

    if c_tmin is None or c_tmax is None:
        raise ValueError(f"{xlsx_path.name}: colonnes tmin/tmax introuvables parmi {list(df.columns)}")

    out = pd.DataFrame({
        "tmin": pd.to_numeric(df[c_tmin], errors="coerce"),
        "tmax": pd.to_numeric(df[c_tmax], errors="coerce"),
        "label": (df[c_label] if c_label else "unknown"),
    })
    out["label"] = out["label"].astype(str).str.strip().str.lower()

    before = len(out)
    out = out.dropna(subset=["tmin", "tmax"]).reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        print(f"[SANITIZE] {xlsx_path.name}: {dropped} ligne(s) ignorée(s) (tmin/tmax non numériques).")
    return out


# =========================================================
#                 TRAITEMENT PAR PATIENT (sans comparaison .mat)
# =========================================================

def process_patient(fif_path: Path, raw_dir: Path, out_dir: Path, seuil: int):
    # ===== SECTION: PATIENT =====
    patient_id = fif_path.stem.split("_")[0]

    if patient_id == "." or not patient_id.isalnum():
        print(f"[SKIP] Nom de patient invalide : {fif_path.name}")
        return

    out_patient_dir = out_dir / patient_id
    out_patient_dir.mkdir(parents=True, exist_ok=True)

    xlsx_out = out_patient_dir / f"{patient_id}_microstates_{seuil}.xlsx"
    raw_annotated_path = out_patient_dir / f"{patient_id}_annotated_{seuil}.fif"

    # Étape: Vérification des sorties existantes
    outputs_exist = xlsx_out.exists() and raw_annotated_path.exists()

    # Étape: Recherche de l'hypnogramme (txt/csv) si nécessaire
    annot_path = None
    if not outputs_exist:
        for ext in (".txt", ".csv"):
            candidate = raw_dir / patient_id / f"{patient_id}_hypnoEXP{ext}"
            if candidate.exists():
                annot_path = candidate
                break
        if annot_path is None:
            print(f"[SKIP] {patient_id} : pas d'hypnogramme (txt/csv) pour générer les sorties automatiques.")
            if not xlsx_out.exists():
                return

    # Étape: (Re)traitement seulement si nécessaire
    if not outputs_exist:
        # Sécurité disque libre >= 5 Go
        if _free_gb(out_patient_dir) < 5.0:
            print(f"[SKIP] {patient_id}: espace insuffisant (<5 Go) sur {out_patient_dir}")
            return

        # Chargement des signaux + segments REM
        raw, rem_segments = load_signals_and_annotations(fif_path, annot_path)

        if not rem_segments:
            print(f"[SKIP] {patient_id} : aucun segment REM détecté.")
            _free_big(raw)
            return

        print(f"[{patient_id}] {len(rem_segments)} segments REM détectés.")

        # Étape: Streaming fenêtres 4s et détection micro-états
        labels = []
        valid_times = []  # liste de tuples (tmin, tmax)
        skip_to = 0
        for idx, win in enumerate(segment_rem_in_windows(raw, rem_segments, window_sec=4, step_sec=4)):
            if idx < skip_to:
                continue
            label = detect_eog_microstate(win=win, min_pair_amp_uv=seuil)
            if label != "ignore":
                t0 = float(win.first_time)
                valid_times.append((t0, t0 + 4.0))
                labels.append(label)
            # saut de 2 si phasic, sinon 1
            skip_to = idx + (2 if label == "phasic" else 1)
        print(f"[{patient_id}] Fenêtres retenues: {len(valid_times)} ({labels.count('phasic')} phasic / {labels.count('tonic')} tonic)")

        # Étape: Annotation du Raw et sauvegardes atomiques (raw + Excel)
        with tempfile.TemporaryDirectory(prefix=f"{patient_id}_", dir=out_patient_dir) as td:
            tdir = Path(td)

            # Ajoute les annotations sans écraser l'existant
            annotate_microstates(
                raw, valid_times, labels, window_sec=4,
                mode="add",
                prefix="REM_"
            )

            # 1) Sauvegarde du Raw annoté -> temp -> destination
            tmp_fif = tdir / raw_annotated_path.name
            raw.save(tmp_fif, overwrite=True)
            os.replace(tmp_fif, raw_annotated_path)

            # 2) Export tabulaire des fenêtres retenues -> temp -> destination
            df = pd.DataFrame({"tmin": [a for a, _ in valid_times],
                               "tmax": [b for _, b in valid_times],
                               "label": labels})
            tmp_xlsx = tdir / xlsx_out.name
            df.to_excel(tmp_xlsx, index=False, engine="xlsxwriter")
            os.replace(tmp_xlsx, xlsx_out)

        _free_big(raw, df, valid_times, labels)

    else:
        # Étape: Sorties déjà présentes -> relecture de l'Excel
        try:
            df = _load_microstates_excel(xlsx_out)
        except Exception as e:
            print(f"[{patient_id}] Impossible de lire {xlsx_out.name} : {e}")
            return

    print(f"[{patient_id}] Traitement terminé.\n")


# =========================================================
#  FONCTION WORKER TOP-LEVEL (picklable pour spawn)
# =========================================================

def run_one(path_str: str, raw_dir_str: str, out_dir_str: str, seuil: int):
    """Wrapper: exécution par patient (mémoire isolée)."""
    try:
        mpl_cache = os.path.join(tempfile.gettempdir(), f"mplcache_{os.getpid()}")
        os.environ["MPLCONFIGDIR"] = mpl_cache
        os.makedirs(mpl_cache, exist_ok=True)
    except Exception:
        pass
    try:
        return process_patient(
            Path(path_str), Path(raw_dir_str), Path(out_dir_str), seuil
        )
    except Exception as e:
        raise RuntimeError(f"Worker error on {Path(path_str).name}: {e}\n{traceback.format_exc()}") from e


# =========================================================
#                        MAIN
# =========================================================

if __name__ == "__main__":
    # ===== SECTION: CONFIGURATION =====

    # Étape: Parsing des arguments CLI minimaux
    parser = argparse.ArgumentParser()
    parser.add_argument("--seuil", type=int, default=150,
                        help="Valeur de seuil pour le nommage des fichiers (ex: 100 ou 150)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Nb de processus en parallèle (0 => CPU-1)")
    args = parser.parse_args()
    seuil = args.seuil

    # Étape: Montage fixé à bipolaire (info)
    print("[INFO] Montage fixé: bipolaire")

    # Étape: Détection/choix du disque (forcé)
    """
    system = platform.system()
    if system == "Darwin":
        disque = "/Volumes/Crucial X6"
    elif system == "Windows":
        disque = "D:"
    elif system == "Linux":
        disque = "/media/darryld/Crucial X6"
    else:
        raise RuntimeError("Système non supporté.")
    """
    disque = "/home/darryld/documents"

    # Étape: Racines d'E/S
    # Lecture: dossiers patients sous 1_noArtefacts/gp2/
    root_preproc = Path(f"{disque}/EEG/preprocessed/bipolaire/1_noArtefacts/gp2/")
    # Sortie: vers 2_rem_only/gp2/<patient>/
    root_out     = Path(f"{disque}/EEG/preprocessed/bipolaire/2_rem_only/gp2/")
    # Hypnogrammes (inchangé)
    root_raw     = Path(f"{disque}/EEG/raw")

    # Étape: Vérification des dossiers d'entrée/sortie
    for p in [root_preproc, root_raw]:
        if not p.exists():
            raise SystemExit(f"[CONFIG] Dossier introuvable: {p}")
    root_out.mkdir(parents=True, exist_ok=True)

    # Étape: Warm-up Matplotlib (éviter concurrence cache)
    def _warmup_matplotlib():
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import font_manager as fm
        matplotlib.get_cachedir()
        fm.findfont('DejaVu Sans', rebuild_if_missing=True)
        fig = plt.figure()
        plt.plot([0, 1], [0, 1])
        with open(os.devnull, "wb") as f:
            fig.savefig(f, format="png")
        plt.close(fig)

    _warmup_matplotlib()

    # ===== SECTION: DÉCOUVERTE & MULTIPROCESS =====

    # Étape: Découverte des fichiers FIF
    # On cible les fichiers contenant 'art_annotated'
    patterns = ("*art_annotated*.fif",)
    found = []
    for pat in patterns:
        found += [
            p for p in root_preproc.rglob(pat)
            if not p.name.startswith("._") and not p.name.startswith(".")
        ]
    # dédoublonner tout en conservant l'ordre
    seen = set()
    fif_paths = []
    for p in found:
        if p not in seen:
            fif_paths.append(p)
            seen.add(p)

    print(f"{len(fif_paths)} fichier(s) trouvés dans {root_preproc} (patterns: {', '.join(patterns)})")
    if not fif_paths:
        raise SystemExit(0)

    # Étape: Configuration du multiprocess
    cpu = os.cpu_count() or 1
    max_workers = (cpu - 1) if args.workers in (0, None) else max(1, args.workers)
    max_workers = min(max_workers, len(fif_paths))
    print(f"[INFO] Lancement en multiprocess avec {max_workers} worker(s) (CPU={cpu}) (maxtasksperchild=1)")

    # Préparation des arguments picklables
    args_list = [(str(p), str(root_raw), str(root_out), seuil) for p in fif_paths]

    # Étape: Exécution parallèle (pool.starmap)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max_workers, maxtasksperchild=1) as pool:
        try:
            for _ in pool.starmap(run_one, args_list, chunksize=1):
                pass
        except Exception as e:
            print(f"[POOL ERROR] {e}\n{traceback.format_exc()}")
