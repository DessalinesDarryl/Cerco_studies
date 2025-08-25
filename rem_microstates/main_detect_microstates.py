# ==========================
#   Parallélisation & CPU
# ==========================
# À définir AVANT d'importer numpy/scipy/mne
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # macOS/Accelerate

import multiprocessing as mp
import faulthandler; faulthandler.enable()  # log des crashes natifs

# ==========================
#   Imports standard
# ==========================
import platform
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend non interactif, sûr en multiprocess
import matplotlib.pyplot as plt
import argparse
import math
import re
import shutil
import tempfile
import traceback
import gc

# ==========================
#   Imports projet
# ==========================
from signal_processing.loader import load_signals_and_annotations
from signal_processing.windowing import segment_rem_in_windows  # générateur WindowProxy
from signal_processing.eog_analysis import detect_eog_microstate
from signal_processing.annotation import annotate_microstates
from signal_processing.filters import apply_custom_filters  # si utilisé ailleurs
from utils.preprocessing_bip import apply_custom_bipolar_montage  # si utilisé ailleurs

# --- Import comparaison .mat robuste
from utils.compare_mat import compare_with_mat, _load_rem_phasic_intervals as load_rem_intervals

import mne
mne.set_config('MNE_MEMMAP_MIN_SIZE', '1M', set_env=True)  # favorise memmap


# -------------------------------------------------------------------
# Utilitaires
# -------------------------------------------------------------------

def _is_inside(start: float, end: float, intervals, tol: float = 0.5) -> bool:
    s = float(start); e = float(end)
    for a, b in intervals:
        if (a - tol) <= s and e <= (b + tol):
            return True
    return False

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
    # Lecture (engine explicite si dispo)
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
#                 TRAITEMENT PAR PATIENT
# =========================================================
def process_patient(fif_path: Path, raw_dir: Path, out_dir: Path, montage: str, seuil: int):
    patient_id = fif_path.stem.split("_")[0]

    if patient_id == "." or not patient_id.isalnum():
        print(f"[SKIP] Nom de patient invalide : {fif_path.name}")
        return

    out_patient_dir = out_dir / patient_id
    out_patient_dir.mkdir(parents=True, exist_ok=True)

    xlsx_out = out_patient_dir / f"{patient_id}_microstates_{seuil}.xlsx"
    raw_annotated_path = out_patient_dir / f"{patient_id}_annotated_{seuil}.fif"

    # --- Détecte si on a déjà les sorties automatiques ---
    outputs_exist = xlsx_out.exists() and raw_annotated_path.exists()

    # --- Trouve l'annotation hypnogramme seulement si on doit (re)traiter ---
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

    # --- (Re)traitement seulement si nécessaire ---
    if not outputs_exist:
        # Espace disque minimal (sécurité)
        if _free_gb(out_patient_dir) < 5.0:
            print(f"[SKIP] {patient_id}: espace insuffisant (<5 Go) sur {out_patient_dir}")
            return

        # Charge les signaux + segments REM (memmap, preload=False dans loader)
        raw, rem_segments = load_signals_and_annotations(fif_path, annot_path)

        if not rem_segments:
            print(f"[SKIP] {patient_id} : aucun segment REM détecté.")
            _free_big(raw)
            return

        print(f"[{patient_id}] {len(rem_segments)} segments REM détectés.")

        # --- Streaming des fenêtres: pas de liste 'windows' en RAM
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

        print(f"[{patient_id}] {len(valid_times)} fenêtres retenues "
              f"({labels.count('phasic')} phasic / {labels.count('tonic')} tonic)")

        # Écriture via répertoire temporaire + moves atomiques
        with tempfile.TemporaryDirectory(prefix=f"{patient_id}_", dir=out_patient_dir) as td:
            tdir = Path(td)

            # Ajoute les annotations sans écraser l'existant
            annotate_microstates(
                raw, valid_times, labels, window_sec=4,
                mode="add",
                prefix="microstate_"
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

        # Libère RAM
        _free_big(raw, df, valid_times, labels)

    else:
        print(f"[{patient_id}] Sorties auto déjà présentes, on saute le pipeline et on charge l'Excel.")
        try:
            df = _load_microstates_excel(xlsx_out)
        except Exception as e:
            print(f"[{patient_id}] Impossible de lire {xlsx_out.name} pour la comparaison : {e}")
            return

    # =========================
    # Comparaison avec .mat (robuste via compare_mat)
    # =========================
    mat_candidates = sorted((raw_dir / patient_id).glob(f"events_{patient_id}*.mat"))
    print(f"[{patient_id}] {len(mat_candidates)} fichiers .mat trouvés pour comparaison. chemin = {mat_candidates}")
    if mat_candidates:
        mat_path = max(mat_candidates, key=lambda p: p.stat().st_mtime)

        # 1) Sauvegarde des erreurs via compare_with_mat (suffixe avec seuil, comme avant)
        errors_suffix = f"_microstates_errors_{seuil}.xlsx"
        _ = compare_with_mat(
            df_windows=df,
            raw_dir=raw_dir,
            patient_id=patient_id,
            out_patient_dir=out_patient_dir,
            glob_pattern="events_{pid}*.mat",
            inside_tol_sec=0.5,
            pad_sec_if_single_time=2.0,
            errors_xlsx_suffix=errors_suffix,
            verbose=True,
        )

        # 2) Stats + figure locales (loader robuste)
        rem_phasic = load_rem_intervals(mat_path, pad_sec=2.0)

        erreurs = []
        bons = []
        for _, row in df.iterrows():
            start, end, pred = float(row["tmin"]), float(row["tmax"]), str(row["label"]).strip().lower()
            true = "phasic" if _is_inside(start, end, rem_phasic, tol=0.5) else "tonic"
            result = {"tmin": start, "tmax": end, "auto_label": pred, "true_label": true}
            (bons if pred == true else erreurs).append(result)

        # --- Stats globales & par classe ---
        n_good = len(bons)
        n_bad = len(erreurs)
        n_eval = n_good + n_bad
        n_total = len(df)
        pct_good = (100.0 * n_good / n_eval) if n_eval else 0.0
        pct_bad  = (100.0 * n_bad  / n_eval) if n_eval else 0.0

        from collections import Counter
        true_counts = Counter([r["true_label"] for r in bons + erreurs])
        good_counts = Counter([r["true_label"] for r in bons])
        bad_counts  = Counter([r["true_label"] for r in erreurs])

        def _pct(x, denom): return (100.0 * x / denom) if denom else 0.0

        print(
            f"[{patient_id}] Comparaison .mat — "
            f"Évaluées: {n_eval}/{n_total} | "
            f"Bonnes: {n_good} ({pct_good:.2f}%) | "
            f"Mauvaises: {n_bad} ({pct_bad:.2f}%)"
        )
        for cls in ("phasic", "tonic"):
            tc = true_counts.get(cls, 0)
            gc = good_counts.get(cls, 0)
            bc = bad_counts.get(cls, 0)
            print(
                f"    {cls}: vrais={tc} | "
                f"bons={gc} ({_pct(gc, tc):.2f}%) | "
                f"mauvais={bc} ({_pct(bc, tc):.2f}%)"
            )

        # --- Visualisation du summary (PNG) ---
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            categories = ["Bonnes", "Mauvaises"]
            values = [pct_good, pct_bad]
            bars = ax.bar(categories, values)
            ax.set_ylim(0, 100)
            ax.set_ylabel("Pourcentage (%)")
            ax.set_title(f"{patient_id} — microstates ({seuil})\nÉvaluées: {n_eval}/{n_total}")
            for rect, count in zip(bars, [n_good, n_bad]):
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2.0, height,
                        f"{height:.1f}%\n(n={count})",
                        ha="center", va="bottom", fontsize=9)
            out_png = out_patient_dir / f"{patient_id}_microstates_eval_summary_{seuil}.png"
            with tempfile.TemporaryDirectory(prefix=f"{patient_id}_", dir=out_patient_dir) as td:
                tmp_png = Path(td) / out_png.name
                fig.savefig(tmp_png, dpi=150, bbox_inches="tight")
                os.replace(tmp_png, out_png)
            plt.close(fig)
            print(f"[{patient_id}] Visualisation summary sauvegardée → {out_png.name}")
        except Exception as e:
            print(f"[{patient_id}] Plot skipped: {e}")

        _free_big(erreurs, bons, rem_phasic)

    else:
        print(f"[{patient_id}] Pas d’événements .mat pour comparaison manuelle. (comparaison sautée)")

    print(f"[{patient_id}] Traitement terminé.\n")


# =========================================================
#  FONCTION WORKER TOP-LEVEL (picklable pour spawn)
# =========================================================
def run_one(path_str: str, raw_dir_str: str, out_dir_str: str, montage: str, seuil: int):
    """Wrapper: exécution par patient (mémoire isolée)."""
    try:
        mpl_cache = os.path.join(tempfile.gettempdir(), f"mplcache_{os.getpid()}")
        os.environ["MPLCONFIGDIR"] = mpl_cache
        os.makedirs(mpl_cache, exist_ok=True)
    except Exception:
        pass
    try:
        return process_patient(
            Path(path_str), Path(raw_dir_str), Path(out_dir_str), montage, seuil
        )
    except Exception as e:
        # Trace complète depuis le worker (remonte côté parent)
        raise RuntimeError(f"Worker error on {Path(path_str).name}: {e}\n{traceback.format_exc()}") from e


# =========================================================
#         AGRÉGAT GLOBAL (radar multi-seuils + Excel)
# =========================================================
def aggregate_and_plot_overview_all_seuils(raw_dir: Path, out_dir: Path):
    """
    Scanne out_dir/*/ pour tous les *_microstates_*.xlsx.
    Recalcule les stats à partir des .mat, exporte un Excel global,
    et trace un radar multi-polygones (un par seuil).
    """
    from collections import defaultdict, Counter

    def _pct(x, denom): return (100.0 * x / denom) if denom else 0.0
    pattern = re.compile(r"_microstates_(\d+)\.xlsx$", re.IGNORECASE)

    overview_rows = []
    radar_data = defaultdict(dict)   # seuil -> {patient_id: pct_good}
    processed_any = False

    for pdir in sorted([p for p in out_dir.iterdir() if p.is_dir()]):
        pid = pdir.name
        for xlsx_path in sorted(pdir.glob(f"{pid}_microstates_*.xlsx")):
            m = pattern.search(xlsx_path.name)
            if not m:
                continue
            seuil = int(m.group(1))

            # Charger l'xlsx (sanitizer)
            try:
                df = _load_microstates_excel(xlsx_path)
            except Exception as e:
                print(f"[{pid}] lecture {xlsx_path.name} impossible: {e}")
                continue

            # .mat candidats
            mat_candidates = sorted((raw_dir / pid).glob(f"events_{pid}*.mat"))
            if not mat_candidates:
                print(f"[{pid}] pas de .mat — ignoré pour stats, errors_{seuil}.xlsx vide créé.")
                pd.DataFrame(columns=["tmin", "tmax", "auto_label", "true_label"]).to_excel(
                    pdir / f"{pid}_microstates_errors_{seuil}.xlsx", index=False, engine="xlsxwriter"
                )
                continue

            mat_path = max(mat_candidates, key=lambda q: q.stat().st_mtime)

            # Intervalles robustes
            rem_phasic = load_rem_intervals(mat_path, pad_sec=2.0)

            erreurs, bons = [], []
            for _, row in df.iterrows():
                start, end, pred = float(row["tmin"]), float(row["tmax"]), str(row["label"]).strip().lower()
                true = "phasic" if _is_inside(start, end, rem_phasic, tol=0.5) else "tonic"
                result = {"tmin": float(start), "tmax": float(end), "auto_label": pred, "true_label": true}
                (bons if pred == true else erreurs).append(result)

            n_good, n_bad = len(bons), len(erreurs)
            n_eval = n_good + n_bad
            n_total = len(df)
            pct_good = (100.0 * n_good / n_eval) if n_eval else np.nan
            pct_bad  = 100.0 - pct_good if n_eval and not np.isnan(pct_good) else np.nan

            from collections import Counter
            true_counts = Counter([r["true_label"] for r in bons + erreurs])
            good_counts = Counter([r["true_label"] for r in bons])
            bad_counts  = Counter([r["true_label"] for r in erreurs])

            overview_rows.append({
                "patient_id": pid,
                "seuil": seuil,
                "mat_file": str(mat_path.name),
                "n_total_windows": n_total,
                "n_evaluated": n_eval,
                "n_good": n_good,
                "n_bad": n_bad,
                "pct_good": pct_good,
                "pct_bad": pct_bad,
                "true_phasic": true_counts.get("phasic", 0),
                "true_tonic":  true_counts.get("tonic", 0),
                "good_phasic": good_counts.get("phasic", 0),
                "good_tonic":  good_counts.get("tonic", 0),
                "bad_phasic":  bad_counts.get("phasic", 0),
                "bad_tonic":   bad_counts.get("tonic", 0),
            })

            # Fichier d'erreurs (même si vide)
            pd.DataFrame(erreurs, columns=["tmin", "tmax", "auto_label", "true_label"]).to_excel(
                pdir / f"{pid}_microstates_errors_{seuil}.xlsx", index=False, engine="xlsxwriter"
            )
            print(f"[{pid}] errors_{seuil}.xlsx sauvegardé ({len(erreurs)} erreurs).")

            radar_data[seuil][pid] = pct_good
            processed_any = True

    if not processed_any:
        print("[AGRÉGAT] Aucun couple (patient, seuil) éligible (xlsx + .mat).")
        return

    # --- Export Excel global (tous seuils)
    df_overview = pd.DataFrame(overview_rows)
    out_excel = out_dir / "microstates_eval_overview_all.xlsx"
    df_overview.to_excel(out_excel, index=False, engine="xlsxwriter")
    print(f"[AGRÉGAT] Export récap → {out_excel}")

    # --- Radar multi-polygones (un par seuil) ---
    all_patients = sorted({pid for d in radar_data.values() for pid in d.keys()})
    N = len(all_patients)
    angles = np.linspace(0, 2 * math.pi, N, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(7, 7))
    ax.set_ylim(0, 100)
    ax.set_xticks(angles)
    ax.set_xticklabels(all_patients, fontsize=9)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels([str(v) for v in [20, 40, 60, 80, 100]])
    ax.set_title("Pourcentage de bonnes prédictions — tous seuils", va="bottom")

    seuils_sorted = sorted(radar_data.keys())
    two_standard = (len(seuils_sorted) == 2 and set(seuils_sorted) == {100, 150})
    color_map = {100: "tab:blue", 150: "tab:orange"} if two_standard else {}

    for s in seuils_sorted:
        data = radar_data[s]
        vals = [data.get(pid, np.nan) for pid in all_patients]
        vals_closed = vals + [vals[0]]
        color = color_map.get(s, None)  # None => couleur auto
        ax.plot(angles_closed, vals_closed, linewidth=2, label=f"seuil {s}", color=color)
        ax.fill(angles_closed, vals_closed, alpha=0.15, color=color)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    out_png = out_dir / "microstates_eval_radar_all.png"
    with tempfile.TemporaryDirectory(prefix="_overview_", dir=out_dir) as td:
        tmp_png = Path(td) / out_png.name
        fig.savefig(tmp_png, dpi=150, bbox_inches="tight")
        os.replace(tmp_png, out_png)
    plt.close(fig)
    print(f"[AGRÉGAT] Radar multi-seuils sauvegardé → {out_png}")


# =========================================================
#                        MAIN
# =========================================================
if __name__ == "__main__":
    # --- Parsing des arguments CLI ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--seuil", type=int, default=150,
                        help="Valeur de seuil pour le nommage des fichiers (ex: 100 ou 150)")
    parser.add_argument("--aggregate", action="store_true",
                        help="Mode agrégat : radar multi-seuils + Excel récap, sans retraiter les FIF.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Nb de processus en parallèle (0 => CPU-1)")
    parser.add_argument("--montage", choices=["bipolaire", "monopolaire"], default=None,
                        help="Montage à utiliser sans invite interactive.")
    args = parser.parse_args()
    seuil = args.seuil

    # --- Choix du montage ---
    def choose_montage(args):
        if args.aggregate:
            return args.montage or "bipolaire"
        if args.montage in {"bipolaire", "monopolaire"}:
            return args.montage
        resp = input("Montage bipolaire ? (y/n) : ").strip().lower()
        if resp not in {"y", "n"}:
            print("Réponse invalide. Tape 'y' ou 'n'."); raise SystemExit(1)
        return "bipolaire" if resp == "y" else "monopolaire"

    montage = choose_montage(args)

    # --- Détection du disque selon OS (ici forcé) ---
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

    root_preproc = Path(f"{disque}/EEG/preprocessed/{montage}/full/")
    root_raw     = Path(f"{disque}/EEG/raw")
    root_out     = Path(f"{disque}/EEG/preprocessed/{montage}/rem_only")

    # Vérif d'existence + création
    for p in [root_preproc, root_raw]:
        if not p.exists():
            raise SystemExit(f"[CONFIG] Dossier introuvable: {p}")
    root_out.mkdir(parents=True, exist_ok=True)

    # --- Warm-up Matplotlib pour éviter la création concurrente du cache
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

    if args.aggregate:
        aggregate_and_plot_overview_all_seuils(raw_dir=root_raw, out_dir=root_out)
    else:
        fif_paths = [
            p for p in root_preproc.rglob("*_preprocessed_*.fif")
            if not p.name.startswith("._") and not p.name.startswith(".")
        ]
        print(f"{len(fif_paths)} fichiers trouvés dans {root_preproc}")

        if not fif_paths:
            raise SystemExit(0)

        # Calcul du nombre de workers
        cpu = os.cpu_count() or 1
        max_workers = (cpu - 1) if args.workers in (0, None) else max(1, args.workers)
        max_workers = min(max_workers, len(fif_paths))
        print(f"[INFO] Lancement en multiprocess avec {max_workers} worker(s) (CPU={cpu}) (maxtasksperchild=1)")

        # Prépare les arguments (strings picklables)
        args_list = [(str(p), str(root_raw), str(root_out), montage, seuil) for p in fif_paths]

        # Pool recyclable + Option A: STARMAP (pas de wrapper local)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=max_workers, maxtasksperchild=1) as pool:
            try:
                # starmap renvoie les résultats dans l'ordre; on itère pour exécuter/propager les erreurs
                for _ in pool.starmap(run_one, args_list, chunksize=1):
                    pass
            except Exception as e:
                print(f"[POOL ERROR] {e}\n{traceback.format_exc()}")
