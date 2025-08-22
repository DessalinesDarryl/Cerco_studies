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

from concurrent.futures import ProcessPoolExecutor, as_completed
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

# ==========================
#   Imports projet
# ==========================
from signal_processing.loader import load_signals_and_annotations
from signal_processing.windowing import segment_rem_in_windows
from signal_processing.eog_analysis import detect_eog_microstate
from signal_processing.annotation import annotate_microstates
from signal_processing.filters import apply_custom_filters  # si utilisé ailleurs
from utils.preprocessing_bip import apply_custom_bipolar_montage  # si utilisé ailleurs

# --- Import comparaison .mat robuste
from utils.compare_mat import compare_with_mat, _load_rem_phasic_intervals as load_rem_intervals

import mne


# Petit helper commun (tolérance d’inclusion)
def _is_inside(start: float, end: float, intervals, tol: float = 0.5) -> bool:
    s = float(start); e = float(end)
    for a, b in intervals:
        if (a - tol) <= s and e <= (b + tol):
            return True
    return False


# =========================================================
#                 TRAITEMENT PAR PATIENT
# =========================================================
def process_patient(fif_path, raw_dir, out_dir, montage, seuil):
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
        # Charge les signaux + segments REM et exécute le pipeline
        raw, rem_segments = load_signals_and_annotations(fif_path, annot_path)

        if not rem_segments:
            print(f"[SKIP] {patient_id} : aucun segment REM détecté.")
            return

        print(f"[{patient_id}] {len(rem_segments)} segments REM détectés.")
        windows = segment_rem_in_windows(raw, rem_segments, window_sec=4, step_sec=4)

        labels = []
        valid_windows = []
        i = 0
        while i < len(windows):
            win = windows[i]
            label = detect_eog_microstate(win)
            if label != "ignore":
                labels.append(label)
                valid_windows.append(win)
            i += 2 if label == "phasic" else 1

        print(f"[{patient_id}] {len(valid_windows)} fenêtres retenues "
              f"({labels.count('phasic')} phasic / {labels.count('tonic')} tonic)")

        # Ajoute les annotations sans écraser l'existant
        annotate_microstates(
            raw, valid_windows, labels, window_sec=4,
            mode="add",
            prefix="microstate_"
        )

        # Sauvegarde du Raw annoté
        raw.save(raw_annotated_path, overwrite=True)

        # Export tabulaire des fenêtres retenues
        df = pd.DataFrame({
            "tmin": [win.first_time for win in valid_windows],
            "tmax": [win.first_time + 4 for win in valid_windows],
            "label": labels
        })
        df.to_excel(xlsx_out, index=False)
    else:
        print(f"[{patient_id}] Sorties auto déjà présentes, on saute le pipeline et on charge l'Excel.")
        try:
            df = pd.read_excel(xlsx_out)
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

        # 2) Stats + figure locales (en réutilisant le loader robuste)
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

        if n_eval > 0:
            pct_good = 100.0 * n_good / n_eval
            pct_bad  = 100.0 * n_bad  / n_eval
        else:
            pct_good = pct_bad = 0.0

        from collections import Counter
        true_counts = Counter([r["true_label"] for r in bons + erreurs])
        good_counts = Counter([r["true_label"] for r in bons])
        bad_counts  = Counter([r["true_label"] for r in erreurs])

        def _pct(x, denom):
            return (100.0 * x / denom) if denom else 0.0

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
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[{patient_id}] Visualisation summary sauvegardée → {out_png.name}")
        except Exception as e:
            print(f"[{patient_id}] Plot skipped: {e}")

    else:
        print(f"[{patient_id}] Pas d’événements .mat pour comparaison manuelle. (comparaison sautée)")


    print(f"[{patient_id}] Traitement terminé.\n")


# =========================================================
#  FONCTION WORKER TOP-LEVEL (picklable pour spawn)
# =========================================================
def run_one(path_str: str, raw_dir_str: str, out_dir_str: str, montage: str, seuil: int):
    """Wrapper isolant le cache Matplotlib, pour exécution en multiprocess."""
    try:
        mpl_cache = f"/tmp/mplcache_{os.getpid()}"
        os.environ["MPLCONFIGDIR"] = mpl_cache
        os.makedirs(mpl_cache, exist_ok=True)
    except Exception:
        pass
    return process_patient(
        Path(path_str),
        Path(raw_dir_str),
        Path(out_dir_str),
        montage,
        seuil,
    )


# =========================================================
#         AGRÉGAT GLOBAL (radar multi-seuils + Excel)
# =========================================================
def aggregate_and_plot_overview_all_seuils(raw_dir: Path, out_dir: Path):
    """
    Scanne out_dir/*/ pour tous les *_microstates_*.xlsx.
    Pour chaque fichier, retrouve events_{ID}*.mat, recalcule les stats,
    écrit errors_{seuil}.xlsx par (patient, seuil),
    exporte un Excel global unique (tous seuils),
    et trace un radar multi-polygones (un par seuil).
    """
    from collections import defaultdict, Counter

    def _pct(x, denom):
        return (100.0 * x / denom) if denom else 0.0

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

            # Charger l'xlsx auto
            try:
                df = pd.read_excel(xlsx_path)
            except Exception as e:
                print(f"[{pid}] lecture {xlsx_path.name} impossible: {e} — errors_{seuil}.xlsx vide créé.")
                pd.DataFrame(columns=["tmin", "tmax", "auto_label", "true_label"]).to_excel(
                    pdir / f"{pid}_microstates_errors_{seuil}.xlsx", index=False
                )
                continue

            # .mat candidats
            mat_candidates = sorted((raw_dir / pid).glob(f"events_{pid}*.mat"))
            if not mat_candidates:
                print(f"[{pid}] pas de .mat — ignoré pour stats, errors_{seuil}.xlsx vide créé.")
                pd.DataFrame(columns=["tmin", "tmax", "auto_label", "true_label"]).to_excel(
                    pdir / f"{pid}_microstates_errors_{seuil}.xlsx", index=False
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
                pdir / f"{pid}_microstates_errors_{seuil}.xlsx", index=False
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
    df_overview.to_excel(out_excel, index=False)
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

    # Couleurs: si exactement {100,150} -> bleu/orange ; sinon couleurs auto
    seuils_sorted = sorted(radar_data.keys())
    two_standard = (len(seuils_sorted) == 2 and set(seuils_sorted) == {100, 150})
    color_map = {100: "tab:blue", 150: "tab:orange"} if two_standard else {}

    for seuil in seuils_sorted:
        data = radar_data[seuil]
        vals = [data.get(pid, np.nan) for pid in all_patients]
        vals_closed = vals + [vals[0]]
        color = color_map.get(seuil, None)  # None => couleur auto
        ax.plot(angles_closed, vals_closed, linewidth=2, label=f"seuil {seuil}", color=color)
        ax.fill(angles_closed, vals_closed, alpha=0.15, color=color)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    out_png = out_dir / "microstates_eval_radar_all.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
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
    parser.add_argument("--workers", type=int, default=10,
                        help="Nb de processus en parallèle (0 => CPU-1)")
    args = parser.parse_args()
    seuil = args.seuil

    # --- Choix du montage ---
    response = input("Montage bipolaire ? (y/n) : ").strip().lower()
    if response not in {"y", "n"}:
        print("Réponse invalide. Tape 'y' ou 'n'.")
        raise SystemExit(1)
    montage = "bipolaire" if response == "y" else "monopolaire"

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

    # --- Warm-up Matplotlib pour éviter la création concurrente du cache
    def _warmup_matplotlib():
        import os
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import font_manager as fm
        matplotlib.get_cachedir()
        fm.findfont('DejaVu Sans', rebuild_if_missing=True)
        fig = plt.figure()
        plt.plot([0, 1], [0, 1])
        with open(os.devnull, "wb") as f:
            fig.savefig(f, format="png")  # pas d’extension => pas de '/dev/null.png'
        plt.close(fig)

    _warmup_matplotlib()

    if args.aggregate:
        # Agrégat multi-seuils (scanne tous les *_microstates_*.xlsx)
        aggregate_and_plot_overview_all_seuils(raw_dir=root_raw, out_dir=root_out)
    else:
        # Mode traitement patient par patient
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
        print(f"[INFO] Lancement en multiprocess avec {max_workers} worker(s) (CPU={cpu})")

        ctx = mp.get_context("spawn")  # plus sûr avec NumPy/MKL

        # Prépare les arguments (strings picklables)
        args_list = [(str(p), str(root_raw), str(root_out), montage, seuil) for p in fif_paths]

        # Pool de processus
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            futures = {ex.submit(run_one, *a): a[0] for a in args_list}
            for fut in as_completed(futures):
                pstr = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"[ERROR] {Path(pstr).name}: {e}")
