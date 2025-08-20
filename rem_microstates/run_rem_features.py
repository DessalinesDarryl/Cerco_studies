#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline REM (EEG) — extraction de caractéristiques spectrales et micro-états phasic/tonic.

Fonctionnalités :
- Parcourt un dossier de patients (un sous-dossier/patient).
- Charge les époques REM et les labels micro-états alignés.
- Calcule, de manière vectorisée, les puissances de bandes (delta→gamma) + score gamma/bêta par époque.
- Calcule des indicateurs résumés micro-états (phasic_ratio, alternations, etc.).
- Sauvegarde des CSV par époque et des JSON résumés par patient.
- Produit des figures par patient et des figures de groupe.
- Parallélisation multi-process configurable via CLI.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Limiter dès le début les threads internes BLAS/OpenMP
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # Accelerate (macOS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")  # backend non interactif (nécessaire en multi-process)
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

from signal_processing.eeg_features import (  # noqa: E402
    compute_band_powers_epochs,
    compute_microstate_features,
)
from signal_processing.utils import (  # noqa: E402
    align_labels_to_epochs,
    load_epochs,
    load_labels_any,
)


# =========================
# Utilitaires CPU / MP
# =========================

def pick_n_workers(workers: Optional[int], leave_free: int = 1) -> int:
    """
    Détermine le nombre de workers à utiliser.

    Parameters
    ----------
    workers : Optional[int]
        Valeur CLI explicite. Si None, calcule os.cpu_count() - leave_free.
    leave_free : int
        Nombre de cœurs logiques laissés libres.

    Returns
    -------
    int
        Nombre de workers (>= 1).
    """
    if workers and workers >= 1:
        return int(workers)
    ncpu = os.cpu_count() or 1
    return max(1, ncpu - int(leave_free))


def get_mp_context() -> mp.context.BaseContext:
    """
    Contexte multiprocessing portable.

    Returns
    -------
    mp.context.BaseContext
        Contexte 'spawn' si disponible, sinon défaut du système.
    """
    try:
        return mp.get_context("spawn")
    except ValueError:
        return mp.get_context()


# =========================
# Calculs par patient
# =========================

def compute_scores_per_epoch_vectorized(epochs: mne.Epochs) -> Dict[str, np.ndarray]:
    """
    Calcule les puissances de bandes (delta→gamma) et le score gamma/bêta pour toutes les époques.

    Parameters
    ----------
    epochs : mne.Epochs

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionnaire : 'delta', 'theta', 'alpha', 'beta', 'gamma', 'score'
        (toutes les valeurs sont des arrays de shape (n_epochs,)).
    """
    return compute_band_powers_epochs(epochs)


def summarize_patient(
    bands: Dict[str, np.ndarray],
    labels: List[str],
    window_sec: float,
) -> Dict[str, float]:
    """
    Résume les mesures par patient : statistiques du score et des micro-états,
    plus la moyenne/écart-type des bandes.

    Parameters
    ----------
    bands : Dict[str, np.ndarray]
        Puissances de bandes et score par époque.
    labels : List[str]
        Labels micro-états alignés aux époques.
    window_sec : float
        Durée d'une époque (s), utilisée dans les features micro-états.

    Returns
    -------
    Dict[str, float]
        Dictionnaire de résumés par patient.
    """
    feats = compute_microstate_features(labels, window_sec=window_sec)
    score = bands["score"]

    out = {
        "score_mean": float(np.mean(score)) if len(score) else np.nan,
        "score_std": float(np.std(score, ddof=1)) if len(score) > 1 else 0.0,
        "score_median": float(np.median(score)) if len(score) else np.nan,
    }

    for band in ("delta", "theta", "alpha", "beta", "gamma"):
        vals = bands[band]
        out[f"{band}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
        out[f"{band}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    out.update(feats)
    return out


# =========================
# Visualisation
# =========================

def plot_patient_figures(
    patient_id: str,
    bands: Dict[str, np.ndarray],
    labels: List[str],
    out_dir: Path,
) -> None:
    """
    Génère 3 figures pour un patient :
    - Histogramme des scores (gamma/bêta)
    - Série temporelle des scores colorée par micro-état
    - Répartition phasic/tonic
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = bands["score"]

    # A) Histogramme
    plt.figure(figsize=(6, 4))
    plt.hist(scores, bins=30, alpha=0.9)
    if len(scores):
        plt.axvline(np.mean(scores), linestyle="--", linewidth=2)
    plt.title(f"{patient_id} — Distribution du score gamma/bêta")
    plt.xlabel("Score spectral (gamma/bêta, 1–45 Hz)")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.savefig(out_dir / f"{patient_id}_hist_scores.png", dpi=150)
    plt.close()

    # B) Série temporelle
    color_map = {"phasic": "tab:red", "tonic": "tab:blue"}
    colors = [color_map.get(lab, "gray") for lab in labels]
    x = np.arange(len(scores))
    plt.figure(figsize=(10, 3.8))
    plt.scatter(x, scores, s=12, c=colors)
    if len(scores) >= 5:
        mov = np.convolve(scores, np.ones(5) / 5, mode="same")
        plt.plot(x, mov, linewidth=2)
    plt.title(f"{patient_id} — Score par époque (couleur = micro-état)")
    plt.xlabel("Époque (4 s)")
    plt.ylabel("Score gamma/bêta")
    plt.tight_layout()
    plt.savefig(out_dir / f"{patient_id}_timeseries_scores.png", dpi=150)
    plt.close()

    # C) Répartition phasic/tonic
    vals, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(5, 4))
    plt.bar(vals, counts)
    plt.title(f"{patient_id} — Répartition des micro-états")
    plt.ylabel("Nombre d'époques")
    plt.tight_layout()
    plt.savefig(out_dir / f"{patient_id}_bar_labels.png", dpi=150)
    plt.close()


def plot_group_figures(
    per_epoch_df: pd.DataFrame,
    per_patient_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Figures de groupe :
      - Boxplot score par patient
      - Scatter phasic_ratio vs score_mean avec corrélation de Pearson
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Boxplot score par patient
    plt.figure(figsize=(10, 4))
    order = per_patient_df.sort_values("score_mean")["patient"].tolist()
    data = [
        per_epoch_df.loc[per_epoch_df["patient"] == p, "score"].values
        for p in order
    ]
    plt.boxplot(data, labels=order, showfliers=False)
    plt.xticks(rotation=60, ha="right")
    plt.title("Distribution du score par patient")
    plt.ylabel("Score gamma/bêta")
    plt.tight_layout()
    plt.savefig(out_dir / "group_box_scores_by_patient.png", dpi=150)
    plt.close()

    # Scatter + corrélation
    x = per_patient_df["phasic_ratio"].values
    y = per_patient_df["score_mean"].values
    mask = np.isfinite(x) & np.isfinite(y)
    r, p = (np.nan, np.nan)
    if mask.sum() >= 3:
        r, p = pearsonr(x[mask], y[mask])

    plt.figure(figsize=(5.6, 4.6))
    plt.scatter(x, y)
    for _, rrow in per_patient_df.iterrows():
        plt.annotate(
            rrow["patient"],
            (rrow["phasic_ratio"], rrow["score_mean"]),
            fontsize=7,
            alpha=0.7,
        )
    plt.xlabel("Phasic ratio")
    plt.ylabel("Score moyen (gamma/bêta)")
    title = "Phasic ratio vs score spectral moyen"
    if np.isfinite(r):
        title += f" (r={r:.2f}, p={p:.3g})"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / "group_scatter_phasicratio_vs_scoremean.png", dpi=150)
    plt.close()


# =========================
# Utilitaires locaux
# =========================

def find_first(root: Path, patterns: Iterable[str]) -> Optional[Path]:
    """Retourne le premier fichier existant dans root qui matche un des patterns."""
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_file():
                return p
    return None


# =========================
# Traitement patient
# =========================

def process_patient(patient_dir: Path, window_sec: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Traite un patient et retourne (df_epoch, summary).

    Steps :
    - charge les époques (mne.Epochs) et labels (csv/txt/json),
    - calcule les puissances de bandes vectorisées + score,
    - produit les figures patient,
    - sauvegarde CSV/JSON.
    """
    pid = patient_dir.name

    fif = find_first(patient_dir, ("*rem*epochs*.fif", "*.fif"))
    if fif is None:
        raise FileNotFoundError(f"[{pid}] Aucun .fif d'époques trouvé dans {patient_dir}")

    labels_path = find_first(
        patient_dir,
        ("labels.csv", "*labels*.csv", "labels.txt", "*labels*.txt", "labels.json"),
    )
    if labels_path is None:
        raise FileNotFoundError(f"[{pid}] Aucun fichier de labels (csv/txt/json) trouvé dans {patient_dir}")

    epochs = load_epochs(fif)

    # Calcul vectorisé des bandes et du score
    bands = compute_scores_per_epoch_vectorized(epochs)  # dict (n_epochs,)
    scores = bands["score"]

    # Sanity log
    if len(scores) > 0:
        print(f"[{pid}] score min/max: {scores.min():.3f} / {scores.max():.3f}")

    # Labels alignés
    labels = load_labels_any(labels_path)
    labels = align_labels_to_epochs(labels, len(epochs))
    assert len(labels) == len(epochs), "Labels et époques non alignés après align_labels_to_epochs."

    # Per-epoch dataframe (inclure les bandes)
    df_ep = pd.DataFrame({"patient": pid, "epoch_idx": np.arange(len(scores))})
    for band in ("delta", "theta", "alpha", "beta", "gamma", "score"):
        df_ep[band] = bands[band]
    df_ep["label"] = labels

    # Résumé patient
    summary = summarize_patient(bands, labels, window_sec=window_sec)
    summary["patient"] = pid

    # Plots
    figs_dir = patient_dir / "figs_rem_features"
    plot_patient_figures(pid, bands, labels, figs_dir)

    # Sauvegardes locales
    df_ep.to_csv(patient_dir / f"{pid}_rem_features_per_epoch.csv", index=False)
    with open(patient_dir / f"{pid}_rem_features_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return df_ep, summary


# =========================
# Main (CLI)
# =========================

def main() -> None:
    """Point d'entrée CLI avec parallélisation configurable."""
    parser = argparse.ArgumentParser(
        description="Extraction de caractéristiques REM + micro-états (phasic/tonic) avec parallélisation."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Nombre de workers (process). Par défaut: cpu_count - leave-free.",
    )
    parser.add_argument(
        "--leave-free",
        type=int,
        default=1,
        help="Nombre de cœurs logiques laissés libres (défaut: 1).",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=4.0,
        help="Durée d'une époque (s) utilisée pour les indicateurs micro-états.",
    )
    args = parser.parse_args()

    # Choix du montage (interaction type main_detect_microstates.py)
    response = input("Montage bipolaire ? (y/n) : ").strip().lower()
    if response not in {"y", "n"}:
        print("Réponse invalide. Tape 'y' ou 'n'.")
        sys.exit(1)
    montage = "bipolaire" if response == "y" else "monopolaire"

    # Détection du système (mêmes chemins que ton autre script)
    system = platform.system()
    if system == "Darwin":
        disque = "/Volumes/Crucial X6"
    elif system == "Windows":
        disque = "D:"
    elif system == "Linux":
        disque = "/media/darryld/Crucial X6"
    else:
        raise RuntimeError("Système non supporté.")

    base_dir = Path(f"{disque}/EEG/preprocessed/{montage}/rem_only").resolve()
    out_dir = base_dir / "_rem_features_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Un sous-dossier par patient dans base_dir
    patient_dirs = [
        p for p in base_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and not p.name.startswith("._")
    ]
    if not patient_dirs:
        print(f"Aucun sous-dossier patient trouvé dans {base_dir}", file=sys.stderr)
        sys.exit(1)

    # Parallélisation
    n_workers = pick_n_workers(args.workers, leave_free=args.leave_free)
    ctx = get_mp_context()
    print(f"Parallélisation : {n_workers} workers (sur {os.cpu_count()} cœurs logiques) [spawn]")

    all_epoch_rows: List[pd.DataFrame] = []
    all_summaries: List[Dict[str, float]] = []

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as exe:
        futures = {
            exe.submit(process_patient, pdir, args.window_sec): pdir
            for pdir in sorted(patient_dirs)
        }
        for fut in as_completed(futures):
            pdir = futures[fut]
            try:
                df_ep, summary = fut.result()
            except Exception as exc:
                print(f"[SKIP] {pdir.name}: {exc}", file=sys.stderr)
                continue
            all_epoch_rows.append(df_ep)
            all_summaries.append(summary)
            print(f"[OK] {pdir.name} — {len(df_ep)} époques traitées")

    if not all_epoch_rows:
        print("Rien à agréger (aucun patient traité).", file=sys.stderr)
        sys.exit(2)

    per_epoch_df = pd.concat(all_epoch_rows, ignore_index=True)
    per_patient_df = pd.DataFrame(all_summaries)

    # Sauvegardes globales
    per_epoch_df.to_csv(out_dir / "all_patients_rem_features_per_epoch.csv", index=False)
    per_patient_df.to_csv(out_dir / "all_patients_rem_features_summary.csv", index=False)

    # Figures de groupe
    plot_group_figures(per_epoch_df, per_patient_df, out_dir)

    print(
        "\nTerminé. Résultats :"
        f"\n- {out_dir / 'all_patients_rem_features_per_epoch.csv'}"
        f"\n- {out_dir / 'all_patients_rem_features_summary.csv'}"
        f"\n- Figures dans {out_dir}"
    )


if __name__ == "__main__":
    main()
