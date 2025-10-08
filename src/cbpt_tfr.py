#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CBPT sur cartes TFR (temps×fréquence) entre catégories de patients.

Prérequis I/O:
  - Pour chaque catégorie CAT, stage ST et canal CH:
      IN_ROOT/CAT/ST/stack_CH.npy   # (n_subj, n_freq, n_time)
      IN_ROOT/CAT/ST/times.npy
      IN_ROOT/CAT/ST/freqs.npy

Sorties:
  OUT_ROOT/{CATA}_vs_{CATB}/{ST}/{CH}/
    - cbpt_{CATA}_vs_{CATB}_{ST}_{CH}.png         # heatmap Δ (dB ou % rel) + contours clusters p<alpha
    - cbpt_{CATA}_vs_{CATB}_{ST}_{CH}_clusters.csv
    - cbpt_{CATA}_vs_{CATB}_{ST}_{CH}_sigmask.npy # masque binaire (clusters significatifs)

Usage (exemples):
  python cbpt_tfr.py \
    --in-root "/home/darryld/documents/EEG/preprocessed/bipolaire/3_results_analysis/gp2/_by_patient_category" \
    --out-root "/home/darryld/documents/EEG/preprocessed/bipolaire/3_results_analysis/gp2/_by_patient_category/_cbpt_between_groups" \
    --groups syn narco \
    --stages REM N2 N3 \
    --diff-mode rel \
    --n-perm 5000 --cdt-p 0.05 --alpha 0.05

  # Toutes paires de catégories présentes, tous canaux communs:
  python cbpt_tfr.py --in-root .../_by_patient_category --auto-pairs --diff-mode rel

Notes:
  - Test non apparié (Student, variances égales), df = nA+nB-2, CDT bilatéral.
  - Connexité 8 (diagonales incluses).
  - Option de contrôle FWER global par max pooling inter-canaux (--pool-channels).
"""

from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from scipy.ndimage import label, generate_binary_structure

# ---------------- Utils ----------------
def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "", s)

def robust_sym_limits(A: np.ndarray, pct_lo=5, pct_hi=95):
    A = A[np.isfinite(A)]
    if A.size == 0:
        return (-1.0, 1.0)
    lo, hi = np.percentile(A, [pct_lo, pct_hi])
    m = max(abs(lo), abs(hi))
    if m == 0:
        m = 1e-6
    return (-m, m)

def load_stack(in_dir: Path, ch: str):
    stack_p = in_dir / f"stack_{sanitize(ch)}.npy"
    times_p = in_dir / "times.npy"
    freqs_p = in_dir / "freqs.npy"
    if not (stack_p.exists() and times_p.exists() and freqs_p.exists()):
        return None, None, None
    X = np.load(stack_p)  # (n, F, T)
    times = np.load(times_p)
    freqs = np.load(freqs_p)
    return X, times, freqs

def align_min(A: np.ndarray, B: np.ndarray, timesA, freqsA, timesB, freqsB):
    F = min(A.shape[1], B.shape[1], len(freqsA), len(freqsB))
    T = min(A.shape[2], B.shape[2], len(timesA), len(timesB))
    return (A[:, :F, :T].astype(float, copy=False),
            B[:, :F, :T].astype(float, copy=False),
            np.asarray(timesA[:T]),
            np.asarray(freqsA[:F]))

def t_map_independent_equalvar(XA: np.ndarray, XB: np.ndarray, eps=1e-12):
    """
    XA: (nA, F, T), XB: (nB, F, T). Student non apparié, variances égales.
    Retourne (t(F,T), df int, delta(F,T) en dB, d(F,T) Cohen).
    """
    nA, nB = XA.shape[0], XB.shape[0]
    mA = XA.mean(axis=0)
    mB = XB.mean(axis=0)
    vA = XA.var(axis=0, ddof=1)
    vB = XB.var(axis=0, ddof=1)

    # variance poolée
    sp2 = ((nA - 1) * vA + (nB - 1) * vB) / max(nA + nB - 2, 1)
    sp2 = np.maximum(sp2, eps)
    se = np.sqrt(sp2 * (1.0 / nA + 1.0 / nB))
    t = (mA - mB) / np.maximum(se, eps)

    # Cohen's d (Hedges' g optionnel)
    s_pooled = np.sqrt(sp2)
    d = (mA - mB) / np.maximum(s_pooled, eps)

    return t, (nA + nB - 2), (mA - mB), d

def form_clusters(t: np.ndarray, tcrit: float):
    """Retourne clusters pos/neg: listes de (label_id, mask_bool, mass_float)."""
    conn = generate_binary_structure(2, 2)  # 8-connexité
    pos_mask = (t >= tcrit)
    neg_mask = (t <= -tcrit)

    clusters = []

    for mask, sign in [(pos_mask, +1), (neg_mask, -1)]:
        lab, nlab = label(mask, structure=conn)
        for k in range(1, nlab + 1):
            m = (lab == k)
            mass = (t[m].sum() if sign > 0 else (-t[m].sum()))
            clusters.append((sign, m, float(mass)))

    # max absolu pour la FWER
    max_abs_mass = 0.0
    if clusters:
        max_abs_mass = max(abs(c[2]) for c in clusters)

    return clusters, max_abs_mass

def permutation_max_mass(XA: np.ndarray, XB: np.ndarray, tcrit: float, n_perm: int, rng: np.random.Generator):
    """Renvoie un vecteur (n_perm,) des maxima de masse (absolue) sous H0."""
    nA, nB = XA.shape[0], XB.shape[0]
    allX = np.concatenate([XA, XB], axis=0)  # (N, F, T)
    N = nA + nB
    idx = np.arange(N)

    max_masses = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        rng.shuffle(idx)
        iA = idx[:nA]
        iB = idx[nA:]
        t_pi, _, _, _ = t_map_independent_equalvar(allX[iA], allX[iB])
        _, max_abs_mass = form_clusters(t_pi, tcrit)
        max_masses[i] = max_abs_mass
    return max_masses

def db_delta_to_rel_percent(delta_db: np.ndarray) -> np.ndarray:
    """Convertit un delta en dB vers une différence relative en % sur l'échelle linéaire."""
    return 100.0 * (np.power(10.0, delta_db / 10.0) - 1.0)

def summarize_cluster(mask: np.ndarray, mass: float,
                      delta_db: np.ndarray, d: np.ndarray,
                      times: np.ndarray, freqs: np.ndarray):
    """
    Stats résumées par cluster.
    - delta_db : carte delta en dB (A-B)
    - d        : carte de Cohen's d
    Retourne aussi la moyenne de la différence relative en % (calculée sur l'échelle linéaire).
    """
    ys, xs = np.where(mask)
    fmin, fmax = freqs[ys.min()], freqs[ys.max()]
    tmin, tmax = times[xs.min()], times[xs.max()]

    delta_rel = db_delta_to_rel_percent(delta_db)  # 100*(10^(ΔdB/10)-1)

    mean_delta_db  = float(delta_db[mask].mean())
    mean_delta_rel = float(delta_rel[mask].mean())
    mean_d         = float(d[mask].mean())
    n_pix          = int(mask.sum())

    return {
        "mass": float(mass),
        "n_pixels": n_pix,
        "f_min_hz": float(fmin),
        "f_max_hz": float(fmax),
        "t_min_s": float(tmin),
        "t_max_s": float(tmax),
        "delta_db_mean": mean_delta_db,
        "delta_rel_percent_mean": mean_delta_rel,
        "cohen_d_mean": mean_d,
    }

def plot_with_clusters(out_png: Path, delta: np.ndarray, times: np.ndarray, freqs: np.ndarray,
                       sig_masks: list[np.ndarray], title: str, cbar_label: str,
                       vmin: float | None = None, vmax: float | None = None):
    if vmin is None or vmax is None:
        vmin, vmax = robust_sym_limits(delta)
        m = max(abs(vmin), abs(vmax)) or 1e-6
        vmin, vmax = -m, m
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    im = ax.imshow(delta, aspect="auto", origin="lower",
                   extent=[times[0], times[-1], freqs[0], freqs[-1]],
                   cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Fréquence (Hz)")
    cbar = plt.colorbar(im, ax=ax); cbar.set_label(cbar_label)
    ax.set_title(title)

    # Contours
    for m in sig_masks:
        if m is None or not m.any():
            continue
        ax.contour(np.linspace(times[0], times[-1], m.shape[1]),
                   np.linspace(freqs[0], freqs[-1], m.shape[0]),
                   m.astype(float), levels=[0.5], linewidths=1.2, colors="k")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root",  type=str, required=False,
                    default="/home/darryld/documents/EEG/preprocessed/bipolaire/3_results_analysis/gp2/_by_patient_category",
                    help="Racine _by_patient_category")
    ap.add_argument("--out-root", type=str, required=False,
                    default="/home/darryld/documents/EEG/preprocessed/bipolaire/3_results_analysis/gp2/_by_patient_category/_cbpt_between_groups",
                    help="Dossier de sortie pour CBPT")
    ap.add_argument("--groups",   type=str, nargs=2, default=None,
                    help="Deux catégories à comparer (ex: syn narco)")
    ap.add_argument("--auto-pairs", action="store_true",
                    help="Si défini, compare toutes les paires de catégories présentes.")
    ap.add_argument("--stages",   type=str, nargs="*", default=["REM","N2","N3"])
    ap.add_argument("--channels", type=str, nargs="*", default=None,
                    help="Si fourni, restreint aux canaux listés (noms exacts).")
    ap.add_argument("--n-perm",   type=int, default=5000)
    ap.add_argument("--cdt-p",    type=float, default=0.05,
                    help="p non corrigé pour le seuil de formation de clusters (bilatéral).")
    ap.add_argument("--alpha",    type=float, default=0.05,
                    help="Seuil de significativité des clusters (FWER).")
    ap.add_argument("--seed",     type=int, default=13)
    ap.add_argument("--pool-channels", action="store_true",
                    help="Contrôler FWER global en prenant le max de masse sur tous les canaux à chaque permutation.")
    ap.add_argument("--diff-mode", type=str, default="rel",
                    choices=["abs", "rel"],
                    help="Mode d'affichage : 'abs' = delta dB (A - B), 'rel' = 100*(10^(ΔdB/10)-1) = delta % linéaire (A vs B).")

    args = ap.parse_args()

    IN = Path(args.in_root)
    OUT = Path(args.out_root)

    if not IN.exists():
        raise SystemExit(f"[IN] Introuvable: {IN}")

    rng = np.random.default_rng(args.seed)

    # Détection catégories
    cats = sorted([d.name for d in IN.iterdir() if d.is_dir() and not d.name.startswith("_")])
    if args.groups and args.auto_pairs:
        raise SystemExit("Utilise --groups A B ou --auto-pairs, pas les deux.")
    if args.groups:
        pairs = [tuple(args.groups)]
    elif args.auto_pairs:
        pairs = list(combinations(cats, 2))
        if not pairs:
            raise SystemExit("Aucune paire de catégories détectée.")
    else:
        raise SystemExit("Spécifie --groups A B ou --auto-pairs.")

    for (catA, catB) in pairs:
        for stage in args.stages:
            dirA = IN / sanitize(catA) / stage
            dirB = IN / sanitize(catB) / stage
            if not (dirA.exists() and dirB.exists()):
                print(f"[INFO] {catA} vs {catB} / {stage}: dossier manquant -> skip")
                continue

            # Liste des canaux communs (via stacks présents)
            chansA = sorted({re.sub(r"^stack_(.+)\.npy$", r"\1", p.name)
                             for p in dirA.glob("stack_*.npy")})
            chansB = sorted({re.sub(r"^stack_(.+)\.npy$", r"\1", p.name)
                             for p in dirB.glob("stack_*.npy")})
            common_chans = sorted(set(chansA) & set(chansB))
            if args.channels:
                # restreindre aux canaux demandés
                common_chans = [ch for ch in common_chans if ch in set(args.channels)]
            if not common_chans:
                print(f"[INFO] {catA} vs {catB} / {stage}: aucun canal commun (avec stacks) -> skip")
                continue

            # Si pooling inter-canaux: on pré-initialise conteneur pour max par permutation
            if args.pool_channels:
                pooled_max_perm = None  # sera le max inter-canaux, construit au fil des canaux

            for ch in common_chans:
                XA, timesA, freqsA = load_stack(dirA, ch)
                XB, timesB, freqsB = load_stack(dirB, ch)
                if XA is None or XB is None:
                    print(f"[WARN] stack manquante pour {stage}/{ch} -> skip")
                    continue
                if XA.shape[0] < 2 or XB.shape[0] < 2:
                    print(f"[WARN] {stage}/{ch}: nA={XA.shape[0]} nB={XB.shape[0]} insuffisant -> skip")
                    continue

                XA, XB, times, freqs = align_min(XA, XB, timesA, freqsA, timesB, freqsB)

                # t-map & delta (en dB)
                t_obs, df, delta_db, d = t_map_independent_equalvar(XA, XB)

                # Moyennes de groupe en dB (après align_min)
                A_mean_db = XA.mean(axis=0)  # (F,T)
                B_mean_db = XB.mean(axis=0)  # (F,T)

                # Différences à afficher
                if args.diff_mode == "rel":
                    delta_to_plot = db_delta_to_rel_percent(delta_db)
                    cbar_label = "Delta Power (% rel)  (A vs B)"
                else:
                    delta_to_plot = delta_db
                    cbar_label = "Delta Power (dB)  (A - B)"



                # Seuil CDT et clusters observés
                tcrit = stats.t.isf(args.cdt_p / 2.0, df)  # bilatéral
                clusters_obs, _ = form_clusters(t_obs, tcrit)

                # Permutations -> distribution nulle du max de masse
                if args.pool_channels:
                    max_perm_this_ch = permutation_max_mass(XA, XB, tcrit, args.n_perm, rng)
                    pooled_max_perm = max_perm_this_ch if pooled_max_perm is None else np.maximum(pooled_max_perm, max_perm_this_ch)
                else:
                    max_perm = permutation_max_mass(XA, XB, tcrit, args.n_perm, rng)

                # p-valeurs des clusters observés
                rows = []
                sig_masks = []
                if clusters_obs:
                    null_max = pooled_max_perm if args.pool_channels else max_perm
                    for (sign, mask, mass) in clusters_obs:
                        # p bilatéral via max absolu
                        p = (1.0 + np.sum(null_max >= abs(mass))) / (1.0 + null_max.size)
                        summ = summarize_cluster(mask, mass, delta_db=delta_db, d=d, times=times, freqs=freqs)
                        rows.append({
                            "stage": stage,
                            "channel": ch,
                            "sign": "+" if sign > 0 else "-",
                            "p_value": float(p),
                            **summ,
                        })
                        if p < args.alpha:
                            sig_masks.append(mask)

                # Sauvegardes
                out_dir = OUT / f"{sanitize(catA)}_vs_{sanitize(catB)}" / stage / sanitize(ch)

                np.save(out_dir / "group_A_mean_db.npy", A_mean_db.astype(np.float32))
                np.save(out_dir / "group_B_mean_db.npy", B_mean_db.astype(np.float32))
                np.save(out_dir / "times.npy",  np.asarray(times, dtype=np.float32))
                np.save(out_dir / "freqs.npy",  np.asarray(freqs, dtype=np.float32))
                np.save(out_dir / "group_diff_db.npy", (A_mean_db - B_mean_db).astype(np.float32))

                out_dir.mkdir(parents=True, exist_ok=True)

                # CSV clusters (schéma stable même si 0 cluster)
                df_out = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
                    "stage","channel","sign","p_value","mass","n_pixels",
                    "f_min_hz","f_max_hz","t_min_s","t_max_s",
                    "delta_db_mean","delta_rel_percent_mean","cohen_d_mean"
                ])
                df_out.to_csv(out_dir / f"cbpt_{sanitize(catA)}_vs_{sanitize(catB)}_{stage}_{sanitize(ch)}_clusters.csv",
                              index=False)

                # Masque significatif (union)
                if sig_masks:
                    sig_union = np.zeros_like(delta_db, dtype=bool)
                    for m in sig_masks:
                        sig_union |= m
                    np.save(out_dir / f"cbpt_{sanitize(catA)}_vs_{sanitize(catB)}_{stage}_{sanitize(ch)}_sigmask.npy",
                            sig_union.astype(np.uint8))
                else:
                    sig_union = None

                # Figure
                title = (f"{catA} vs {catB} - {stage} - {ch} \n"
                         f"CDT p={args.cdt_p:.3f} | Nperm={args.n_perm} | alpha={args.alpha:.3f}")
                out_png = out_dir / f"cbpt_{sanitize(catA)}_vs_{sanitize(catB)}_{stage}_{sanitize(ch)}.png"
                plot_with_clusters(
                    out_png,
                    delta=delta_to_plot,            
                    times=times,
                    freqs=freqs,
                    sig_masks=[sig_union] if sig_union is not None else [],
                    title=title,
                    cbar_label=cbar_label,
                )

                print(f"[OK] CBPT {catA} vs {catB} / {stage} / {ch} -> {out_png.name}")

            
    print("[DONE] CBPT terminé.")

if __name__ == "__main__":
    main()
