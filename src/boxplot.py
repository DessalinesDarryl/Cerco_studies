#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boxplots par groupe à partir de all_band_powers.csv

Entrée:
  - CSV: out_root/all_band_powers.csv (colonnes: base, group, total_abs, abs_*, rel_*)

Sorties (dans --out-root):
  - boxplot_total_abs.png
  - boxplot_abs_<Band>.png
  - boxplot_rel_<Band>.png
  - (optionnel) grilles récapitulatives:
      - grid_boxplots_abs.png
      - grid_boxplots_rel.png

Usage:
  python boxplots_by_group.py --csv /chemin/vers/all_band_powers.csv --out-root /chemin/sorties --kind both --logy
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Palette & utilitaires ----------
PALETTE = {
    "autoimmune encephalitides": "#C9D175",  # AI  (jaune-vert)
    "narcolepsy":                "#F15854",  # Narco (rouge)
    "synucleopathy":             "#44AA99",  # Syn (bleu-vert)
    "tcspi":                     "#BEBEBE",  # TCSP (gris)
    "unknown":                   "#000000",  # fallback
}

GROUP_ORDER = ["autoimmune encephalitides", "narcolepsy", "synucleopathy", "tcspi", "unknown"]
OUT_ROOT_DEFAULT = "/home/darryld/documents/EEG/preprocessed/bipolaire/3_results_analysis/gp1"

def color_for_group(label: str) -> str:
    return PALETTE.get(str(label).strip().lower(), PALETTE["unknown"])

def short_label(label: str) -> str:
    lab = str(label).strip().lower()
    return {
        "autoimmune encephalitides": "AI",
        "narcolepsy": "Narco",
        "synucleopathy": "Syn",
        "tcspi": "TCSP",
        "unknown": "Unknown",
    }.get(lab, label)

def sanitize_name(x: str) -> str:
    return "".join(c for c in str(x) if c.isalnum() or c in ("_", "-")).strip("_-").lower()

# ---------- Détection des bandes dans le CSV ----------
def detect_bands(df: pd.DataFrame, prefix: str) -> list[str]:
    # colonnes du type "abs_Delta", "rel_Theta", etc.
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    bands = [c.split("_", 1)[1] for c in cols]
    # conserver l'ordre standard si possible
    order = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    bands_sorted = [b for b in order if b in bands] + [b for b in bands if b not in order]
    return bands_sorted

# ---------- Boxplot helper ----------
def boxplot_one_band(df: pd.DataFrame, band: str, kind: str, out_path: Path, logy: bool = False):
    """
    Crée un boxplot par groupe pour une bande donnée.
    kind: "abs" ou "rel"
    """
    assert kind in ("abs", "rel")
    col = f"{kind}_{band}"
    if col not in df.columns:
        print(f"[WARN] Colonne absente: {col} -> skip")
        return

    # Groupes présents, ordonnés selon GROUP_ORDER
    groups_present = [g for g in GROUP_ORDER if g in df["group"].unique().tolist()]
    # fallback si rien ne matche
    if not groups_present:
        groups_present = sorted(df["group"].unique())

    data = []
    colors = []
    xticks = []
    for g in groups_present:
        vals = df.loc[df["group"] == g, col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
        if vals.size == 0:
            # éviter boxplots vides : mettre un NaN qui sera ignoré par la box
            vals = np.array([np.nan])
        data.append(vals)
        colors.append(color_for_group(g))
        n = int(np.sum(df["group"] == g))
        xticks.append(f"{short_label(g)} (n={n})")

    fig = plt.figure(figsize=(max(7, 1.2 * len(groups_present)), 5))
    bp = plt.boxplot(
        data,
        patch_artist=True,
        showfliers=True,
        widths=0.6,
        medianprops=dict(linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=1.2),
    )
    # couleurs par groupe
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)

    plt.xticks(np.arange(1, len(xticks) + 1), xticks, rotation=20, ha="right")
    if kind == "abs":
        plt.ylabel("Band power absolu (µV²)")
        if logy:
            plt.yscale("log")
        title = f"Boxplot par groupe — {band} (absolu)"
    else:
        plt.ylabel("Band power relatif")
        title = f"Boxplot par groupe — {band} (relatif)"

    plt.title(title)
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=230)
    plt.close(fig)
    print(f"→ {out_path}")

def boxplot_total_abs(df: pd.DataFrame, out_path: Path, logy: bool = False):
    if "total_abs" not in df.columns:
        print("[WARN] Colonne 'total_abs' absente -> skip total_abs")
        return

    groups_present = [g for g in GROUP_ORDER if g in df["group"].unique().tolist()]
    if not groups_present:
        groups_present = sorted(df["group"].unique())

    data, colors, xticks = [], [], []
    for g in groups_present:
        vals = df.loc[df["group"] == g, "total_abs"].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
        if vals.size == 0:
            vals = np.array([np.nan])
        data.append(vals)
        colors.append(color_for_group(g))
        n = int(np.sum(df["group"] == g))
        xticks.append(f"{short_label(g)} (n={n})")

    fig = plt.figure(figsize=(max(7, 1.2 * len(groups_present)), 5))
    bp = plt.boxplot(
        data,
        patch_artist=True,
        showfliers=True,
        widths=0.6,
        medianprops=dict(linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=1.2),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55)

    plt.xticks(np.arange(1, len(xticks) + 1), xticks, rotation=20, ha="right")
    plt.ylabel("Total (µV²)")
    if logy:
        plt.yscale("log")
    plt.title("Boxplot par groupe — Total absolu")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=230)
    plt.close(fig)
    print(f"→ {out_path}")

# ---------- Grilles récapitulatives (facultatives) ----------
def grid_boxplots(df: pd.DataFrame, bands: list[str], kind: str, out_path: Path, logy: bool = False):
    assert kind in ("abs", "rel")
    if not bands:
        return
    n = len(bands)
    nrows = (n + 2) // 3
    ncols = min(3, n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols + 1, 4.5 * nrows))
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = [axes]

    groups_present = [g for g in GROUP_ORDER if g in df["group"].unique().tolist()]
    if not groups_present:
        groups_present = sorted(df["group"].unique())

    for i, band in enumerate(bands):
        ax = axes[i]
        col = f"{kind}_{band}"
        if col not in df.columns:
            ax.set_visible(False); continue

        data, colors, xticks = [], [], []
        for g in groups_present:
            vals = df.loc[df["group"] == g, col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
            if vals.size == 0:
                vals = np.array([np.nan])
            data.append(vals)
            colors.append(color_for_group(g))
            n_g = int(np.sum(df["group"] == g))
            xticks.append(f"{short_label(g)}\n(n={n_g})")

        bp = ax.boxplot(
            data, patch_artist=True, showfliers=False, widths=0.6,
            medianprops=dict(linewidth=1.5),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            boxprops=dict(linewidth=1.0),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.55)
        ax.set_xticks(np.arange(1, len(xticks) + 1))
        ax.set_xticklabels(xticks, rotation=0)
        if kind == "abs":
            ax.set_ylabel("µV²")
            if logy:
                ax.set_yscale("log")
        else:
            ax.set_ylabel("Relatif")
        ax.set_title(band)
        ax.grid(axis="y", alpha=0.2)

    # masquer cases inutilisées
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    supt = "Récap boxplots — ABSOLU" if kind == "abs" else "Récap boxplots — RELATIF"
    fig.suptitle(supt, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=230)
    plt.close(fig)
    print(f"→ {out_path}")

# ---------- Main ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root",  type=str, default=OUT_ROOT_DEFAULT)
    p.add_argument("--kind", type=str, default="both", choices=["abs", "rel", "both"],
                   help="Type de boxplots à produire")
    p.add_argument("--logy", action="store_true", help="Échelle Y logarithmique pour l'absolu")
    return p.parse_args()

def main():
    args = parse_args()
    csv_path = Path("/home/darryld/documents/EEG/preprocessed/bipolaire/3_results_analysis/gp1/all_band_powers.csv")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise SystemExit(f"CSV introuvable: {csv_path}")

    df = pd.read_csv(csv_path)
    # normaliser noms de groupes (minuscule) et filtrer colonnes attendues
    if "group" not in df.columns:
        raise SystemExit("Colonne 'group' absente du CSV.")
    df["group"] = df["group"].astype(str).str.strip().str.lower()

    # Détecter bandes disponibles
    bands_abs = detect_bands(df, "abs")
    bands_rel = detect_bands(df, "rel")

    # total_abs
    boxplot_total_abs(df, out_root / "boxplot_total_abs.png", logy=args.logy)

    # boxplots par bande
    if args.kind in ("abs", "both"):
        for b in bands_abs:
            boxplot_one_band(df, b, kind="abs", out_path=out_root / f"boxplot_abs_{sanitize_name(b)}.png", logy=args.logy)
        if bands_abs:
            grid_boxplots(df, bands_abs, kind="abs", out_path=out_root / "grid_boxplots_abs.png", logy=args.logy)

    if args.kind in ("rel", "both"):
        for b in bands_rel:
            boxplot_one_band(df, b, kind="rel", out_path=out_root / f"boxplot_rel_{sanitize_name(b)}.png")
        if bands_rel:
            grid_boxplots(df, bands_rel, kind="rel", out_path=out_root / "grid_boxplots_rel.png")

    print("[OK] Terminé.")

if __name__ == "__main__":
    main()
