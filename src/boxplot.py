#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boxplots par groupe à partir de all_band_powers.csv
+ (optionnel) boxplots **par canal** à partir de per_channel_band_powers.csv
+ Détection d'outliers IQR (Tukey) globale et par canal

Entrées:
  - CSV global: all_band_powers.csv (colonnes: base, group, total_abs, abs_*, rel_*)
  - CSV par canal (optionnel): per_channel_band_powers.csv
      colonnes attendues: base, channel, band, abs, rel, total_abs_channel, group

Sorties (dans --out-root):
  - boxplot_total_abs.png
  - boxplot_abs_<Band>.png / boxplot_rel_<Band>.png
  - grid_boxplots_abs.png / grid_boxplots_rel.png (optionnel)
  - outliers_iqr.csv (global: band × group, abs/rel)
  - (si --per-channel-csv fourni)
      - boxplot_abs_chan_<CANAL>_<BAND>.png
      - boxplot_rel_chan_<CANAL>_<BAND>.png
      - (optionnel) grid_boxplots_abs_chan_<CANAL>.png
      - (optionnel) grid_boxplots_rel_chan_<CANAL>.png
      - outliers_per_channel.csv (channel × band × group, abs/rel)
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
    "autoimmune encephalitides": "#C9D175",  # AI
    "narcolepsy":                "#F15854",  # Narco
    "synucleopathy":             "#44AA99",  # Syn
    "tcspi":                     "#BEBEBE",  # TCSP
    "unknown":                   "#000000",
}

GROUP_ORDER = ["autoimmune encephalitides", "narcolepsy", "synucleopathy", "tcspi", "unknown"]
OUT_ROOT_DEFAULT = "/home/darryld/documents/EEG/preprocessed/bipolaire/3_results_analysis/gp2"

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
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    bands = [c.split("_", 1)[1] for c in cols]
    order = ["Delta", "Theta", "Alpha", "Beta", "Gamma", "Gamma_Bas", "Gamma_Haut"]
    bands_sorted = [b for b in order if b in bands] + [b for b in bands if b not in order]
    return bands_sorted

# ---------- Boxplot helper (global patient) ----------
def boxplot_one_band(df: pd.DataFrame, band: str, kind: str, out_path: Path, logy: bool = False):
    assert kind in ("abs", "rel")
    col = f"{kind}_{band}"
    if col not in df.columns:
        print(f"[WARN] Colonne absente: {col} -> skip")
        return

    groups_present = [g for g in GROUP_ORDER if g in df["group"].unique().tolist()] or sorted(df["group"].unique())

    data, colors, xticks = [], [], []
    for g in groups_present:
        vals = df.loc[df["group"] == g, col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
        if vals.size == 0:
            vals = np.array([np.nan])
        data.append(vals)
        colors.append(color_for_group(g))
        n = int(np.sum(df["group"] == g))
        xticks.append(f"{short_label(g)} (n={n})")

    fig = plt.figure(figsize=(max(7, 1.2 * len(groups_present)), 5))
    bp = plt.boxplot(
        data, patch_artist=True, showfliers=True, widths=0.6,
        medianprops=dict(linewidth=1.8), whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2), boxprops=dict(linewidth=1.2),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55)

    plt.xticks(np.arange(1, len(xticks) + 1), xticks, rotation=20, ha="right")
    if kind == "abs":
        plt.ylabel("Band power absolu (µV²)")
        if logy: plt.yscale("log")
        title = f"Boxplot par groupe - {band} (absolu)"
    else:
        plt.ylabel("Band power relatif")
        title = f"Boxplot par groupe - {band} (relatif)"
    plt.title(title); plt.grid(axis="y", alpha=0.2)
    plt.tight_layout(); fig.savefig(out_path, dpi=230); plt.close(fig)
    print(f">>> {out_path}")

def boxplot_total_abs(df: pd.DataFrame, out_path: Path, logy: bool = False):
    if "total_abs" not in df.columns:
        print("[WARN] Colonne 'total_abs' absente -> skip total_abs")
        return

    groups_present = [g for g in GROUP_ORDER if g in df["group"].unique().tolist()] or sorted(df["group"].unique())

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
        data, patch_artist=True, showfliers=True, widths=0.6,
        medianprops=dict(linewidth=1.8), whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2), boxprops=dict(linewidth=1.2),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55)

    plt.xticks(np.arange(1, len(xticks) + 1), xticks, rotation=20, ha="right")
    plt.ylabel("Total (µV²)")
    if logy: plt.yscale("log")
    plt.title("Boxplot par groupe - Total absolu")
    plt.grid(axis="y", alpha=0.2); plt.tight_layout()
    fig.savefig(out_path, dpi=230); plt.close(fig)
    print(f">>> {out_path}")

# ---------- Grilles (global patient) ----------
def grid_boxplots(df: pd.DataFrame, bands: list[str], kind: str, out_path: Path, logy: bool = False):
    assert kind in ("abs", "rel")
    if not bands: return
    n = len(bands); nrows = (n + 2) // 3; ncols = min(3, n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols + 1, 4.5 * nrows))
    axes = axes.ravel() if isinstance(axes, np.ndarray) else [axes]

    groups_present = [g for g in GROUP_ORDER if g in df["group"].unique().tolist()] or sorted(df["group"].unique())

    for i, band in enumerate(bands):
        ax = axes[i]
        col = f"{kind}_{band}"
        if col not in df.columns:
            ax.set_visible(False); continue

        data, colors, xticks = [], [], []
        for g in groups_present:
            vals = df.loc[df["group"] == g, col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
            if vals.size == 0: vals = np.array([np.nan])
            data.append(vals); colors.append(color_for_group(g))
            n_g = int(np.sum(df["group"] == g)); xticks.append(f"{short_label(g)}\n(n={n_g})")

        bp = ax.boxplot(
            data, patch_artist=True, showfliers=False, widths=0.6,
            medianprops=dict(linewidth=1.5), whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0), boxprops=dict(linewidth=1.0),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.55)
        ax.set_xticks(np.arange(1, len(xticks) + 1)); ax.set_xticklabels(xticks, rotation=0)
        if kind == "abs":
            ax.set_ylabel("µV²")
            if logy: ax.set_yscale("log")
        else:
            ax.set_ylabel("Relatif")
        ax.set_title(band); ax.grid(axis="y", alpha=0.2)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    supt = "Récap boxplots - ABSOLU" if kind == "abs" else "Récap boxplots - RELATIF"
    fig.suptitle(supt, fontsize=14); plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=230); plt.close(fig)
    print(f">>> {out_path}")

# ===================== PARTIE **PAR CANAL** =====================
def detect_eligible_channels(df_ch: pd.DataFrame, min_patients: int, restrict_to: list[str] | None = None) -> list[str]:
    """
    Compte le nombre de patients par 'channel' (nom complet) et renvoie les canaux
    avec au moins `min_patients` patients. Optionnellement restreint à une liste.
    """
    cnt = df_ch.groupby("channel")["base"].nunique()
    eligible = [ch for ch, n in cnt.items() if n >= min_patients]
    if restrict_to:
        ref = {c.strip() for c in restrict_to}
        eligible = [c for c in eligible if c in ref]
    return sorted(eligible, key=lambda s: s.upper())

def boxplot_one_band_per_channel(df_ch: pd.DataFrame, channel: str, band: str, kind: str,
                                 out_path: Path, logy: bool = False):
    """
    Boxplot par **groupe** pour une bande donnée, restreint à un canal donné.
    df_ch: colonnes = base, channel, band, abs, rel, total_abs_channel, group
    """
    assert kind in ("abs", "rel")
    value_col = "abs" if kind == "abs" else "rel"
    sub = df_ch[(df_ch["channel"] == channel) & (df_ch["band"] == band)].copy()
    if sub.empty:
        print(f"[WARN] Aucun data pour canal={channel}, bande={band} -> skip")
        return

    groups_present = [g for g in GROUP_ORDER if g in sub["group"].unique().tolist()] or sorted(sub["group"].unique())

    data, colors, xticks = [], [], []
    for g in groups_present:
        vals = sub.loc[sub["group"] == g, value_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
        if vals.size == 0: vals = np.array([np.nan])
        data.append(vals); colors.append(color_for_group(g))
        n = int(np.sum(sub["group"] == g)); xticks.append(f"{short_label(g)} (n={n})")

    fig = plt.figure(figsize=(max(7, 1.2 * len(groups_present)), 5))
    bp = plt.boxplot(
        data, patch_artist=True, showfliers=True, widths=0.6,
        medianprops=dict(linewidth=1.8), whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2), boxprops=dict(linewidth=1.2),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55)

    plt.xticks(np.arange(1, len(xticks) + 1), xticks, rotation=20, ha="right")
    if kind == "abs":
        plt.ylabel("PSD absolue (µV²)")
        if logy: plt.yscale("log")
        title = f"{band} - Canal {channel} (absolu)"
    else:
        plt.ylabel("PSD relative"); title = f"{band} - Canal {channel} (relatif)"
    plt.title(title); plt.grid(axis="y", alpha=0.2); plt.tight_layout()
    fig.savefig(out_path, dpi=230); plt.close(fig)
    print(f">>> {out_path}")

def grid_boxplots_per_channel(df_ch: pd.DataFrame, channel: str, bands: list[str], kind: str,
                              out_path: Path, logy: bool = False):
    """
    Grille de boxplots (toutes les bandes) pour un canal donné.
    """
    assert kind in ("abs", "rel")
    if not bands: return
    value_col = "abs" if kind == "abs" else "rel"

    sub_c = df_ch[df_ch["channel"] == channel].copy()
    if sub_c.empty:
        print(f"[WARN] Aucun data pour canal={channel} -> skip grille"); return

    n = len(bands); nrows = (n + 2) // 3; ncols = min(3, n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols + 1, 4.5 * nrows))
    axes = axes.ravel() if isinstance(axes, np.ndarray) else [axes]

    groups_present = [g for g in GROUP_ORDER if g in sub_c["group"].unique().tolist()] or sorted(sub_c["group"].unique())

    for i, band in enumerate(bands):
        ax = axes[i]
        sub = sub_c[sub_c["band"] == band]
        if sub.empty:
            ax.set_visible(False); continue

        data, colors, xticks = [], [], []
        for g in groups_present:
            vals = sub.loc[sub["group"] == g, value_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
            if vals.size == 0: vals = np.array([np.nan])
            data.append(vals); colors.append(color_for_group(g))
            n_g = int(np.sum(sub["group"] == g)); xticks.append(f"{short_label(g)}\n(n={n_g})")

        bp = ax.boxplot(
            data, patch_artist=True, showfliers=False, widths=0.6,
            medianprops=dict(linewidth=1.5), whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0), boxprops=dict(linewidth=1.0),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.55)
        ax.set_xticks(np.arange(1, len(xticks) + 1)); ax.set_xticklabels(xticks)
        if kind == "abs":
            ax.set_ylabel("µV²")
            if logy: ax.set_yscale("log")
        else:
            ax.set_ylabel("Relatif")
        ax.set_title(band); ax.grid(axis="y", alpha=0.2)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    supt = f"Canal {channel} - ABSOLU" if kind == "abs" else f"Canal {channel} - RELATIF"
    fig.suptitle(supt, fontsize=14); plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=230); plt.close(fig)
    print(f">>> {out_path}")

# ===================== OUTLIERS IQR (helpers) =====================
def _iqr_bounds(arr: np.ndarray, k: float = 1.5):
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    return q1, q3, iqr, q1 - k * iqr, q3 + k * iqr

def detect_outliers_iqr_long(df_long: pd.DataFrame,
                             group_cols: list[str],
                             value_col: str = "value",
                             metric_col: str = "metric",
                             k: float = 1.5,
                             extreme_k: float = 3.0) -> pd.DataFrame:
    """
    IQR par groupes (ex: ['band','group'] ou ['channel','band','group']).
    df_long doit contenir: base, value_col, metric_col, group_cols...
    """
    required = {"base", value_col, metric_col, *group_cols}
    missing = required - set(df_long.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour outliers IQR: {missing}")

    out = []
    for key, sub in df_long.groupby(group_cols + [metric_col], dropna=False):
        vals = sub[value_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
        if len(vals) < 4:
            continue
        q1, q3, iqr, low, up = _iqr_bounds(vals, k=k)
        _, _, _, low_e, up_e = _iqr_bounds(vals, k=extreme_k)
        mask_out = (sub[value_col] < low) | (sub[value_col] > up)
        mask_ext = (sub[value_col] < low_e) | (sub[value_col] > up_e)
        for idx in sub.index[mask_out]:
            row = {
                "base": sub.at[idx, "base"], value_col: sub.at[idx, value_col],
                "q1": q1, "q3": q3, "iqr": iqr, "lower_bound": low, "upper_bound": up,
                "is_outlier": True, "is_extreme": bool(mask_ext.loc[idx]),
                metric_col: sub.at[idx, metric_col],
            }
            for gc in group_cols:
                row[gc] = sub.at[idx, gc]
            out.append(row)
    return pd.DataFrame(out)

# (ancienne version simple, conservée si besoin direct)
def detect_outliers_iqr(df: pd.DataFrame, value_col: str,
                        group_col: str = "group", band_col: str = "band",
                        k: float = 1.5, extreme_k: float = 3.0) -> pd.DataFrame:
    out_rows = []
    for (band, group), sub in df.groupby([band_col, group_col]):
        vals = sub[value_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) < 4:
            continue
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        low, up = q1 - k * iqr, q3 + k * iqr
        low_ext, up_ext = q1 - extreme_k * iqr, q3 + extreme_k * iqr
        mask_out = (sub[value_col] < low) | (sub[value_col] > up)
        mask_ext = (sub[value_col] < low_ext) | (sub[value_col] > up_ext)
        for idx in sub.index[mask_out]:
            out_rows.append({
                "base": sub.at[idx, "base"],
                "group": group,
                "band": band,
                "metric": value_col,
                "value": sub.at[idx, value_col],
                "q1": q1, "q3": q3, "iqr": iqr,
                "lower_bound": low, "upper_bound": up,
                "is_outlier": True,
                "is_extreme": bool(mask_ext.loc[idx])
            })
    return pd.DataFrame(out_rows)

# ===================== MAIN =====================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=False,
                   default="/home/darryld/documents/EEG/preprocessed/bipolaire/3_results_analysis/gp2/all_band_powers.csv")
    p.add_argument("--out-root",  type=str, default=OUT_ROOT_DEFAULT)
    p.add_argument("--kind", type=str, default="both", choices=["abs", "rel", "both"],
                   help="Type de boxplots globaux à produire")
    p.add_argument("--logy", action="store_true", help="Échelle Y logarithmique pour l'absolu")
    # --- Options PAR CANAL ---
    p.add_argument("--per-channel-csv", type=str, required=False,
                   default="/home/darryld/documents/EEG/preprocessed/bipolaire/3_results_analysis/gp2/per_channel_band_powers.csv")
    p.add_argument("--min-patients-per-channel", type=int, default=5,
                   help="Min de patients ayant un canal pour l’inclure")
    p.add_argument("--channels", type=str, nargs="*", default=None,
                   help="Limite l’analyse à ces canaux (noms complets, sensibles à la casse)")
    p.add_argument("--per-channel-grid", action="store_true",
                   help="Générer aussi une grille récap par canal (toutes bandes)")
    return p.parse_args()

def main():
    args = parse_args()
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    # ---------- GLOBAL (par patient)
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV introuvable: {csv_path}")

    df = pd.read_csv(csv_path)
    if "group" not in df.columns:
        raise SystemExit("Colonne 'group' absente du CSV global.")
    df["group"] = df["group"].astype(str).str.strip().str.lower()

    bands_abs = detect_bands(df, "abs")
    bands_rel = detect_bands(df, "rel")

    # Boxplots (inchangé)
    boxplot_total_abs(df, out_root / "boxplot_total_abs.png", logy=args.logy)
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

    # ---------- OUTLIERS IQR (GLOBAL) ----------
    abs_cols = [c for c in df.columns if c.startswith("abs_")]
    rel_cols = [c for c in df.columns if c.startswith("rel_")]

    outliers_parts = []

    if abs_cols:
        long_abs = df.melt(id_vars=["base", "group"], value_vars=abs_cols,
                           var_name="metric", value_name="value")
        long_abs["band"] = long_abs["metric"].str.replace("^abs_", "", regex=True)
        long_abs["metric"] = "abs"
        out_abs = detect_outliers_iqr_long(
            long_abs, group_cols=["band", "group"],
            value_col="value", metric_col="metric", k=1.5, extreme_k=3.0
        )
        if not out_abs.empty:
            outliers_parts.append(out_abs)

    if rel_cols:
        long_rel = df.melt(id_vars=["base", "group"], value_vars=rel_cols,
                           var_name="metric", value_name="value")
        long_rel["band"] = long_rel["metric"].str.replace("^rel_", "", regex=True)
        long_rel["metric"] = "rel"
        out_rel = detect_outliers_iqr_long(
            long_rel, group_cols=["band", "group"],
            value_col="value", metric_col="metric", k=1.5, extreme_k=3.0
        )
        if not out_rel.empty:
            outliers_parts.append(out_rel)

    if outliers_parts:
        outliers_df = pd.concat(outliers_parts, ignore_index=True)
        out_path_global = out_root / "outliers_iqr.csv"
        outliers_df.to_csv(out_path_global, index=False)
        print(f">>> Outliers globaux sauvegardés: {out_path_global}")
    else:
        print("[INFO] Aucun outlier global détecté (ou pas de colonnes abs_/rel_).")

    # ---------- PAR CANAL ----------
    if args.per_channel_csv:
        ch_path = Path(args.per_channel_csv)  # <-- utilise l'argument CLI
        if not ch_path.exists():
            print(f"[WARN] per-channel CSV introuvable: {ch_path} -> skip partie par canal")
        else:
            df_ch = pd.read_csv(ch_path)
            if "group" not in df_ch.columns:
                print("[WARN] Colonne 'group' absente de per_channel_band_powers.csv -> skip partie par canal")
            else:
                df_ch["group"] = df_ch["group"].astype(str).str.strip().str.lower()

                # bandes disponibles (depuis la colonne 'band')
                bands_ch = [b for b in df_ch["band"].astype(str).unique().tolist() if b and b == b]
                bands_ch_sorted = [b for b in ["Delta","Theta","Alpha","Beta","Gamma_Bas","Gamma_Haut","Gamma"] if b in bands_ch] + \
                                  [b for b in bands_ch if b not in ["Delta","Theta","Alpha","Beta","Gamma_Bas","Gamma_Haut","Gamma"]]

                eligible = detect_eligible_channels(
                    df_ch, min_patients=args.min_patients_per_channel, restrict_to=args.channels
                )
                if not eligible:
                    print("[WARN] Aucun canal éligible (vérifie --min-patients-per-channel / --channels).")
                else:
                    print(f"[INFO] Canaux éligibles ({len(eligible)}): {eligible}")

                # Boxplots par canal (inchangé)
                for ch in eligible:
                    for b in bands_ch_sorted:
                        out_abs = out_root / f"boxplot_abs_chan_{sanitize_name(ch)}_{sanitize_name(b)}.png"
                        boxplot_one_band_per_channel(df_ch, ch, b, kind="abs", out_path=out_abs, logy=args.logy)
                        out_rel = out_root / f"boxplot_rel_chan_{sanitize_name(ch)}_{sanitize_name(b)}.png"
                        boxplot_one_band_per_channel(df_ch, ch, b, kind="rel", out_path=out_rel, logy=False)

                    if args.per_channel_grid and bands_ch_sorted:
                        out_g_abs = out_root / f"grid_boxplots_abs_chan_{sanitize_name(ch)}.png"
                        out_g_rel = out_root / f"grid_boxplots_rel_chan_{sanitize_name(ch)}.png"
                        grid_boxplots_per_channel(df_ch, ch, bands_ch_sorted, kind="abs", out_path=out_g_abs, logy=args.logy)
                        grid_boxplots_per_channel(df_ch, ch, bands_ch_sorted, kind="rel", out_path=out_g_rel, logy=False)

                # ---------- OUTLIERS IQR (PAR CANAL) ----------
                needed = {"base", "group", "channel", "band", "abs", "rel"}
                if not needed.issubset(df_ch.columns):
                    print(f"[WARN] Colonnes manquantes pour outliers par canal: {needed - set(df_ch.columns)}")
                else:
                    ch_abs = df_ch.loc[:, ["base", "group", "channel", "band", "abs"]].rename(columns={"abs": "value"}).copy()
                    ch_abs["metric"] = "abs"
                    ch_rel = df_ch.loc[:, ["base", "group", "channel", "band", "rel"]].rename(columns={"rel": "value"}).copy()
                    ch_rel["metric"] = "rel"
                    long_ch = pd.concat([ch_abs, ch_rel], ignore_index=True)

                    # (Optionnel) Restreindre aux canaux éligibles :
                    # long_ch = long_ch[long_ch["channel"].isin(eligible)]

                    outliers_ch = detect_outliers_iqr_long(
                        long_ch, group_cols=["channel", "band", "group"],
                        value_col="value", metric_col="metric", k=1.5, extreme_k=3.0
                    )
                    if not outliers_ch.empty:
                        out_path_ch = out_root / "outliers_per_channel.csv"
                        outliers_ch.to_csv(out_path_ch, index=False)
                        print(f">>> Outliers par canal sauvegardés: {out_path_ch}")
                    else:
                        print("[INFO] Aucun outlier (par canal) détecté.")

    print("[OK] Terminé.")

if __name__ == "__main__":
    main()
