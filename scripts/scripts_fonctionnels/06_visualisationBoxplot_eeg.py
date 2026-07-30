from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
A_eeg_spectral_optimized.py - Boxplots + stats EEG par bande/canal/groupe
=============================================================================

Version .py optimisée du notebook A_eeg_spectral.ipynb.

Optimisations appliquées par rapport au notebook :
  1. DÉDOUBLONNAGE DES STATS
     Le notebook original calculait Shapiro/Kruskal/Mann-Whitney+Holm
     jusqu'à 3 fois pour la même combinaison (band, channel, value) :
     une fois pour la figure "relative" (cellule 7), une fois pour la
     figure "absolue" (cellule 8), une fois pour l'export CSV (cellule 9).
     Ici, chaque combinaison n'est calculée qu'UNE SEULE FOIS ; le résultat
     est réutilisé à la fois pour la figure et pour l'export.

  2. PARALLÉLISATION
     Le traitement est distribué par (band, channel) sur plusieurs coeurs
     via ProcessPoolExecutor, comme dans 01/03/04_*.py du pipeline.

  3. FIGURES EN PDF (VECTORIEL) PAR DÉFAUT
     Format PDF conservé comme dans le notebook, pour la qualité publication.
     SVG est aussi supporté (changez simplement FIG_FORMAT="svg"). Le rendu
     vectoriel avec de nombreux points de stripplot reste plus lent qu'un
     PNG raster - c'est le compromis qualité/vitesse assumé ici. Repassez
     FIG_FORMAT="png" si le temps d'exécution redevient un problème.

Usage :
  python A_eeg_spectral_optimized.py

Dépendances : pandas, numpy, scipy, statsmodels, seaborn, matplotlib, openpyxl
"""

# ============================================================================
# PARAMÈTRES  <<<  À MODIFIER SELON LA CONFIGURATION SOUHAITÉE
# ============================================================================

EEG_FEATURES_CSV = r"c:\dev\Cerco_studies\data\eeg_features.csv"
LABELS_TXT       = r"c:\dev\Cerco_studies\data\patients_label.txt"
OUTPUT_FIGS      = r"c:\dev\Cerco_studies\data\visualisation2\eeg\boxplots"
OUTPUT_STATS     = r"c:\dev\Cerco_studies\data\visualisation2\eeg\stats"

SAVE_FIGS  = True
FIG_FORMAT = "pdf"   # "pdf" ou "svg" (vectoriel, qualité publication) 
DPI        = 300     
ALPHA      = 0.05

N_WORKERS = 4  

PALETTE = {
    "EAI":   "#C9D175",
    "Narco": "#F15854",
    "SYN":   "#44AA99",
    "TCSPi": "#BEBEBE",
}
# Ordre d'affichage voulu : EAI > Narco > SYN > TCSPi
GROUP_ORDER = ["EAI", "Narco", "SYN", "TCSPi"]

# Libellés affichés (axes, légendes) — les codes internes ci-dessus restent
# utilisés pour la fusion avec patients_label.txt et le filtrage des données.
DISPLAY_LABELS = {
    "SYN":   "Syn",
    "Narco": "Narco",
    "TCSPi": "iRBD",
    "EAI":   "AI",
}

BANDS = ["delta", "theta", "alpha", "beta", "gamma_bas", "gamma_haut"]
BAND_FREQS = {
    "delta":      "0.5–4 Hz",
    "theta":      "4–8 Hz",
    "alpha":      "8–12 Hz",
    "beta":       "12–30 Hz",
    "gamma_bas":  "30–50 Hz",
    "gamma_haut": "50–80 Hz",
}
BIPOLAR_CHANNELS = [
    "Fp1-T3", "Fp1-C3", "T3-O1",
    "Fp2-T4", "Fp2-C4", "T4-O2",
    "Fp1-A1", "Fp2-A1", "T3-A1",
    "C3-A1",  "T4-A1",  "C4-A1",
]

VALUE_COLS = ["rel", "abs"]
YLABELS = {"rel": "Relative power (a.u.)", "abs": "Absolute power (V\u00b2/Hz)"}
TITLES  = {"rel": "Relative spectral power", "abs": "Absolute spectral power"}

# ============================================================================
# IMPORTS
# ============================================================================

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


def _get_logger(name: str = "eeg_spectral") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


# ============================================================================
# CHARGEMENT ET FUSION (identique à la logique du notebook)
# ============================================================================

def _load_and_merge() -> Tuple[pd.DataFrame, pd.DataFrame]:
    feat = pd.read_csv(EEG_FEATURES_CSV)
    feat["patient_id"] = feat["patient_id"].astype(str).str.strip()

    try:
        lbl = pd.read_csv(LABELS_TXT)
        cols = {c.lower(): c for c in lbl.columns}
        id_col  = cols.get("patient_id") or cols.get("identifiant")
        lbl_col = cols.get("label_str") or cols.get("diagnostic") or cols.get("label")
        lbl = lbl[[id_col, lbl_col]].rename(columns={id_col: "patient_id", lbl_col: "group"})
    except Exception:
        lbl = pd.read_csv(LABELS_TXT, header=None, names=["patient_id", "group"])

    lbl["patient_id"] = lbl["patient_id"].astype(str).str.strip()
    lbl["group"]      = lbl["group"].astype(str).str.strip()

    def to_macro(s: str) -> str:
        u = s.upper()
        if any(k in u for k in ("PARK", "MPI", "AMS", "DCL", "DLB", "PAF")):
            return "SYN"
        if "NARCO" in u:
            return "Narco"
        if "TCSP" in u or "RBDI" in u:
            return "TCSPi"
        if "EAI" in u or "ENCEPHALITE" in u:
            return "EAI"
        return s

    lbl["group"] = lbl["group"].map(to_macro)

    df = feat.merge(lbl, on="patient_id", how="left")
    df_before_filter = df.copy()
    df = df[df["group"].isin(GROUP_ORDER)].copy()

    excluded = df_before_filter[~df_before_filter["patient_id"].isin(df["patient_id"])]
    return df, excluded


def _reshape_long(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruit un tableau long : 1 ligne = patient x canal x bande."""
    records = []
    for _, row in df.iterrows():
        pid, group = row["patient_id"], row["group"]
        for ch in BIPOLAR_CHANNELS:
            for band in BANDS:
                records.append({
                    "patient_id": pid, "group": group, "channel": ch, "band": band,
                    "abs": row.get(f"eeg_{ch}_{band}_bp_abs_mean", np.nan),
                    "rel": row.get(f"eeg_{ch}_{band}_bp_rel_mean", np.nan),
                })
    return pd.DataFrame(records).dropna(subset=["group"])


# ============================================================================
# HELPERS STATISTIQUES (calculés UNE SEULE FOIS par combinaison)
# ============================================================================

def iqr_bounds(a: np.ndarray):
    q1, q3 = np.percentile(a, [25, 75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def p_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def run_stats_block(data: pd.DataFrame, value_col: str, group_col: str = "group"):
    """Retourne (normality_ok, kw_p, pairwise_df)."""
    groups = {g: sub[value_col].dropna().values
              for g, sub in data.groupby(group_col) if len(sub) >= 3}
    if len(groups) < 2:
        return False, np.nan, pd.DataFrame()

    normal = all(
        st.shapiro(v)[1] >= ALPHA
        for v in groups.values() if 3 <= len(v) <= 5000
    )
    kw_stat, kw_p = st.kruskal(*groups.values())

    pw_rows = []
    for (g1, x), (g2, y) in combinations(groups.items(), 2):
        u, p_raw = st.mannwhitneyu(x, y, alternative="two-sided")
        pw_rows.append({"group1": g1, "group2": g2,
                         "n1": len(x), "n2": len(y),
                         "U": u, "p_raw": p_raw,
                         "r_rb": 1 - 2 * u / (len(x) * len(y))})

    pw = pd.DataFrame(pw_rows)
    if not pw.empty:
        _, p_holm, _, _ = multipletests(pw["p_raw"], method="holm")
        pw["p_holm"] = p_holm
        pw["stars"]  = pw["p_holm"].map(p_stars)

    return normal, kw_p, pw


def filter_iqr(sub: pd.DataFrame, value_col: str) -> pd.DataFrame:
    filtered = []
    for g, grp in sub.groupby("group"):
        lo, hi = iqr_bounds(grp[value_col].dropna().values)
        filtered.append(grp[(grp[value_col] >= lo) & (grp[value_col] <= hi)])
    return pd.concat(filtered, ignore_index=True) if filtered else sub.iloc[0:0]


# ============================================================================
# WORKER - traite UNE combinaison (band, channel) : stats "rel" + "abs" + figures
# ============================================================================

def _process_band_channel(args):
    band, channel, sub_all = args

    # Backend non-interactif obligatoire dans les processus workers
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    kw_rows, pw_rows, means_rows = [], [], []

    for value_col in VALUE_COLS:
        sub = sub_all.dropna(subset=[value_col])
        if sub["group"].nunique() < 2:
            continue

        sub_f = filter_iqr(sub, value_col)

        # --- calcul UNIQUE, réutilisé pour la figure ET l'export CSV ---
        normal, kw_p, pw = run_stats_block(sub_f, value_col)

        kw_rows.append({"band": band, "channel": channel, "value": value_col, "p_kw": kw_p})
        for g, grp in sub.groupby("group"):
            vals = grp[value_col].dropna()
            means_rows.append({"band": band, "channel": channel, "value": value_col,
                                "group": g, "mean": vals.mean(), "sd": vals.std(), "n": len(vals)})
        if not pw.empty:
            for _, r in pw.iterrows():
                pw_rows.append({"band": band, "channel": channel, "value": value_col, **r.to_dict()})

        if SAVE_FIGS:
            _make_boxplot(sub_f, value_col, channel, band, kw_p, pw, plt, sns)

    return kw_rows, pw_rows, means_rows


def _make_boxplot(sub_f, value_col, channel, band, kw_p, pw, plt, sns):
    fig, ax = plt.subplots(figsize=(7, 5))
    order = [g for g in GROUP_ORDER if g in sub_f["group"].unique()]
    display_order = [DISPLAY_LABELS.get(g, g) for g in order]
    palette_display = {DISPLAY_LABELS.get(g, g): PALETTE[g] for g in order}

    sub_f = sub_f.copy()
    sub_f["group_display"] = sub_f["group"].map(DISPLAY_LABELS).fillna(sub_f["group"])

    sns.boxplot(data=sub_f, x="group_display", y=value_col, order=display_order,
                palette=palette_display, showfliers=False, ax=ax, legend=False)
    sns.stripplot(data=sub_f, x="group_display", y=value_col, order=display_order,
                  color="black", alpha=0.55, size=4, jitter=True, ax=ax)

    if not pw.empty:
        sig = pw[pw["stars"] != ""]
        if not sig.empty:
            ymax = sub_f[value_col].max()
            yrange = sub_f[value_col].max() - sub_f[value_col].min()
            step = yrange * 0.12 if yrange > 0 else 1.0
            for k, (_, row) in enumerate(sig.iterrows()):
                # Les positions x restent indexées sur "order" (codes internes) :
                # la position i correspond à display_order[i], donc l'index est identique.
                x1, x2 = order.index(row["group1"]), order.index(row["group2"])
                y_ = ymax + step * (k + 1)
                ax.plot([x1, x1, x2, x2], [y_ - step * 0.2, y_, y_, y_ - step * 0.2],
                        lw=1.2, color="black")
                ax.text((x1 + x2) / 2, y_, row["stars"], ha="center", va="bottom",
                        fontsize=13, fontweight="bold")

    kw_label = f"Kruskal\u2013Wallis p={kw_p:.3f}" if not np.isnan(kw_p) else ""
    ax.set_title(f"{TITLES[value_col]}\n{channel} | {BAND_FREQS.get(band, band)}\n{kw_label}",
                 fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel(YLABELS[value_col])
    plt.tight_layout()

    save_path = Path(OUTPUT_FIGS) / f"bp_{value_col}_{band}_{channel.replace('-', '')}.{FIG_FORMAT}"
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# HEATMAPS DE P-VALUES (rapide, exécuté une seule fois en fin, séquentiel)
# ============================================================================

def _make_heatmaps(pw_df: pd.DataFrame, log) -> None:
    if pw_df.empty:
        log.info("Pas de comparaisons -> heatmaps ignorées.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    for vcol in VALUE_COLS:
        sub = pw_df[pw_df["value"] == vcol].copy()
        if sub.empty:
            continue
        sub["pair"] = sub["group1"].map(DISPLAY_LABELS).fillna(sub["group1"]) + " vs " + \
                      sub["group2"].map(DISPLAY_LABELS).fillna(sub["group2"])
        pivot = sub.pivot_table(index="pair", columns=["channel", "band"],
                                 values="p_holm", aggfunc="min")

        fig, ax = plt.subplots(figsize=(max(14, len(pivot.columns) * 0.6), max(4, len(pivot) * 0.8)))
        sns.heatmap(-np.log10(pivot.fillna(1).clip(lower=1e-10)),
                    cmap="YlOrRd", ax=ax, linewidths=0.3,
                    cbar_kws={"label": "\u2212log\u2081\u2080(p_holm)"})
        ax.axhline(y=-np.log10(ALPHA), color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Significativit\u00e9 Mann\u2013Whitney Holm \u2014 puissance {vcol}", fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("Paires de groupes")
        plt.tight_layout()
        fig.savefig(Path(OUTPUT_FIGS) / f"heatmap_pvalues_{vcol}.{FIG_FORMAT}",
                    dpi=DPI, bbox_inches="tight")
        plt.close(fig)


# ============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================================

def main() -> None:
    log = _get_logger()
    Path(OUTPUT_FIGS).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_STATS).mkdir(parents=True, exist_ok=True)

    log.info("Chargement et fusion des features + labels...")
    df, excluded = _load_and_merge()
    if not excluded.empty:
        log.warning(f"{len(excluded)} patients exclus (groupe non reconnu) :")
        for _, r in excluded.iterrows():
            log.warning(f"  - {r['patient_id']} : {r['group']}")
    log.info(f"Patients retenus : {df['patient_id'].nunique()} | "
             f"Groupes : {df['group'].value_counts().to_dict()}")

    log.info("Reshape en tableau long...")
    long_df = _reshape_long(df)
    log.info(f"Tableau long : {long_df.shape}")

    # Découpage en petites tranches (band, channel) - chaque tranche est légère
    # (n_patients lignes), donc le passage par argument aux workers est peu coûteux.
    combos: List[Tuple[str, str, pd.DataFrame]] = []
    for band in BANDS:
        for ch in BIPOLAR_CHANNELS:
            sub_all = long_df[(long_df["channel"] == ch) & (long_df["band"] == band)].copy()
            combos.append((band, ch, sub_all))

    log.info(f"{len(combos)} combinaisons band\u00d7channel \u00e0 traiter avec {N_WORKERS} worker(s).")

    kw_rows, pw_rows, means_rows = [], [], []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_process_band_channel, c): c for c in combos}
        for fut in as_completed(futures):
            band, ch, _ = futures[fut]
            try:
                kw_r, pw_r, means_r = fut.result()
                kw_rows.extend(kw_r)
                pw_rows.extend(pw_r)
                means_rows.extend(means_r)
            except Exception as e:
                log.error(f"[ERREUR] {band}/{ch} : {e}")

    means_df = pd.DataFrame(means_rows)
    kw_df    = pd.DataFrame(kw_rows)
    pw_df    = pd.DataFrame(pw_rows)

    means_df.to_csv(Path(OUTPUT_STATS) / "eeg_means_per_band_channel_group.csv", index=False)
    kw_df.to_csv(Path(OUTPUT_STATS) / "eeg_kruskal_results.csv", index=False)
    pw_df.to_csv(Path(OUTPUT_STATS) / "eeg_mannwhitney_posthoc.csv", index=False)

    sig_df = pw_df[pw_df["stars"] != ""].copy() if not pw_df.empty else pd.DataFrame()
    if not sig_df.empty:
        sig_df.to_excel(Path(OUTPUT_STATS) / "eeg_significant_comparisons.xlsx", index=False)
        log.info(f"{len(sig_df)} comparaisons significatives export\u00e9es.")
    else:
        log.info("Aucune comparaison significative.")

    log.info("G\u00e9n\u00e9ration des heatmaps de p-values...")
    _make_heatmaps(pw_df, log)

    log.info(f"Stats \u00e9crites dans {OUTPUT_STATS}")
    log.info(f"Figures \u00e9crites dans {OUTPUT_FIGS}")
    log.info("Termin\u00e9.")


if __name__ == "__main__":
    main()