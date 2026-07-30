from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
C_emg_boxplots.py - Boxplots EMG/RSWA par canal, métrique et groupe clinique
=============================================================================

Ce script est AUTONOME. Il reprend la même méthodologie que
A_eeg_spectral_optimized.py (stats calculées une seule fois par combinaison,
parallélisation par ProcessPoolExecutor, figures vectorielles PDF/SVG) mais
appliquée aux métriques EMG/RSWA issues de 03_segment_rswa.py, avec un
niveau de détail PAR CANAL EMG (Menton, JAMBD, JAMBG, EMG1, EMG2) - pas
d'agrégation tous canaux confondus.

Ce qu'il fait :
  1. Charge le CSV produit par 03_segment_rswa.py, ne garde que les lignes
     type == "REM_EPOCH_4S"
  2. Agrège chaque métrique au niveau patient x canal (moyenne sur les époques
     REM 4s de ce patient, pour ce canal)
  3. Fusionne avec les labels cliniques (même mapping macro-classes que
     A_eeg_spectral)
  4. Pour chaque combinaison (canal, métrique) : boxplot + stats
     (Shapiro -> Kruskal-Wallis -> Mann-Whitney post-hoc + Holm), calculées
     UNE SEULE FOIS et réutilisées pour la figure ET l'export CSV
  5. Exporte les tables de stats (moyennes, Kruskal-Wallis, post-hoc) +
     un Excel récapitulatif des comparaisons significatives

Métriques visualisées (une figure par canal x métrique, soit 5x6 = 30 figures) :
  - rswa_fraction         : % d'époques REM classées RSWA (tonic_ratio>1.3 OU tonic_eog)
  - phasic_ratio          : ratio moyen de temps occupé par l'activité phasique
  - tonic_ratio           : ratio médiane REM / médiane NREM de l'enveloppe EMG
  - tonic_eog_fraction    : % d'époques avec tonicité détectée via EOG
  - tonic_events_per_min  : débit d'époques toniques (événements) / minute de REM
  - phasic_events_per_min : débit de bursts phasiques détectés / minute de REM

Comment utiliser ce script :
  1. Modifier les chemins dans la section "PARAMÈTRES" ci-dessous
  2. Lancer :  python C_emg_boxplots.py

Dépendances requises :
  pip install pandas numpy scipy statsmodels seaborn matplotlib openpyxl
=============================================================================
"""

# ============================================================================
# PARAMÈTRES  <<<  À MODIFIER SELON VOTRE CONFIGURATION
# ============================================================================
 
RBD_EMG_CSV  = r"c:\dev\Cerco_studies\data\rbd_emg_events_and_summary_4s_per_channel.csv"
LABELS_TXT   = r"c:\dev\Cerco_studies\data\patients_label.txt"
OUTPUT_FIGS  = r"c:\dev\Cerco_studies\data\visualisation2\emg\boxplots"
OUTPUT_STATS = r"c:\dev\Cerco_studies\data\visualisation2\emg\stats"
 
SAVE_FIGS  = True
FIG_FORMAT = "pdf"   # "pdf" ou "svg" (vectoriel) ; "png" pour un rendu plus rapide
DPI        = 300
ALPHA      = 0.05
 
N_WORKERS = 4   # cohérent avec le reste du pipeline
 
MIN_N_FOR_PLOT = 2   # un groupe avec moins de patients que ce seuil sur une combinaison
                      # (canal, métrique) donnée est exclu de la figure et des stats
                      # (ex: EAI avec un seul point sur un canal donné)
 
PALETTE = {
    "EAI":   "#C9D175",
    "Narco": "#F15854",
    "SYN":   "#44AA99",
    "TCSPi": "#BEBEBE",
}
# Ordre d'affichage voulu : EAI > Narco > SYN > TCSPi
GROUP_ORDER = ["EAI", "Narco", "SYN", "TCSPi"]
 
# Libellés affichés (axes) - les codes internes ci-dessus restent utilisés
# pour la fusion avec patients_label.txt et le filtrage des données.
DISPLAY_LABELS = {
    "SYN":   "Syn",
    "Narco": "Narco",
    "TCSPi": "iRBD",
    "EAI":   "AI",
}
 
EMG_CHANNELS = ["Menton", "JAMBD", "JAMBG", "EMG1", "EMG2"]
 
# Métriques calculées par 03_segment_rswa.py, agrégées ici au niveau patient x canal.
# "is_fraction=True" -> converti en % (x100) pour la lisibilité de l'axe Y.
METRICS = {
    "rswa_fraction":      {"source_col": "rswa",         "agg": "mean", "is_fraction": True,
                            "label": "Époques RSWA (%)"},
    "phasic_ratio":       {"source_col": "phasic_ratio",  "agg": "mean", "is_fraction": False,
                            "label": "Ratio phasique moyen (a.u.)"},
    "tonic_ratio":        {"source_col": "tonic_ratio",   "agg": "mean", "is_fraction": False,
                            "label": "Ratio tonique (médiane REM / médiane NREM)"},
    "tonic_eog_fraction": {"source_col": "tonic_eog",     "agg": "mean", "is_fraction": True,
                            "label": "Époques toniques EOG (%)"},
    "tonic_events_per_min":  {"source_col": "rswa",       "agg": "rate", "is_fraction": False,
                               "label": "Débit d'époques toniques (événements/min)"},
    "phasic_events_per_min": {"source_col": "type=PHASIC", "agg": "rate", "is_fraction": False,
                               "label": "Débit d'événements phasiques (bursts/min)"},
}
 
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
 
 
def _get_logger(name: str = "emg_boxplots") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger
 
 
# ============================================================================
# CHARGEMENT DES LABELS (identique aux scripts EEG précédents)
# ============================================================================
 
def _load_labels() -> pd.DataFrame:
    # patients_label.txt n'a PAS de ligne d'en-tête : 2 colonnes, patient_id puis group.
    # (confirmé : la première ligne réelle est une donnée, ex. "AE129,NARCO", pas un header)
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
    return lbl
 
 
# ============================================================================
# CHARGEMENT + AGRÉGATION PATIENT x CANAL (depuis le CSV de 03_segment_rswa.py)
# ============================================================================
 
def _to_bool_or_nan(x) -> float:
    """Convertit une valeur bool/str en 1.0/0.0/NaN, en préservant les NaN
    (important pour tonic_eog qui peut être np.nan pour certaines époques)."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return 1.0
    if s in ("0", "false", "f", "no", "n"):
        return 0.0
    return np.nan
 
 
def _load_and_aggregate() -> pd.DataFrame:
    df_all = pd.read_csv(RBD_EMG_CSV)
    df = df_all[df_all["type"] == "REM_EPOCH_4S"].copy()
    if df.empty:
        raise RuntimeError(f"Aucune ligne REM_EPOCH_4S dans {RBD_EMG_CSV}.")

    df["rswa"]      = df["rswa"].map(_to_bool_or_nan)
    df["tonic_eog"] = df["tonic_eog"].map(_to_bool_or_nan)

    agg = df.groupby(["patient_id", "channel"]).agg(
        rswa_fraction=("rswa", "mean"),
        phasic_ratio=("phasic_ratio", "mean"),
        tonic_ratio=("tonic_ratio", "mean"),
        tonic_eog_fraction=("tonic_eog", "mean"),
    ).reset_index()

    # --- Duree de sommeil REM (en minutes) par patient x canal ---
    # Denominateur commun aux deux debits d'evenements ci-dessous.
    epoch_len_col = df["epoch_len_sec"] if "epoch_len_sec" in df.columns else pd.Series(4.0, index=df.index)
    rem_minutes = (df.assign(_epoch_len=epoch_len_col)
                     .groupby(["patient_id", "channel"])["_epoch_len"]
                     .sum() / 60.0)
    rem_minutes = rem_minutes.rename("rem_minutes").reset_index()
    agg = agg.merge(rem_minutes, on=["patient_id", "channel"], how="left")

    # --- Debit d'epoques "toniques" par minute de REM (meme critere que rswa) ---
    n_tonic = (df[df["rswa"] == 1.0]
               .groupby(["patient_id", "channel"]).size()
               .rename("n_tonic_events").reset_index())
    agg = agg.merge(n_tonic, on=["patient_id", "channel"], how="left")
    agg["n_tonic_events"] = agg["n_tonic_events"].fillna(0.0)
    agg["tonic_events_per_min"] = agg["n_tonic_events"] / agg["rem_minutes"].replace(0, np.nan)

    # --- Debit d'evenements phasiques (bursts distincts, lignes type="PHASIC") par minute ---
    df_phasic = df_all[df_all["type"] == "PHASIC"]
    if not df_phasic.empty and {"patient_id", "channel"}.issubset(df_phasic.columns):
        n_phasic = (df_phasic.groupby(["patient_id", "channel"]).size()
                    .rename("n_phasic_events").reset_index())
        agg = agg.merge(n_phasic, on=["patient_id", "channel"], how="left")
        agg["n_phasic_events"] = agg["n_phasic_events"].fillna(0.0)
        agg["phasic_events_per_min"] = agg["n_phasic_events"] / agg["rem_minutes"].replace(0, np.nan)
    else:
        agg["phasic_events_per_min"] = np.nan

    for m, cfg in METRICS.items():
        if cfg["is_fraction"] and m in agg.columns:
            agg[m] = agg[m] * 100.0

    return agg
 
 
def _reshape_long(agg: pd.DataFrame, lbl: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    agg["patient_id"] = agg["patient_id"].astype(str).str.strip()
    merged = agg.merge(lbl, on="patient_id", how="left")
    merged_before_filter = merged.copy()
    merged = merged[merged["group"].isin(GROUP_ORDER)].copy()
 
    records = []
    for _, row in merged.iterrows():
        for metric in METRICS:
            records.append({
                "patient_id": row["patient_id"], "group": row["group"],
                "channel": row["channel"], "metric": metric, "value": row.get(metric, np.nan),
            })
    long_df = pd.DataFrame(records).dropna(subset=["group"])
    return long_df, merged_before_filter
 
 
# ============================================================================
# HELPERS STATISTIQUES (identiques à A_eeg_spectral_optimized.py)
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
# WORKER - traite UNE combinaison (channel, metric) : stats + figure
# ============================================================================
 
def _process_channel_metric(args):
    channel, metric, sub_all = args
 
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
 
    kw_rows, pw_rows, means_rows = [], [], []
 
    sub = sub_all.dropna(subset=["value"])
 
    # Exclut les groupes trop peu représentés sur CETTE combinaison canal x métrique
    # (ex: EAI avec un seul patient ayant une valeur non-NaN pour ce canal)
    counts = sub.groupby("group").size()
    valid_groups = counts[counts >= MIN_N_FOR_PLOT].index.tolist()
    dropped = counts[counts < MIN_N_FOR_PLOT]
    sub = sub[sub["group"].isin(valid_groups)]
 
    if not dropped.empty:
        log = _get_logger()
        for g, n in dropped.items():
            log.info(f"[{channel}/{metric}] Groupe '{g}' exclu de la figure (n={n} < {MIN_N_FOR_PLOT}).")
 
    if sub["group"].nunique() < 2:
        return kw_rows, pw_rows, means_rows
 
    sub_f = filter_iqr(sub, "value")
    normal, kw_p, pw = run_stats_block(sub_f, "value")
 
    kw_rows.append({"channel": channel, "metric": metric, "p_kw": kw_p})
    for g, grp in sub.groupby("group"):
        vals = grp["value"].dropna()
        means_rows.append({"channel": channel, "metric": metric,
                            "group": g, "mean": vals.mean(), "sd": vals.std(), "n": len(vals)})
    if not pw.empty:
        for _, r in pw.iterrows():
            pw_rows.append({"channel": channel, "metric": metric, **r.to_dict()})
 
    if SAVE_FIGS:
        _make_boxplot(sub_f, channel, metric, kw_p, pw, plt, sns)
 
    return kw_rows, pw_rows, means_rows
 
 
def _make_boxplot(sub_f, channel, metric, kw_p, pw, plt, sns):
    fig, ax = plt.subplots(figsize=(7, 5))
    order = [g for g in GROUP_ORDER if g in sub_f["group"].unique()]
    display_order = [DISPLAY_LABELS.get(g, g) for g in order]
    palette_display = {DISPLAY_LABELS.get(g, g): PALETTE[g] for g in order}
 
    sub_f = sub_f.copy()
    sub_f["group_display"] = sub_f["group"].map(DISPLAY_LABELS).fillna(sub_f["group"])
 
    sns.boxplot(data=sub_f, x="group_display", y="value", order=display_order,
                palette=palette_display, showfliers=False, ax=ax, legend=False)
    sns.stripplot(data=sub_f, x="group_display", y="value", order=display_order,
                  color="black", alpha=0.55, size=4, jitter=True, ax=ax)
 
    if not pw.empty:
        sig = pw[pw["stars"] != ""]
        if not sig.empty:
            ymax = sub_f["value"].max()
            yrange = sub_f["value"].max() - sub_f["value"].min()
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
    ax.set_title(f"{METRICS[metric]['label']}\n{channel}\n{kw_label}", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel(METRICS[metric]["label"])
    plt.tight_layout()
 
    save_path = Path(OUTPUT_FIGS) / f"bp_{metric}_{channel}.{FIG_FORMAT}"
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
 
 
# ============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================================
 
def main() -> None:
    log = _get_logger()
    Path(OUTPUT_FIGS).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_STATS).mkdir(parents=True, exist_ok=True)
 
    log.info("Chargement et agrégation patient x canal depuis le CSV EMG...")
    agg = _load_and_aggregate()
    log.info(f"Agrégation : {agg.shape} (patient x canal)")
 
    lbl = _load_labels()
    long_df, merged_before_filter = _reshape_long(agg, lbl)
 
    excluded = merged_before_filter[~merged_before_filter["patient_id"].isin(long_df["patient_id"])]
    if not excluded.empty:
        excluded_unique = excluded.drop_duplicates("patient_id")
        log.warning(f"{len(excluded_unique)} patients exclus (groupe non reconnu) :")
        for _, r in excluded_unique.iterrows():
            log.warning(f"  - {r['patient_id']} : {r['group']}")
 
    log.info(f"Patients retenus : {long_df['patient_id'].nunique()} | "
             f"Groupes : {long_df.drop_duplicates('patient_id')['group'].value_counts().to_dict()}")
    log.info(f"Tableau long : {long_df.shape}")
 
    combos: List[Tuple[str, str, pd.DataFrame]] = []
    for channel in EMG_CHANNELS:
        for metric in METRICS:
            sub_all = long_df[(long_df["channel"] == channel) & (long_df["metric"] == metric)].copy()
            combos.append((channel, metric, sub_all))
 
    log.info(f"{len(combos)} combinaisons canal\u00d7m\u00e9trique \u00e0 traiter avec {N_WORKERS} worker(s).")
 
    kw_rows, pw_rows, means_rows = [], [], []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_process_channel_metric, c): c for c in combos}
        for fut in as_completed(futures):
            channel, metric, _ = futures[fut]
            try:
                kw_r, pw_r, means_r = fut.result()
                kw_rows.extend(kw_r)
                pw_rows.extend(pw_r)
                means_rows.extend(means_r)
            except Exception as e:
                log.error(f"[ERREUR] {channel}/{metric} : {e}")
 
    means_df = pd.DataFrame(means_rows)
    kw_df    = pd.DataFrame(kw_rows)
    pw_df    = pd.DataFrame(pw_rows)
 
    means_df.to_csv(Path(OUTPUT_STATS) / "emg_means_per_channel_metric_group.csv", index=False)
    kw_df.to_csv(Path(OUTPUT_STATS) / "emg_kruskal_results.csv", index=False)
    pw_df.to_csv(Path(OUTPUT_STATS) / "emg_mannwhitney_posthoc.csv", index=False)
 
    sig_df = pw_df[pw_df["stars"] != ""].copy() if not pw_df.empty else pd.DataFrame()
    if not sig_df.empty:
        sig_df.to_excel(Path(OUTPUT_STATS) / "emg_significant_comparisons.xlsx", index=False)
        log.info(f"{len(sig_df)} comparaisons significatives export\u00e9es.")
    else:
        log.info("Aucune comparaison significative.")
 
    log.info(f"Stats \u00e9crites dans {OUTPUT_STATS}")
    log.info(f"Figures \u00e9crites dans {OUTPUT_FIGS}")
    log.info("Termin\u00e9.")
 
 
if __name__ == "__main__":
    main()