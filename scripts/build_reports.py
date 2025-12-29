#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_reports.py

Construit un rapport Markdown en intégrant automatiquement :
  - les figures de performances globales (confusion matrix + ROC),
  - les figures XAI (globales + par groupe + par patient),
  - un résumé chiffré des métriques de classification,
  - un résumé des statistiques disponibles (CSV).

Entrée :
  --xai-root   : dossier avec les sorties XAI
  --stats-root : dossier avec les résultats statistiques (CSV)
  --fig-root   : dossier racine des figures (metrics/, xai/, ...)
  --out        : dossier de sortie des rapports (ex: outputs/reports)

Sortie :
  - {out}/report.md
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse
import glob

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Allow "import src.*"
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.utils.logging import get_logger


# =====================================================================
# FIGURES DISCOVERY
# =====================================================================
def find_metrics_figures(fig_root: Path):
    metrics_dir = fig_root 
    if not metrics_dir.exists():
        return None, None

    cm_candidates = list(metrics_dir.glob("*confusion*.png"))
    roc_candidates = list(metrics_dir.glob("*roc*.png"))

    cm = cm_candidates[0] if cm_candidates else None
    roc = roc_candidates[0] if roc_candidates else None
    return cm, roc



def find_xai_figures(fig_root: Path):
    """
    Cherche les figures XAI dans fig_root/xai :
      - importance globale (SHAP / permutation)
      - heatmap par groupe
      - attributions locales par patient
    """
    xai_dir = fig_root / "xai"
    if not xai_dir.exists():
        return None, None, []

    global_candidates = (
        list(xai_dir.glob("*global*importance*.png"))
        + list(xai_dir.glob("shap_global*.png"))
        + list(xai_dir.glob("permutation_importance*.png"))
    )
    global_fig = global_candidates[0] if global_candidates else None

    group_candidates = (
        list(xai_dir.glob("*group*heatmap*.png"))
        + list(xai_dir.glob("*group*.png"))
    )
    group_fig = group_candidates[0] if group_candidates else None

    patient_figs = sorted(
        Path(p) for p in glob.glob(str(xai_dir / "xai_patient_*_contrib.png"))
    )

    return global_fig, group_fig, patient_figs


# =====================================================================
# STATS SUMMARY
# =====================================================================
def summarize_stats(stats_root: Path) -> str:
    """Résumé simple des CSV de statistiques disponibles."""
    if not stats_root.exists():
        return "Aucun fichier de statistiques trouvé.\n"

    csv_files = sorted(stats_root.glob("*.csv"))
    if not csv_files:
        return "Aucun fichier de statistiques (CSV) trouvé.\n"

    lines = ["Statistiques disponibles :\n"]
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            lines.append(
                f"- **{csv_path.name}** : {df.shape[0]} lignes, {df.shape[1]} colonnes"
            )
        except Exception:
            lines.append(
                f"- **{csv_path.name}** (lecture impossible)"
            )

    return "\n".join(lines) + "\n"


def summarize_metrics(stats_root: Path) -> str:
    """Résumé numérique des performances moyennes (cross-validation)."""
    metrics_file = stats_root / "train_metrics_rf.csv"
    if not metrics_file.exists():
        return ""

    df = pd.read_csv(metrics_file)
    mean_metrics = df.mean(numeric_only=True)

    return (
        "### Performances moyennes (cross-validation)\n\n"
        f"- Accuracy moyenne : **{mean_metrics.get('accuracy', np.nan):.3f}**\n"
        f"- Balanced accuracy : **{mean_metrics.get('balanced_accuracy', np.nan):.3f}**\n"
        f"- F1-score macro : **{mean_metrics.get('f1_macro', np.nan):.3f}**\n"
        f"- Recall macro : **{mean_metrics.get('recall_macro', np.nan):.3f}**\n\n"
    )


# =====================================================================
# REPORT BUILDING
# =====================================================================
def build_markdown_report(
    out_dir: Path,
    fig_root: Path,
    xai_root: Path,
    stats_root: Path,
):
    """
    Construit le rapport Markdown et l’écrit dans out_dir/report.md.
    Les chemins d’images sont relatifs au rapport.
    """
    cm_path, roc_path = find_metrics_figures(fig_root)
    xai_global, xai_group, xai_patient_figs = find_xai_figures(fig_root)

    stats_summary = summarize_stats(stats_root)
    metrics_summary = summarize_metrics(stats_root)

    md_lines = []

    md_lines.append("# Rapport d’analyse – Classification RBD / EEG–EMG\n")

    # ------------------------------------------------------------------
    # Section 1 : Performances globales
    # ------------------------------------------------------------------
    md_lines.append("## 1. Performances globales du modèle\n")
    md_lines.append(
        "Cette section résume les performances du modèle (RandomForest) sur les features "
        "extraits des segments REM (EEG / EMG).\n"
    )

    if metrics_summary:
        md_lines.append(metrics_summary)

    if cm_path is not None:
        rel_cm = Path("..") / cm_path.relative_to(out_dir.parent)
        md_lines.append("### 1.1 Matrice de confusion\n")
        md_lines.append(f"![Matrice de confusion]({rel_cm.as_posix()})\n")
    else:
        md_lines.append("- Matrice de confusion non trouvée.\n")

    if roc_path is not None:
        rel_roc = Path("..") / roc_path.relative_to(out_dir.parent)
        md_lines.append("### 1.2 Courbes ROC multiclasses\n")
        md_lines.append(f"![ROC multiclasses]({rel_roc.as_posix()})\n")
    else:
        md_lines.append("- Courbes ROC non trouvées.\n")

    # ------------------------------------------------------------------
    # Section 2 : Explicabilité globale
    # ------------------------------------------------------------------
    md_lines.append("\n## 2. Explicabilité globale (XAI)\n")

    if xai_global is not None:
        rel_xai_global = Path("..") / xai_global.relative_to(out_dir.parent)
        md_lines.append("### 2.1 Importances globales des features\n")
        md_lines.append(f"![XAI global]({rel_xai_global.as_posix()})\n")
    else:
        md_lines.append("- Aucune figure XAI globale trouvée.\n")

    if xai_group is not None:
        rel_xai_group = Path("..") / xai_group.relative_to(out_dir.parent)
        md_lines.append("### 2.2 Importances par groupe de patients\n")
        md_lines.append(f"![XAI groupes]({rel_xai_group.as_posix()})\n")
    else:
        md_lines.append("- Heatmap XAI par groupe non trouvée.\n")

    # ------------------------------------------------------------------
    # Section 3 : Attributions locales
    # ------------------------------------------------------------------
    md_lines.append("\n## 3. Attributions locales par patient\n")

    if xai_patient_figs:
        for fig_path in xai_patient_figs:
            pid = fig_path.stem.replace("xai_patient_", "").replace("_contrib", "")
            rel_fig = Path("..") / fig_path.relative_to(out_dir.parent)
            md_lines.append(f"### Patient {pid}\n")
            md_lines.append(f"![XAI patient {pid}]({rel_fig.as_posix()})\n")
    else:
        md_lines.append("- Aucune attribution locale trouvée.\n")

    # ------------------------------------------------------------------
    # Section 4 : Statistiques
    # ------------------------------------------------------------------
    md_lines.append("\n## 4. Statistiques de groupe\n")
    md_lines.append(stats_summary)

    top_feats = stats_root / "shap_top_features.csv"
    if top_feats.exists():
        df = pd.read_csv(top_feats)
        top_names = ", ".join(df["feature"].head(3))
        md_lines.append(
            f"\nLes features les plus contributives sont principalement : **{top_names}**.\n"
        )

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "report.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    return out_md


# =====================================================================
# CLI
# =====================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Construire un rapport Markdown XAI + métriques.")
    ap.add_argument("--xai-root", default="outputs/xai")
    ap.add_argument("--stats-root", default="outputs/stats")
    ap.add_argument("--fig-root", default="outputs/figures")
    ap.add_argument("--out", default="outputs/reports")
    return ap.parse_args()


def main():
    log = get_logger("build_reports")

    args = parse_args()
    out_md = build_markdown_report(
        out_dir=Path(args.out),
        fig_root=Path(args.fig_root),
        xai_root=Path(args.xai_root),
        stats_root=Path(args.stats_root),
    )

    log.info(f"Rapport écrit : {out_md}")


if __name__ == "__main__":
    main()
