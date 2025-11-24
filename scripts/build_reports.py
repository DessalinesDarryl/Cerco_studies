#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_reports.py

Construit un rapport Markdown en intégrant automatiquement :
  - les figures de performances globales (confusion matrix + ROC),
  - les figures XAI (globales + par groupe + par patient),
  - éventuellement quelques infos de stats si disponibles.

Entrée (via arguments CLI, cohérent avec le Makefile) :
  --xai-root   : dossier avec les sorties XAI (si besoin plus tard)
  --stats-root : dossier avec les résultats statistiques (CSV, etc.)
  --fig-root   : dossier racine des figures (metrics/, xai/, ...)
  --out        : dossier de sortie des rapports (ex: outputs/reports)

Sortie :
  - {out}/report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
import glob

import pandas as pd

from src.utils.logging import get_logger


def find_metrics_figures(fig_root: Path):
    """Retourne les chemins vers confusion_matrix.png et roc_multiclass.png si présents."""
    metrics_dir = fig_root / "metrics"
    cm = metrics_dir / "confusion_matrix.png"
    roc = metrics_dir / "roc_multiclass.png"

    cm = cm if cm.exists() else None
    roc = roc if roc.exists() else None
    return cm, roc


def find_xai_figures(fig_root: Path):
    """
    Cherche les figures XAI dans fig_root/xai :
      - global_feature_importance
      - group_feature_heatmap
      - xai_patient_*_contrib
    """
    xai_dir = fig_root / "xai"
    if not xai_dir.exists():
        return None, None, []

    global_fig = xai_dir / "xai_global_feature_importance.png"
    group_fig = xai_dir / "xai_group_feature_heatmap.png"

    global_fig = global_fig if global_fig.exists() else None
    group_fig = group_fig if group_fig.exists() else None

    patient_figs = sorted(
        Path(p) for p in glob.glob(str(xai_dir / "xai_patient_*_contrib.png"))
    )

    return global_fig, group_fig, patient_figs


def summarize_stats(stats_root: Path):
    """
    Optionnel : résume quelques CSV de stats s'ils existent.
    Retourne une petite section texte.
    """
    if not stats_root.exists():
        return "Aucun fichier de statistiques trouvé.\n"

    csv_files = sorted(stats_root.glob("*.csv"))
    if not csv_files:
        return "Aucun fichier de statistiques (CSV) trouvé.\n"

    lines = ["Statistiques disponibles :\n"]
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            lines.append(f"- **{csv_path.name}** : {df.shape[0]} lignes, {df.shape[1]} colonnes")
        except Exception:
            lines.append(f"- **{csv_path.name}** (lecture impossible, format non standard)")

    return "\n".join(lines) + "\n"


def build_markdown_report(
    out_dir: Path,
    fig_root: Path,
    xai_root: Path,
    stats_root: Path,
):
    """
    Construit le contenu Markdown du rapport et l'écrit dans out_dir/report.md.
    Les chemins d'images sont exprimés en relatif par rapport au rapport.
    """
    # Pour les chemins relatifs, le rapport est dans out_dir,
    # les figures sont dans fig_root (ex: outputs/figures).
    rel_fig_base = Path("..") / fig_root.relative_to(fig_root.parent)

    # 1) Figures metrics
    cm_path, roc_path = find_metrics_figures(fig_root)
    # 2) Figures XAI
    xai_global, xai_group, xai_patient_figs = find_xai_figures(fig_root)
    # 3) Stats
    stats_summary = summarize_stats(stats_root)

    md_lines = []

    md_lines.append("# Rapport d’analyse – Classification RBD / EEG–EMG\n")

    # ------------------------------------------------------------------
    # Section 1 : Performances globales
    # ------------------------------------------------------------------
    md_lines.append("## 1. Performances globales du modèle\n")
    md_lines.append(
        "Cette section résume les performances du modèle (RandomForest) sur les features "
        "extraits des segments REM (EEG/EMG/ECG).\n"
    )

    if cm_path is not None:
        rel_cm = Path("..") / cm_path.relative_to(out_dir.parent)
        md_lines.append("### 1.1 Matrice de confusion\n")
        md_lines.append(
            "La matrice de confusion (normalisée par classe réelle) permet de visualiser "
            "les erreurs de classification entre les différentes catégories de patients.\n"
        )
        md_lines.append(f"![Matrice de confusion]({rel_cm.as_posix()})\n")
    else:
        md_lines.append("- Matrice de confusion non trouvée (confusion_matrix.png).\n")

    if roc_path is not None:
        rel_roc = Path("..") / roc_path.relative_to(out_dir.parent)
        md_lines.append("### 1.2 Courbes ROC multiclasses\n")
        md_lines.append(
            "Les courbes ROC (one-vs-rest) et les AUC associées quantifient la capacité "
            "du modèle à discriminer chaque catégorie par rapport aux autres.\n"
        )
        md_lines.append(f"![ROC multiclasses]({rel_roc.as_posix()})\n")
    else:
        md_lines.append("- Courbes ROC non trouvées (roc_multiclass.png).\n")

    md_lines.append("\n")

    # ------------------------------------------------------------------
    # Section 2 : Importances globales des features (XAI)
    # ------------------------------------------------------------------
    md_lines.append("## 2. Explicabilité globale (XAI)\n")
    md_lines.append(
        "Cette section résume les features les plus importantes pour le modèle, "
        "ainsi que leurs différences selon les groupes de patients.\n"
    )

    if xai_global is not None:
        rel_xai_global = Path("..") / xai_global.relative_to(out_dir.parent)
        md_lines.append("### 2.1 Importances globales des features\n")
        md_lines.append(
            "Les barres représentent l’importance moyenne de chaque feature dans le RandomForest.\n"
        )
        md_lines.append(f"![Importances globales]({rel_xai_global.as_posix()})\n")
    else:
        md_lines.append("- Figure d’importances globales (xai_global_feature_importance.png) non trouvée.\n")

    if xai_group is not None:
        rel_xai_group = Path("..") / xai_group.relative_to(out_dir.parent)
        md_lines.append("### 2.2 Importances par groupe de patients\n")
        md_lines.append(
            "La heatmap montre, pour chaque groupe de patients, quelles features contribuent le plus "
            "aux différences par rapport à la moyenne globale.\n"
        )
        md_lines.append(f"![Importances par groupe]({rel_xai_group.as_posix()})\n")
    else:
        md_lines.append("- Heatmap XAI par groupe (xai_group_feature_heatmap.png) non trouvée.\n")

    md_lines.append("\n")

    # ------------------------------------------------------------------
    # Section 3 : Attributions locales par patient
    # ------------------------------------------------------------------
    md_lines.append("## 3. Attributions locales par patient\n")
    md_lines.append(
        "Pour certains patients sélectionnés, on visualise les contributions (signées) des features "
        "qui tirent la prédiction vers une catégorie ou une autre.\n"
    )

    if xai_patient_figs:
        for fig_path in xai_patient_figs:
            # extraire l'ID patient du nom de fichier : xai_patient_{ID}_contrib.png
            name = fig_path.name
            patient_id = name.replace("xai_patient_", "").replace("_contrib.png", "")
            rel_fig = Path("..") / fig_path.relative_to(out_dir.parent)

            md_lines.append(f"### 3.x Patient {patient_id}\n")
            md_lines.append(
                "Barres positives : features qui augmentent la probabilité de la classe prédite ; "
                "barres négatives : features qui la diminuent.\n"
            )
            md_lines.append(f"![Attributions patient {patient_id}]({rel_fig.as_posix()})\n")
    else:
        md_lines.append(
            "- Aucune figure d’attribution locale trouvée (xai_patient_*_contrib.png). "
            "Tu peux en générer via `scripts/xai_plots.py`.\n"
        )

    md_lines.append("\n")

    # ------------------------------------------------------------------
    # Section 4 : Statistiques de groupe
    # ------------------------------------------------------------------
    md_lines.append("## 4. Statistiques de groupe\n")
    md_lines.append(
        "Résumé des fichiers de statistiques produits (tests de groupe, comparaisons inter-catégories, etc.).\n\n"
    )
    md_lines.append(stats_summary)

    # ------------------------------------------------------------------
    # Écriture du rapport
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "report.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    return out_md


def parse_args():
    ap = argparse.ArgumentParser(description="Construire un rapport Markdown avec les figures XAI + métriques.")
    ap.add_argument("--xai-root", type=str, required=False, default="outputs/xai",
                    help="Dossier des sorties XAI (non strictement nécessaire pour les figures).")
    ap.add_argument("--stats-root", type=str, required=False, default="outputs/stats",
                    help="Dossier contenant les CSV de stats.")
    ap.add_argument("--fig-root", type=str, required=False, default="outputs/figures",
                    help="Dossier racine des figures (metrics/, xai/, ...).")
    ap.add_argument("--out", type=str, required=False, default="outputs/reports",
                    help="Dossier de sortie des rapports.")
    return ap.parse_args()


def main():
    log = get_logger("build_reports")

    args = parse_args()
    xai_root = Path(args.xai_root)
    stats_root = Path(args.stats_root)
    fig_root = Path(args.fig_root)
    out_dir = Path(args.out)

    log.info(f"xai_root   = {xai_root}")
    log.info(f"stats_root = {stats_root}")
    log.info(f"fig_root   = {fig_root}")
    log.info(f"out_dir    = {out_dir}")

    out_md = build_markdown_report(
        out_dir=out_dir,
        fig_root=fig_root,
        xai_root=xai_root,
        stats_root=stats_root,
    )

    log.info(f"Rapport écrit : {out_md}")


if __name__ == "__main__":
    main()
