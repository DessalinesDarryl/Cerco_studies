#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
xai_plots.py
============

Objectif
--------
Générer des figures XAI (prêtes pour un rapport / papier) à partir des sorties
produites par `xai_attributions.py`.

Figures générées
----------------
1) Barplot des **Top-k** features selon la permutation importance
   - Fichier attendu : permutation_importance.csv
   - Score : importance_mean (moyenne sur n_repeats permutations)
   - Interprétation : plus c'est grand, plus la feature est importante au sens
     "si je casse cette feature, la performance baisse".

2) Barplot des **Top-k** features selon **SHAP mean |value|**
   - Fichier attendu : shap_global_meanabs.csv
   - Score : mean_abs_shap (moyenne des contributions absolues)
   - Interprétation : plus c'est grand, plus la feature contribue en moyenne à la décision,
     toutes classes / tous échantillons confondus.

Entrées
-------
Les chemins viennent du YAML, typiquement :

- perm_csv : outputs/xai/attributions/permutation_importance.csv
- shap_csv : outputs/xai/attributions/shap_global_meanabs.csv
- out_dir  : outputs/figures/xai
- top_k    : 20

Sorties
-------
Dans out_dir :
- permutation_importance_topk.png
- shap_global_topk.png

Dépendances
-----------
- pandas
- matplotlib
- seaborn

Notes / bonnes pratiques
------------------------
- Le script ne plante pas si l'un des CSV est absent : il log un warning.
- Le barplot est orienté "horizontal" (x=score, y=feature) pour lire facilement les noms.
- La taille de figure est adaptée au nombre de features (hauteur proportionnelle à top_k).

Exécution
---------
python scripts/xai_plots.py --config configs/xai/plots.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.utils.config import load_yaml, add_common_args
from src.utils.logging import get_logger


def plot_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    out_path: Path,
    *,
    figsize_w: float = 8.0,
    row_height: float = 0.25,
    min_h: float = 4.0,
    dpi: int = 200,
) -> None:
    """
    Trace et sauvegarde un barplot horizontal (score en abscisse, features en ordonnée).

    Pourquoi cette fonction ?
    -------------------------
    - Mutualiser la logique de plotting (permutation + SHAP).
    - Assurer une figure lisible même quand top_k est grand (hauteur variable).
    - Centraliser la sauvegarde (dpi, tight_layout, close).

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame déjà filtré / trié (souvent top_k lignes).
    x : str
        Nom de la colonne numérique à utiliser comme score (axe X).
        Exemples : "importance_mean", "mean_abs_shap"
    y : str
        Nom de la colonne texte utilisée comme étiquette (axe Y).
        Exemples : "feature"
    title : str
        Titre de la figure.
    out_path : Path
        Chemin du fichier image de sortie (ex: .png).
    figsize_w : float, optionnel
        Largeur figure en pouces.
    row_height : float, optionnel
        Hauteur par ligne (feature) en pouces. Utilisé pour adapter automatiquement la figure.
    min_h : float, optionnel
        Hauteur minimale de figure en pouces.
    dpi : int, optionnel
        Résolution du rendu.

    Sortie
    ------
    None
        Écrit un fichier image sur disque.

    Remarques
    ---------
    - La fonction suppose que df[x] est numérique.
    - En cas de df vide, la fonction ne trace rien.
    """
    if df is None or df.empty:
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig_h = max(min_h, float(len(df)) * float(row_height))

    plt.figure(figsize=(figsize_w, fig_h))
    sns.barplot(data=df, x=x, y=y)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def _load_and_prepare(
    csv_path: Path,
    score_col: str,
    feature_col: str,
    top_k: int,
    log,
) -> Optional[pd.DataFrame]:
    """
    Charge un CSV d'attributions (permutation ou SHAP), vérifie les colonnes,
    trie décroissant sur le score et garde les top_k.

    Paramètres
    ----------
    csv_path : Path
        Chemin vers le CSV.
    score_col : str
        Colonne contenant le score d'importance.
    feature_col : str
        Colonne contenant le nom de la feature.
    top_k : int
        Nombre de features à conserver.
    log : logger

    Retour
    ------
    Optional[pd.DataFrame]
        - DataFrame filtré (top_k) si OK
        - None si fichier absent ou colonnes manquantes
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        log.warning(f"CSV introuvable : {csv_path}")
        return None

    log.info(f"Lecture CSV : {csv_path}")
    df = pd.read_csv(csv_path)

    missing = [c for c in (score_col, feature_col) if c not in df.columns]
    if missing:
        log.warning(f"Colonnes manquantes dans {csv_path.name} : {missing}")
        return None

    df = df.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[score_col, feature_col])

    df = df.sort_values(score_col, ascending=False)
    df_top = df.head(int(top_k)).reset_index(drop=True)
    return df_top


def main(cfg: dict) -> None:
    """
    Point d'entrée du script.

    Étapes
    ------
    1) Lecture des chemins (perm_csv, shap_csv) et paramètres (out_dir, top_k).
    2) Génération du barplot de permutation importance (si perm_csv existe).
    3) Génération du barplot SHAP global (si shap_csv existe).
    4) Journalisation finale des sorties générées.

    Paramètres
    ----------
    cfg : dict
        Configuration YAML, attend au minimum :
          - perm_csv : str
          - shap_csv : str
          - out_dir : str
        et optionnel :
          - top_k : int (défaut 20)
    """
    log = get_logger("xai_plots")

    perm_csv = Path(cfg["perm_csv"])
    shap_csv = Path(cfg["shap_csv"])
    out_dir = Path(cfg["out_dir"])
    top_k = int(cfg.get("top_k", 20))

    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # 1) Permutation importance
    # -------------------------------
    df_perm_top = _load_and_prepare(
        csv_path=perm_csv,
        score_col="importance_mean",
        feature_col="feature",
        top_k=top_k,
        log=log,
    )
    if df_perm_top is not None and not df_perm_top.empty:
        out = out_dir / "permutation_importance_topk.png"
        log.info(f"Plot permutation importance → {out}")
        plot_bar(
            df_perm_top,
            x="importance_mean",
            y="feature",
            title=f"Top {top_k} - Permutation Importance",
            out_path=out,
        )
    else:
        log.warning("Permutation importance non tracée (CSV absent ou invalide).")

    # -------------------------------
    # 2) SHAP global
    # -------------------------------
    df_shap_top = _load_and_prepare(
        csv_path=shap_csv,
        score_col="mean_abs_shap",
        feature_col="feature",
        top_k=top_k,
        log=log,
    )
    if df_shap_top is not None and not df_shap_top.empty:
        out = out_dir / "shap_global_topk.png"
        log.info(f"Plot SHAP global → {out}")
        plot_bar(
            df_shap_top,
            x="mean_abs_shap",
            y="feature",
            title=f"Top {top_k} - SHAP mean |value|",
            out_path=out,
        )
    else:
        log.warning("SHAP global non tracé (CSV absent ou invalide).")

    log.info("Figures XAI générées.")


if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    main(cfg)
