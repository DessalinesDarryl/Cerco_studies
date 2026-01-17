#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
group_stats.py

Stats globales sur les features à partir des sorties de xai_attributions.py :
  - agrégats SHAP par bande de fréquence
  - top-k features les plus importantes

Entrées (via YAML) :
  - xai_root : racine des sorties XAI (ex: outputs/xai)
  - out_dir  : dossier de sortie pour les stats (ex: outputs/stats)
  - top_k    : (optionnel) nombre de features dans le top SHAP (par défaut 20)

Fichiers attendus dans xai_root/attributions :
  - shap_global_meanabs.csv
  - (optionnel) permutation_importance.csv
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import os
import argparse
import re

import numpy as np
import pandas as pd

from src.utils.config import add_common_args, load_yaml
from src.utils.logging import get_logger


def infer_band_from_feature(feat: str) -> str:
    """
    Essaie de déduire une bande de fréquence à partir du nom de la feature.

    Règles :
      - si 'delta'/'theta'/'alpha'/'beta'/'gamma_bas'/'gamma_haut' dans le nom >>> ces bandes
      - si pattern numérique du type '85_95Hz' ou '85-95Hz' >>> '85.0–95.0Hz'
      - si spec_entropy / spec_centroid / peak_freq / rms / zc >>> '0.5–80.0Hz'
      - sinon >>> 'broadband'
    """
    if not isinstance(feat, str):
        return "broadband"

    f = feat.lower()

    # bandes nominales
    if "delta" in f:
        return "delta"
    if "theta" in f:
        return "theta"
    if "alpha" in f:
        return "alpha"
    if "beta" in f:
        return "beta"
    if "gamma_bas" in f:
        return "gamma_bas"
    if "gamma_haut" in f:
        return "gamma_haut"

    # pattern numérique : ex "85_95hz" ou "85-95hz"
    m = re.search(r"(\d+\.?\d*)[._-](\d+\.?\d*)hz", f)
    if m:
        lo = float(m.group(1))
        hi = float(m.group(2))
        return f"{lo:.1f}–{hi:.1f}Hz"

    # features globales sur [0.5–80] Hz
    if any(kw in f for kw in ["spec_entropy", "spec_centroid", "peak_freq", "rms", "zc"]):
        return "0.5–80.0Hz"

    # fallback
    return "broadband"


def main(cfg):
    log = get_logger("group_stats")

    xai_root = Path(cfg["xai_root"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    attr_dir = xai_root / "attributions"
    shap_path = attr_dir / "shap_global_meanabs.csv"
    perm_path = attr_dir / "permutation_importance.csv"

    if not shap_path.exists():
        raise FileNotFoundError(
            f"{shap_path} introuvable. "
            f"Assure-toi d'avoir lancé `make xai_attr` avant `make stats`."
        )

    log.info(f"Lecture SHAP global : {shap_path}")
    df_shap = pd.read_csv(shap_path)

    if "feature" not in df_shap.columns or "mean_abs_shap" not in df_shap.columns:
        raise ValueError(
            f"shap_global_meanabs.csv doit contenir au moins "
            f"les colonnes 'feature' et 'mean_abs_shap'. Colonnes trouvées : {list(df_shap.columns)}"
        )

    # Merge éventuel avec permutation importance (si dispo)
    if perm_path.exists():
        log.info(f"Lecture permutation importance : {perm_path}")
        df_perm = pd.read_csv(perm_path)
        if "feature" in df_perm.columns:
            df = df_shap.merge(df_perm, on="feature", how="left")
        else:
            log.warning("permutation_importance.csv sans colonne 'feature' >>> ignoré.")
            df = df_shap.copy()
    else:
        log.info("permutation_importance.csv introuvable >>> seules les stats SHAP seront utilisées.")
        df = df_shap.copy()

    # ---------- Gestion de la colonne 'band' ----------
    if "band" in df.columns:
        # On ne modifie que les 'other'
        df["band"] = df["band"].astype(str)
        mask_other = df["band"] == "other"
        if mask_other.any():
            log.info(f"{mask_other.sum()} features avec band='other' >>> remplacement par bande exacte.")
            df.loc[mask_other, "band"] = df.loc[mask_other, "feature"].apply(infer_band_from_feature)
    else:
        # Pas de bande du tout >>> on la reconstruit pour toutes les features
        log.warning("Colonne 'band' absente, reconstruction des bandes à partir des noms de features.")
        df["band"] = df["feature"].apply(infer_band_from_feature)

    # ---------- Stats par bande ----------
    band_stats = (
        df.groupby("band")
          .agg(
              mean_shap=("mean_abs_shap", "mean"),
              max_shap=("mean_abs_shap", "max"),
              n_feats=("feature", "nunique"),
          )
          .reset_index()
          .sort_values("mean_shap", ascending=False)
    )

    band_path = out_dir / "shap_band_stats.csv"
    band_stats.to_csv(band_path, index=False)
    log.info(f"Stats SHAP par bande écrites dans {band_path}")

    # ---------- Top-k features ----------
    top_k = int(cfg.get("top_k", 20))
    df_top = df.sort_values("mean_abs_shap", ascending=False).head(top_k)
    top_path = out_dir / "shap_top_features.csv"
    df_top.to_csv(top_path, index=False)
    log.info(f"Top-{top_k} features SHAP écrit dans {top_path}")


if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    main(cfg)
