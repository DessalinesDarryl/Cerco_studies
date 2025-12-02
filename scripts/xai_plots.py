#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
xai_plots.py
Génère les figures XAI à partir des sorties de xai_attributions.py :

- Barplot permutation importance
- Barplot SHAP mean |value|
- Top-k features

Config (configs/xai/plots.yaml) attend :
    perm_csv: outputs/xai/attributions/permutation_importance.csv
    shap_csv: outputs/xai/attributions/shap_global_meanabs.csv
    out_dir: outputs/figures/xai
    top_k: 20
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.utils.config import load_yaml, add_common_args
from src.utils.logging import get_logger


def plot_bar(df, x, y, title, out_path):
    plt.figure(figsize=(8, max(4, len(df) * 0.25)))
    sns.barplot(data=df, x=x, y=y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main(cfg):
    log = get_logger("xai_plots")

    perm_csv = Path(cfg["perm_csv"])
    shap_csv = Path(cfg["shap_csv"])
    out_dir = Path(cfg["out_dir"])
    top_k = int(cfg.get("top_k", 20))

    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # 1) Permutation importance
    # -------------------------------
    if perm_csv.exists():
        log.info(f"Lecture permutation_importance: {perm_csv}")
        df_perm = pd.read_csv(perm_csv).sort_values("importance_mean", ascending=False)
        df_top = df_perm.head(top_k)

        out = out_dir / "permutation_importance_topk.png"
        log.info(f"Plot → {out}")
        plot_bar(df_top, x="importance_mean", y="feature",
                 title=f"Top {top_k} - Permutation Importance",
                 out_path=out)
    else:
        log.warning("Permutation importance CSV non trouvé.")

    # -------------------------------
    # 2) SHAP global
    # -------------------------------
    if shap_csv.exists():
        log.info(f"Lecture SHAP global: {shap_csv}")
        df_shap = pd.read_csv(shap_csv).sort_values("mean_abs_shap", ascending=False)
        df_top = df_shap.head(top_k)

        out = out_dir / "shap_global_topk.png"
        log.info(f"Plot → {out}")
        plot_bar(df_top, x="mean_abs_shap", y="feature",
                 title=f"Top {top_k} - SHAP mean |value|",
                 out_path=out)
    else:
        log.warning("SHAP global CSV non trouvé.")

    log.info("Figures XAI générées.")


if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    main(cfg)
