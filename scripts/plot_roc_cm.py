#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_roc_cm.py
Génère :
  - matrice de confusion
  - ROC multiclasses

Inputs :
  - modèle RandomForest checkpoint (joblib)
  - features.csv
  - labels Excel

Config YAML : configs/eval/plots_metrics.yaml
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.utils.config import load_yaml, add_common_args
from src.viz.metrics_plots import (
    plot_confusion_matrix,
    plot_multiclass_roc
)
from src.utils.logging import get_logger


def load_labels(cfg, log):
    excel = Path(cfg["labels_excel"])
    sheet = cfg.get("labels_sheet", 0)
    label_id_col = cfg["label_id_col"]
    label_col = cfg["label_col"]
    header = cfg.get("labels_header_row", 0)

    df = pd.read_excel(excel, sheet_name=sheet, header=header)
    df = df[[label_id_col, label_col]].dropna()
    df[label_id_col] = df[label_id_col].astype(str)
    df[label_col] = df[label_col].astype(str)
    return df


def main(cfg):

    log = get_logger("plot_metrics")

    # ---- Load paths ----
    ckpt = Path(cfg["ckpt_path"])
    features_csv = Path(cfg["features_csv"])
    fig_dir = Path(cfg["fig_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    log.info(f"Loading model checkpoint : {ckpt}")
    bundle = joblib.load(ckpt)
    model = bundle["model"]
    feat_cols = bundle["features"]

    # ---- Load features ----
    df_feat = pd.read_csv(features_csv)
    if "patient_id" not in df_feat.columns:
        raise ValueError("features.csv must contain a 'patient_id' column.")

    # ---- Load labels ----
    df_lab = load_labels(cfg, log)

    df_all = df_feat.merge(
        df_lab,
        on="patient_id",
        how="inner"
    )
    log.info(f"{df_all.shape[0]} lignes après merge features + labels.")

    X = df_all[feat_cols].values.astype(float)
    y = df_all[cfg["label_col"]].values

    class_names = sorted(np.unique(y))

    # ---- Predictions ----
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    # ---- Confusion matrix ----
    out_cm = fig_dir / "confusion_matrix.png"
    log.info(f"Saving confusion matrix → {out_cm}")

    plot_confusion_matrix(
        y_true=y,
        y_pred=y_pred,
        class_names=class_names,
        out_path=out_cm,
    )

    # ---- ROC multiclasses ----
    out_roc = fig_dir / "roc_multiclass.png"
    log.info(f"Saving ROC curves → {out_roc}")

    plot_multiclass_roc(
        y_true=y,
        y_proba=y_proba,
        class_names=class_names,
        out_path=out_roc,
    )


if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    main(cfg)
