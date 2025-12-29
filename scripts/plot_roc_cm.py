#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_roc_cm.py

Génère :
  - matrice de confusion (globale, sur tout dataset_final)
  - ROC multiclasses

Inputs :
  - checkpoint RandomForest (bundle joblib : {"model", "features", "label_map"})
  - dataset_final.csv (features + label_id/label_str)

Config YAML : configs/eval/plots_metrics.yaml
  ckpt_path: models/checkpoint_rf.joblib
  features_csv: data/processed/features/dataset_final.csv
  label_col: "label_id"        # ou "label_str"
  fig_dir: outputs/figures/metrics
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.utils.config import load_yaml, add_common_args
from src.viz.metrics_plots import (
    plot_confusion_matrix,
    plot_multiclass_roc,
)
from src.utils.logging import get_logger


def load_dataset(cfg, bundle, log):
    """
    Charge dataset_final.csv et reconstruit X, y cohérents avec
    les features utilisées à l'entraînement.

    - Utilise cfg["label_col"] (par défaut "label_id")
    - Filtre les lignes sans label
    """
    features_csv = Path(cfg["features_csv"])
    if not features_csv.exists():
        raise FileNotFoundError(f"features_csv introuvable : {features_csv}")

    log.info(f"Lecture dataset_final : {features_csv}")
    df = pd.read_csv(features_csv)

    if "patient_id" not in df.columns:
        raise ValueError("dataset_final.csv doit contenir une colonne 'patient_id'.")

    label_col = cfg.get("label_col", "label_id")
    if label_col not in df.columns:
        raise ValueError(
            f"Colonne label '{label_col}' absente de dataset_final.csv. "
            f"Colonnes dispo : {list(df.columns)}"
        )

    # Filtrer les lignes sans label
    before = df.shape[0]
    df = df.dropna(subset=[label_col])
    after = df.shape[0]
    log.info(f"Lignes avec label ({label_col}) : {before} -> {after}")

    # y : on force en int si possible, sinon factorisation
    if np.issubdtype(df[label_col].dtype, np.number):
        df[label_col] = df[label_col].astype(int)
        y = df[label_col].to_numpy(dtype=int)
    else:
        df[label_col], uniques = pd.factorize(df[label_col].astype(str))
        y = df[label_col].to_numpy(dtype=int)
        log.info(f"Labels factorisés automatiquement depuis '{label_col}' : {dict(enumerate(uniques))}")

    # Features utilisées pendant le training
    feat_cols = bundle["features"]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Colonnes de features manquantes dans dataset_final : {missing}"
        )

    X = df[feat_cols].values.astype(float)

    log.info(f"Shape X: {X.shape}, y: {y.shape}, nb_features: {len(feat_cols)}")
    return X, y, df


def build_class_names(model, bundle, log):
    """
    Construit les noms de classes dans l'ordre de model.classes_.
    Si bundle['label_map'] est dispo (str -> int), on le renverse pour avoir int -> str.
    """
    classes = model.classes_
    label_map = bundle.get("label_map", None)

    if label_map:
        inv_map = {int(v): str(k) for k, v in label_map.items()}
        class_names = [inv_map.get(int(c), str(c)) for c in classes]
        log.info(f"Classes (model.classes_) et noms : {list(zip(classes, class_names))}")
    else:
        class_names = [str(c) for c in classes]
        log.info(f"Classes sans label_map : {classes}")

    return class_names


def main(cfg):
    log = get_logger("plot_metrics")

    # ---- Paths ----
    ckpt = Path(cfg["ckpt_path"])
    fig_dir = Path(cfg["fig_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model bundle ----
    log.info(f"Loading model checkpoint : {ckpt}")
    bundle = joblib.load(ckpt)
    model = bundle["model"]

    # ---- Dataset (X, y) ----
    X, y, df = load_dataset(cfg, bundle, log)

    # ---- Class names alignés avec model.classes_ ----
    class_names = build_class_names(model, bundle, log)

    # ---- Prédictions ----
    log.info("Calcul des prédictions et probabilités...")
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

    log.info("Métriques (CM + ROC) générées.")


if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    main(cfg)
