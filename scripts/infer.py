#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer.py

Objectif :
  - Charger le checkpoint RandomForest entraîné (checkpoint_rf.joblib).
  - Charger le CSV de features (dataset_final.csv).
  - Utiliser les mêmes colonnes de features que pendant l'entraînement.
  - Produire un CSV avec :
      * patient_id
      * y_pred (id de classe)
      * y_pred_str (nom de classe si label_map dispo)
      * proba par classe (colonnes p_<label> ou p_class_<id>)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse
import numpy as np
import pandas as pd
import joblib

from src.utils.config import load_yaml, add_common_args
from src.utils.logging import get_logger


def main(cfg):
    log = get_logger("infer")

    # 1) Config
    features_csv = Path(cfg["features_csv"])  # ex: data/processed/features/dataset_final.csv
    ckpt_path = Path(cfg["ckpt_path"])       # ex: models/checkpoint_rf.joblib
    out_csv = Path(cfg["out_csv"])           # ex: outputs/infer/predictions_rf.csv
    label_col = cfg.get("label_col", "label_id")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 2) Chargement features
    if not features_csv.exists():
        raise FileNotFoundError(f"features_csv introuvable : {features_csv}")
    df = pd.read_csv(features_csv)

    if "patient_id" not in df.columns:
        raise ValueError("La colonne 'patient_id' est absente de features_csv.")

    log.info(f"Features (infer) : {df.shape[0]} lignes, {df.shape[1]} colonnes.")

    # 3) Chargement checkpoint RF
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {ckpt_path}")
    bundle = joblib.load(ckpt_path)

    model = bundle["model"]
    feat_cols = bundle["features"]
    label_map = bundle.get("label_map", None)  # dict texte -> int, si dispo

    # 4) Sélection des features comme au train
    missing_feats = [c for c in feat_cols if c not in df.columns]
    if missing_feats:
        raise ValueError(
            f"Colonnes de features manquantes dans features_csv : {missing_feats}"
        )

    X = df[feat_cols].values.astype(float)

    # 5) Prédictions
    log.info(f"Prédiction sur {X.shape[0]} échantillons avec {X.shape[1]} features.")
    y_pred = model.predict(X)

    # Probas si disponibles
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
    else:
        proba = None
        log.warning("Le modèle ne supporte pas predict_proba -> pas de colonnes de probabilité.")

    # 6) Construction DataFrame de sortie
    out = pd.DataFrame()
    out["patient_id"] = df["patient_id"]
    out["y_pred"] = y_pred

    # Si on a un label_map texte -> int, on peut reconstituer un y_pred_str
    inv_label_map = None
    if label_map is not None:
        inv_label_map = {v: k for k, v in label_map.items()}
        out["y_pred_str"] = out["y_pred"].map(inv_label_map)
        log.info(f"label_map utilisé pour y_pred_str : {label_map}")
    else:
        log.info("Pas de label_map dans le checkpoint -> pas de y_pred_str.")

    # Colonnes de probas
    if proba is not None:
        n_classes = proba.shape[1]
        # On tente d'utiliser les noms de classes du label_map si dispo
        if inv_label_map is not None and len(inv_label_map) == n_classes:
            # On ordonne les labels par id
            for cls_id in range(n_classes):
                cls_name = inv_label_map.get(cls_id, f"class_{cls_id}")
                col_name = f"p_{cls_name}"
                out[col_name] = proba[:, cls_id]
        else:
            # Fallback : p_class_0, p_class_1, ...
            for cls_id in range(n_classes):
                out[f"p_class_{cls_id}"] = proba[:, cls_id]

    # 7) Sauvegarde
    out.to_csv(out_csv, index=False)
    log.info(f"Prédictions sauvegardées dans {out_csv}")


if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    main(cfg)
