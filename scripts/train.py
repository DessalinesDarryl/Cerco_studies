#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py

Objectif :
  - Charger les features par patient (features.csv).
  - Charger les labels patients depuis l'Excel clinique.
  - Faire un split par patient (GroupKFold).
  - Entraîner un modèle (RandomForest) et évaluer (balanced accuracy, F1 macro).
  - Sauvegarder le modèle final + méta-données (features, label_map) dans un checkpoint .joblib
    compatible avec xai_attributions.py.

Config YAML attendue (ex: configs/eval/train_rf.yaml) :
  features_csv: data/processed/features/features.csv
  labels_excel: data/BDD_RBD_patients_updated.xlsx
  labels_sheet: "classification"
  labels_header_row: 3
  label_id_col: "identifiant"
  label_col: "groupe_diag"
  label_map:
    "CTRL": 0
    "RBD_idiopathique": 1
    "RBD_synucléinopathie": 2
    "Autre": 3

  out_dir: models
  ckpt_name: checkpoint_rf.joblib

  cv:
    n_splits: 5
    shuffle: true
    random_state: 42

  rf_params:
    n_estimators: 300
    max_depth: null
    n_jobs: -1
    random_state: 42
    class_weight: "balanced"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from src.utils.config import load_yaml, add_common_args
from src.utils.logging import get_logger


# ======================================================================
#  Utils chargement features / labels
# ======================================================================

def load_features(features_csv: Path, log):
    if not features_csv.exists():
        raise FileNotFoundError(f"features_csv introuvable : {features_csv}")
    df = pd.read_csv(features_csv)
    if "patient_id" not in df.columns:
        raise ValueError("La colonne 'patient_id' est absente de features.csv.")
    log.info(f"Features : {df.shape[0]} lignes, {df.shape[1] - 1} colonnes (incluant patient_id).")
    return df


def load_labels_from_excel(cfg, log) -> Tuple[pd.DataFrame, str, str, Optional[dict]]:
    """
    Retourne :
      - df_labels avec colonnes [label_id_col, label_col]
      - label_id_col
      - label_col
      - label_map_effective (dict texte->int si fourni, sinon mapping factorisé)
    """
    excel_path = Path(cfg["labels_excel"])
    sheet_name = cfg.get("labels_sheet", 0)
    label_id_col = cfg.get("label_id_col", "identifiant")
    label_col = cfg.get("label_col", "groupe_diag")
    header_row = cfg.get("labels_header_row", 0)

    if not excel_path.exists():
        raise FileNotFoundError(f"Fichier Excel labels introuvable : {excel_path}")

    log.info(f"Lecture Excel labels : {excel_path} (sheet={sheet_name}, header={header_row})")
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=header_row)

    for col in (label_id_col, label_col):
        if col not in df.columns:
            raise ValueError(
                f"Colonne '{col}' introuvable dans l'Excel. "
                f"Colonnes dispo : {list(df.columns)}"
            )

    df = df[[label_id_col, label_col]].copy()
    df = df.dropna(subset=[label_id_col, label_col])

    label_map_cfg = cfg.get("label_map", None)
    label_map_effective = None

    if label_map_cfg is not None:
        # mapping explicite texte -> int
        df[label_col] = df[label_col].astype(str).str.strip()
        map_norm = {str(k): int(v) for k, v in label_map_cfg.items()}
        before = df[label_col].unique()
        df[label_col] = df[label_col].map(map_norm)
        after = df[label_col].dropna().unique()
        log.info(f"Mapping labels (Excel -> int) : {before} -> {after}")
        df = df.dropna(subset=[label_col])
        df[label_col] = df[label_col].astype(int)
        label_map_effective = map_norm
    else:
        # factorisation automatique
        df[label_col], uniques = pd.factorize(df[label_col])
        label_map_effective = {str(u): int(i) for i, u in enumerate(uniques)}
        log.info(f"Labels factorisés automatiquement : {label_map_effective}")

    return df, label_id_col, label_col, label_map_effective


def merge_features_labels(df_feat, df_lab, label_id_col, label_col, log):
    merged = df_feat.merge(
        df_lab,
        left_on="patient_id",
        right_on=label_id_col,
        how="left",
    )

    n_pat_feat = df_feat["patient_id"].nunique()
    n_pat_lab = df_lab[label_id_col].nunique()
    n_pat_merged_lab = merged.dropna(subset=[label_col])["patient_id"].nunique()
    log.info(
        f"Merge features/labels : "
        f"{n_pat_feat} patients avec features, "
        f"{n_pat_lab} avec labels, "
        f"{n_pat_merged_lab} patients labellisés."
    )

    # On ne garde que les lignes avec label (les autres ne serviront pas à l'entraînement)
    merged = merged.dropna(subset=[label_col]).reset_index(drop=True)
    merged[label_col] = merged[label_col].astype(int)

    return merged


# ======================================================================
#  Modèle & CV
# ======================================================================

def build_model(cfg, log) -> RandomForestClassifier:
    rf_params = cfg.get("rf_params", {})
    log.info(f"RandomForest params : {rf_params}")
    model = RandomForestClassifier(**rf_params)
    return model


def train_with_cv(df, label_col, cfg, log):
    """
    df : DataFrame avec 'patient_id', features, et 'label_col'
    """
    all_cols = [c for c in df.columns if c not in ("patient_id", label_col)]
    feat_cols = all_cols

    X = df[feat_cols].values.astype(float)
    y = df[label_col].astype(int).values
    groups = df["patient_id"].values

    n_splits = cfg.get("cv", {}).get("n_splits", 5)
    do_group = cfg.get("cv", {}).get("group_by_patient", True)
    shuffle = cfg.get("cv", {}).get("shuffle", False)
    seed = cfg.get("cv", {}).get("random_state", 42)

    folds = []
    if do_group:
        log.info(f"CV : GroupKFold par patient (n_splits={n_splits})")
        gkf = GroupKFold(n_splits=n_splits)
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            folds.append((train_idx, val_idx))
    else:
        log.info(f"CV : StratifiedKFold (n_splits={n_splits}, shuffle={shuffle}, random_state={seed})")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        for train_idx, val_idx in skf.split(X, y):
            folds.append((train_idx, val_idx))

    metrics_rows = []

    for k, (tr, va) in enumerate(folds):
        log.info(f"Fold {k+1}/{len(folds)} : train={len(tr)} / val={len(va)}")
        model = build_model(cfg, log)
        model.fit(X[tr], y[tr])

        y_pred = model.predict(X[va])
        bal_acc = balanced_accuracy_score(y[va], y_pred)
        f1_macro = f1_score(y[va], y_pred, average="macro")

        log.info(f"  Fold {k+1} : BAC={bal_acc:.3f}, F1_macro={f1_macro:.3f}")
        metrics_rows.append({
            "fold": k + 1,
            "n_train": len(tr),
            "n_val": len(va),
            "balanced_accuracy": bal_acc,
            "f1_macro": f1_macro,
        })

    # Moyenne des folds
    df_metrics = pd.DataFrame(metrics_rows)
    if not df_metrics.empty:
        mean_row = {
            "fold": 0,
            "n_train": df_metrics["n_train"].mean(),
            "n_val": df_metrics["n_val"].mean(),
            "balanced_accuracy": df_metrics["balanced_accuracy"].mean(),
            "f1_macro": df_metrics["f1_macro"].mean(),
        }
        df_metrics = pd.concat([df_metrics, pd.DataFrame([mean_row])], ignore_index=True)
        log.info(
            f"CV global : BAC={mean_row['balanced_accuracy']:.3f}, "
            f"F1_macro={mean_row['f1_macro']:.3f}"
        )
    else:
        log.warning("Pas de métriques CV (df_metrics vide).")

    return feat_cols, df_metrics


def main(cfg):
    log = get_logger("train")

    features_csv = Path(cfg["features_csv"])
    out_dir = Path(cfg.get("out_dir", "models"))
    ckpt_name = cfg.get("ckpt_name", "checkpoint_rf.joblib")
    metrics_out = Path(cfg.get("metrics_out", "outputs/stats/train_metrics.csv"))

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1) Features ----
    df_feat = load_features(features_csv, log)

    # ---- 2) Labels Excel ----
    df_lab, label_id_col, label_col, label_map_effective = load_labels_from_excel(cfg, log)

    # ---- 3) Merge ----
    df_train = merge_features_labels(df_feat, df_lab, label_id_col, label_col, log)
    if df_train.empty:
        raise RuntimeError("Aucune ligne avec label après merge features/labels.")

    # ---- 4) CV + métriques ----
    feat_cols, df_metrics = train_with_cv(df_train, label_col, cfg, log)

    # ---- 5) Sauvegarde métriques ----
    df_metrics.to_csv(metrics_out, index=False)
    log.info(f"Métriques CV sauvegardées dans {metrics_out}")

    # ---- 6) Entraînement final sur tout le dataset ----
    X_full = df_train[feat_cols].values.astype(float)
    y_full = df_train[label_col].astype(int).values

    model = build_model(cfg, log)
    model.fit(X_full, y_full)
    log.info(f"Modèle final entraîné sur {len(y_full)} échantillons.")

    # ---- 7) Checkpoint pour XAI ----
    ckpt_path = out_dir / ckpt_name
    bundle = {
        "model": model,
        "features": feat_cols,
        "label_map": label_map_effective,
    }
    joblib.dump(bundle, ckpt_path)
    log.info(f"Checkpoint RF sauvegardé dans {ckpt_path}")


if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    main(cfg)
