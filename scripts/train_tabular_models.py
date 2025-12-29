#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_tabular_models.py

Objectif :
  - Charger les features par patient (dataset_final.csv).
  - Utiliser directement les labels déjà présents dans ce CSV
    (label_id et éventuellement label_str).
  - Faire un split par patient (GroupKFold).
  - Entraîner un modèle tabulaire (RandomForest, XGBoost, KNN).
  - Évaluer (balanced accuracy, F1 macro) en CV.
  - Sauvegarder le modèle final + méta-données (features, label_map) dans un checkpoint .joblib
    compatible avec xai_attributions.py si besoin.

Config YAML attendue (ex: configs/eval/train_rf.yaml, train_xgb.yaml, train_knn.yaml) :
  features_csv: data/processed/features/dataset_final.csv
  label_col: "label_id"

  model_type: "rf" | "xgb" | "knn"

  out_dir: models
  ckpt_name: checkpoint_rf.joblib
  metrics_out: outputs/stats/train_metrics_rf.csv

  cv:
    n_splits: 5
    group_by_patient: true
    shuffle: false
    random_state: 42

  rf_params:
    n_estimators: 300
    max_depth: null
    n_jobs: -1
    random_state: 42
    class_weight: "balanced"

  xgb_params:
    n_estimators: 400
    max_depth: 5
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8
    objective: "multi:softprob"
    n_jobs: -1
    reg_lambda: 1.0
    reg_alpha: 0.0

  knn_params:
    n_neighbors: 7
    weights: "distance"
    metric: "minkowski"
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from src.utils.config import load_yaml, add_common_args
from src.utils.logging import get_logger


# ======================================================================
#  Utils chargement features / labels
# ======================================================================

def load_features(features_csv: Path, log) -> pd.DataFrame:
    """
    Lit le CSV de features (dataset_final.csv).
    Attend au minimum une colonne 'patient_id' et une colonne de label
    (par ex. 'label_id' ou 'label_str', cf. cfg['label_col']).
    """
    if not features_csv.exists():
        raise FileNotFoundError(f"features_csv introuvable : {features_csv}")
    df = pd.read_csv(features_csv)
    if "patient_id" not in df.columns:
        raise ValueError("La colonne 'patient_id' est absente de features_csv.")
    log.info(f"Features : {df.shape[0]} lignes, {df.shape[1] - 1} colonnes (incluant patient_id).")
    return df


def prepare_train_dataframe(
    df_feat: pd.DataFrame,
    cfg: dict,
    log,
) -> Tuple[pd.DataFrame, str, Optional[Dict[str, int]]]:
    """
    À partir de df_feat (dataset_final.csv), construit df_train avec :
      - une colonne 'patient_id'
      - une colonne de label (cfg['label_col'])
      - toutes les colonnes de features numériques.

    Retourne :
      - df_train (lignes avec label non NaN)
      - label_col (nom de la colonne label utilisée)
      - label_map_effective (dict texte->int si info dispo)
    """
    label_col = cfg.get("label_col", "label_id")
    if label_col not in df_feat.columns:
        raise ValueError(
            f"Colonne label '{label_col}' absente de features_csv. "
            f"Colonnes disponibles : {list(df_feat.columns)}"
        )

    df = df_feat.copy()
    n_pat_total = df["patient_id"].nunique()

    before_rows = df.shape[0]
    df = df.dropna(subset=[label_col])
    after_rows = df.shape[0]

    log.info(
        f"Filtrage lignes sans label ({label_col}) : {before_rows} -> {after_rows} lignes."
    )

    # Si on utilise un label numérique (recommandé : label_id)
    if np.issubdtype(df[label_col].dtype, np.number):
        df[label_col] = df[label_col].astype(int)
    else:
        # Si c'est une string (type label_str), factorisation automatique
        df[label_col], uniques = pd.factorize(df[label_col])
        df[label_col] = df[label_col].astype(int)
        log.info(f"Labels factorisés automatiquement depuis '{label_col}' : {dict(enumerate(uniques))}")

    n_pat_labeled = df["patient_id"].nunique()
    log.info(
        f"Patients avec features : {n_pat_total}, "
        f"patients avec label ({label_col}) : {n_pat_labeled}."
    )

    # Construire un label_map_effective pour le checkpoint XAI
    label_map_effective = None
    if "label_str" in df_feat.columns and "label_id" in df_feat.columns:
        pairs = (
            df_feat[["label_str", "label_id"]]
            .dropna()
            .drop_duplicates()
        )
        try:
            pairs["label_id"] = pairs["label_id"].astype(int)
        except Exception:
            pass
        label_map_effective = dict(zip(pairs["label_str"], pairs["label_id"]))
        log.info(f"label_map_effective construit depuis dataset_final : {label_map_effective}")
    else:
        # Sinon, on regarde si un label_map est fourni dans la config
        label_map_cfg = cfg.get("label_map", None)
        if label_map_cfg is not None:
            label_map_effective = {str(k): int(v) for k, v in label_map_cfg.items()}
            log.info(f"label_map_effective pris depuis la config : {label_map_effective}")
        else:
            log.info("Aucun label_map explicite disponible (ni label_str/label_id, ni config).")

    return df, label_col, label_map_effective


# ======================================================================
#  Modèles tabulaires
# ======================================================================

def build_model(cfg, log):
    """
    Construit un modèle en fonction de cfg['model_type']:
      - 'rf'  : RandomForestClassifier
      - 'xgb' : XGBoost (XGBClassifier)
      - 'knn' : KNeighborsClassifier
    """
    model_type = cfg.get("model_type", "rf").lower()

    if model_type == "rf":
        rf_params = cfg.get("rf_params", {})
        log.info(f"Modèle = RandomForest, params : {rf_params}")
        return RandomForestClassifier(**rf_params)

    elif model_type == "xgb":
        from xgboost import XGBClassifier
        xgb_params = cfg.get("xgb_params", {})
        # Ajout de quelques valeurs par défaut raisonnables
        xgb_default = dict(
            objective="multi:softprob",
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
        )
        xgb_default.update(xgb_params)
        log.info(f"Modèle = XGBoost, params : {xgb_default}")
        return XGBClassifier(**xgb_default)

    elif model_type == "knn":
        knn_params = cfg.get("knn_params", {})
        knn_default = dict(
            n_neighbors=7,
            weights="distance",
            metric="minkowski",
        )
        knn_default.update(knn_params)
        log.info(f"Modèle = KNN, params : {knn_default}")
        return KNeighborsClassifier(**knn_default)

    else:
        raise ValueError(f"model_type inconnu : {model_type}. Attendu : 'rf' | 'xgb' | 'knn'.")


def train_with_cv(df, label_col, cfg, log):
    """
    df : DataFrame avec 'patient_id', features, et 'label_col'
    Retourne :
      - feat_cols : liste des colonnes utilisées comme features
      - df_metrics : métriques CV par fold + moyenne
    """
    # On retire les colonnes non-features, et on ne garde que les colonnes numériques
    all_cols = [c for c in df.columns if c not in ("patient_id", label_col, "label_str")]
    feat_cols = [c for c in all_cols if np.issubdtype(df[c].dtype, np.number)]

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

        from sklearn.impute import SimpleImputer

        if cfg.get("model_type") == "knn":
            imputer = SimpleImputer(strategy="median")
            X_tr = imputer.fit_transform(X[tr])
            X_va = imputer.transform(X[va])
        else:
            X_tr = X[tr]
            X_va = X[va]

        model.fit(X_tr, y[tr])
        y_va = y[va]
        y_pred = model.predict(X_va)
        bal_acc = balanced_accuracy_score(y_va, y_pred)
        f1_macro = f1_score(y_va, y_pred, average="macro")
        recall_macro = recall_score(y_va, y_pred, average="macro")

        log.info(f"  Fold {k+1} : BAC={bal_acc:.3f}, F1_macro={f1_macro:.3f}")
        metrics_rows.append({
            "fold": k + 1,
            "n_train": len(tr),
            "n_val": len(va),
            "balanced_accuracy": bal_acc,
            "f1_macro": f1_macro,
            "recall_macro": recall_macro,
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
            "recall_macro": df_metrics["recall_macro"].mean(),
        }
        df_metrics = pd.concat([df_metrics, pd.DataFrame([mean_row])], ignore_index=True)
        log.info(
            f"CV global : BAC={mean_row['balanced_accuracy']:.3f}, "
            f"F1_macro={mean_row['f1_macro']:.3f}, "
            f"Recall_macro={mean_row['recall_macro']:.3f}"
        )
    else:
        log.warning("Pas de métriques CV (df_metrics vide).")

    return feat_cols, df_metrics


# ======================================================================
#  Main
# ======================================================================

def main(cfg):
    log = get_logger("train_tabular")

    features_csv = Path(cfg["features_csv"])
    out_dir = Path(cfg.get("out_dir", "models"))
    ckpt_name = cfg.get("ckpt_name", "checkpoint_tabular.joblib")
    metrics_out = Path(cfg.get("metrics_out", "outputs/stats/train_metrics_tabular.csv"))

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1) Features (incluant labels) ----
    df_feat = load_features(features_csv, log)

    # ---- 2) Préparation df_train à partir du CSV (plus d'Excel) ----
    df_train, label_col, label_map_effective = prepare_train_dataframe(df_feat, cfg, log)
    if df_train.empty:
        raise RuntimeError("Aucune ligne avec label après filtrage des features.")

    # ---- 3) CV + métriques ----
    feat_cols, df_metrics = train_with_cv(df_train, label_col, cfg, log)

    # ---- 4) Sauvegarde métriques ----
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(metrics_out, index=False)
    log.info(f"Métriques CV sauvegardées dans {metrics_out}")

    # ---- 5) Entraînement final sur tout le dataset ----
    X_full = df_train[feat_cols].values.astype(float)
    y_full = df_train[label_col].astype(int).values

    model = build_model(cfg, log)
    if cfg.get("model_type") == "knn":
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="median")
        X_full = imputer.fit_transform(X_full)

    model.fit(X_full, y_full)
    log.info(f"Modèle final entraîné sur {len(y_full)} échantillons.")

    # ---- 6) Checkpoint pour XAI / inference ----
    ckpt_path = out_dir / ckpt_name
    bundle = {
        "model": model,
        "features": feat_cols,
        "label_map": label_map_effective,
    }
    joblib.dump(bundle, ckpt_path)
    log.info(f"Checkpoint modèle sauvegardé dans {ckpt_path}")


if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    main(cfg)
