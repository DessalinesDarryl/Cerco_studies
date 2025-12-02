#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
xai_attributions.py

XAI global sur ton modèle classique (RF / XGBoost, etc.) :

- Lit le dataset complet (dataset_final.csv) qui contient déjà :
    * patient_id
    * features EEG/EMG
    * label_id / label_str
- Charge le checkpoint RF (bundle joblib : model + features + label_map)
- Calcule :
    * Permutation importance (sklearn)
    * SHAP global (TreeExplainer si modèle d'arbres, sinon KernelExplainer)

Config YAML attendue (ex: configs/xai/attributions.yaml) :
  model_path: models/checkpoint_rf.joblib
  features_csv: data/processed/features/dataset_final.csv
  label_col: "label_id"      # ou "label_str"

  out_dir: outputs/xai/attributions

  # options XAI :
  do_permutation: true
  n_perm_repeats: 30
  random_state: 42

  max_samples: 2000
  n_background: 200
  nsamples_kernel: 200
  top_k_features: 20
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

from sklearn.inspection import permutation_importance
import shap  # à mettre dans requirements

from src.utils.config import load_yaml, add_common_args
from src.utils.logging import get_logger


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _is_tree_model(model) -> bool:
    """
    Heuristique simple pour savoir si on peut utiliser TreeExplainer.
    """
    name = model.__class__.__name__.lower()
    module = model.__class__.__module__.lower()
    tree_keywords = ["forest", "xgb", "xgboost", "gradientboost", "extratrees", "gbm"]
    return any(k in name or k in module for k in tree_keywords)


def load_dataset_from_cfg(cfg, bundle, log):
    """
    Charge dataset_final.csv et reconstruit X, y en respectant les
    mêmes colonnes de features que lors de l'entraînement.

    Retourne :
      - X (ndarray)
      - y (ndarray)
      - feature_names (list[str])
      - df complet (pour debug éventuel)
    """
    feat_path = Path(cfg["features_csv"])
    if not feat_path.exists():
        raise FileNotFoundError(f"features_csv introuvable: {feat_path}")

    log.info(f"Lecture features (dataset_final): {feat_path}")
    df = pd.read_csv(feat_path)

    if "patient_id" not in df.columns:
        raise KeyError("Colonne 'patient_id' absente de dataset_final.csv")

    label_col = cfg.get("label_col", "label_id")
    if label_col not in df.columns:
        raise KeyError(
            f"Colonne de label '{label_col}' absente de dataset_final.csv. "
            f"Colonnes dispo: {list(df.columns)}"
        )

    # On enlève les lignes sans label
    before = df.shape[0]
    df = df.dropna(subset=[label_col])
    after = df.shape[0]
    log.info(f"Lignes avec label ({label_col}) : {before} -> {after}")

    # y : on force en int si possible
    if np.issubdtype(df[label_col].dtype, np.number):
        df[label_col] = df[label_col].astype(int)
        y = df[label_col].to_numpy(dtype=int)
    else:
        # factorisation si texte
        df[label_col], uniques = pd.factorize(df[label_col].astype(str))
        y = df[label_col].to_numpy(dtype=int)
        log.info(f"Labels factorisés automatiquement : {dict(enumerate(uniques))}")

    # Récupérer les features utilisées à l'entraînement
    feat_cols = bundle["features"]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Colonnes de features manquantes dans dataset_final : {missing}"
        )

    # On ne garde que les colonnes numériques de ces features
    df_feat_only = df[feat_cols].select_dtypes(include=[np.number])
    feature_names = list(df_feat_only.columns)

    X = df_feat_only.to_numpy(dtype=float)
    log.info(f"Shape X: {X.shape}, y: {y.shape}, nb_features: {len(feature_names)}")

    return X, y, feature_names, df


def compute_permutation_importance(model, X, y, feature_names, cfg, log, out_dir: Path):
    if not cfg.get("do_permutation", True):
        log.info("Permutation importance désactivée dans la config.")
        return

    n_repeats = int(cfg.get("n_perm_repeats", 30))
    random_state = int(cfg.get("random_state", 42))

    log.info(f"Permutation importance (n_repeats={n_repeats}, random_state={random_state})...")
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    df_perm = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    out_path = out_dir / "permutation_importance.csv"
    df_perm.to_csv(out_path, index=False)
    log.info(f"Permutation importance sauvegardée → {out_path}")


def compute_shap_global(model, X, feature_names, cfg, log, out_dir: Path):
    max_samples = int(cfg.get("max_samples", 2000))
    n_background = int(cfg.get("n_background", 200))
    nsamples_kernel = int(cfg.get("nsamples_kernel", 200))

    n_samples = X.shape[0]
    if n_samples > max_samples:
        log.info(f"Sous-échantillonnage SHAP: {n_samples} → {max_samples} échantillons")
        rng = np.random.RandomState(cfg.get("random_state", 42))
        idx = rng.choice(n_samples, size=max_samples, replace=False)
        X_use = X[idx]
    else:
        X_use = X

    log.info(f"SHAP sur {X_use.shape[0]} échantillons, {X_use.shape[1]} features")

    try:
        if _is_tree_model(model):
            log.info("Modèle détecté comme arbre → TreeExplainer")
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_use)
        else:
            log.info("Modèle non-arbre → KernelExplainer (peut être lent)")
            # arrière-plan
            if X_use.shape[0] > n_background:
                background = shap.sample(X_use, n_background, random_state=cfg.get("random_state", 42))
            else:
                background = X_use
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_vals = explainer.shap_values(X_use, nsamples=nsamples_kernel)
    except Exception as e:
        log.warning(f"Échec SHAP ({e}) → pas de shap_global_meanabs.csv")
        return

    # Gestion des formats SHAP (multiclass ou non)
    if isinstance(shap_vals, list):
        # liste de (n_samples, n_features) par classe
        shap_arr = np.stack(shap_vals, axis=-1)  # (n_classes, n_samples, n_features)
        shap_arr = np.transpose(shap_arr, (1, 2, 0))  # (n_samples, n_features, n_classes)
    else:
        shap_arr = np.array(shap_vals)

    if shap_arr.ndim == 3:
        # moyenne |SHAP| sur samples et classes
        mean_abs = np.mean(np.abs(shap_arr), axis=(0, 2))
    elif shap_arr.ndim == 2:
        mean_abs = np.mean(np.abs(shap_arr), axis=0)
    else:
        log.warning(f"Format SHAP inattendu: shape={shap_arr.shape}")
        return

    df_shap = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)

    out_path = out_dir / "shap_global_meanabs.csv"
    df_shap.to_csv(out_path, index=False)
    log.info(f"SHAP global sauvegardé → {out_path}")

    # fichier texte simple pour ton rapport
    top_k = int(cfg.get("top_k_features", 10))
    top = df_shap.head(top_k)
    txt_path = out_dir / "shap_topk.txt"
    with txt_path.open("w") as f:
        f.write(f"Top {top_k} features (SHAP mean |value|):\n")
        for _, row in top.iterrows():
            f.write(f"- {row['feature']}: {row['mean_abs_shap']:.4g}\n")
    log.info(f"Top-{top_k} features SHAP → {txt_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(cfg):
    log = get_logger("xai_attributions")

    out_dir = Path(cfg.get("out_dir", "outputs/xai/attributions"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Chargement checkpoint RF (bundle)
    model_path = Path(cfg["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"model_path introuvable: {model_path}")
    log.info(f"Chargement checkpoint RF : {model_path}")
    bundle = joblib.load(model_path)

    model = bundle["model"]

    # 2) Dataset (dataset_final + features du bundle)
    X, y, feature_names, df_merge = load_dataset_from_cfg(cfg, bundle, log)

    # 3) Permutation importance
    compute_permutation_importance(model, X, y, feature_names, cfg, log, out_dir)

    # 4) SHAP global
    compute_shap_global(model, X, feature_names, cfg, log, out_dir)

    log.info("XAI (attributions globales) terminé.")


if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    main(cfg)
