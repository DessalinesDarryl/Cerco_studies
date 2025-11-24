#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
xai_attributions.py

XAI global sur ton modèle classique (RF / SVM / XGBoost) :

- Lit les features par époque (features.csv)
- Aligne avec les labels patients depuis l'Excel
- Charge le modèle entraîné (joblib)
- Calcule :
    * Permutation importance (sklearn)
    * SHAP global (TreeExplainer si modèle d'arbres, sinon KernelExplainer)

Sorties dans out_dir :
    - permutation_importance.csv
    - shap_global_meanabs.csv
    - shap_topk.txt (liste concise pour ton rapport)
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.inspection import permutation_importance

import shap  # pense à l'ajouter dans requirements

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


def load_dataset_from_cfg(cfg, log):
    """
    Charge features.csv + Excel labels, aligne sur patient_id et
    retourne X (ndarray), y (ndarray), feature_names (list) et df complet.
    """
    feat_path = Path(cfg["features_csv"])
    if not feat_path.exists():
        raise FileNotFoundError(f"features_csv introuvable: {feat_path}")

    log.info(f"Lecture features: {feat_path}")
    df_feat = pd.read_csv(feat_path)

    # colonnes d'identifiant
    feat_id_col = cfg.get("features_id_col", "patient_id")
    if feat_id_col not in df_feat.columns:
        raise KeyError(f"Colonne '{feat_id_col}' absente de features.csv")

    # Excel labels
    labels_excel = Path(cfg["labels_excel"])
    if not labels_excel.exists():
        raise FileNotFoundError(f"labels_excel introuvable: {labels_excel}")

    labels_sheet = cfg.get("labels_sheet", 0)
    header_row = cfg.get("labels_header_row", 0)
    id_col = cfg["label_id_col"]
    label_col = cfg["label_col"]

    log.info(f"Lecture labels Excel: {labels_excel} (sheet={labels_sheet}, header_row={header_row})")
    df_lab = pd.read_excel(
        labels_excel,
        sheet_name=labels_sheet,
        header=header_row,
    )

    if id_col not in df_lab.columns or label_col not in df_lab.columns:
        raise KeyError(
            f"Colonnes '{id_col}' ou '{label_col}' non trouvées dans {labels_excel}. "
            f"Colonnes dispo: {list(df_lab.columns)}"
        )

    # On harmonise en string pour la jointure patient
    df_feat["_pid"] = df_feat[feat_id_col].astype(str).str.strip()
    df_lab["_pid"] = df_lab[id_col].astype(str).str.strip()

    df_merge = df_feat.merge(
        df_lab[["_pid", label_col]],
        on="_pid",
        how="inner",
        suffixes=("", "_lab"),
    )

    log.info(f"Époques fusionnées (features ∩ labels): {len(df_merge)}/{len(df_feat)}")

    if len(df_merge) == 0:
        raise RuntimeError("Fusion features/labels vide : vérifier identifiants patients.")

    # mapping texte -> int
    label_map = cfg.get("label_map", None)
    if label_map:
        df_merge["y"] = df_merge[label_col].map(label_map)
    else:
        # factorisation automatique
        classes, y = np.unique(df_merge[label_col].astype(str), return_inverse=True)
        log.info(f"Label_map généré automatiquement: {dict(enumerate(classes))}")
        df_merge["y"] = y

    # On enlève les colonnes non-features
    drop_cols = ["_pid", feat_id_col, label_col, "y"]
    non_feature_cols = [c for c in drop_cols if c in df_merge.columns]
    feature_cols = [c for c in df_merge.columns if c not in non_feature_cols]

    # Garder uniquement les colonnes numériques pour X
    df_feat_only = df_merge[feature_cols].select_dtypes(include=[np.number])
    feature_names = list(df_feat_only.columns)

    X = df_feat_only.to_numpy(dtype=float)
    y = df_merge["y"].to_numpy(dtype=int)

    log.info(f"Shape X: {X.shape}, y: {y.shape}, nb_features: {len(feature_names)}")

    return X, y, feature_names, df_merge


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
        log.info(f"Sous-échantillonnage SHAP: {n_samples} → {max_samples} epochs")
        idx = np.random.RandomState(cfg.get("random_state", 42)).choice(
            n_samples, size=max_samples, replace=False
        )
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

    # 1) Dataset
    X, y, feature_names, df_merge = load_dataset_from_cfg(cfg, log)

    # 2) Modèle
    model_path = Path(cfg["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"model_path introuvable: {model_path}")
    log.info(f"Chargement modèle: {model_path}")
    model = joblib.load(model_path)

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
