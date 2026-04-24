#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py

Objectif :
  - Charger les features par patient (dataset_final.csv).
  - Utiliser directement les labels déjà présents dans ce CSV
    (label_id et éventuellement label_str).
  - Faire un split par patient (GroupKFold).
  - Entraîner un modèle (RandomForest) et évaluer (balanced accuracy, F1 macro).
  - Sauvegarder le modèle final + méta-données (features, label_map) dans un checkpoint .joblib
    compatible avec xai_attributions.py.

Config YAML attendue (ex: configs/eval/train_rf.yaml) :
  features_csv: data/processed/features/dataset_final.csv

  # Colonne de label à utiliser pour l'entraînement
  label_col: "label_id"   # ou "label_str" si tu veux utiliser les labels texte

  # (optionnel) mapping texte -> int si tu veux l'imposer,
  # mais en pratique on va le déduire depuis label_str/label_id dans le CSV.
  # label_map:
  #   "SYN": 0
  #   "Narco": 1
  #   "TCSPi": 2
  #   "EAI": 3

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
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.model_selection import GroupKFold, StratifiedKFold

from src.utils.config import load_yaml, add_common_args
from src.utils.logging import get_logger


# ======================================================================
# Chargement des features et préparation des labels
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
    cfg,
    log,
) -> Tuple[pd.DataFrame, str, Optional[Dict[str, int]]]:
    """
    À partir de df_feat (dataset_final.csv), construit df_train avec :
      - une colonne 'patient_id'
      - une colonne de label (cfg['label_col'])
      - toutes les colonnes de features.

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

    # 1) Suppression des lignes sans label exploitable
    before_rows = df.shape[0]
    df = df.dropna(subset=[label_col])
    after_rows = df.shape[0]

    log.info(
        f"Filtrage lignes sans label ({label_col}) : {before_rows} -> {after_rows} lignes."
    )

    # 2) Construction d’un label numérique utilisable pour l’entraînement
    if np.issubdtype(df[label_col].dtype, np.number):
        df[label_col] = df[label_col].astype(int)
    else:
        # Si c'est une string (type label_str), on fait une factorisation automatique
        df[label_col], uniques = pd.factorize(df[label_col])
        df[label_col] = df[label_col].astype(int)
        log.info(f"Labels factorisés automatiquement depuis '{label_col}' : {dict(enumerate(uniques))}")

    n_pat_labeled = df["patient_id"].nunique()
    log.info(
        f"Patients avec features : {n_pat_total}, "
        f"patients avec label ({label_col}) : {n_pat_labeled}."
    )

    # 3) Construction d’un label_map utile pour l’inférence/XAI
    label_map_effective = None

    # Cas idéal : le dataset contient déjà label_str et label_id
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
        # Sinon, on essaie de récupérer un mapping dans la config YAML
        label_map_cfg = cfg.get("label_map", None)
        if label_map_cfg is not None:
            label_map_effective = {str(k): int(v) for k, v in label_map_cfg.items()}
            log.info(f"label_map_effective pris depuis la config : {label_map_effective}")
        else:
            log.info("Aucun label_map explicite disponible (ni label_str/label_id, ni config).")

    return df, label_col, label_map_effective


# ======================================================================
# Construction du modèle et validation croisée
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
    # 1) Sélection des colonnes de features numériques
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
    y_true_all = []
    y_pred_all = []

    for k, (tr, va) in enumerate(folds):
        log.info(f"Fold {k+1}/{len(folds)} : train={len(tr)} / val={len(va)}")
        model = build_model(cfg, log)
        model.fit(X[tr], y[tr])

        y_va = y[va]
        y_pred = model.predict(X[va])

        # On stocke pour la matrice de confusion globale
        y_true_all.append(y_va)
        y_pred_all.append(y_pred)

        # Calcul des métriques globales
        bal_acc = balanced_accuracy_score(y_va, y_pred)
        f1_macro = f1_score(y_va, y_pred, average='macro', zero_division=0)

        # Scores par classe
        prec_per_class = precision_score(y_va, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_va, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_va, y_pred, average=None, zero_division=0)

        # On stocke dans les logs
        for i, (p, r, f) in enumerate(zip(prec_per_class, recall_per_class, f1_per_class)):
            log.info(f"    Classe {i} : Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}")

        log.info(f"  Fold {k+1} : BAC={bal_acc:.3f}, F1_macro={f1_macro:.3f}")
        metrics_rows.append({
            "fold": k + 1,
            "n_train": len(tr),
            "n_val": len(va),
            "balanced_accuracy": bal_acc,
            "f1_macro": f1_macro,
            **{f"precision_class_{i}": prec_per_class[i] for i in range(len(prec_per_class))},
            **{f"recall_class_{i}": recall_per_class[i] for i in range(len(recall_per_class))},
            **{f"f1_class_{i}": f1_per_class[i] for i in range(len(f1_per_class))}
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

    # Matrice de confusion globale (tous folds confondus)
    if y_true_all:
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        labels_sorted = np.unique(y_true_all)
        cm = confusion_matrix(y_true_all, y_pred_all, labels=labels_sorted)
        df_cm = pd.DataFrame(
            cm,
            index=[f"true_{l}" for l in labels_sorted],
            columns=[f"pred_{l}" for l in labels_sorted],
        )
    else:
        df_cm = pd.DataFrame()

    return feat_cols, df_metrics, df_cm


def main(cfg):
    log = get_logger("train")

    features_csv = Path(cfg["features_csv"])
    out_dir = Path(cfg.get("out_dir", "models"))
    ckpt_name = cfg.get("ckpt_name", "checkpoint_rf.joblib")
    metrics_out = Path(cfg.get("metrics_out", "outputs/stats/train_metrics_rf.csv"))

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1) Features (incluant labels) ----
    df_feat = load_features(features_csv, log)

    # ---- 2) Préparation df_train à partir du CSV (plus d'Excel) ----
    df_train, label_col, label_map_effective = prepare_train_dataframe(df_feat, cfg, log)
    if df_train.empty:
        raise RuntimeError("Aucune ligne avec label après filtrage des features.")

    # ---- 3) CV + métriques + matrice de confusion ----
    feat_cols, df_metrics, df_cm = train_with_cv(df_train, label_col, cfg, log)

    # ---- 4) Sauvegarde métriques ----
    df_metrics.to_csv(metrics_out, index=False)
    log.info(f"Métriques CV sauvegardées dans {metrics_out}")

    # Sauvegarde matrice de confusion globale
    cm_out = Path(str(metrics_out).replace(".csv", "_confusion.csv"))
    if not df_cm.empty:
        df_cm.to_csv(cm_out, index=True)
        log.info(f"Matrice de confusion globale sauvegardée dans {cm_out}")
    else:
        log.warning("Matrice de confusion vide : pas de sauvegarde.")

    # ---- 4 bis) Figure : heatmap de la matrice de confusion ----
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig_out = Path("outputs/figures/RF_confusion.png")
        fig_out.parent.mkdir(parents=True, exist_ok=True)

        # Inverser le label_map pour avoir un mapping id -> nom
        if label_map_effective:
            id_to_label = {v: k for k, v in label_map_effective.items()}
        else:
            # Fallback si pas de label_map
            id_to_label = None

        # Récupérer les labels des axes (ex: ["true_0", "true_1", ...])
        # et extraire les ids numériques
        if id_to_label:
            true_labels = [id_to_label.get(int(idx.split('_')[1]), idx.split('_')[1]) 
                          for idx in df_cm.index]
            pred_labels = [id_to_label.get(int(col.split('_')[1]), col.split('_')[1]) 
                          for col in df_cm.columns]
        else:
            true_labels = df_cm.index
            pred_labels = df_cm.columns

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            df_cm,
            annot=True,
            fmt="d",  # Format entier pour afficher le nombre exact
            cmap="Blues",
            cbar=True,
            xticklabels=pred_labels,
            yticklabels=true_labels
        )

        plt.title("Matrice de confusion – Validation croisée (nombre de patients)")
        plt.ylabel("Classe réelle")
        plt.xlabel("Classe prédite")

        plt.tight_layout()
        plt.savefig(fig_out, dpi=150)
        plt.close()

        log.info(f"Figure matrice de confusion sauvegardée dans {fig_out}")

    except Exception as e:
        log.error(f"Échec génération de la figure matrice de confusion : {e}")


    # ---- 5) Entraînement final sur tout le dataset ----
    X_full = df_train[feat_cols].values.astype(float)
    y_full = df_train[label_col].astype(int).values

    model = build_model(cfg, log)
    model.fit(X_full, y_full)
    log.info(f"Modèle final entraîné sur {len(y_full)} échantillons.")

    # ---- 6) Checkpoint pour XAI ----
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
