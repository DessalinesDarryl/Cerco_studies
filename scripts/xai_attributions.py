#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
xai_attributions.py
===================

Objectif
--------
Réaliser une analyse XAI **globale** (au niveau du dataset) pour un modèle classique
(RandomForest / XGBoost / GradientBoosting / ExtraTrees / etc.) entraîné sur ton
dataset final `dataset_final.csv`.

Le script calcule deux familles d'explications :

1) Permutation Importance (sklearn)
   - Mesure l'importance d'une feature via la baisse de performance lorsque la feature est permutée.
   - Interprétation : plus la performance chute quand on casse la feature, plus elle est importante.
   - Avantage : simple, model-agnostic.
   - Limite : sensible aux corrélations entre features (les features redondantes peuvent paraître moins importantes).

2) SHAP Global (SHapley Additive exPlanations)
   - Calcule des contributions (valeurs de Shapley) par feature et par échantillon.
   - Ici on en extrait une version *globale* : moyenne de |SHAP| (mean_abs_shap) par feature.
   - Choix de l'explainer :
       * TreeExplainer si modèle d'arbres (rapide et fiable)
       * KernelExplainer sinon (générique mais souvent lent)

Entrées
-------
- Dataset complet (CSV) : `dataset_final.csv`
    Colonnes attendues :
      * patient_id
      * features EEG/EMG (numériques)
      * label_id / label_str (ou autre colonne définie par label_col)

- Checkpoint modèle (joblib) : un "bundle" contenant au minimum :
      bundle["model"]    -> modèle sklearn entraîné
      bundle["features"] -> liste des colonnes utilisées pendant l'entraînement
    Optionnel :
      bundle["label_map"] / mapping classes (pas requis ici)

Sorties
-------
Dans `out_dir` :
- permutation_importance.csv
    Colonnes : feature, importance_mean, importance_std

- shap_global_meanabs.csv
    Colonnes : feature, mean_abs_shap

- shap_topk.txt
    Liste texte des top-k features selon mean_abs_shap, pratique pour un rapport.

Configuration YAML attendue
---------------------------
Exemple (configs/xai/attributions.yaml) :

  model_path: models/checkpoint_rf.joblib
  features_csv: data/processed/features/dataset_final.csv
  label_col: "label_id"      # ou "label_str"

  out_dir: outputs/xai/attributions

  # options XAI :
  do_permutation: true
  n_perm_repeats: 30
  random_state: 42

  # options SHAP :
  max_samples: 2000          # sous-échantillonnage pour limiter le coût SHAP
  n_background: 200          # background pour KernelExplainer
  nsamples_kernel: 200       # nb d'évaluations approx KernelExplainer
  top_k_features: 20

Notes importantes / pièges fréquents
------------------------------------
- Alignement features : on utilise STRICTEMENT bundle["features"] pour reconstruire X.
  => si dataset_final.csv a changé, tu détecteras des colonnes manquantes.

- Colonnes non numériques :
  Le script sélectionne uniquement les colonnes numériques parmi bundle["features"].
  => si tu as des features encodées en string, elles seront exclues (et donc X changera).
  Recommandation : garder tes features en float/int dans dataset_final.csv.

- SHAP multiclasses :
  shap_values peut être une liste (une matrice par classe) ou une matrice (binaire).
  On agrège en moyenne de |SHAP| sur (samples, classes) lorsque pertinent.

Exécution
---------
python scripts/xai_attributions.py --config configs/xai/attributions.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.inspection import permutation_importance
import shap  # doit être dans requirements

from src.utils.config import load_yaml, add_common_args
from src.utils.logging import get_logger


# ------------------------------------------------------------------
# Fonctions utilitaires pour la sélection de l’explainer et du dataset
# ------------------------------------------------------------------

def _is_tree_model(model: Any) -> bool:
    """
    Détecte (heuristique) si un modèle est probablement un modèle d'arbres
    (et donc compatible avec SHAP TreeExplainer).

    Pourquoi ?
    ---------
    - TreeExplainer est très efficace/rapide sur les modèles d'arbres (RF, XGB, GBDT).
    - KernelExplainer est générique mais peut être très lent.

    Méthode
    -------
    On inspecte :
      - model.__class__.__name__
      - model.__class__.__module__
    et on cherche des mots-clés typiques.

    Paramètres
    ----------
    model : Any
        Modèle scikit-learn ou assimilé.

    Retour
    ------
    bool
        True si on suppose que c'est un modèle d'arbres, False sinon.

    Limites
    -------
    - Ce n'est pas une preuve formelle. Certains wrappers peuvent passer à travers.
    - Si tu utilises un modèle exotique, adapte tree_keywords.
    """
    name = model.__class__.__name__.lower()
    module = model.__class__.__module__.lower()
    tree_keywords = ["forest", "xgb", "xgboost", "gradientboost", "extratrees", "gbm"]
    return any(k in name or k in module for k in tree_keywords)


def load_dataset_from_cfg(
    cfg: dict,
    bundle: dict,
    log,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Charge `dataset_final.csv` et reconstruit (X, y) en respectant EXACTEMENT
    les features utilisées lors de l'entraînement (bundle["features"]).

    Étapes
    ------
    1) Lecture du CSV `features_csv` depuis la config.
    2) Vérification présence de 'patient_id' et de la colonne de label (cfg['label_col']).
    3) Suppression des lignes sans label.
    4) Construction de y :
       - si label numérique => cast int
       - sinon factorize (label texte)
    5) Vérification que toutes les features attendues (bundle["features"]) existent.
    6) Sélection des colonnes numériques parmi ces features
       (sécurité si certaines colonnes sont non-numériques dans le CSV).
    7) Construction de X (float).

    Paramètres
    ----------
    cfg : dict
        Configuration YAML. Champs requis :
          - features_csv : chemin vers dataset_final.csv
          - label_col (optionnel) : défaut "label_id"
    bundle : dict
        Checkpoint joblib chargé. Doit contenir :
          - "features" : list[str] (colonnes utilisées pendant l'entraînement)
    log : logger

    Retours
    -------
    X : np.ndarray
        Matrice des features (n_samples, n_features_numeric).
    y : np.ndarray
        Vecteur labels (n_samples,).
    feature_names : list[str]
        Liste des noms de colonnes réellement utilisées dans X (numériques).
    df : pd.DataFrame
        DataFrame complet filtré (utile pour debug).

    Raises
    ------
    FileNotFoundError
        Si features_csv est introuvable.
    KeyError
        Si patient_id ou label_col est absent, ou si des features manquent.
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
    before = int(df.shape[0])
    df = df.dropna(subset=[label_col])
    after = int(df.shape[0])
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
    if "features" not in bundle:
        raise KeyError("Le bundle joblib ne contient pas la clé 'features' (liste des colonnes utilisées).")

    feat_cols = list(bundle["features"])
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Colonnes de features manquantes dans dataset_final : {missing}")

    # On ne garde que les colonnes numériques de ces features
    df_feat_only = df[feat_cols].select_dtypes(include=[np.number])
    feature_names = list(df_feat_only.columns)

    X = df_feat_only.to_numpy(dtype=float)
    log.info(f"Shape X: {X.shape}, y: {y.shape}, nb_features: {len(feature_names)}")

    if len(feature_names) == 0:
        raise ValueError(
            "Aucune feature numérique trouvée parmi bundle['features']. "
            "Vérifie les dtypes dans dataset_final.csv."
        )

    return X, y, feature_names, df


def compute_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cfg: dict,
    log,
    out_dir: Path,
) -> None:
    """
    Calcule et sauvegarde la permutation importance (sklearn).

    Principe
    --------
    Pour chaque feature :
      - on permute aléatoirement la colonne
      - on mesure la baisse de score du modèle
    Ici, on laisse sklearn choisir le score par défaut selon le modèle,
    mais en classification, ça correspond souvent à accuracy.
    (Tu peux le contrôler via `scoring=` si tu veux BAC/f1.)

    Paramètres
    ----------
    model : Any
        Modèle sklearn entraîné (doit implémenter predict / score).
    X : np.ndarray
        Features.
    y : np.ndarray
        Labels.
    feature_names : list[str]
        Noms des features correspondant à X.
    cfg : dict
        Doit contenir éventuellement :
          - do_permutation (bool) : activer/désactiver
          - n_perm_repeats (int) : nb répétitions
          - random_state (int)
    log : logger
    out_dir : Path
        Dossier où sauvegarder permutation_importance.csv

    Sorties
    -------
    - out_dir/permutation_importance.csv
    """
    if not cfg.get("do_permutation", True):
        log.info("Permutation importance désactivée dans la config.")
        return

    n_repeats = int(cfg.get("n_perm_repeats", 30))
    random_state = int(cfg.get("random_state", 42))

    log.info(f"Permutation importance (n_repeats={n_repeats}, random_state={random_state})...")

    # NOTE: si tu veux une métrique spécifique :
    # result = permutation_importance(model, X, y, scoring="balanced_accuracy", ...)
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    df_perm = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    out_path = out_dir / "permutation_importance.csv"
    df_perm.to_csv(out_path, index=False)
    log.info(f"Permutation importance sauvegardée → {out_path}")


def compute_shap_global(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    cfg: dict,
    log,
    out_dir: Path,
) -> None:
    """
    Calcule une attribution SHAP globale et la sauvegarde.

    Stratégie
    ---------
    1) Sous-échantillonne X si X est trop grand (max_samples).
    2) Choisit l'explainer :
       - TreeExplainer si modèle d'arbres (rapide)
       - KernelExplainer sinon (lent mais général)
    3) Calcule shap_values.
    4) Agrège en score global par feature :
       - mean_abs_shap = moyenne de |SHAP| sur :
           * les échantillons
           * et les classes si multiclass

    Paramètres
    ----------
    model : Any
        Modèle entraîné. Si KernelExplainer, il faut model.predict_proba.
    X : np.ndarray
        Features (n_samples, n_features).
    feature_names : list[str]
        Noms des features.
    cfg : dict
        Paramètres SHAP :
          - max_samples (int) : limite d'échantillons
          - n_background (int) : taille du background (KernelExplainer)
          - nsamples_kernel (int) : budget d'échantillonnage KernelExplainer
          - random_state (int)
          - top_k_features (int) : pour shap_topk.txt
    log : logger
    out_dir : Path
        Dossier des sorties.

    Sorties
    -------
    - out_dir/shap_global_meanabs.csv
    - out_dir/shap_topk.txt

    Gestion erreurs
    --------------
    Si SHAP échoue (ex: modèle non compatible, coût trop élevé),
    on log un warning et on n'écrit pas les fichiers SHAP.
    """
    max_samples = int(cfg.get("max_samples", 2000))
    n_background = int(cfg.get("n_background", 200))
    nsamples_kernel = int(cfg.get("nsamples_kernel", 200))
    random_state = int(cfg.get("random_state", 42))

    n_samples = int(X.shape[0])

    # Sous-échantillonnage (utile si gros dataset)
    if n_samples > max_samples:
        log.info(f"Sous-échantillonnage SHAP: {n_samples} → {max_samples} échantillons")
        rng = np.random.RandomState(random_state)
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

            # Background : petit sous-ensemble de X_use
            if X_use.shape[0] > n_background:
                background = shap.sample(X_use, n_background, random_state=random_state)
            else:
                background = X_use

            # KernelExplainer en classification utilise souvent predict_proba
            if not hasattr(model, "predict_proba"):
                raise AttributeError("KernelExplainer nécessite model.predict_proba (absent).")

            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_vals = explainer.shap_values(X_use, nsamples=nsamples_kernel)

    except Exception as e:
        log.warning(f"Échec SHAP ({e}) → pas de shap_global_meanabs.csv")
        return

    # ---------- Normalisation formats SHAP ----------
    # Cas multiclasses : shap_vals est souvent une liste (une matrice par classe)
    # Cas binaire/régression : shap_vals est souvent une matrice (n_samples, n_features)
    if isinstance(shap_vals, list):
        # liste de matrices (n_samples, n_features) par classe
        # => on crée un tableau (n_samples, n_features, n_classes)
        shap_arr = np.stack(shap_vals, axis=-1)
    else:
        shap_arr = np.asarray(shap_vals)

    # ---------- Agrégation globale ----------
    if shap_arr.ndim == 3:
        # (n_samples, n_features, n_classes)
        mean_abs = np.mean(np.abs(shap_arr), axis=(0, 2))
    elif shap_arr.ndim == 2:
        # (n_samples, n_features)
        mean_abs = np.mean(np.abs(shap_arr), axis=0)
    else:
        log.warning(f"Format SHAP inattendu: shape={shap_arr.shape}")
        return

    df_shap = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    out_path = out_dir / "shap_global_meanabs.csv"
    df_shap.to_csv(out_path, index=False)
    log.info(f"SHAP global sauvegardé → {out_path}")

    # ---------- Top-k pour rapport ----------
    top_k = int(cfg.get("top_k_features", 10))
    top = df_shap.head(top_k)

    txt_path = out_dir / "shap_topk.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"Top {top_k} features (SHAP mean |value|):\n")
        for _, row in top.iterrows():
            f.write(f"- {row['feature']}: {row['mean_abs_shap']:.6g}\n")

    log.info(f"Top-{top_k} features SHAP → {txt_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(cfg: dict) -> None:
    """
    Point d'entrée principal.

    Étapes
    ------
    1) Initialise le logger et le dossier de sortie.
    2) Charge le bundle joblib contenant le modèle et la liste des features.
    3) Charge le dataset final et reconstruit X, y.
    4) Lance permutation importance (si activée).
    5) Lance SHAP global.
    6) Termine.

    Paramètres
    ----------
    cfg : dict
        Configuration YAML (voir docstring du module).

    Raises
    ------
    FileNotFoundError
        Si model_path est introuvable.
    KeyError / ValueError
        Si bundle ou dataset ne sont pas conformes (features manquantes, etc.).
    """
    log = get_logger("xai_attributions")

    out_dir = Path(cfg.get("out_dir", "outputs/xai/attributions"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Chargement checkpoint (bundle joblib)
    model_path = Path(cfg["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"model_path introuvable: {model_path}")

    log.info(f"Chargement checkpoint modèle : {model_path}")
    bundle = joblib.load(model_path)

    if "model" not in bundle:
        raise KeyError("Le bundle joblib ne contient pas la clé 'model'.")

    model = bundle["model"]

    # 2) Dataset (dataset_final + features du bundle)
    X, y, feature_names, _df = load_dataset_from_cfg(cfg, bundle, log)

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
