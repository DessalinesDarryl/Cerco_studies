#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_deep_models.py
====================

But
---
Entraîner et évaluer (validation croisée) des **modèles deep** simples sur des **features tabulaires**
(dérivées EEG/EMG), puis entraîner un modèle final sur tout le dataset et sauvegarder un checkpoint.

Ce script est adapté à ton contexte :
- Dataset final : `dataset_final.csv` (1 ligne = 1 patient, ou 1 ligne = 1 epoch selon ton pipeline)
- Labels : déjà présents dans une colonne numérique `label_id` (ou autre)
- Split : **GroupKFold par patient** (recommandé) pour éviter la fuite d’information entre train/val
- Modèles :
  - CNN1D sur vecteur de features (vu comme un "signal" 1D de longueur n_features)
  - GRU (RNN) sur séquence de longueur n_features (1 valeur par time-step)

Ce que fait le script
---------------------
1) Charge le CSV de features.
2) Nettoie / prépare le DataFrame d'entraînement (filtre lignes sans label, encode labels si besoin).
3) Sélectionne les colonnes numériques comme features.
4) Effectue une cross-validation :
   - GroupKFold (par patient) si configuré
   - sinon StratifiedKFold
5) Pour chaque fold :
   - construit un modèle deep (CNN ou GRU)
   - entraîne sur train
   - évalue sur val à chaque epoch
   - garde le meilleur état selon la Balanced Accuracy (BAC)
6) Sauvegarde les métriques CV dans un CSV.
7) Entraîne un modèle final sur toutes les données.
8) Sauvegarde un checkpoint `.pt` avec :
   - poids du modèle
   - type de modèle
   - dimensions
   - liste des colonnes utilisées
   - mapping id -> label_str (si dispo)
   - configuration d'entraînement

Config YAML attendue
--------------------
Exemple :

features_csv: data/processed/features/dataset_final.csv
label_col: label_id
model_type: "cnn"  # ou "rnn"

out_dir: models
ckpt_name: checkpoint_cnn.pt
metrics_out: outputs/stats/train_metrics_cnn.csv

cv:
  n_splits: 5
  group_by_patient: true
  shuffle: false
  random_state: 42

training:
  n_epochs: 50
  batch_size: 32
  lr: 1e-3
  weight_decay: 0.0
  device: "cuda"    # ou "cpu"

cnn_params:
  n_filters: 64
  kernel_size: 5
  hidden_dim: 128
  dropout: 0.3

rnn_params:
  hidden_dim: 64
  num_layers: 1
  bidirectional: false
  dropout: 0.3

Notes importantes / limites
---------------------------
- Ces modèles ne sont pas "deep learning" au sens séquence EEG brute : on applique du DL sur des features tabulaires.
- Le CNN1D fait un conv1d + global average pooling : c’est plus proche d’un MLP "structuré" qu’un CNN temps-signal.
- Le GRU traite les features comme une séquence arbitraire : l’ordre des features influence le résultat (ordre des colonnes).
  => si tu changes l’ordre des colonnes, tu changes la "séquence". (Pour du tabulaire, ce n’est pas forcément souhaitable.)
- Aucune normalisation (scaling) n’est appliquée ici. Souvent, un StandardScaler améliore beaucoup.
  => On peut l’ajouter proprement fold-by-fold (scaler fit sur train uniquement).

Exécution
---------
python scripts/train_deep_models.py --config configs/train_cnn.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Initialisation du chemin projet pour permettre les imports internes.
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse
from typing import Tuple, Dict, Optional, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from src.utils.config import load_yaml, add_common_args
from src.utils.logging import get_logger


# ======================================================================
# Dataset PyTorch et fonctions utilitaires de préparation
# ======================================================================

class FeatureDataset(Dataset):
    """
    Dataset PyTorch minimal pour du tabulaire.

    Paramètres
    ----------
    X : np.ndarray
        Matrice (n_samples, n_features).
    y : np.ndarray
        Vecteur (n_samples,) de labels entiers.

    Renvoie
    -------
    (x_i, y_i) :
        x_i : torch.FloatTensor shape (n_features,)
        y_i : torch.LongTensor scalar
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        """Nombre d'échantillons."""
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        """Retourne un échantillon (features, label) à l’indice idx."""
        return self.X[idx], self.y[idx]


def load_features_deep(features_csv: Path, log) -> pd.DataFrame:
    """
    Charge le CSV de features utilisé pour l'entraînement deep.

    Exigences
    ---------
    - Le fichier doit exister
    - Il doit contenir une colonne 'patient_id' (utilisée pour GroupKFold / groupes)

    Paramètres
    ----------
    features_csv : Path
        Chemin du fichier dataset final (features + labels).
    log : logger
        Logger du projet.

    Retours
    -------
    df : pd.DataFrame
        DataFrame brut chargé depuis le CSV.

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    ValueError
        Si la colonne 'patient_id' est absente.
    """
    if not features_csv.exists():
        raise FileNotFoundError(f"features_csv introuvable : {features_csv}")

    df = pd.read_csv(features_csv)

    if "patient_id" not in df.columns:
        raise ValueError("La colonne 'patient_id' est absente de features_csv.")

    log.info(
        f"Features (deep) : {df.shape[0]} lignes, {df.shape[1] - 1} colonnes (incluant patient_id)."
    )
    return df


def prepare_train_dataframe_deep(
    df_feat: pd.DataFrame,
    cfg: dict,
    log,
) -> Tuple[pd.DataFrame, str, Dict[int, str]]:
    """
    Prépare le DataFrame pour l’entraînement.

    Étapes
    ------
    1) Vérifie la présence de la colonne de label (cfg['label_col'], défaut label_id)
    2) Supprime les lignes sans label
    3) Cast label en int (si déjà numérique) ou factorise (si label texte)
    4) Construit un mapping id_to_name (utile pour l’interprétation) :
       - si (label_id,label_str) sont présents : mapping basé sur ces colonnes
       - sinon : fallback trivial id->str(id)

    Paramètres
    ----------
    df_feat : pd.DataFrame
        Features brutes.
    cfg : dict
        Configuration YAML.
    log : logger

    Retours
    -------
    df : pd.DataFrame
        DataFrame filtré (sans NaN labels) avec labels int.
    label_col : str
        Nom de la colonne utilisée comme label.
    id_to_name : dict[int, str]
        Dictionnaire pour interpréter les classes.

    Raises
    ------
    ValueError
        Si la colonne label est absente.
    """
    label_col = cfg.get("label_col", "label_id")
    if label_col not in df_feat.columns:
        raise ValueError(
            f"Colonne label '{label_col}' absente de features_csv. "
            f"Colonnes disponibles : {list(df_feat.columns)}"
        )

    df = df_feat.copy()

    before_rows = int(df.shape[0])
    df = df.dropna(subset=[label_col])
    after_rows = int(df.shape[0])

    log.info(f"Filtrage lignes sans label ({label_col}) : {before_rows} -> {after_rows} lignes.")

    # S'assure que y est un int (CrossEntropyLoss attend des entiers)
    if np.issubdtype(df[label_col].dtype, np.number):
        df[label_col] = df[label_col].astype(int)
    else:
        df[label_col], uniques = pd.factorize(df[label_col])
        df[label_col] = df[label_col].astype(int)
        log.info(f"Labels factorisés automatiquement : {dict(enumerate(uniques))}")

    # Mapping id -> nom (utile pour debug / interprétation)
    id_to_name: Dict[int, str] = {}
    if "label_str" in df.columns and "label_id" in df.columns:
        tmp = df[["label_id", "label_str"]].dropna().drop_duplicates()
        tmp["label_id"] = tmp["label_id"].astype(int)
        id_to_name = dict(zip(tmp["label_id"], tmp["label_str"]))
    else:
        uniques = np.unique(df[label_col].values)
        id_to_name = {int(i): str(i) for i in uniques}

    return df, label_col, id_to_name


# ======================================================================
# Modèles CNN / RNN
# ======================================================================

class CNN1DClassifier(nn.Module):
    """
    Classifieur CNN1D sur vecteur de features tabulaires.

    Idée
    ----
    On interprète le vecteur de features (feat_dim) comme un "signal" 1D,
    avec 1 canal. On applique une convolution 1D, puis un pooling global
    sur l’axe longueur, puis un petit MLP.

    Architecture
    ------------
    Input : (batch, feat_dim)
      -> unsqueeze : (batch, 1, feat_dim)
      -> Conv1d(1 -> n_filters)
      -> BatchNorm + ReLU
      -> Global average pooling (mean sur dim=-1)
      -> Dropout
      -> FC hidden + ReLU
      -> Dropout
      -> FC logits (n_classes)

    Paramètres
    ----------
    input_dim : int
        Dimension du vecteur de features (feat_dim). (pas utilisé directement ici,
        mais utile si tu veux ajouter des couches dépendantes de input_dim)
    n_classes : int
        Nombre de classes.
    n_filters : int
        Nombre de filtres convolutionnels.
    kernel_size : int
        Taille du noyau conv1d.
    hidden_dim : int
        Dimension de la couche fully-connected.
    dropout : float
        Taux de dropout.

    Notes
    -----
    - La convolution dépend de l’ordre des features (comme une séquence).
    - Le global pooling rend le modèle invariant à certaines positions, mais pas entièrement.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        n_filters: int = 64,
        kernel_size: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm1d(n_filters)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(n_filters, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Paramètres
        ----------
        x : torch.Tensor
            Shape (batch, feat_dim)

        Retours
        -------
        logits : torch.Tensor
            Shape (batch, n_classes)
        """
        x = x.unsqueeze(1)          # (batch, 1, feat_dim)
        x = self.conv(x)            # (batch, n_filters, feat_dim)
        x = self.bn(x)
        x = self.act(x)
        x = x.mean(dim=-1)          # (batch, n_filters)
        x = self.dropout(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


class GRUClassifier(nn.Module):
    """
    Classifieur GRU sur vecteur de features tabulaires vu comme une séquence.

    Idée
    ----
    On transforme un vecteur de features (feat_dim) en une séquence de longueur feat_dim
    avec 1 valeur par time-step :
        (batch, feat_dim) -> (batch, feat_dim, 1)
    Puis on applique un GRU et on utilise le dernier état caché pour classifier.

    Paramètres
    ----------
    input_dim : int
        Dimension du vecteur de features (feat_dim).
        (Gardé pour cohérence, le GRU utilise input_size=1).
    n_classes : int
        Nombre de classes.
    hidden_dim : int
        Taille de l’état caché.
    num_layers : int
        Nombre de couches GRU empilées.
    bidirectional : bool
        Si True, GRU bi-directionnel.
    dropout : float
        Dropout entre layers (uniquement si num_layers > 1).

    Attention
    ---------
    - L’ordre des features devient l’ordre temporel de la séquence.
      En tabulaire, cet ordre est arbitraire -> résultats potentiellement instables si colonnes réordonnées.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        d = 2 if bidirectional else 1
        self.fc = nn.Linear(d * hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Paramètres
        ----------
        x : torch.Tensor
            Shape (batch, feat_dim)

        Retours
        -------
        logits : torch.Tensor
            Shape (batch, n_classes)
        """
        x = x.unsqueeze(-1)         # (batch, feat_dim, 1)
        _, h_n = self.rnn(x)        # h_n: (num_layers*d, batch, hidden)

        if self.bidirectional:
            # concat des deux directions du dernier layer
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (batch, 2*hidden)
        else:
            h_last = h_n[-1]        # (batch, hidden)

        logits = self.fc(h_last)
        return logits


def build_deep_model(input_dim: int, n_classes: int, cfg: dict, log) -> nn.Module:
    """
    Construit un modèle deep (CNN ou GRU) à partir de la config.

    Paramètres
    ----------
    input_dim : int
        Nombre de features (dimension d’entrée).
    n_classes : int
        Nombre de classes.
    cfg : dict
        Configuration YAML.
    log : logger

    Retours
    -------
    model : nn.Module

    Raises
    ------
    ValueError
        Si model_type est inconnu.
    """
    model_type = str(cfg.get("model_type", "cnn")).lower()

    if model_type == "cnn":
        params = cfg.get("cnn_params", {})
        default = dict(n_filters=64, kernel_size=5, hidden_dim=128, dropout=0.3)
        default.update(params)
        log.info(f"Modèle = CNN1D, params : {default}")
        return CNN1DClassifier(input_dim=input_dim, n_classes=n_classes, **default)

    if model_type == "rnn":
        params = cfg.get("rnn_params", {})
        default = dict(hidden_dim=64, num_layers=1, bidirectional=False, dropout=0.3)
        default.update(params)
        log.info(f"Modèle = GRU, params : {default}")
        return GRUClassifier(input_dim=input_dim, n_classes=n_classes, **default)

    raise ValueError(f"model_type inconnu pour deep : {model_type} (attendu : 'cnn' | 'rnn').")


# ======================================================================
# Training loop
# ======================================================================

def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: dict,
    log,
) -> Tuple[float, float]:
    """
    Entraîne un modèle sur un fold et retourne les métriques du meilleur epoch.

    Logique de sélection "best model"
    ---------------------------------
    On garde l’état du modèle (state_dict) qui maximise la Balanced Accuracy sur val.

    Paramètres
    ----------
    model : nn.Module
        Modèle (CNN/GRU) non entraîné ou réinitialisé pour ce fold.
    train_loader : DataLoader
        Dataloader train.
    val_loader : DataLoader
        Dataloader validation.
    device : torch.device
        Device PyTorch ("cuda" ou "cpu").
    cfg : dict
        Config YAML (training hyperparams).
    log : logger

    Retours
    -------
    best_bal_acc : float
        Balanced Accuracy du meilleur epoch.
    best_f1_macro : float
        F1 macro du meilleur epoch (attention : renvoyé ici tel quel au dernier best_bal_acc).

    Notes
    -----
    - Ici, `best_f1_macro` correspond au F1 du même epoch que best_bal_acc.
    - CrossEntropyLoss attend des labels entiers [0..n_classes-1].
    """
    training_cfg = cfg.get("training", {})
    n_epochs = int(training_cfg.get("n_epochs", 50))
    lr = float(training_cfg.get("lr", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)

    best_bal_acc = -np.inf
    best_f1_macro = -np.inf
    best_state = None

    for epoch in range(1, n_epochs + 1):
        # ---- train ----
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # ---- val ----
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                preds = torch.argmax(logits, dim=1)

                y_true.append(yb.cpu().numpy())
                y_pred.append(preds.cpu().numpy())

        y_true = np.concatenate(y_true) if y_true else np.array([])
        y_pred = np.concatenate(y_pred) if y_pred else np.array([])

        if y_true.size == 0:
            log.warning("Validation vide (aucun batch).")
            continue

        bal_acc = balanced_accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")

        log.info(f"    Epoch {epoch:03d} : BAC={bal_acc:.3f}, F1_macro={f1_macro:.3f}")

        if bal_acc > best_bal_acc:
            best_bal_acc = float(bal_acc)
            best_f1_macro = float(f1_macro)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_bal_acc, best_f1_macro


def train_with_cv_deep(df: pd.DataFrame, label_col: str, cfg: dict, log):
    """
    Lance une cross-validation deep et retourne :
    - feat_cols (liste des features utilisées)
    - df_metrics (métriques par fold + moyenne)
    - input_dim (nb features)
    - n_classes

    Choix des features
    ------------------
    On prend toutes les colonnes numériques sauf :
      - patient_id
      - label_col
      - label_str

    CV
    --
    - GroupKFold si cfg['cv']['group_by_patient'] = True
      => split par patient_id pour limiter fuite d’info
    - sinon StratifiedKFold

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame préparé (labels int, patient_id présent).
    label_col : str
        Colonne label (int).
    cfg : dict
        Config YAML.
    log : logger

    Retours
    -------
    feat_cols : list[str]
    df_metrics : pd.DataFrame
    input_dim : int
    n_classes : int
    """
    # Colonnes candidates
    all_cols = [c for c in df.columns if c not in ("patient_id", label_col, "label_str")]
    feat_cols = [c for c in all_cols if np.issubdtype(df[c].dtype, np.number)]

    X = df[feat_cols].values.astype(np.float32)
    y = df[label_col].astype(int).values
    groups = df["patient_id"].astype(str).values

    n_classes = int(len(np.unique(y)))
    input_dim = int(X.shape[1])

    cv_cfg = cfg.get("cv", {})
    n_splits = int(cv_cfg.get("n_splits", 5))
    do_group = bool(cv_cfg.get("group_by_patient", True))
    shuffle = bool(cv_cfg.get("shuffle", False))
    seed = int(cv_cfg.get("random_state", 42))

    training_cfg = cfg.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 32))
    device_str = str(training_cfg.get("device", "cpu"))
    device = torch.device(device_str if (torch.cuda.is_available() and device_str.startswith("cuda")) else "cpu")

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    if do_group:
        log.info(f"CV deep : GroupKFold par patient (n_splits={n_splits})")
        gkf = GroupKFold(n_splits=n_splits)
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            folds.append((train_idx, val_idx))
    else:
        log.info(f"CV deep : StratifiedKFold (n_splits={n_splits}, shuffle={shuffle}, random_state={seed})")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        for train_idx, val_idx in skf.split(X, y):
            folds.append((train_idx, val_idx))

    metrics_rows = []

    for k, (tr, va) in enumerate(folds):
        log.info(f"Fold {k+1}/{len(folds)} : train={len(tr)} / val={len(va)}")

        X_tr, y_tr = X[tr], y[tr]
        X_va, y_va = X[va], y[va]

        ds_tr = FeatureDataset(X_tr, y_tr)
        ds_va = FeatureDataset(X_va, y_va)

        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False)

        model = build_deep_model(input_dim, n_classes, cfg, log)

        best_bal_acc, best_f1 = train_one_model(
            model=model,
            train_loader=dl_tr,
            val_loader=dl_va,
            device=device,
            cfg=cfg,
            log=log,
        )

        log.info(f"  Fold {k+1} (best) : BAC={best_bal_acc:.3f}, F1_macro={best_f1:.3f}")
        metrics_rows.append({
            "fold": k + 1,
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "balanced_accuracy": float(best_bal_acc),
            "f1_macro": float(best_f1),
        })

    df_metrics = pd.DataFrame(metrics_rows)

    # Ajout ligne moyenne fold=0
    if not df_metrics.empty:
        mean_row = {
            "fold": 0,
            "n_train": float(df_metrics["n_train"].mean()),
            "n_val": float(df_metrics["n_val"].mean()),
            "balanced_accuracy": float(df_metrics["balanced_accuracy"].mean()),
            "f1_macro": float(df_metrics["f1_macro"].mean()),
        }
        df_metrics = pd.concat([df_metrics, pd.DataFrame([mean_row])], ignore_index=True)
        log.info(f"CV deep global : BAC={mean_row['balanced_accuracy']:.3f}, F1_macro={mean_row['f1_macro']:.3f}")
    else:
        log.warning("Pas de métriques CV deep (df_metrics vide).")

    return feat_cols, df_metrics, input_dim, n_classes


# ======================================================================
# Point d’entrée principal
# ======================================================================

def main(cfg: dict) -> None:
    """
    Point d’entrée principal (piloté par YAML).

    Étapes
    ------
    1) Charge features_csv
    2) Prépare df_train (labels int)
    3) Fait CV (GroupKFold recommandé)
    4) Sauve métriques
    5) Entraîne modèle final sur tout le dataset
    6) Sauve checkpoint (poids + métadonnées)

    Paramètres
    ----------
    cfg : dict
        Configuration YAML.
    """
    log = get_logger("train_deep")

    features_csv = Path(cfg["features_csv"])
    out_dir = Path(cfg.get("out_dir", "models"))
    ckpt_name = str(cfg.get("ckpt_name", "checkpoint_deep.pt"))
    metrics_out = Path(cfg.get("metrics_out", "outputs/stats/train_metrics_deep.csv"))

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1) Chargement du dataset final de features ----
    df_feat = load_features_deep(features_csv, log)

    # ---- 2) Préparation du jeu d'entraînement et des labels ----
    df_train, label_col, id_to_name = prepare_train_dataframe_deep(df_feat, cfg, log)
    if df_train.empty:
        raise RuntimeError("Aucune ligne avec label après filtrage des features (deep).")

    # ---- 3) Validation croisée du modèle deep ----
    feat_cols, df_metrics, input_dim, n_classes = train_with_cv_deep(df_train, label_col, cfg, log)

    df_metrics.to_csv(metrics_out, index=False)
    log.info(f"Métriques deep CV sauvegardées dans {metrics_out}")

    # ---- 4) Entraînement final sur l'ensemble des données ----
    X_full = df_train[feat_cols].values.astype(np.float32)
    y_full = df_train[label_col].astype(int).values

    training_cfg = cfg.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 32))
    device_str = str(training_cfg.get("device", "cpu"))
    device = torch.device(device_str if (torch.cuda.is_available() and device_str.startswith("cuda")) else "cpu")

    ds_full = FeatureDataset(X_full, y_full)
    dl_full = DataLoader(ds_full, batch_size=batch_size, shuffle=True)

    model = build_deep_model(input_dim, n_classes, cfg, log)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg.get("lr", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )

    n_epochs = int(training_cfg.get("n_epochs", 50))

    model.to(device)
    for epoch in range(1, n_epochs + 1):
        model.train()
        for xb, yb in dl_full:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    log.info(f"Modèle deep final entraîné sur {len(y_full)} échantillons.")

    # ---- 5) Sauvegarde du checkpoint et des métadonnées utiles ----
    ckpt_path = out_dir / ckpt_name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": cfg.get("model_type", "cnn"),
            "input_dim": int(input_dim),
            "n_classes": int(n_classes),
            "feat_cols": feat_cols,
            "id_to_name": id_to_name,
            "training_cfg": training_cfg,
        },
        ckpt_path,
    )
    log.info(f"Checkpoint deep sauvegardé dans {ckpt_path}")


if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    main(cfg)
