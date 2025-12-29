#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_deep_models.py

Objectif :
  - Charger les features par patient (dataset_final.csv).
  - Utiliser directement les labels (label_id).
  - Faire un split par patient (GroupKFold).
  - Entraîner un modèle deep (CNN 1D ou RNN/GRU) sur les features tabulaires.
  - Évaluer (balanced accuracy, F1 macro) en CV.
  - Sauvegarder le modèle final et les méta-données.

Config YAML attendue (ex: train_cnn.yaml / train_rnn.yaml) :
  features_csv: data/processed/features/dataset_final.csv
  label_col: label_id

  model_type: "cnn" | "rnn"

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
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse
from typing import Tuple, Dict, Optional

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
#  Dataset & utils
# ======================================================================

class FeatureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_features_deep(features_csv: Path, log) -> pd.DataFrame:
    if not features_csv.exists():
        raise FileNotFoundError(f"features_csv introuvable : {features_csv}")
    df = pd.read_csv(features_csv)
    if "patient_id" not in df.columns:
        raise ValueError("La colonne 'patient_id' est absente de features_csv.")
    log.info(f"Features (deep) : {df.shape[0]} lignes, {df.shape[1] - 1} colonnes (incluant patient_id).")
    return df


def prepare_train_dataframe_deep(
    df_feat: pd.DataFrame,
    cfg: dict,
    log,
) -> Tuple[pd.DataFrame, str, Dict[int, str]]:
    label_col = cfg.get("label_col", "label_id")
    if label_col not in df_feat.columns:
        raise ValueError(
            f"Colonne label '{label_col}' absente de features_csv. "
            f"Colonnes disponibles : {list(df_feat.columns)}"
        )

    df = df_feat.copy()
    before_rows = df.shape[0]
    df = df.dropna(subset=[label_col])
    after_rows = df.shape[0]

    log.info(
        f"Filtrage lignes sans label ({label_col}) : {before_rows} -> {after_rows} lignes."
    )

    if np.issubdtype(df[label_col].dtype, np.number):
        df[label_col] = df[label_col].astype(int)
    else:
        df[label_col], uniques = pd.factorize(df[label_col])
        df[label_col] = df[label_col].astype(int)
        log.info(f"Labels factorisés automatiquement : {dict(enumerate(uniques))}")

    # mapping id -> nom (utile pour interpréter confusion matrix, etc.)
    id_to_name = {}
    if "label_str" in df.columns and "label_id" in df.columns:
        tmp = df[["label_id", "label_str"]].dropna().drop_duplicates()
        tmp["label_id"] = tmp["label_id"].astype(int)
        id_to_name = dict(zip(tmp["label_id"], tmp["label_str"]))
    else:
        # fallback simple
        uniques = np.unique(df[label_col].values)
        id_to_name = {int(i): str(i) for i in uniques}

    return df, label_col, id_to_name


# ======================================================================
#  Modèles CNN / RNN
# ======================================================================

class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int,
                 n_filters: int = 64, kernel_size: int = 5,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Conv1d(1, n_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(n_filters)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(n_filters, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # x : (batch, feat_dim)
        x = x.unsqueeze(1)          # (batch, 1, feat_dim)
        x = self.conv(x)            # (batch, n_filters, feat_dim)
        x = self.bn(x)
        x = self.act(x)
        x = x.mean(dim=-1)          # Global average pooling → (batch, n_filters)
        x = self.dropout(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int,
                 hidden_dim: int = 64, num_layers: int = 1,
                 bidirectional: bool = False, dropout: float = 0.3):
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

    def forward(self, x):
        # x : (batch, feat_dim)
        # On le voit comme séquence de longueur feat_dim, avec 1 feature par time-step
        x = x.unsqueeze(-1)  # (batch, feat_dim, 1)
        out, h_n = self.rnn(x)  # out: (batch, feat_dim, hidden), h_n: (num_layers*d, batch, hidden)
        # On prend le dernier état caché
        if self.bidirectional:
            # Concat des deux directions pour le dernier layer
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (batch, 2*hidden)
        else:
            h_last = h_n[-1]  # (batch, hidden)
        logits = self.fc(h_last)
        return logits


def build_deep_model(input_dim: int, n_classes: int, cfg, log) -> nn.Module:
    model_type = cfg.get("model_type", "cnn").lower()

    if model_type == "cnn":
        params = cfg.get("cnn_params", {})
        default = dict(
            n_filters=64,
            kernel_size=5,
            hidden_dim=128,
            dropout=0.3,
        )
        default.update(params)
        log.info(f"Modèle = CNN1D, params : {default}")
        model = CNN1DClassifier(
            input_dim=input_dim,
            n_classes=n_classes,
            **default
        )
        return model

    elif model_type == "rnn":
        params = cfg.get("rnn_params", {})
        default = dict(
            hidden_dim=64,
            num_layers=1,
            bidirectional=False,
            dropout=0.3,
        )
        default.update(params)
        log.info(f"Modèle = GRU, params : {default}")
        model = GRUClassifier(
            input_dim=input_dim,
            n_classes=n_classes,
            **default
        )
        return model

    else:
        raise ValueError(f"model_type inconnu pour deep : {model_type} (attendu : 'cnn' | 'rnn').")


# ======================================================================
#  Training loop
# ======================================================================

def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: dict,
    log,
) -> Tuple[float, float]:
    training_cfg = cfg.get("training", {})
    n_epochs = int(training_cfg.get("n_epochs", 50))
    lr = float(training_cfg.get("lr", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)

    best_bal_acc = -np.inf
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

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        bal_acc = balanced_accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")

        log.info(f"    Epoch {epoch:03d} : BAC={bal_acc:.3f}, F1_macro={f1_macro:.3f}")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_state = model.state_dict()

    # On remet le meilleur état
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_bal_acc, f1_macro


def train_with_cv_deep(df, label_col, cfg, log):
    # Colonnes de features numériques
    all_cols = [c for c in df.columns if c not in ("patient_id", label_col, "label_str")]
    feat_cols = [c for c in all_cols if np.issubdtype(df[c].dtype, np.number)]

    X = df[feat_cols].values.astype(np.float32)
    y = df[label_col].astype(int).values
    groups = df["patient_id"].values

    n_classes = len(np.unique(y))
    input_dim = X.shape[1]

    n_splits = cfg.get("cv", {}).get("n_splits", 5)
    do_group = cfg.get("cv", {}).get("group_by_patient", True)
    shuffle = cfg.get("cv", {}).get("shuffle", False)
    seed = cfg.get("cv", {}).get("random_state", 42)

    training_cfg = cfg.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 32))
    device_str = training_cfg.get("device", "cpu")
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

    folds = []
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
            "n_train": len(tr),
            "n_val": len(va),
            "balanced_accuracy": best_bal_acc,
            "f1_macro": best_f1,
        })

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
            f"CV deep global : BAC={mean_row['balanced_accuracy']:.3f}, "
            f"F1_macro={mean_row['f1_macro']:.3f}"
        )
    else:
        log.warning("Pas de métriques CV deep (df_metrics vide).")

    return feat_cols, df_metrics, input_dim, n_classes


# ======================================================================
#  Main
# ======================================================================

def main(cfg):
    log = get_logger("train_deep")

    features_csv = Path(cfg["features_csv"])
    out_dir = Path(cfg.get("out_dir", "models"))
    ckpt_name = cfg.get("ckpt_name", "checkpoint_deep.pt")
    metrics_out = Path(cfg.get("metrics_out", "outputs/stats/train_metrics_deep.csv"))

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1) Features ----
    df_feat = load_features_deep(features_csv, log)

    # ---- 2) Préparation df_train ----
    df_train, label_col, id_to_name = prepare_train_dataframe_deep(df_feat, cfg, log)
    if df_train.empty:
        raise RuntimeError("Aucune ligne avec label après filtrage des features (deep).")

    # ---- 3) CV ----
    feat_cols, df_metrics, input_dim, n_classes = train_with_cv_deep(df_train, label_col, cfg, log)

    df_metrics.to_csv(metrics_out, index=False)
    log.info(f"Métriques deep CV sauvegardées dans {metrics_out}")

    # ---- 4) Entraînement final sur tout le dataset ----
    X_full = df_train[feat_cols].values.astype(np.float32)
    y_full = df_train[label_col].astype(int).values

    training_cfg = cfg.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 32))
    device_str = training_cfg.get("device", "cpu")
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

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

    # ---- 5) Sauvegarde checkpoint ----
    ckpt_path = out_dir / ckpt_name
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_type": cfg.get("model_type", "cnn"),
        "input_dim": input_dim,
        "n_classes": n_classes,
        "feat_cols": feat_cols,
        "id_to_name": id_to_name,
        "training_cfg": training_cfg,
    }, ckpt_path)
    log.info(f"Checkpoint deep sauvegardé dans {ckpt_path}")


if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    main(cfg)
