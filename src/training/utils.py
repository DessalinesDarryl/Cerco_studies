"""Utilitaires communs pour l'entraînement et l'évaluation.

Le module regroupe les métriques de classification utilisées pendant
la validation des modèles tabulaires et deep.
"""

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def compute_metrics(y_true, y_prob):
    """Calcule les métriques de classification à partir des probabilités prédites."""
    y_pred = y_prob.argmax(1)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bac": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
