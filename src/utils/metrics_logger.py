"""Persistance des métriques d'entraînement et d'évaluation.

Ce module exporte les métriques au format JSON/CSV avec timestamp pour
assurer la traçabilité des expériences.
"""

import csv
import json
import os

from datetime import datetime


def save_metrics(metrics: dict, out_dir: str, name: str = "metrics.json"):
    """Enregistre un dictionnaire de métriques au format JSON avec timestamp."""
    os.makedirs(out_dir, exist_ok=True)
    metrics = {"timestamp": datetime.now().isoformat(), **metrics}
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_roc_csv(fpr, tpr, thresholds, out_csv: str):
    """Exporte une courbe ROC sous forme tabulaire pour archivage ou replot."""
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fpr", "tpr", "threshold"])
        for a, b, c in zip(fpr, tpr, thresholds):
            w.writerow([a, b, c])
