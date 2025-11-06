import json, os, csv
from datetime import datetime

def save_metrics(metrics: dict, out_dir: str, name: str = "metrics.json"):
    os.makedirs(out_dir, exist_ok=True)
    metrics = {"timestamp": datetime.now().isoformat(), **metrics}
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(metrics, f, indent=2)

def save_roc_csv(fpr, tpr, thresholds, out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fpr", "tpr", "threshold"])
        for a, b, c in zip(fpr, tpr, thresholds):
            w.writerow([a, b, c])
