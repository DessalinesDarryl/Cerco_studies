#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
knn_diagnostic.py

Diagnostic géométrique de l’espace des features :
- distances intra-classe
- distances inter-classe
- ratio de séparabilité

Entrée :
- CSV patient-level (dataset_final.csv)
- label_col
- k voisins

Sorties :
- CSV diagnostic
- résumé console
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from src.utils.logging import get_logger
log = get_logger("knn_diagnostic")


def knn_diagnostic(
    df: pd.DataFrame,
    feature_cols: list,
    label_col: str,
    patient_col: str = "patient_id",
    k: int = 5,
):
    # 1) Extraction de la matrice de features et des labels
    y = df[label_col].values

    X = df[feature_cols].values

    n_nan = np.isnan(X).sum()
    log.info(f"Nombre total de NaN dans X : {n_nan}")


    # 2) Imputation robuste des valeurs manquantes
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    assert not np.isnan(X).any(), "NaN restants après imputation"

    # 3) Standardisation des features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    # 4) Recherche des plus proches voisins dans l’espace normalisé
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # 5) Calcul des distances intra-classe et inter-classe pour chaque patient
    rows = []

    for i in range(len(df)):
        label_i = y[i]

        intra_dists = []
        inter_dists = []

        for dist, j in zip(distances[i][1:], indices[i][1:]):
            if y[j] == label_i:
                intra_dists.append(dist)
            else:
                inter_dists.append(dist)

        rows.append({
            "patient_id": df.iloc[i][patient_col],
            "label": label_i,
            "intra_distance": np.mean(intra_dists) if intra_dists else np.nan,
            "inter_distance": np.mean(inter_dists) if inter_dists else np.nan,
        })

    out = pd.DataFrame(rows)
    # 6) Ratio de séparabilité : plus il est faible, meilleure est la séparation
    out["separation_ratio"] = out["intra_distance"] / out["inter_distance"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_csv", required=True)
    parser.add_argument("--label_col", default="label_id")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default="outputs/stats/knn_diagnostic.csv")
    args = parser.parse_args()

    log = get_logger("knn_diagnostic")

    df = pd.read_csv(args.features_csv)
    assert "patient_id" in df.columns

    # Sélection des features numériques utilisées pour le diagnostic
    feature_cols = [
        c for c in df.columns
        if c not in ("patient_id", args.label_col, "label_str")
        and np.issubdtype(df[c].dtype, np.number)
    ]

    log.info(f"{len(feature_cols)} features utilisées")
    log.info(f"{df.shape[0]} patients")

    diag = knn_diagnostic(
        df,
        feature_cols=feature_cols,
        label_col=args.label_col,
        k=args.k,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    diag.to_csv(args.out, index=False)

    log.info("=== RÉSUMÉ DIAGNOSTIC KNN ===")
    log.info(f"Intra distance moyenne : {diag['intra_distance'].mean():.3f}")
    log.info(f"Inter distance moyenne : {diag['inter_distance'].mean():.3f}")
    log.info(f"Ratio moyen : {diag['separation_ratio'].mean():.3f}")

    print("\nPar classe :")
    print(
        diag.groupby("label")[["intra_distance", "inter_distance", "separation_ratio"]]
        .mean()
        .round(3)
    )


if __name__ == "__main__":
    main()
