# train_model.py

import argparse
import os
import sys

import numpy as np
import pandas as pd
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
)

from models import get_model  # ton get_model() doit retourner un estimateur sklearn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["random_forest", "svm", "xgboost"],
        required=True,
        help="Choisir le modèle d'apprentissage",
    )
    args = parser.parse_args()

    # Choix du montage
    response = input("Le montage est-il bipolaire ? (y/n) : ").strip().lower()
    if response not in {"y", "n"}:
        print("Réponse invalide. Veuillez entrer 'y' pour oui ou 'n' pour non.")
        sys.exit(1)
    montage = "bipolaire" if response == "y" else "monopolaire"
    print(f"montage défini={montage}")

    # ---- Chargement
    print("[INFO] Chargement des données...")
    path_csv = f"data/features_{montage}.csv"
    if not os.path.exists(path_csv):
        print(f"[ERROR] Fichier introuvable: {path_csv}")
        sys.exit(1)

    df = pd.read_csv(path_csv)

    # ---- Vérifications de base
    required = {"id_patient", "segment", "label", "age", "genre"}
    if not required.issubset(df.columns):
        raise ValueError(f"Le fichier doit contenir les colonnes : {sorted(required)}")

    # ---- Encodages / nettoyage colonnes info
    print("[INFO] Encodage du genre et nettoyage...")
    # On tolère 'h'/'f' uniquement (le build supprime normalement les NaN déjà)
    df["genre"] = df["genre"].map({"h": 0, "f": 1})
    # Supprime toute ligne résiduelle incomplète côté infos (sécurité)
    before = len(df)
    df.dropna(subset=["label", "age", "genre"], inplace=True)
    removed = before - len(df)
    if removed > 0:
        print(f"[INFO] Lignes supprimées (infos manquantes): {removed}")

    # Label encoding
    print("[INFO] Encodage du label...")
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])
    print(f"[INFO] Labels encodés : {list(le.classes_)}")

    # ---- Features / Target
    print("[INFO] Préparation des features et target...")
    drop_cols = ["id_patient", "segment", "label", "label_encoded"]
    # On garde 'age' et 'genre' en features (si tu ne veux pas, retire-les ici)
    X = df.drop(columns=drop_cols)
    y = df["label_encoded"].copy()

    # ---- Split groupé par patient
    print("[INFO] Split train/test groupé par patient...")
    groups = df["id_patient"].values
    if len(np.unique(groups)) < 2:
        print("[ERROR] Moins de 2 patients uniques, impossible de faire un split.")
        sys.exit(1)

    gss = GroupShuffleSplit(test_size=0.2, random_state=42, n_splits=1)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups[train_idx]

    print(f"[INFO] Patients train: {df.iloc[train_idx]['id_patient'].nunique()} | "
          f"Patients test: {df.iloc[test_idx]['id_patient'].nunique()}")
    print(f"[INFO] Tailles -> X_train: {X_train.shape}, X_test: {X_test.shape}")

    # ---- Modèle de base
    base_model = get_model(args.model)

    # Essaie d'activer class_weight='balanced' pour RF/SVM si possible
    try:
        if args.model in {"random_forest", "svm"}:
            # Certains modèles n'acceptent pas class_weight (ex: SVC l'accepte, SVR non).
            # On teste la présence du paramètre:
            params = base_model.get_params(deep=False)
            if "class_weight" in params and params.get("class_weight", None) is None:
                base_model.set_params(class_weight="balanced")
                print("[INFO] class_weight='balanced' activé.")
    except Exception as e:
        print(f"[WARNING] Impossible de fixer class_weight='balanced': {e}")

    # ---- Pipeline (imputation -> variance threshold -> scaler (pour SVM) -> clf)
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("varth", VarianceThreshold(threshold=1e-6)),
    ]
    if args.model == "svm":
        # Standardisation utile pour SVM. with_mean=True attend un dense;
        # si tu utilises des matrices clairsemées, mets with_mean=False.
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", base_model))

    pipeline = Pipeline(steps)

    # ---- Entraînement
    print(f"[INFO] Entraînement du modèle {args.model}...")
    pipeline.fit(X_train, y_train)

    # ---- Évaluation test
    print("[INFO] Évaluation (test set)...")
    y_pred = pipeline.predict(X_test)

    ba = balanced_accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    print(f"\nBalanced Accuracy: {ba:.3f}")
    print(f"F1 macro        : {f1m:.3f}\n")

    # Report avec toutes les classes connues (même absentes du test -> colonnes/rows à 0)
    print("Classification Report (toutes classes connues):\n")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=list(range(len(le.classes_))),
            target_names=le.classes_,
            zero_division=0,
        )
    )

    # Matrice de confusion normalisée par classe réelle
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=list(range(len(le.classes_))),
        normalize="true",
    )
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title("Matrice de confusion (normalisée)")
    plt.tight_layout()
    plt.show()

    # ---- Cross-validation groupée (sur le TRAIN uniquement)
    print("[INFO] Validation croisée (GroupKFold) sur le TRAIN ...")
    n_groups_train = len(np.unique(groups_train))
    n_splits = min(5, n_groups_train)  # évite d'avoir plus de splits que de groupes
    if n_splits < 2:
        print("[WARNING] Trop peu de patients en train pour une CV (n_splits < 2). Skip.")
        cv_scores = None
    else:
        gkf = GroupKFold(n_splits=n_splits)
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=gkf.split(X_train, y_train, groups=groups_train),
            scoring="balanced_accuracy",
            n_jobs=-1,
        )
        print(f"\nCV (GroupKFold, BA): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ---- Sauvegarde
    print("[INFO] Sauvegarde du pipeline et du label encoder...")
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{args.model}_pipeline.joblib"
    le_path = f"models/{args.model}_label_encoder.joblib"
    joblib.dump(pipeline, model_path)
    joblib.dump(le, le_path)
    print(f"[OK] Pipeline sauvegardé -> {model_path}")
    print(f"[OK] LabelEncoder sauvegardé -> {le_path}")


if __name__ == "__main__":
    main()
