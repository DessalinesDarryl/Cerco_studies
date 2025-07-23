import pandas as pd
import argparse
import os
import sys
import joblib
import platform
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from models import get_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["random_forest", "svm", "xgboost"], required=True)
    args = parser.parse_args()

    # Choix du montage
    response = input("Le montage est-il bipolaire ? (y/n) : ").strip().lower()
    if response not in {"y", "n"}:
        print("Réponse invalide. Veuillez entrer 'y' ou 'n'.")
        sys.exit(1)

    montage = "bipolaire" if response == "y" else "monopolaire"
    print(f"montage défini={montage}")

    # Détection système
    system = platform.system()
    if system == "Darwin":
        disque = "/Volumes/Crucial X6"
    elif system == "Windows":
        disque = "D:"
    else:
        raise RuntimeError("Système non supporté.")

    # Chargement du fichier CSV
    print("[INFO] Chargement des données...")
    path_csv = f"data/features_{montage}.csv"
    df = pd.read_csv(path_csv)

    # Vérification colonnes
    required = {"id_patient", "label", "age", "genre"}
    if not required.issubset(df.columns):
        raise ValueError(f"Le fichier doit contenir les colonnes : {required}")

    # Préprocessing
    print("[INFO] Encodage du genre et nettoyage...")
    df["genre"] = df["genre"].map({"h": 0, "f": 1})
    df.dropna(subset=["label", "age", "genre"], inplace=True)

    print("[INFO] Encodage du label...")
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])
    print(f"[INFO] Labels encodés : {list(le.classes_)}")

    print("[INFO] Préparation des features et target...")
    X = df.drop(columns=["id_patient", "segment", "label", "label_encoded"])
    y = df["label_encoded"]

    n_nan_X = X.isna().sum().sum()
    n_nan_y = y.isna().sum()
    print(f"[DEBUG] Nombre total de NaN dans X : {n_nan_X}")
    print(f"[DEBUG] Nombre total de NaN dans y : {n_nan_y}")

    if n_nan_y > 0:
        print("[WARNING] Des valeurs NaN sont présentes dans y. Elles seront supprimées.")
        nan_idx = y.isna()
        X = X[~nan_idx]
        y = y[~nan_idx]

    if n_nan_X > 0:
        print("[INFO] Remplissage des NaN dans X avec 0")
        X.fillna(0, inplace=True)

    # Split
    print("[INFO] Split train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Entraînement
    print(f"[INFO] Entraînement du modèle {args.model}...")
    model = get_model(args.model)
    model.fit(X_train, y_train)

    # Prédictions
    print("[INFO] Évaluation...")
    y_pred = model.predict(X_test)

    # Interprétation
    X_test_full = df.loc[X_test.index, ["id_patient", "segment", "label", "label_encoded"]].copy()
    X_test_full["y_pred"] = y_pred
    X_test_full["y_true"] = y_test.values
    X_test_full["correct"] = X_test_full["y_pred"] == X_test_full["y_true"]
    X_test_full["predicted_label"] = le.inverse_transform(X_test_full["y_pred"])
    X_test_full["true_label"] = le.inverse_transform(X_test_full["y_true"])

    print("\nClassification Report:\n")
    print(classification_report(
        y_test,
        y_pred,
        labels=le.transform(le.classes_),
        target_names=le.classes_,
        zero_division=0
    ))

    print(f"[INFO] Classes dans y_train : {sorted(set(y_train))}")
    print(f"[INFO] Classes dans y_test : {sorted(set(y_test))}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.show()

    print("\n[INFO] Segments mal classés :")
    print(X_test_full[~X_test_full["correct"]][["id_patient", "segment", "true_label", "predicted_label"]].head(10))

    per_patient = X_test_full.groupby("id_patient").agg({
        "correct": "mean",
        "true_label": "first"
    }).sort_values(by="correct")

    print("\n[INFO] Patients les plus souvent mal classés :")
    print(per_patient.head(10))

    print("[INFO] Validation croisée (5-fold)...")
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"\nCross-validation accuracy (5-fold): {scores.mean():.4f} ± {scores.std():.4f}")

    # Sauvegarde du modèle et encoder
    print("[INFO] Sauvegarde du modèle...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{args.model}_model.joblib")
    joblib.dump(le, f"models/{args.model}_label_encoder.joblib")
    print(f"[OK] Modèle et encoder sauvegardés dans 'models/'")

    # Feature importance
    if hasattr(model, "feature_importances_"):
        print("\n[INFO] Top 20 features les plus importantes :")
        importances = model.feature_importances_
        feature_names = X.columns
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        print(importance_df.head(20))
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.head(20), x="importance", y="feature")
        plt.title(f"Top 20 features - {args.model}")
        plt.tight_layout()
        plt.show()

        importance_df.to_csv(f"models/feature_importance_{args.model}.csv", index=False)
        print(f"[INFO] Importances exportées dans models/feature_importance_{args.model}.csv")

    # Sauvegarde des erreurs
    X_test_full.to_csv(f"models/misclassified_segments_{args.model}.csv", index=False)
    print(f"[INFO] Erreurs exportées dans models/misclassified_segments_{args.model}.csv")
