import pandas as pd
import argparse
import os
import joblib
import platform
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from models import get_model  # suppose que tu as un fichier models.py avec get_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["random_forest", "svm", "xgboost"], required=True)
    args = parser.parse_args()

    # On demande à l'utilisateur si le montage est bipolaire
    response = input("Le montage est-il bipolaire ? (y/n) : ").strip().lower()
    if response not in {"y", "n"}:
        print("Réponse invalide. Veuillez entrer 'y' pour oui ou 'n' pour non.")
        sys.exit(1)

    montage = "bipolaire" if response == "y" else "monopolaire"
    print(f"montage défini={montage}")

    # Adaptation système
    system = platform.system()
    if system == "Darwin":  # macOS
        disque = "/Volumes/Crucial X6"
    elif system == "Windows":
        disque = "D:"
    else:
        raise RuntimeError("Système non supporté.")

    print("[INFO] Chargement des données...")
    path_csv = f"data/features_{montage}.csv"
    df = pd.read_csv(path_csv)

    # Vérification des colonnes nécessaires
    required = {"id_patient", "label", "age", "genre"}
    if not required.issubset(df.columns):
        raise ValueError(f"Le fichier doit contenir les colonnes : {required}")

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


    print("[INFO] Split train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    print(f"[INFO] Entraînement du modèle {args.model}...")
    model = get_model(args.model)
    model.fit(X_train, y_train)

    print("[INFO] Évaluation...")
    y_pred = model.predict(X_test)
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

    print("[INFO] Validation croisée (5-fold)...")
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"\nCross-validation accuracy (5-fold): {scores.mean():.4f} ± {scores.std():.4f}")

    print("[INFO] Sauvegarde du modèle...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{args.model}_model.joblib")
    joblib.dump(le, f"models/{args.model}_label_encoder.joblib")
    print(f"[OK] Modèle et encoder sauvegardés dans 'models/'")

