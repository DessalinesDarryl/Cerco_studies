import pandas as pd
import argparse
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

from cohort import load_patient_groups
from models import get_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["random_forest", "svm", "xgboost"], required=True)
    args = parser.parse_args()

    print("[INFO] Chargement des features EEG...")
    features_df = pd.read_csv("../data/dataset_features_labels.csv")

    if "id_patient" not in features_df.columns:
        features_df.columns = ["id_patient"] + features_df.columns.tolist()[1:]

    features_df["id_patient"] = features_df["id_patient"].astype(str).str.strip()
    print(f"[INFO] Nb lignes features_df : {len(features_df)}")

    print("[INFO] Chargement des métadonnées et labels...")
    _, group_map, demographics_map = load_patient_groups("../data/Tableau_synthese_patients.xlsx", sheet_index=0)

    print("[DEBUG] Quelques valeurs de group_map :", list(group_map.items())[:5])
    print("[DEBUG] Quelques valeurs de demographics_map :", list(demographics_map.items())[:5])

    full_info_map = {
        pid: {"label": group_map[pid], **demographics_map.get(pid, {})}
        for pid in group_map
    }

    info_df = pd.DataFrame.from_dict(full_info_map, orient='index')
    info_df.index.name = 'id_patient'
    info_df.reset_index(inplace=True)
    info_df["id_patient"] = info_df["id_patient"].astype(str).str.strip()

    print(f"[INFO] Nb lignes info_df : {len(info_df)}")
    print("[INFO] Colonnes de info_df :", info_df.columns.tolist())
    print(info_df.head())

    print("[INFO] Fusion des deux tables sur 'id_patient'...")
    df_ml = pd.merge(features_df, info_df, on="id_patient", how="inner")
    print(f"[DEBUG] Nb de lignes après merge : {len(df_ml)}")

    for col in ["label", "age", "genre"]:
        if f"{col}_x" in df_ml.columns and f"{col}_y" in df_ml.columns:
            print(f"[DEBUG] Colonne dupliquée détectée : {col}")
            df_ml[col] = df_ml[f"{col}_x"].combine_first(df_ml[f"{col}_y"])
            df_ml.drop(columns=[f"{col}_x", f"{col}_y"], inplace=True)
        elif col in df_ml.columns:
            print(f"[DEBUG] Colonne unique trouvée : {col}")
        else:
            raise ValueError(f"[ERREUR] La colonne '{col}' est absente du merge.")

    df_ml["genre"] = df_ml["genre"].fillna("unknown")
    df_ml = df_ml[df_ml["genre"].isin(["h", "f"])]
    print(f"[DEBUG] Nb de lignes après filtre genre ['h','f'] : {len(df_ml)}")
    df_ml["genre"] = df_ml["genre"].map({"h": 0, "f": 1})

    df_ml.dropna(subset=["label", "age", "genre"], inplace=True)
    print(f"[DEBUG] Nb de lignes après dropna : {len(df_ml)}")

    if df_ml.empty:
        raise ValueError("[ERREUR] Le DataFrame final est vide. Vérifie les colonnes 'label', 'age', 'genre'.")

    print("[INFO] Préparation des données pour le ML...")
    X = df_ml.drop(columns=["id_patient", "label"])
    y = df_ml["label"]

    print(f"[INFO] Shape finale X : {X.shape}")
    print(f"[INFO] Labels : {y.unique()}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"[INFO] Labels encodés : {list(le.classes_)}")

    print("[INFO] Split train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42
    )
    print(f"[DEBUG] Train size: {len(y_train)} - Test size: {len(y_test)}")

    print(f"[INFO] Entraînement du modèle : {args.model}")
    model = get_model(args.model)
    model.fit(X_train, y_train)

    print("[INFO] Prédictions et évaluation...")
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    print("[INFO] Validation croisée 5-fold...")
    scores = cross_val_score(model, X, y_encoded, cv=5, scoring="accuracy")
    print(f"\nCross-validation accuracy (5-fold): {scores.mean():.4f} ± {scores.std():.4f}")

    print("[INFO] Sauvegarde du modèle et de l'encodeur...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{args.model}_model.joblib")
    joblib.dump(le, f"models/{args.model}_label_encoder.joblib")
    print(f"[INFO] Modèle '{args.model}' sauvegardé dans le dossier 'models/'.")
