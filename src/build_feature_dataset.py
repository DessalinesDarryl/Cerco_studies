# build_feature_dataset.py

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

def flatten_npz_dict(d):
    """Transforme un dict .npz (matriciel) en dict 1D à plat."""
    flat = {}
    for k, v in d.items():
        v = np.atleast_1d(v)
        if v.ndim == 1:
            for i in range(len(v)):
                flat[f"{k}_{i}"] = v[i]
        elif v.ndim == 2:
            for i in range(v.shape[0]):
                for j in range(v.shape[1]):
                    flat[f"{k}_{i}_{j}"] = v[i, j]
    return flat

def load_and_clean_patient_info(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv, dtype=str).rename(columns=str.strip)
    required = {"id_patient", "label", "age", "genre"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans {path_csv}: {sorted(missing_cols)}")

    # Trim espaces
    for c in ["id_patient", "label", "genre"]:
        df[c] = df[c].astype(str).str.strip()

    # Harmonise les 'nan' textuels et vides
    df.replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan}, inplace=True)

    # Age numérique
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Genre normalisé (h/f)
    df["genre"] = df["genre"].str.lower().map(
        {"h": "h", "m": "h", "homme": "h", "male": "h",
         "f": "f", "femme": "f", "female": "f"}
    ).fillna(df["genre"])  # si autre codage, on le conserve

    return df

def main():
    # Demande du montage
    response = input("Le montage est-il bipolaire ? (y/n) : ").strip().lower()
    if response not in {"y", "n"}:
        print("Réponse invalide. Veuillez entrer 'y' ou 'n'.")
        sys.exit(1)
    montage = "bipolaire" if response == "y" else "monopolaire"

    features_root = Path(f"features/{montage}/rem_only")
    output_features = f"data/features_{montage}.csv"
    info_csv = "data/patient_info.csv"

    # 1) Lire les infos patients depuis le CSV
    print(f"[INFO] Lecture des infos patients depuis {info_csv} ...")
    df_info = load_and_clean_patient_info(info_csv)

    # 2) Construire les features
    rows = []
    for npz_file in features_root.rglob("*.npz"):
        subject = npz_file.parent.name  # dossier du sujet
        data = dict(np.load(npz_file))
        flat = flatten_npz_dict(data)
        flat["id_patient"] = subject
        flat["segment"] = npz_file.name.replace("_features.npz", "")
        rows.append(flat)

    if not rows:
        print(f"[ERROR] Aucune feature trouvée sous {features_root} (*.npz).")
        sys.exit(1)

    df_features = pd.DataFrame(rows)

    # 3) Sanity checks avant merge
    missing_in_info = sorted(set(df_features["id_patient"]) - set(df_info["id_patient"]))
    if missing_in_info:
        print(f"[WARNING] {len(missing_in_info)} sujet(s) des features absents de {info_csv} : {missing_in_info[:10]}{' ...' if len(missing_in_info)>10 else ''}")

    # 4) Fusion finale
    print("[INFO] Fusion features + info ...")
    df_final = pd.merge(df_features, df_info, on="id_patient", how="left")

    # 5) Supprimer les lignes avec NaN dans label/age/genre
    df_final = df_final.dropna(subset=["label", "age", "genre"])
    print(f"[INFO] Lignes incomplètes supprimées. Dataset final : {df_final.shape[0]} lignes restantes.")


    # 6) Réorganisation des colonnes (5 colonnes info d'abord)
    info_cols = ["id_patient", "segment", "label", "age", "genre"]
    feature_cols = [c for c in df_final.columns if c not in set(info_cols)]
    df_final = df_final[info_cols + feature_cols]

    # Assertion douce pour s'assurer de l'ordre voulu
    expected_head = ["id_patient", "segment", "label", "age", "genre"]
    assert list(df_final.columns[:5]) == expected_head, f"En-tête inattendu: {df_final.columns[:5]}"

    # 7) Sauvegarde
    df_final.to_csv(output_features, index=False)
    print(f"[OK] Fichier final : {output_features} ({df_final.shape[0]} lignes, {df_final.shape[1]} colonnes)")
    print(f"[OK] 5 premières colonnes : {expected_head}")
    

if __name__ == "__main__":
    main()
