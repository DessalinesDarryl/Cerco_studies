# build_feature_dataset.py

import sys
import os
import numpy as np
import pandas as pd
import platform
from pathlib import Path
from cohort import load_patient_groups

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

def main():
    # Demande du montage
    response = input("Le montage est-il bipolaire ? (y/n) : ").strip().lower()
    if response not in {"y", "n"}:
        print("Réponse invalide. Veuillez entrer 'y' ou 'n'.")
        sys.exit(1)
    montage = "bipolaire" if response == "y" else "monopolaire"

    # Chemins
    excel_path = "data/Tableau_synthese_patients.xlsx"
    features_root = Path(f"features/{montage}/rem_only")
    output_features = f"data/features_{montage}.csv"
    output_info = "data/patient_info.csv"

    # 1. Charger les métadonnées
    print("[INFO] Chargement des métadonnées...")
    df_excel, group_map, demographics_map = load_patient_groups(excel_path, sheet_index=0)

    full_info_map = {}
    for patient_id in group_map:
        label = group_map.get(patient_id)
        demo = demographics_map.get(patient_id)
        if label is not None and demo is not None:
            full_info_map[patient_id] = {
                "label": label,
                "age": demo.get("age", np.nan),
                "genre": demo.get("genre", np.nan)
            }

    # 2. Construction des features
    rows = []
    info_rows = {}
    for npz_file in features_root.rglob("*.npz"):
        subject = npz_file.parent.name
        if subject not in full_info_map:
            print(f"[WARNING] Sujet {subject} absent du fichier Excel. Ignoré.")
            continue

        data = dict(np.load(npz_file))
        flat = flatten_npz_dict(data)
        flat["id_patient"] = subject
        flat["segment"] = npz_file.name.replace("_features.npz", "")
        rows.append(flat)

        if subject not in info_rows:
            info = {"id_patient": subject}
            info.update(full_info_map[subject])
            info_rows[subject] = info

    df_features = pd.DataFrame(rows)
    df_info = pd.DataFrame(info_rows.values())

    # 3. Jointure finale
    print("[INFO] Fusion features + info pour reconstituer le dataset final...")
    df_final = pd.merge(df_features, df_info, on="id_patient", how="left")

    # Vérification des colonnes
    if df_final[["label", "age", "genre"]].isnull().any().any():
        print("[WARNING] Des colonnes label/age/genre contiennent des NaNs après fusion.")

    # Réorganisation des colonnes
    feature_cols = [col for col in df_final.columns if col not in {"id_patient", "label", "age", "genre"}]
    ordered_cols = ["id_patient", "label", "age", "genre"] + feature_cols
    df_final = df_final[ordered_cols]

    # 4. Sauvegarde
    df_final.to_csv(output_features, index=False)
    df_info.to_csv(output_info, index=False)

    print(f"[OK] Fichier final : {output_features} ({df_final.shape[0]} lignes)")
    print(f"[OK] Infos patient (unique) : {output_info} ({df_info.shape[0]} patients)")

if __name__ == "__main__":
    main()

