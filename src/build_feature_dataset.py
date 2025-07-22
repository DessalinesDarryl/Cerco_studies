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

    excel_path = "data/Tableau_synthese_patients.xlsx"
    if not Path(excel_path).exists():
        raise FileNotFoundError(f" Le fichier Excel est introuvable : {excel_path}")

    features_root = Path(f"features/{montage}/rem_only")  
    output_csv = "data/dataset_features_labels.csv"

    # 1. Chargement des groupes + métadonnées 
    print("Chargement des données...")
    df_excel, group_map, demographics_map = load_patient_groups(excel_path, sheet_index=0)

    # 2. Fusion labels + infos démographiques
    print("Fusion des labels et des données demographiques avec les features...")
    full_info_map = {}
    for patient_id in group_map:
        full_info_map[patient_id] = {'label': group_map[patient_id]}
        if patient_id in demographics_map:
            full_info_map[patient_id].update(demographics_map[patient_id])

    rows = []
    for npz_file in features_root.rglob("*.npz"):
        subject = npz_file.parent.name  # Nom du dossier patient
        data = dict(np.load(npz_file))
        flat = flatten_npz_dict(data)
        flat["subject"] = subject
        flat["segment"] = npz_file.name.replace("_features.npz", "")

        if subject in full_info_map:
            flat.update(full_info_map[subject])
        else:
            print(f"Sujet {subject} introuvable dans le fichier Excel. Segment ignoré.")
            continue

        rows.append(flat)

    # 3. DataFrame final 
    print("Sauvegarde...")
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Fichier CSV généré : {output_csv} ({df.shape[0]} lignes, {df.shape[1]} colonnes)")

if __name__ == "__main__":
    main()

