import pandas as pd
import os
import glob

def load_patient_groups(xlsx_path, sheet_index=None):
    """
    Charge la feuille d'un fichier Excel contenant les identifiants et pathologies,
    retourne un DataFrame et un mapping {identifiant_edf: pathologie}.
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_index)

    # Renomme proprement les colonnes attendues
    df.rename(columns={
        df.columns[0]: 'identifiant_edf',
        df.columns[5]: 'label'
    }, inplace=True)

    df['identifiant_edf'] = df['identifiant_edf'].astype(str).str.strip()
    df['label'] = df['label'].astype(str).str.lower().str.strip()

    group_map = dict(zip(df['identifiant_edf'], df['label']))
    return df, group_map



def group_edf_files_by_pathology(raw_dir, group_map):
    """
    Regroupe les fichiers .edf dans un dict {label: [liste de chemins EDF]}.
    """
    edf_files = glob.glob(os.path.join(raw_dir, "*.edf"))
    grouped = {}

    for edf_path in edf_files:
        basename = os.path.splitext(os.path.basename(edf_path))[0]
        if basename in group_map:
            patho = group_map[basename]
            grouped.setdefault(patho, []).append(edf_path)
        else:
            print(f" Fichier {basename}.edf non référencé dans l'Excel.")

    return grouped
