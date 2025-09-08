import pandas as pd
import os
import glob

def load_patient_groups(xlsx_path, sheet_index=None):
    """
    Charge la feuille d'un fichier Excel contenant les identifiants, âges, genres et pathologies.
    Retourne :
      - un DataFrame complet,
      - un mapping {id_patient: pathologie},
      - un mapping {id_patient: {'age': ..., 'genre': ...}}.
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_index)

    # Renommage explicite des colonnes clés
    df.rename(columns={
        df.columns[0]: 'id_patient',
        df.columns[4]: 'label',
        df.columns[5]: 'genre'
    }, inplace=True)

    # Nettoyage des champs
    df['id_patient'] = df['id_patient'].astype(str).str.strip()
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    df['genre'] = df['genre'].astype(str).str.lower().str.strip()

    # Remplacement des noms de groupes
    df['label'] = df['label'].replace({
        'syn_patients': 'synucleopathy',
        'tcsp_patients': 'tcspi',
        'narco_patients': 'narcolepsy',
        'autres_patients': 'autoimmune encephalitides'
    })

    # Mapping des pathologies
    group_map = dict(zip(df['id_patient'], df['label']))

    # Mapping des infos démographiques
    demographics_map = df.set_index('id_patient')[['genre']].to_dict(orient='index')

    return df, group_map, demographics_map




def group_edf_files_by_pathology(raw_dir, group_map):
    """
    Regroupe les fichiers .edf dans un dict {pathologie: [liste de chemins EDF]}.
    """
    edf_files = glob.glob(os.path.join(raw_dir, "*.edf"))
    grouped = {}

    for edf_path in edf_files:
        basename = os.path.splitext(os.path.basename(edf_path))[0].strip()
        if basename in group_map:
            patho = group_map[basename]
            grouped.setdefault(patho, []).append(edf_path)
        else:
            print(f"[WARN] Fichier {basename}.edf non référencé dans le mapping fourni.")

    return grouped
