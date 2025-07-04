import os
import glob
import pandas as pd
import mne

def load_annotation_file(txt_path):
    """
    Charge un fichier .txt d'annotations et extrait les intervalles REM.
    Suppose les colonnes : start_s, temps, phase, index.
    """
    df = pd.read_csv(txt_path, sep="\t", names=["start", "temps", "stage", "index"])

    # Supprimer les lignes incomplètes
    df = df.dropna(subset=["start", "stage"])

    # Calculer les durées en secondes : diff entre start courant et suivant
    df["duration"] = df["start"].shift(-1) - df["start"]
    df = df[:-1]  # retirer la dernière ligne (durée inconnue)

    # Filtrer les segments REM
    rem_df = df[df["stage"].str.upper().str.strip() == "REM"]
    return rem_df[["start", "duration"]].values  # array (N, 2)

def extract_rem_segments(raw, rem_intervals):
    """
    Coupe le Raw entre les intervalles REM fournis (en secondes).
    Renvoie un Raw contenant uniquement les segments REM concaténés.
    """
    rem_segments = []
    for start, dur in rem_intervals:
        stop = start + dur
        try:
            rem_seg = raw.copy().crop(tmin=start, tmax=stop)
            rem_segments.append(rem_seg)
        except Exception as e:
            print(f" Erreur sur intervalle [{start}-{stop}] : {e}")

    if rem_segments:
        return mne.concatenate_raws(rem_segments)
    else:
        return None

def segment_all_rem(preprocessed_dir=None,
                    annot_root=None,
                    save_dir=None):
    os.makedirs(save_dir, exist_ok=True)
    fif_files = glob.glob(os.path.join(preprocessed_dir, "*_eeg_cleaned_raw.fif"))

    for fif_path in fif_files:
        base = os.path.basename(fif_path).replace("_eeg_cleaned_raw.fif", "")
        out_path = os.path.join(save_dir, f"{base}_REM_raw.fif")

        # Vérifie si le fichier a déjà été traité
        if os.path.exists(out_path):
            print(f" Fichier déjà existant pour {base}, on saute.")
            continue

        raw = mne.io.read_raw_fif(fif_path, preload=True)

        # Recherche d'un fichier .txt d'annotation correspondant (même nom de base)
        txt_candidates = glob.glob(os.path.join(annot_root, "**", f"{base}hypnoEXP.txt"), recursive=True)

        if not txt_candidates:
            print(f" Aucun fichier d'annotation trouvé pour {base}")
            continue

        txt_path = txt_candidates[0]
        rem_intervals = load_annotation_file(txt_path)

        if len(rem_intervals) == 0:
            print(f" Pas de phase REM trouvée pour {base}")
            continue

        rem_raw = extract_rem_segments(raw, rem_intervals)
        if rem_raw is None:
            print(f" Impossible d'extraire les segments pour {base}")
            continue

        rem_raw.save(out_path, overwrite=True)
        print(f" Segments REM extraits et sauvegardés pour {base}")

