# src/utils.py

import os

def list_fif_files(directory):
    """
    Liste les fichiers .fif présents dans un dossier donné.

    Args:
        directory (str): Chemin vers le dossier.

    Returns:
        list: Liste triée des fichiers .fif
    """
    fif_files = []
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith('.fif'):
                full_path = os.path.join(dirpath, f)
                fif_files.append(os.path.relpath(full_path, directory))  
    return fif_files


def get_base_name(file_path, suffix="_eeg_cleaned_raw.fif"):
    """
    Supprime le suffixe standard pour obtenir le nom de base du fichier.

    Args:
        file_path (str): Nom de fichier
        suffix (str): Suffixe à retirer

    Returns:
        str: Nom de base sans suffixe
    """
    return os.path.basename(file_path).replace(suffix, "")

def find_first(path, patterns):
    for pat in patterns:
        matches = list(path.glob(pat))
        if matches:
            return matches[0]
    return None

def load_epochs(fif_path):
    epochs = mne.read_epochs(fif_path, preload=True, verbose="ERROR")
    # On garde uniquement EEG si multi-modal
    if 'eeg' in epochs.get_channel_types():
        epochs.pick_types(eeg=True)
    return epochs

def load_labels_any(labels_path):
    """
    Accepte :
      - CSV avec colonne 'label' (phasic/tonic)
      - TXT avec un label par ligne
      - JSON liste de strings
    """
    if labels_path.suffix.lower() == ".csv":
        df = pd.read_csv(labels_path)
        if 'label' not in df.columns:
            raise ValueError(f"{labels_path} doit contenir une colonne 'label'")
        return df['label'].astype(str).str.lower().tolist()
    elif labels_path.suffix.lower() == ".txt":
        with open(labels_path, 'r', encoding='utf-8') as f:
            return [line.strip().lower() for line in f if line.strip()]
    elif labels_path.suffix.lower() == ".json":
        with open(labels_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON de labels doit être une liste de strings.")
        return [str(x).lower() for x in data]
    else:
        raise ValueError(f"Extension non gérée pour labels: {labels_path.suffix}")

def align_labels_to_epochs(labels, n_epochs):
    """Tronque/complète silencieusement en 'tonic' pour matcher le nb d'époques."""
    if len(labels) >= n_epochs:
        return labels[:n_epochs]
    return labels + ["tonic"] * (n_epochs - len(labels))
