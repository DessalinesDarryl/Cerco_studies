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
                fif_files.append(os.path.relpath(full_path, directory))  # chemin relatif
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
