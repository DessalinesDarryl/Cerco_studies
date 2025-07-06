import os

def list_fif_files(directory):
    """
    Liste les fichiers .fif présents dans un dossier donné.

    Args:
        directory (str): Chemin vers le dossier.

    Returns:
        list: Liste triée des fichiers .fif
    """
    if not os.path.isdir(directory):
        return []
    return sorted([f for f in os.listdir(directory) if f.endswith(".fif")])

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
