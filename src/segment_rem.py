# src/segment_rem.py

import os
import mne
import platform
from pathlib import Path
from annotations import get_rem_annotations


def extract_rem_segments(fif_path, annot_dir, output_root):
    """
    Extrait et sauvegarde les segments de sommeil REM à partir d’un enregistrement EEG au format FIF,
    en utilisant un fichier d’annotations `.txt` correspondant (=hypnogramme).

    Paramètres
    ----------
    fif_path : str ou Path
        Chemin vers le fichier EEG prétraité au format `.fif`.

    annot_dir : str ou Path
        Répertoire contenant les fichiers d’annotations REM au format `.txt`. Le fichier doit
        correspondre au début du nom du fichier `.fif` (ex. `AN123` dans `AN123_preprocessed.fif`).

    output_root : str ou Path
        Répertoire racine dans lequel seront sauvegardés les segments REM extraits. 
        Un sous-dossier sera créé automatiquement pour chaque sujet.

    Comportement
    ------------
    - Charge le fichier `.fif` contenant les données EEG prétraitées.
    - Localise le fichier d’annotations REM correspondant dans `annot_dir`.
    - Pour chaque segment REM détecté, extrait la portion du signal correspondante et la sauvegarde
      dans un nouveau fichier `.fif` nommé `{base_name}_REM_{i}.fif`.
    - Ignore les segments déjà extraits (présents dans le dossier de sortie).
    - Affiche des messages de progression et gère les erreurs éventuelles.

    """
    base_name = Path(fif_path).stem.split('_')[0]
    output_dir = Path(output_root) / base_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraitement de {base_name}")
    print(f"Chargement du signal : {fif_path}")

    # Chargement du fichier FIF prétraité
    raw = mne.io.read_raw_fif(fif_path, preload=True)

    # Chargement des périodes REM à partir du fichier .txt
    rem_annotations = get_rem_annotations(base_name, annot_dir)
    if rem_annotations is None or len(rem_annotations) == 0:
        print(f"Aucune période REM trouvée pour {base_name}")
        return
    else: 
        print(f"{base_name} : {len(rem_annotations)} segments REM trouvés.")

    for i, (onset, duration, _) in enumerate(zip(rem_annotations.onset, rem_annotations.duration, rem_annotations.description), 1):
        try:
            print(f">>> Segment REM {i}: {onset:.1f}s à {onset+duration:.1f}s")
            segment_path = output_dir / f"{base_name}_REM_{i}.fif"
            if segment_path.exists():
                print(f" {segment_path.name} existe déjà, segment ignoré.")
                continue

            segment = raw.copy().crop(tmin=onset, tmax=onset + duration)
            segment.save(segment_path, overwrite=True)

        except Exception as e:
            print(f"Erreur pour le segment {i}: {e}")

if __name__ == "__main__":
    # On demande à l'utilisateur si le montage est bipolaire
    response = input("Le montage est-il bipolaire ? (y/n) : ").strip().lower()
    if response not in {"y", "n"}:
        print("Réponse invalide. Veuillez entrer 'y' pour oui ou 'n' pour non.")
        sys.exit(1)

    montage = "bipolaire" if response == "y" else "monopolaire"
    suffix = "bip" if response == "y" else "monop"
    print(f"montage défini={montage}")

    # Détection automatique du système
    system = platform.system()
    if system == "Darwin":  # MacOS
        disque = "/Volumes/Crucial X6"
    elif system == "Windows":
        disque = "D:"
    else:
        raise RuntimeError("Système non supporté.")

    # Chemins correctement interpolés
    fif_dir = Path(f"{disque}/EEG/preprocessed/{montage}/full")
    annot_dir = Path(f"{disque}/EEG/raw")
    output_root = Path(f"{disque}/EEG/preprocessed/{montage}/rem_only")

    # Boucle sur tous les fichiers .fif
    for fif_file in fif_dir.glob(f"*_preprocessed_{suffix}.fif"):
        if not fif_file.name.endswith(".fif") or fif_file.name.startswith("._"): #ignore les fichiers fantômes et autres que .fif
            continue
        extract_rem_segments(fif_file, annot_dir, output_root)
