from signal_processing.loader import load_signals_and_annotations
from signal_processing.windowing import segment_rem_in_windows
from signal_processing.eog_analysis import detect_eog_bursts
from signal_processing.eeg_features import extract_spectral_features
from signal_processing.emg_analysis import evaluate_emg_twitching
from utils.microstate_annotation import annotate_microstates_to_raw
from pathlib import Path
import platform
import sys
import mne  

def main():
    # On demande à l'utilisateur si le montage est bipolaire
    response = input("Le montage est-il bipolaire ? (y/n) : ").strip().lower()
    if response not in {"y", "n"}:
        print("Réponse invalide. Veuillez entrer 'y' pour oui ou 'n' pour non.")
        sys.exit(1)

    montage = "bipolaire" if response == "y" else "monopolaire"
    print(f"montage défini = {montage}")

    # Détection automatique du système
    system = platform.system()
    if system == "Darwin":  # MacOS
        disque = "/Volumes/Crucial X6"
    elif system == "Windows":
        disque = "D:"
    elif system == "Linux":
        disque = "/media/darryld/Crucial X6"
    else:
        raise RuntimeError("Système non supporté.")

    edf_path = Path(f"{disque}/EEG/raw/MN143/MN143_raw.edf")
    annot_path = Path(f"{disque}/EEG/raw/MN143/MN143_hypnoEXP.txt")
    
    print("Chargement des fichiers...")
    raw, rem_segments = load_signals_and_annotations(edf_path, annot_path)
    print("Identification des segments REM...")
    windows = segment_rem_in_windows(raw, rem_segments)

    print("Détection des REM toniques/phasiques...")
    labels = []
    for win in windows:
        eog_score = detect_eog_bursts(win)
        eeg_score = extract_spectral_features(win)
        emg_score = evaluate_emg_twitching(win)

        score = 0.5 * eog_score + 0.3 * eeg_score + 0.2 * emg_score
        labels.append("phasic" if score > 0.5 else "tonic")

    print("Annotation dans le fichier MNE en cours...")
    raw_annotated = annotate_microstates_to_raw(raw, windows, labels, win_len_sec=4.0)

    output_path = edf_path.with_name(edf_path.stem + "_annotated.fif")
    raw_annotated.save(output_path, overwrite=True)
    print(f"Fichier annoté sauvegardé : {output_path}")

if __name__ == "__main__":
    main()
