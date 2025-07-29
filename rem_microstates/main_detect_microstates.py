from signal_processing.loader import load_signals_and_annotations
from signal_processing.windowing import segment_rem_in_windows
from signal_processing.eog_analysis import detect_eog_bursts
from signal_processing.eeg_features import extract_spectral_features
from signal_processing.emg_analysis import evaluate_emg_twitching
from utils.plot_segments import plot_microstates
from pathlib import Path
import platform
import sys


def main():
    # On demande à l'utilisateur si le montage est bipolaire
    response = input("Le montage est-il bipolaire ? (y/n) : ").strip().lower()
    if response not in {"y", "n"}:
        print("Réponse invalide. Veuillez entrer 'y' pour oui ou 'n' pour non.")
        sys.exit(1)

    montage = "bipolaire" if response == "y" else "monopolaire"
    print(f"montage défini={montage}")

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

    print("Détection des REM toniques/phasiques....")
    labels = []
    for win in windows:
        eog_score = detect_eog_bursts(win)
        eeg_score = extract_spectral_features(win)
        emg_score = evaluate_emg_twitching(win)
        
        score = 0.5 * eog_score + 0.3 * eeg_score + 0.2 * emg_score
        labels.append("phasic" if score > 0.5 else "tonic")

    print("Affichage en cours...")
    plot_microstates(windows, labels)

if __name__ == "__main__":
    main()
