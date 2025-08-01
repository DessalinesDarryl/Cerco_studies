import matplotlib
matplotlib.use("Agg")

from signal_processing.loader import load_signals_and_annotations
from signal_processing.windowing import segment_rem_in_windows
from signal_processing.eog_analysis import detect_eog_microstate
from signal_processing.annotation import annotate_microstates
from signal_processing.filters import apply_custom_filters
from utils.preprocessing_bip import apply_custom_bipolar_montage

from pathlib import Path
import platform
import sys
import pandas as pd
import numpy as np


def main():
    response = input("Le montage est-il bipolaire ? (y/n) : ").strip().lower()
    if response not in {"y", "n"}:
        print("Réponse invalide. Veuillez entrer 'y' ou 'n'.")
        sys.exit(1)
    montage = "bipolaire" if response == "y" else "monopolaire"
    print(f"montage défini={montage}")

    system = platform.system()
    if system == "Darwin":
        disque = "/Volumes/Crucial X6"
    elif system == "Windows":
        disque = "D:"
    elif system == "Linux":
        disque = "/media/darryld/Crucial X6"
    else:
        raise RuntimeError("Système non supporté.")

    #Fichiers sources
    edf_path = Path(f"{disque}/EEG/raw/BB114/BB114_raw.edf")
    annot_path = Path(f"{disque}/EEG/raw/BB114/BB114_hypnoEXP.txt")

    print("Chargement des fichiers...")
    raw, rem_segments = load_signals_and_annotations(edf_path, annot_path)

    # Nettoyage des noms si montage bipolaire
    if montage == "bipolaire":
        print("Application du montage bipolaire personnalisé...")
        print(f"Liste des canaux AVANT : {raw.ch_names}")
        raw.rename_channels(lambda name: name.replace("EEG ", ""))
        print(f"Liste des canaux APRÈS : {raw.ch_names}")
        raw = apply_custom_bipolar_montage(raw)

    print("Application des filtres sur les canaux EOG, EMG et EEG...")
    raw = apply_custom_filters(raw)

    print(f"{len(rem_segments)} segments REM détectés.")
    windows = segment_rem_in_windows(raw, rem_segments, window_sec=4, step_sec=4)

    # Labélisation de la fenêtre (tonic/phasic)
    labels = []
    valid_windows = []

    i = 0
    while i < len(windows):
        win = windows[i]
        label = detect_eog_microstate(win)

        if label != "ignore":
            labels.append(label)
            valid_windows.append(win)

        # Saut de 8 s (= 2 fenêtres) si "phasic", sinon on passe à la suivante
        if label == "phasic":
            i += 2
        else:
            i += 1

    print(f"{len(valid_windows)} fenêtres retenues ({labels.count('phasic')} phasic / {labels.count('tonic')} tonic)")

    annotate_microstates(raw, valid_windows, labels, window_sec=4)

    output_dir = Path(f"{disque}/EEG/raw/BB114")
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / f"{edf_path.stem}_annotated.fif"
    raw.save(annotated_path, overwrite=True)
    print(f"[OK] Fichier annoté sauvegardé : {annotated_path}")

    # Sauvegarde du résumé des fenêtres REM labélisées tonic/phasic
    df = pd.DataFrame({
        "tmin": [win.first_time for win in valid_windows],
        "tmax": [win.first_time + 4 for win in valid_windows],
        "label": labels
    })
    df.to_excel(output_dir / f"{edf_path.stem}_microstates.xlsx", index=False)
    print("[OK] Export resumé des fenêtres Excel terminé.")


if __name__ == "__main__":
    main()
