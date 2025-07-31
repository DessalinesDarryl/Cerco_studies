import matplotlib
matplotlib.use("Agg")

from signal_processing.loader import load_signals_and_annotations
from signal_processing.windowing import segment_rem_in_windows
from signal_processing.eog_analysis import detect_eog_microstate
from signal_processing.annotation import annotate_microstates
from signal_processing.filters import apply_custom_filters


from pathlib import Path
import platform
import sys
import pandas as pd



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

    edf_path = Path(f"{disque}/EEG/raw/MN143/MN143_raw.edf")
    annot_path = Path(f"{disque}/EEG/raw/MN143/MN143_hypnoEXP.txt")

    print("Chargement des fichiers...")
    raw, rem_segments = load_signals_and_annotations(edf_path, annot_path)

    print("Application des filtres sur les canaux EOG, EMG et EEG...")
    raw = apply_custom_filters(raw)

    print(f"{len(rem_segments)} segments REM détectés.")
    windows = segment_rem_in_windows(raw, rem_segments, window_sec=4, step_sec=4)

    labels = []
    valid_windows = []
    for win in windows:
        label = detect_eog_microstate(win)
        if label != "ignore":
            labels.append(label)
            valid_windows.append(win)

    print(f"{len(valid_windows)} fenêtres retenues ({labels.count('phasic')} phasic / {labels.count('tonic')} tonic)")

    annotate_microstates(raw, valid_windows, labels, window_sec=4)

    output_dir = Path(f"{disque}/EEG/raw/MN143")
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / f"{edf_path.stem}_annotated.fif"
    raw.save(annotated_path, overwrite=True)
    print(f"[OK] Fichier annoté sauvegardé : {annotated_path}")

    df = pd.DataFrame({
        "tmin": [win.first_time for win in valid_windows],
        "tmax": [win.first_time + 4 for win in valid_windows],
        "label": labels
    })
    df.to_excel(output_dir / f"{edf_path.stem}_microstates.xlsx", index=False)
    print("[OK] Export Excel terminé.")

if __name__ == "__main__":
    main()
