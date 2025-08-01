import mne
import pandas as pd
from pathlib import Path
import platform
from utils.preprocessing_bip import apply_custom_bipolar_montage
from signal_processing.filters import apply_custom_filters


system = platform.system()
if system == "Darwin":
    disque = "/Volumes/Crucial X6"
elif system == "Windows":
    disque = "D:"
elif system == "Linux":
    disque = "/media/darryld/Crucial X6"
else:
    raise RuntimeError("Système non supporté.")

def export_amplitudes_segment(edf_path, tmin, tmax, output_path):
    print(f"[INFO] Chargement du fichier EDF : {edf_path}")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    #Montage bipolaire 
    print("Application du montage bipolaire personnalisé...")
    print(f"Liste des canaux AVANT : {raw.ch_names}")
    raw.rename_channels(lambda name: name.replace("EEG ", ""))
    print(f"Liste des canaux APRÈS : {raw.ch_names}")
    raw = apply_custom_bipolar_montage(raw)

    # Filtre 0.3-15Hz
    print("Application des filtres sur les canaux EOG, EMG et EEG...")
    raw = apply_custom_filters(raw)

    # Vérification des bornes
    tmin = max(0, tmin)
    tmax = min(tmax, raw.times[-1])

    # Découpage du signal
    raw_crop = raw.copy().crop(tmin=tmin, tmax=tmax)

    # Conversion µV
    data = raw_crop.get_data() * 1e6
    times = raw_crop.times + tmin  # Réajustement absolu du temps

    # Création du DataFrame
    df = pd.DataFrame(data.T, columns=raw_crop.ch_names)
    df.insert(0, "time_sec", times)

    # Export CSV
    df.to_csv(output_path, index=False)
    print(f"[OK] Export CSV terminé : {output_path}")

if __name__ == "__main__":
    # Exemples interactifs
    edf_path = input("Chemin vers le fichier .edf : ").strip()
    tmin = float(input("Temps de début (en secondes) : "))
    tmax = float(input("Temps de fin (en secondes) : "))
    output_path = Path(f"{disque}/EEG/raw/BB114/amplitude_fenetre.csv")


    export_amplitudes_segment(edf_path, tmin, tmax, output_path)
