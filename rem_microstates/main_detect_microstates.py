import platform
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

from signal_processing.loader import load_signals_and_annotations
from signal_processing.windowing import segment_rem_in_windows
from signal_processing.eog_analysis import detect_eog_microstate
from signal_processing.annotation import annotate_microstates
from signal_processing.filters import apply_custom_filters
from utils.preprocessing_bip import apply_custom_bipolar_montage

import mne


def process_patient(fif_path, raw_dir, out_dir, montage):

    patient_id = fif_path.stem.split("_")[0]

    # Vérification d'intégrité avant toute chose
    if patient_id == "." or not patient_id.isalnum():
        print(f"[SKIP] Nom de patient invalide : {fif_path.name}")
        return

    annot_path = raw_dir / patient_id / f"{patient_id}_hypnoEXP.txt"
    print(f"annot_path = {annot_path}")
    
    if not annot_path.exists():
        print(f"[WARN] Annotations manquantes pour {patient_id}, ignoré.")
        return

    out_patient_dir = out_dir / patient_id
    out_patient_dir.mkdir(parents=True, exist_ok=True)

    # Vérifier si déjà traité
    xlsx_out = out_patient_dir / f"{patient_id}_microstates.xlsx"
    if xlsx_out.exists():
        print(f"[SKIP] {patient_id} déjà traité.")
        return

    # Chargement du signal et des annotations
    raw, rem_segments = load_signals_and_annotations(fif_path, annot_path)

    print(f"[{patient_id}] {len(rem_segments)} segments REM détectés.")
    windows = segment_rem_in_windows(raw, rem_segments, window_sec=4, step_sec=4)

    labels = []
    valid_windows = []

    i = 0
    while i < len(windows):
        win = windows[i]
        label = detect_eog_microstate(win)

        if label != "ignore":
            labels.append(label)
            valid_windows.append(win)

        i += 2 if label == "phasic" else 1

    print(f"[{patient_id}] {len(valid_windows)} fenêtres retenues ({labels.count('phasic')} phasic / {labels.count('tonic')} tonic)")

    # Annoter et sauvegarder
    annotate_microstates(raw, valid_windows, labels, window_sec=4)
    raw_annotated_path = out_patient_dir / f"{patient_id}_annotated.fif"
    raw.save(raw_annotated_path, overwrite=True)

    # Résumé des fenêtres
    df = pd.DataFrame({
        "tmin": [win.first_time for win in valid_windows],
        "tmax": [win.first_time + 4 for win in valid_windows],
        "label": labels
    })
    df.to_excel(xlsx_out, index=False)

    # Détection des erreurs : phasic inclus dans REM phasique
    mat_path = raw_dir / patient_id / f"events_{patient_id}_bipolaire.mat"
    if mat_path.exists():
        from scipy.io import loadmat
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        events = mat["events"]
        rem_phasic = [(e.times[0], e.times[1]) for e in events if e.label == "REM phasique"]

        # Fonction d’inclusion
        def is_inside(start, end, intervals, tol=0.5):
            return any((s - tol) <= start and end <= (e + tol) for s, e in intervals)

        erreurs = []
        bons = []

        for i, row in df.iterrows():
            start, end, pred = row["tmin"], row["tmax"], row["label"]
            true = "phasic" if is_inside(start, end, rem_phasic) else "tonic"
            result = {
                "tmin": start,
                "tmax": end,
                "auto_label": pred,
                "true_label": true
            }
            if pred != true:
                erreurs.append(result)
            else:
                bons.append(result)

        if erreurs:
            df_err = pd.DataFrame(erreurs)
            df_err.to_excel(out_patient_dir / f"{patient_id}_microstates_errors.xlsx", index=False)
            print(f"[{patient_id}] {len(erreurs)} erreurs sauvegardées.")
        else:
            print(f"[{patient_id}] 0 erreurs, pas de fichier généré.")
    else:
        print(f"[{patient_id}] Pas d’événements .mat pour comparaison manuelle.")

    print(f"[{patient_id}] Traitement terminé.\n")


if __name__ == "__main__":
    response = input("Montage bipolaire ? (y/n) : ").strip().lower()
    if response not in {"y", "n"}:
        print("Réponse invalide. Tape 'y' ou 'n'.")
        exit()
    montage = "bipolaire" if response == "y" else "monopolaire"

    # Détection du système
    system = platform.system()
    if system == "Darwin":
        disque = "/Volumes/Crucial X6"
    elif system == "Windows":
        disque = "D:"
    elif system == "Linux":
        disque = "/media/darryld/Crucial X6"
    else:
        raise RuntimeError("Système non supporté.")

    root_preproc = Path(f"{disque}/EEG/preprocessed/{montage}/full/")
    root_raw = Path(f"{disque}/EEG/raw")
    root_out = Path(f"{disque}/EEG/preprocessed/{montage}/rem_only")

    fif_paths = [
    p for p in root_preproc.rglob("*_preprocessed_*.fif")
    if not p.name.startswith("._") and not p.name.startswith(".")
    ]

    print(f"{len(fif_paths)} fichiers trouvés dans {root_preproc}")

    for path in fif_paths:
        process_patient(path, root_raw, root_out, montage)
