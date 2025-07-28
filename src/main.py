import os
import sys
import mne
import numpy as np
import platform
from pathlib import Path
from glob import glob

from preprocess_psg import preprocess_psg
from train_model_SleepStages import train_model
from predict_SleepStages import predict
from extract_rem_epochs import extract_rem_epochs
from plot_hypnodensity import plot_hypnodensity
from annotations import load_all_annotations_from_file
from evaluate_model import evaluate_model
from sklearn.model_selection import GroupShuffleSplit


def load_all_patients_data(raw_root, stage_map):
    """
    Charge et concatène tous les enregistrements .edf + hypnogrammes .txt dans un répertoire donné.

    Retourne les données PSG, les labels, et l’ID patient de chaque époque.
    """
    X_list = []
    y_list = []
    patient_ids = []

    patient_dirs = sorted(Path(raw_root).glob("*"))
    print(f"Détection de {len(patient_dirs)} dossiers patients...")

    for patient_dir in patient_dirs:
        edf_files = list(patient_dir.glob("*.edf"))
        txt_files = list(patient_dir.glob("*.txt"))

        if not edf_files or not txt_files:
            print(f" >>> Skipping {patient_dir.name} (missing .edf or .txt)")
            continue

        edf_path = edf_files[0]
        txt_path = txt_files[0]

        print(f"Chargement : {patient_dir.name}")
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            X = preprocess_psg(raw)
            y = load_all_annotations_from_file(txt_path, stage_map=stage_map)

            if len(X) != len(y):
                print(f" >>> Mismatch epochs/labels for {patient_dir.name}, skipping.")
                continue

            X_list.append(X)
            y_list.append(y)
            patient_ids.extend([patient_dir.name] * len(y))

        except Exception as e:
            print(f"Erreur avec {patient_dir.name} : {e}")
            continue

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    patient_ids = np.array(patient_ids)
    print(f"Total époques chargées : {len(y_all)}")
    return X_all, y_all, patient_ids


def train_test_split_by_patient(X_all, y_all, patient_ids, test_size=0.2, random_state=42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X_all, y_all, groups=patient_ids))
    return X_all[train_idx], X_all[test_idx], y_all[train_idx], y_all[test_idx]


def main():
    """
    Main function for training and evaluating the sleep stage classification model.
    """
    response = input("Le montage est-il bipolaire ? (y/n) : ").strip().lower()
    if response not in {"y", "n"}:
        print("Réponse invalide. Veuillez entrer 'y' ou 'n'.")
        sys.exit(1)
    montage = "bipolaire" if response == "y" else "monopolaire"
    print(f"montage défini = {montage}")

    system = platform.system()
    if system == "Darwin":
        disque = "/Volumes/Crucial X6"
    elif system == "Windows":
        disque = "D:"
    elif system == "Linux":
        disque = "/media/darryld/Crucial X6"
    else:
        raise RuntimeError("Système non supporté.")

    raw_root = Path(f"{disque}/EEG/raw")
    use_ground_truth = True

    class_names = ['W', 'N1', 'REM', 'N2', 'N3']
    stage_map = {name: i for i, name in enumerate(class_names)}
    reverse_stage_map = {0: 1, 1: 3, 2: 2, 3: 4, 4: 5}

    # Chargement des données multi-patients
    print("Loading all patient data...")
    X_all, y_all, patient_ids = load_all_patients_data(raw_root, stage_map=stage_map)

    # Split train/test par patient
    print("Splitting data by patient...")
    X_train, X_test, y_train, y_test = train_test_split_by_patient(X_all, y_all, patient_ids)

    # Entraînement
    print("Training model...")
    model = train_model(X_train, y_train, epochs=10, batch_size=16)

    # Prédiction sur le test set
    print("Predicting on test set...")
    y_pred, probas = predict(model, X_test)

    y_pred_original = np.array([reverse_stage_map[y] for y in y_pred])
    y_true_original = np.array([reverse_stage_map[y] for y in y_test])

    # Visualisation hypnodensity
    print("Plotting hypnodensity on test set...")
    plot_hypnodensity(
        probas,
        y_pred_original,
        y_true=y_true_original,
        class_names=class_names[::-1],  # REM en haut
        save_path="models/hypnodensity.png"
    )

    # Extraction des segments REM
    rem_segments = extract_rem_epochs(y_pred, epoch_length=30, fs=200)
    print(f"Detected {len(rem_segments)} REM epochs.")
    for i, (start, end) in enumerate(rem_segments[:5]):
        print(f"  REM segment {i+1}: samples {start} to {end}")

    # Évaluation
    print("Evaluating model performance...")
    class_labels_original = [reverse_stage_map[i] for i in range(len(class_names))]
    evaluate_model(
        y_true_original,
        y_pred_original,
        class_labels=class_labels_original,
        class_names=class_names,
        save_path="models/confusion_matrix.png"
    )


if __name__ == "__main__":
    main()
