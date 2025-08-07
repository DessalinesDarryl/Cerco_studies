# src/preprocessing.py

import os
import sys
from pathlib import Path
import platform
import mne
import numpy as np
from mne.preprocessing import ICA
from utils import apply_custom_bipolar_montage
from filters import apply_custom_filters

# 1. FILTRAGE & RÉFÉRENCEMENT
def preprocess_raw_A1(raw, l_freq=0.3, h_freq=100, notch=50, ref="A1"): 
    """
    Applique un filtrage passe-bande, un filtre notch, puis référence le signal EEG 
    par rapport au canal spécifié (par défaut A1).

    Paramètres
    ----------
    raw : mne.io.Raw
        Données EEG brutes à prétraiter.

    l_freq : float
        Fréquence de coupure basse pour le filtrage.

    h_freq : float
        Fréquence de coupure haute pour le filtrage.

    notch : float
        Fréquence du filtre notch (pour supprimer le bruit secteur).

    ref : str
        Canal de référence EEG à utiliser (ex. : "A1"). Si le canal n’est pas présent, 
        une référence moyenne est utilisée à la place.

    Retour
    ------
    raw_f : mne.io.Raw
        Données EEG filtrées et référencées.
    """
    raw_f = raw.copy().filter(l_freq, h_freq, fir_design="firwin")
    raw_f.notch_filter(notch)
    if ref in raw_f.ch_names:
        raw_f.set_eeg_reference([ref])
    else:
        raw_f.set_eeg_reference("average", projection=False)
    return raw_f

def preprocess_raw_bip(raw, l_freq=0.3, h_freq=200, notch=50):
    """
    Applique un filtrage passe-bande, un filtre notch, puis construit un montage EEG bipolaire 
    à partir d’un ensemble de paires de canaux définies.

    Paramètres
    ----------
    raw : mne.io.Raw
        Données EEG brutes à prétraiter.

    l_freq : float
        Fréquence de coupure basse pour le filtrage.

    h_freq : float
        Fréquence de coupure haute pour le filtrage.

    notch : float
        Fréquence du filtre notch (généralement 50 Hz pour supprimer le bruit secteur).

    Retour
    ------
    raw_f : mne.io.Raw
        Données EEG filtrées avec montage bipolaire appliqué.
    """
    raw_f = raw.copy().filter(l_freq, h_freq, fir_design="firwin")
    raw_f.notch_filter(notch)

    print("Application du montage bipolaire personnalisé...")
    print(f"Liste des canaux AVANT : {raw_f.ch_names}")
    raw_f.rename_channels(lambda name: name.replace("EEG ", ""))
    print(f"Liste des canaux APRÈS : {raw_f.ch_names}")
    raw_f = apply_custom_bipolar_montage(raw)

    print("Application des filtres sur les canaux EOG, EMG et EEG...")
    raw_f = apply_custom_filters(raw)

    return raw_f

# 2. Nettoyage par YASA (art_detect + interpolation)
def detect_and_interpolate_artifacts(raw, edf_path, win_sec=4, method='covar', threshold=3):
    """
    Détecte et interpole les artéfacts dans les données EEG à l'aide de YASA (méthode 'covar').
    Ajoute également des annotations MNE sur les segments corrompus et exporte les fenêtres en CSV + log TXT.

    Paramètres
    ----------
    raw : mne.io.Raw
        Données EEG brutes filtrées.

    edf_path : Path
        Chemin du fichier de sortie (même dossier que le .fif de sortie).

    win_sec : int
        Longueur des fenêtres temporelles (en secondes).

    method : str
        Méthode de détection YASA (par défaut 'covar').

    threshold : float
        Seuil de détection pour les artéfacts.

    Retour
    ------
    raw_interp : mne.io.Raw
        Données EEG nettoyées avec interpolation des artéfacts et annotations ajoutées.

    valid_mask : np.ndarray
        Masque booléen des échantillons valides (True = propre).
    """
    import yasa
    import pandas as pd
    import numpy as np
    import mne

    sfreq = raw.info['sfreq']
    data = raw.get_data()
    n_samples = data.shape[1]
    win_samples = int(win_sec * sfreq)

    # === 1. Détection des artéfacts avec YASA
    art, zscores = yasa.art_detect(data, sf=sfreq, window=win_sec, method=method, threshold=threshold)

    # === 2. Création du masque temporel
    valid_mask = np.full(n_samples, False)
    for i, is_art in enumerate(art):
        if not is_art:
            start = i * win_samples
            end = min(start + win_samples, n_samples)
            valid_mask[start:end] = True

    # === 3. Interpolation linéaire des artéfacts
    interp_data = data.copy()
    x = np.arange(n_samples)
    for ch in range(data.shape[0]):
        good = valid_mask
        bad = ~valid_mask
        interp_data[ch, bad] = np.interp(x[bad], x[good], data[ch, good])

    # === 4. Export CSV des fenêtres d'artéfacts
    art_windows = []
    for i, is_art in enumerate(art):
        if is_art:
            start_sec = i * win_sec
            end_sec = start_sec + win_sec
            art_windows.append((start_sec, end_sec))

    df_art = pd.DataFrame(art_windows, columns=["start_time_s", "end_time_s"])

    out_dir = edf_path.parent
    export_path = out_dir / f"{edf_path.stem}_artifact_windows.csv"
    df_art.to_csv(export_path, index=False)
    print(f"Artéfacts exportés : {len(df_art)} fenêtres -> {export_path.name}")

    # === 5. Ajout d’annotations BAD dans le Raw
    onset = [start for start, end in art_windows]
    duration = [win_sec] * len(onset)
    description = ["BAD_Artifact"] * len(onset)
    annotations = mne.Annotations(onset=onset, duration=duration, description=description)
    
    # === 6. Création d’un nouvel objet Raw avec les données interpolées et annotations
    raw_interp = raw.copy()
    raw_interp._data = interp_data
    raw_interp.set_annotations(annotations)

    # === 7. Log du pourcentage de signal conservé
    pourcentage_conserve = valid_mask.sum() / len(valid_mask) * 100
    log_path = out_dir / "artifact_report.txt"
    with open(log_path, "a") as f:
        f.write(f"{edf_path.stem}.fif : {pourcentage_conserve:.2f}% du signal conservé après suppression des artéfacts.\n")

    return raw_interp, valid_mask



# 3. PIPELINE FICHIER UNIQUE
def process_file(edf_path, out_yasa_dir, new_name, montage):
    """
    Traite un fichier EEG .edf : renommage des canaux, prétraitement (filtrage + référence), 
    ICA, puis sauvegarde au format .fif.

    Paramètres
    ----------
    edf_path : Path
        Chemin vers le fichier .edf à traiter.

    out_yasa_dir : Path
        Répertoire de sortie où le fichier .fif prétraité sera sauvegardé.

    new_name : str
        Nom du fichier .fif de sortie.

    montage : str
        Type de montage EEG : "bipolaire" ou "monopolaire".

    Retour
    ------
    str
        Message de succès ou d'erreur indiquant le statut du traitement.
    """
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

        # On renomme les canaux commençant par "EGG" en supprimant le préfixe
        print(f"Liste des canaux actuels:{raw.ch_names}")
        rename_dict = {
            ch: ch.replace("EEG ", "", 1) for ch in raw.ch_names if ch.startswith("EEG ")
        }
        if rename_dict:
            raw.rename_channels(rename_dict)
            print(f"Liste des canaux à jour : {raw.ch_names}")

        if montage == "bipolaire": 
            raw_p = preprocess_raw_bip(raw)
        else:
            raw_p = preprocess_raw_A1(raw)

        try:
            raw_clean, valid_mask = detect_and_interpolate_artifacts(raw_p, out_yasa_dir)
        except Exception as clean_err:
            return f"{edf_path.name}: erreur YASA -> {clean_err}"


        # Sauvegarde
        out_yasa = out_yasa_dir / new_name
        out_yasa.parent.mkdir(parents=True, exist_ok=True)
        raw_clean.save(out_yasa, overwrite=True, verbose="ERROR")

        return f"{edf_path.name}: OK"

    except Exception as e:
        return f"{edf_path.name}: erreur générale -> {e}"

# 4. MAIN AUTOMATIQUE
if __name__ == "__main__":
    # On demande à l'utilisateur si le montage est bipolaire
    response = input("Le montage est-il bipolaire ? (y/n) : ").strip().lower()
    if response not in {"y", "n"}:
        print("Réponse invalide. Veuillez entrer 'y' pour oui ou 'n' pour non.")
        sys.exit(1)

    montage = "bipolaire" if response == "y" else "monopolaire"
    print(f"montage défini={montage}")

    # Adaptation système
    system = platform.system()
    if system == "Darwin":  # macOS
        disque = "/Volumes/Crucial X6"
    elif system == "Windows":
        disque = "D:"
    else:
        raise RuntimeError("Système non supporté.")

    root_raw = Path(f"{disque}/EEG/raw")
    out_base = Path(f"{disque}/EEG/preprocessed")
    out_ica_dir = out_base / montage / "full"

    edf_paths = list(root_raw.rglob("*.edf"))
    print(f"{len(edf_paths)} fichiers .edf trouvés.")

    for i, path in enumerate(edf_paths):
        parent_name = path.parent.name
        suffix = "bip" if montage == "bipolaire" else "monop"
        new_name = f"{parent_name}_preprocessed_{suffix}.fif"

        print(f"\nTraitement de {path.name} -> sauvegarde sous {new_name}")
        out_fif = out_ica_dir / new_name
        if out_fif.exists():
            print(f"{new_name} déjà traité, ignoré.")
            continue

        res = process_file(
            edf_path=path,
            out_ica_dir=out_ica_dir,
            new_name=new_name,
            montage=montage
        )
        print(res)
