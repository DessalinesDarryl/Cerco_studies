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

# 2. ICA
def run_ica(raw, method="picard", n_comp=0.999, random_state=42, tstep=30.0, resample_hz=150):
    """
    Applique l’Analyse en Composantes Indépendantes (ICA) sur un signal EEG pour détecter
    et exclure les artefacts (EOG, ECG).

    Paramètres
    ----------
    raw : mne.io.Raw
        Données EEG brutes à nettoyer.

    method : str
        Méthode ICA à utiliser : "fastica", "picard", ou "infomax".

    n_comp : float ou int
        Nombre de composantes ICA à extraire. Peut être un float (variance expliquée) ou un entier.

    random_state : int
        Graine de génération aléatoire pour la reproductibilité.

    tstep : float
        Taille des fenêtres en secondes pour l’entraînement ICA.

    resample_hz : float ou None
        Fréquence d’échantillonnage temporaire utilisée pour accélérer l’entraînement ICA.

    Retour
    ------
    raw_clean : mne.io.Raw
        Données EEG avec les artefacts ICA supprimés.

    ica : mne.preprocessing.ICA
        Objet ICA ajusté, contenant les composantes exclues.
    """
    if method not in {"fastica", "picard", "infomax"}:
        raise ValueError("method doit être 'fastica', 'picard' ou 'infomax'")

    # Copie pour entraînement ICA sur EEG uniquement
    raw_fit = raw.copy().pick_types(eeg=True)
    if resample_hz is not None and resample_hz < raw_fit.info["sfreq"]:
        raw_fit = raw_fit.resample(resample_hz, npad="auto")

    fit_params = dict(ortho=False) if method == "picard" else {}
    ica = ICA(method=method, n_components=n_comp, random_state=random_state, fit_params=fit_params)
    ica.fit(raw_fit, tstep=tstep)

    # Détection des composantes EOG/ECG dans le raw complet
    eog_ch = next((c for c in raw.ch_names if "EOG" in c.upper()), None)
    ecg_ch = next((c for c in raw.ch_names if "ECG" in c.upper()), None)
    if eog_ch:
        bad_eog, _ = ica.find_bads_eog(raw, ch_name=eog_ch)
        ica.exclude.extend(bad_eog)
    if ecg_ch:
        bad_ecg, _ = ica.find_bads_ecg(raw, ch_name=ecg_ch)
        ica.exclude.extend(bad_ecg)

    # Application à tout le signal 
    return ica.apply(raw.copy()), ica

# 3. PIPELINE FICHIER UNIQUE
def process_file(edf_path, out_ica_dir, new_name, montage):
    """
    Traite un fichier EEG .edf : renommage des canaux, prétraitement (filtrage + référence), 
    ICA, puis sauvegarde au format .fif.

    Paramètres
    ----------
    edf_path : Path
        Chemin vers le fichier .edf à traiter.

    out_ica_dir : Path
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
            raw_ica, _ = run_ica(raw_p)
        except Exception as ica_err:
            return f"{edf_path.name}: erreur ICA -> {ica_err}"

        # Application ICA
        out_ica = out_ica_dir / new_name
        out_ica.parent.mkdir(parents=True, exist_ok=True)
        raw_ica.save(out_ica, overwrite=True, verbose="ERROR")

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
