import os
import mne

def preprocess_eeg_file(file_path, save_dir=None, n_components=0.99):
    """
    Prétraite un fichier .edf en filtrant, référant, nettoyant par ICA,
    et sauvegarde le fichier .fif nettoyé.

    Étapes :
    - Filtrage passe-bande 10–100 Hz
    - Notch filter à 50 Hz (élimine le bruit secteur)
    - Référencement par le canal A1
    - ICA pour supprimer les artéfacts (EOG, ECG)

    Args:
        file_path (str): Chemin du fichier .edf brut
        save_dir (str): Dossier de sauvegarde des fichiers .fif nettoyés
        n_components (float or int): Nombre de composantes ICA (ex. 0.99 pour 99% de variance)
    """
    # Chargement du fichier
    raw = mne.io.read_raw_edf(file_path, preload=True)

    # Vérification de la présence du canal A1
    if "A1" not in raw.ch_names:
        raise ValueError("Canal A1 non trouvé pour la référence.")

    # Référencement par A1 : signal = signal - A1
    raw.set_eeg_reference(ref_channels=["A1"])

    # Sélection des canaux EEG, EOG, ECG
    picks = [ch for ch in raw.ch_names if ch.startswith("EEG") or ch.startswith("EOG") or ch.startswith("ECG")]
    raw.pick_channels(picks)

    # Filtrage passe-bande 10–100 Hz
    raw.filter(l_freq=10., h_freq=100.)

    # Notch filter à 50 Hz
    raw.notch_filter(freqs=50)

    # ICA pour supprimer les artéfacts
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=42, max_iter="auto")
    ica.fit(raw)

    # Détection des composantes corrélées à l’EOG
    eog_ch = next((ch for ch in raw.ch_names if "EOG" in ch.upper()), None)
    if eog_ch:
        eog_inds, _ = ica.find_bads_eog(raw, ch_name=eog_ch)
        ica.exclude.extend(eog_inds)

    # Détection des composantes corrélées à l’ECG
    ecg_ch = next((ch for ch in raw.ch_names if "ECG" in ch.upper()), None)
    if ecg_ch:
        ecg_inds, _ = ica.find_bads_ecg(raw, ch_name=ecg_ch)
        ica.exclude.extend(ecg_inds)

    # Application de la correction ICA
    raw_clean = raw.copy()
    ica.apply(raw_clean)

    # Sauvegarde du signal nettoyé
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(save_dir, f"{base_name}_eeg_cleaned_raw.fif")
    raw_clean.save(out_path, overwrite=True)

    print(f" {base_name} prétraité et sauvegardé dans {out_path}")
