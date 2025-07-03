import os
import mne

def preprocess_eeg_file(file_path, save_dir="../data/preprocessed", n_components=0.99):
    """
    Prétraite un fichier .edf en sélectionnant les canaux EEG,
    appliquant ICA (si possible), et sauvegarde le signal nettoyé.
    
    Args:
        file_path (str): Chemin vers le fichier .edf brut
        save_dir (str): Dossier où sauvegarder le .fif nettoyé
        n_components (float or int): Nombre de composantes ICA
    """
    # Chargement...
    raw = mne.io.read_raw_edf(file_path, preload=True)
    
    # Sélectionner les canaux EEG + EOG/ECG pour ICA 
    picks = [ch for ch in raw.ch_names if ch.startswith("EEG") or ch.startswith("EOG")]
    raw.pick_channels(picks)
    
    # Filtrage passe-haut à 1 Hz
    raw.filter(l_freq=1., h_freq=None)
    
    # ICA
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=42, max_iter='auto')
    ica.fit(raw)

    # Détection EOG
    eog_ch = next((ch for ch in raw.ch_names if 'EOG' in ch.upper()), None)
    if eog_ch:
        eog_inds, _ = ica.find_bads_eog(raw, ch_name=eog_ch)
        ica.exclude.extend(eog_inds)

    # Détection ECG
    ecg_ch = next((ch for ch in raw.ch_names if 'ECG' in ch.upper()), None)
    if ecg_ch:
        ecg_inds, _ = ica.find_bads_ecg(raw, ch_name=ecg_ch)
        ica.exclude.extend(ecg_inds)

    # Application du nettoyage
    raw_clean = raw.copy()
    ica.apply(raw_clean)

    # Sauvegarde
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(save_dir, f"{base_name}_eeg_cleaned_raw.fif")
    raw_clean.save(out_path, overwrite=True)
    
    print(f" {base_name} prétraité et sauvegardé dans {out_path}")

