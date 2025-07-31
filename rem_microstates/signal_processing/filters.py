import mne

def get_standard_bands():
    """
    Renvoie les bandes de fréquences EEG standards.
    """
    return {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 12),
        "Beta": (12, 30),
        "Gamma": (30, 100)
    }


def filter_band(raw, l_freq, h_freq):
    """
    Applique un filtre passe-bande à l'objet raw (copie).

    Args:
        raw (mne.io.Raw): objet EEG brut
        l_freq (float): fréquence basse (Hz)
        h_freq (float): fréquence haute (Hz)

    Returns:
        mne.io.Raw: objet filtré

    Raises:
        ValueError: si aucun canal EEG/EOG/EMG n’est disponible pour le filtrage.
    """
    # Sélectionne uniquement les canaux EEG, EOG ou EMG (filtrables)
    picks = mne.pick_types(raw.info, eeg=True, eog=True, emg=True)
    if len(picks) == 0:
        raise ValueError("Aucun canal EEG/EOG/EMG disponible pour le filtrage.")

    return raw.copy().filter(l_freq=l_freq, h_freq=h_freq, picks=picks)


def apply_custom_filters(raw):
    """
    Applique un filtrage bande-pass spécifique à chaque groupe de canaux :
    - EOG (gauche/droite) : 0.3–15 Hz
    - EMG (menton, jambes...) : 10–200 Hz
    - EEG (tous les autres canaux EEG) : 0.3–70 Hz
    """
    raw_filt = raw.copy()

    # EOG : 0.3–15 Hz
    eog_channels = [ch for ch in raw.ch_names if ch.startswith('EOG')]
    if eog_channels:
        raw_filt.filter(0.3, 15., picks=eog_channels, fir_design='firwin')

    # EMG : 10–200 Hz
    emg_channels = [ch for ch in raw.ch_names if any(kw in ch.lower() for kw in ["Menton", "JAMBG", "JAMBD", "EMG1", "EMG2"])]
    if emg_channels:
        raw_filt.filter(10., 200., picks=emg_channels, fir_design='firwin')

    # EEG : 0.3–70 Hz
    eeg_channels = [ch for ch in raw.ch_names if ch.startswith('EEG ')]
    if len(eeg_channels) > 0:
        raw_filt.filter(0.3, 70., picks=eeg_channels, fir_design='firwin')

    return raw_filt
