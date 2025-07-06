import mne

def filter_band(raw, l_freq, h_freq):
    """
    Applique un filtre passe-bande entre l_freq et h_freq sur un Raw EEG.

    Args:
        raw (mne.io.Raw): Données EEG brutes.
        l_freq (float): Fréquence basse du filtre (Hz).
        h_freq (float): Fréquence haute du filtre (Hz).

    Returns:
        raw_filtered (mne.io.Raw): Données EEG filtrées.
    """
    raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq)
    return raw_filtered

def get_standard_bands():
    """
    Renvoie un dictionnaire des bandes de fréquences EEG standard.

    Returns:
        dict: Nom de bande -> (l_freq, h_freq)
    """
    return {
        "Delta (0.5–4 Hz)": (0.5, 4),
        "Theta (4–8 Hz)": (4, 8),
        "Alpha (8–13 Hz)": (8, 13),
        "Beta (13–30 Hz)": (13, 30),
        "Gamma (30–45 Hz)": (30, 45)
    }

def filter_all_bands(raw):
    """
    Filtre un signal brut dans toutes les bandes standards et retourne un dict.

    Args:
        raw (mne.io.Raw): Données EEG brutes.

    Returns:
        dict: Nom de bande -> Raw filtré
    """
    bands = get_standard_bands()
    return {name: filter_band(raw, l, h) for name, (l, h) in bands.items()}