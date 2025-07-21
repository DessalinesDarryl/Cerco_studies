# src/filters.py

def get_standard_bands():
    """
    Retourne un dictionnaire des bandes de fréquences EEG standards avec leurs intervalles en Hz.

    Retour
    ------
    dict
        Dictionnaire où les clés sont les noms des bandes ("Delta", "Theta", etc.)
        et les valeurs sont des tuples (basse fréquence, haute fréquence).
    """

    return {
        "Delta (0.5–4 Hz)": (0.5, 4),
        "Theta (4–8 Hz)": (4, 8),
        "Alpha (8–13 Hz)": (8, 13),
        "Beta (13–30 Hz)": (13, 30),
        "Gamma (30–45 Hz)": (30, 45)
    }

def filter_band(raw, l_freq, h_freq):
    """
    Applique un filtrage passe-bande sur les données EEG entre deux fréquences données.

    Paramètres
    ----------
    raw : mne.io.Raw
        Données EEG brutes à filtrer.

    l_freq : float
        Fréquence de coupure basse (Hz).

    h_freq : float
        Fréquence de coupure haute (Hz).

    Retour
    ------
    raw_filtered : mne.io.Raw
        Données EEG filtrées dans la bande spécifiée.
    """

    return raw.copy().filter(l_freq=l_freq, h_freq=h_freq)
