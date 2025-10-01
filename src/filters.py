# src/filters.py

def get_standard_bands():
    """
    Retourne un dictionnaire des bandes de fréquences EEG standards avec leurs intervalles en Hz.

    Retour
    ------
    dict
        Dictionnaire où les clés sont les noms courts des bandes ("Delta", "Theta", etc.)
        et les valeurs sont des tuples (basse fréquence, haute fréquence).
    """
    return {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma Bas": (30, 50),
        "Gamma Haut": (50, 80),
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

def filter_all_bands(raw):
    """
    Applique un filtrage pour toutes les bandes EEG standard et retourne un dictionnaire.

    Paramètres
    ----------
    raw : mne.io.Raw
        Données EEG brutes.

    Retour
    ------
    dict
        Dictionnaire où chaque clé est le nom d'une bande EEG
        et chaque valeur est le signal EEG filtré correspondant (mne.io.Raw).
    """
    bands = get_standard_bands()
    return {name: filter_band(raw, l, h) for name, (l, h) in bands.items()}
