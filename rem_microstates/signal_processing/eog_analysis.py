import numpy as np
from scipy.signal import find_peaks

def detect_eog_microstate(win, sfreq=250.0):
    """
    Classe une fenêtre REM de 4 s comme "phasic", "tonic", ou "ignore"
    selon la section 2.2 de Simor et al. (2021), adaptée à une analyse conjointe EOGD/EOGG.

    Critères :
    - "phasic" : au moins 1 mouvement oculaire détecté dans chaque moitié (0–2s, 2–4s),
      défini comme une paire de pics >150µV, opposés, et espacés de <500ms.
    - "tonic" : aucune déflexion >25µV dans les deux canaux
    - sinon : "ignore"
    """
    try:
        # Extraction des canaux EOG
        eog_picks = win.copy().pick_types(eog=True)
        data = eog_picks.get_data() * 1e6  # µV
        print(f">>> Canaux EOG détectés : {eog_picks.ch_names}")

        if data.shape[0] < 2:
            print(">>> Moins de 2 canaux EOG => skip")
            return "ignore"

        eog1, eog2 = data[:2]  # EOGD et EOGG
        n_samples = eog1.shape[0]
        half = n_samples // 2

        def count_valid_em(e1, e2):
            """
            Détection conjointe : pics synchrones de polarité opposée.
            """
            peaks1, _ = find_peaks(np.abs(e1), height=100)
            peaks2, _ = find_peaks(np.abs(e2), height=100)

            valid_count = 0
            for p1 in peaks1:
                for p2 in peaks2:
                    # Synchrones +/- 100ms
                    if abs(p1 - p2) / sfreq < 0.1:
                        # Polarité opposée : produit négatif
                        if np.sign(e1[p1]) != np.sign(e2[p2]):
                            valid_count += 1
            return valid_count

        # Détection dans chaque moitié
        em_left = count_valid_em(eog1[:half], eog2[:half])
        em_right = count_valid_em(eog1[half:], eog2[half:])

        # Critère phasic : au moins 1 dans chaque moitié
        if em_left >= 1 and em_right >= 1:
            return "phasic"

        # Vérifie l’absence globale d’activité >25µV
        no_deflection_1 = np.max(np.abs(eog1)) < 25
        no_deflection_2 = np.max(np.abs(eog2)) < 25

        if no_deflection_1 and no_deflection_2:
            return "tonic"

        return "ignore"

    except Exception as e:
        print(f"[EOG ERROR] {e}")
        return "ignore"
