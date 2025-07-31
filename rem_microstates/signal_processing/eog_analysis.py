import numpy as np
from scipy.signal import find_peaks

def detect_eog_microstate(win, sfreq=250.0):
    """
    Classe une fenêtre REM de 4 s comme "phasic", "tonic", ou "ignore"
    selon la section 2.2 de l'article (Simor et al., 2021).

    Critères :
    - EOG filtré entre 0.3–10 Hz
    - "phasic" si ≥2 déflexions >150 µV et <500 ms dans chacune des 2 moitiés (2s)
    - "tonic" si aucune déflexion >25 µV dans les 2 moitiés
    - sinon "ignore"
    """
    try:
        # On copie la fenêtre (Raw de 4 s) et on sélectionne uniquement les canaux de type EOG
        eog_picks = win.copy().pick_types(eog=True)
        print(f">>> Canaux EOG : {eog_picks}")

        # Récupère les données brutes sous forme de tableau numpy [n_channels, n_samples]
        data = eog_picks.get_data()*1e6
        print(f">>> Récupération des raw data dans un tableau : \n {data}")

        # Si aucun canal EOG n'est présent, on ignore cette fenêtre
        if data.shape[0] == 0:
            print(">>> Aucun canal EOG n'est présent")
            return "ignore"

        # On utilise le premier canal EOG pour l'analyse (supposé suffisant)
        signal = data[0]*1e6

        # Nombre d’échantillons (doit correspondre à 4 s * fréquence d’échantillonnage)
        n_samples = signal.shape[0]

        # On divise la fenêtre en deux moitiés de 2 s
        half = n_samples // 2

        def count_em(segment):
            """
            Fonction interne pour compter les mouvements oculaires (EM)
            """
            # Cherche des pics (en valeur absolue) de plus de 150 µV dans le segment
            peaks, _ = find_peaks(np.abs(segment), height=150)
            print(f">>> Nombre de pic >150µV : {peaks}")

            # Moins de 2 pics -> ne remplit pas le critère
            if len(peaks) < 2:
                print("Moins de 2 pics -> ne remplit pas les critères")
                return 0

            # Convertit les indices de pics en temps (en secondes)
            peak_times = peaks / sfreq

            # Calcule les durées entre chaque pic
            durations = np.diff(peak_times)

            # Compte combien de pics sont espacés de moins de 0.5 s (= < 500 ms)
            return np.sum(durations < 0.5)

        # Applique la détection de mouvements oculaires dans la première moitié (0–2 s)
        em_left = count_em(signal[:half])

        # Idem dans la deuxième moitié (2–4 s)
        em_right = count_em(signal[half:])

        # Critère phasic : au moins 1 EM détectés dans chaque moitié
        if em_left >= 1 and em_right >= 1:
            return "phasic"

        # Détection de toutes les déflexions > 25 µV (sur l'ensemble de la fenêtre)
        all_peaks, _ = find_peaks(np.abs(signal), height=25)

        # Si aucune ou une seule déflexion > 25 µV -> tonic
        if len(all_peaks) <= 1:
            return "tonic"


        # Si on n'est ni phasic ni tonic selon les critères -> ignorer cette fenêtre
        return "ignore"

    except Exception as e:
        # Si une erreur survient (données malformées, etc.), on ignore la fenêtre et on log l’erreur
        print(f"[EOG ERROR] {e}")
        return "ignore"

