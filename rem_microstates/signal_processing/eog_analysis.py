# eog_analysis.py

import numpy as np
from scipy.signal import find_peaks
import mne


def detect_eog_microstate(
    win,
    sfreq: float | None = None,
    min_pair_amp_uv: float = 100.0,
    sync_tol_ms: float = 100.0,
    silent_max_uv: float = 25.0,
) -> str:
    """
    Classer une fenêtre REM (≈ 4 s) en 'phasic', 'tonic' ou 'ignore' à partir de deux canaux EOG.

    La logique suit l’idée d’une détection conjointe gauche/droite :
    - 'phasic' : présence d'au moins 1 paire de pics (EOGG/EOGD) dans chaque moitié de la fenêtre
                 (0–2 s et 2–4 s), avec polarité opposée et synchronie ± sync_tol_ms.
    - 'tonic'  : absence d’activité notable sur les deux canaux (max absolu < silent_max_uv).
    - sinon    : 'ignore'.

    Paramètres
    ----------
    win : mne.io.BaseRaw | mne.Epochs | mne.Evoked
        Fenêtre MNE contenant les canaux EOG. Si Epochs, elle doit contenir exactement 1 époque.
    sfreq : float | None
        Fréquence d’échantillonnage. Si None, lue depuis win.info['sfreq'].
    min_pair_amp_uv : float
        Amplitude minimale (µV) pour qu’un pic soit pris en compte (détection de mouvements).
    sync_tol_ms : float
        Tolérance de synchronie entre les pics des deux EOG (en millisecondes).
    silent_max_uv : float
        Seuil absolu (µV) en dessous duquel on considère l’activité comme “tonic”.

    Retour
    ------
    str
        'phasic' | 'tonic' | 'ignore'

    Notes
    -----
    - Compatible avec MNE ≥ 1.0 : on sélectionne les canaux via mne.pick_types(...) puis inst.pick(...).
    - Nécessite au minimum 2 canaux EOG (gauche/droit). Si plus de 2 EOG sont présents,
      on utilise les deux premiers.
    """
    try:
        # 1) Copie et sélection des canaux EOG avec l’API recommandée
        inst = win.copy()
        if not hasattr(inst, "pick"):
            raise TypeError("L'objet 'win' doit être un Raw/Epochs/Evoked MNE.")

        eog_idx = mne.pick_types(inst.info, eog=True)
        if eog_idx is None or len(eog_idx) < 2:
            return "ignore"

        # On garde seulement les EOG
        inst.pick(eog_idx)

        # 2) Données (en V) et sfreq
        data = inst.get_data()
        # Raw: (n_chan, n_times), Epochs: (n_epochs, n_chan, n_times)
        if data.ndim == 3:
            if data.shape[0] != 1:
                raise ValueError("Si 'win' est un Epochs, il doit contenir exactement 1 époque.")
            data = data[0]  # (n_chan, n_times)

        # Convertir en µV
        data = data * 1e6

        # On prend les deux premiers EOG (gauche/droit)
        if data.shape[0] < 2:
            return "ignore"
        eog1, eog2 = data[:2]

        # Fréquence d’échantillonnage
        sf = float(sfreq) if sfreq is not None else float(inst.info["sfreq"])
        if not np.isfinite(sf) or sf <= 0:
            raise ValueError("Fréquence d’échantillonnage invalide.")

        # 3) Détection de pics sur |signal|
        def peak_idx(x: np.ndarray) -> np.ndarray:
            idx, _ = find_peaks(np.abs(x), height=min_pair_amp_uv)
            return idx

        # Appariement tolérant : pour chaque pic de eog1, chercher un pic de eog2 proche (± tol)
        tol = int(round((sync_tol_ms / 1000.0) * sf))

        def count_valid_pairs(x1: np.ndarray, x2: np.ndarray) -> int:
            p1 = peak_idx(x1)
            p2 = peak_idx(x2)
            if len(p1) == 0 or len(p2) == 0:
                return 0
            cnt = 0
            j0 = 0
            for i in p1:
                # avance j0 tant que le pic de x2 est trop en retard/avance
                while j0 < len(p2) and p2[j0] < i - tol:
                    j0 += 1
                # vérifier les pics dans la fenêtre [i - tol, i + tol]
                j = j0
                while j < len(p2) and p2[j] <= i + tol:
                    if np.sign(x1[i]) != np.sign(x2[p2[j]]):
                        cnt += 1
                        break
                    j += 1
            return cnt

        # 4) Comptages par moitié (≈ 0–2 s, 2–4 s)
        n_samples = eog1.shape[-1]
        mid = n_samples // 2

        left_pairs = count_valid_pairs(eog1[:mid], eog2[:mid])
        right_pairs = count_valid_pairs(eog1[mid:], eog2[mid:])

        if left_pairs >= 1 and right_pairs >= 1:
            return "phasic"

        # 5) Critère “tonic” : faible activité globale sur chaque EOG
        no_defl_1 = float(np.max(np.abs(eog1))) < silent_max_uv
        no_defl_2 = float(np.max(np.abs(eog2))) < silent_max_uv
        if no_defl_1 and no_defl_2:
            return "tonic"

        return "ignore"

    except Exception as exc:
        print(f"[EOG ERROR] {exc}")
        return "ignore"
