import numpy as np
from scipy.signal import find_peaks
import mne
import gc

def detect_eog_microstate(
    win,
    sfreq: float | None = None,
    min_pair_amp_uv: float = 100,
    sync_tol_ms: float = 100.0,
    silent_max_uv: float = 25.0,
) -> str:
    """
    Classer une fenêtre (~4 s) en 'phasic' / 'tonic' / 'ignore' via deux canaux EOG.

    Compat:
    - win proxy léger avec attributs {raw, tmin, tmax}  -> streaming
    - win MNE (Raw/Epochs/Evoked de 4 s)               -> fallback
    """
    try:
        # --- Chemin streaming : WindowProxy-like (raw + tmin/tmax)
        if hasattr(win, "raw") and hasattr(win, "tmin") and hasattr(win, "tmax"):
            raw = win.raw
            tmin = float(win.tmin); tmax = float(win.tmax)
            eog_idx = mne.pick_types(raw.info, eog=True)
            if eog_idx is None or len(eog_idx) < 2:
                return "ignore"
            data = raw.get_data(picks=eog_idx, tmin=tmin, tmax=tmax)   # (n_eog, n_times)
            sf = float(sfreq) if sfreq is not None else float(raw.info["sfreq"])

        else:
            # --- Fallback: objets MNE (copie potentiellement coûteuse)
            inst = win.copy()
            if not hasattr(inst, "pick"):
                raise TypeError("win doit être un Raw/Epochs/Evoked MNE ou un proxy {raw,tmin,tmax}.")
            eog_idx = mne.pick_types(inst.info, eog=True)
            if eog_idx is None or len(eog_idx) < 2:
                return "ignore"
            inst.pick(eog_idx)
            data = inst.get_data()
            if data.ndim == 3:
                if data.shape[0] != 1:
                    raise ValueError("Si 'win' est un Epochs, il doit contenir exactement 1 époque.")
                data = data[0]  # (n_chan, n_times)
            sf = float(sfreq) if sfreq is not None else float(inst.info["sfreq"])
            del inst; gc.collect()

        # --- µV + float32 pour réduire l'empreinte
        data = (data * 1e6).astype(np.float32, copy=False)
        if data.shape[0] < 2:
            del data; gc.collect()
            return "ignore"
        eog1, eog2 = data[:2]

        if not np.isfinite(sf) or sf <= 0:
            del data; gc.collect()
            raise ValueError("Fréquence d’échantillonnage invalide.")
        tol = int(round((sync_tol_ms / 1000.0) * sf))

        def peak_idx(x: np.ndarray) -> np.ndarray:
            idx, _ = find_peaks(np.abs(x), height=min_pair_amp_uv)
            return idx

        def count_valid_pairs(x1: np.ndarray, x2: np.ndarray) -> int:
            p1 = peak_idx(x1); p2 = peak_idx(x2)
            if len(p1) == 0 or len(p2) == 0:
                return 0
            cnt, j0 = 0, 0
            for i in p1:
                while j0 < len(p2) and p2[j0] < i - tol:
                    j0 += 1
                j = j0
                while j < len(p2) and p2[j] <= i + tol:
                    if np.sign(x1[i]) != np.sign(x2[p2[j]]):
                        cnt += 1
                        break
                    j += 1
            return cnt

        ########## Critère phasic ##########
        # Si au moins une paire de pics synchrones et de signe opposé dans chaque moitié de la fenêtre -> phasic

        # Split en deux moitiés
        mid = eog1.shape[-1] // 2
        left_pairs  = count_valid_pairs(eog1[:mid],  eog2[:mid])
        right_pairs = count_valid_pairs(eog1[mid:], eog2[mid:])
    
        # Valeur absolue max dans la fenêtre (µv)
        max_abs1 = float(np.max(np.abs(eog1)))
        max_abs2 = float(np.max(np.abs(eog2)))

        # Libérations explicites
        del data, eog1, eog2
        gc.collect()

        # Application
        if left_pairs >= 1 and right_pairs >= 1:
            return "phasic"

        ########## Critère tonic ##########
        # Sinon, si aucune déflexion > silent_max_uv = 25µV dans toute la fenêtre -> tonic

        no_defl_1 = max_abs1 < silent_max_uv
        no_defl_2 = max_abs2 < silent_max_uv

        # Application
        if no_defl_1 and no_defl_2:
            return "tonic"
        ####################################
        return "ignore"

    except Exception as exc:
        print(f"[EOG ERROR] {exc}")
        return "ignore"
