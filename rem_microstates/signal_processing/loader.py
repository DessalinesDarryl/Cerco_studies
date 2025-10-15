# signal_processing/loader.py
from pathlib import Path
import numpy as np
import mne

# adapte si ton fichier s'appelle/est situé ailleurs
from segment_rem import extract_rem_segments


def _sanitize_segments(segments, t_end: float):
    """Trie, clippe dans [0, t_end], enlève segments vides/NaN, fusionne les chevauchements."""
    clean = []
    for seg in segments or []:
        try:
            t0, t1 = float(seg[0]), float(seg[1])
        except Exception:
            continue
        if not np.isfinite(t0) or not np.isfinite(t1):
            continue
        # ordre + clipping
        if t1 < t0:
            t0, t1 = t1, t0
        t0 = max(0.0, min(t0, t_end))
        t1 = max(0.0, min(t1, t_end))
        if t1 - t0 > 1e-6:
            clean.append((t0, t1))

    if not clean:
        return []

    # tri
    clean.sort(key=lambda x: x[0])

    # fusion des recouvrements
    merged = [clean[0]]
    for a, b in clean[1:]:
        la, lb = merged[-1]
        if a <= lb:  # chevauchement/adjacent
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def load_signals_and_annotations(fif_path: Path, annot_path: Path | None):
    """
    Charge le Raw FIF en memmap (preload=False) et les segments REM depuis annot_path.
    Retourne:
      - raw : mne.io.Raw (memmap)
      - rem_segments : liste [(tmin, tmax), ...] en secondes, nettoyée et bornée dans [0, fin_fichier]
    """
    # Favoriser memmap (limite la RAM)
    mne.set_config('MNE_MEMMAP_MIN_SIZE', '1M', set_env=True)
    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")

    # Déclarer les canaux EOG si présents
    to_eog = {}
    for name in ("EOGG", "EOGD", "EOG1", "EOG2", "HEOG", "VEOG"):
        if name in raw.ch_names:
            to_eog[name] = "eog"
    if to_eog:
        raw.set_channel_types(to_eog)

    # Lire segments REM
    segments = []
    if annot_path is not None and Path(annot_path).exists():
        segments = extract_rem_segments(annot_path)  # doit renvoyer [(tmin, tmax), ...]
    else:
        segments = []

    # Bornage/tri/fusion des segments
    try:
        t_end = float(raw.times[-1])  # (n_times-1)/sfreq
    except Exception:
        # fallback robuste
        sf = float(raw.info["sfreq"])
        t_end = (raw.first_samp + raw.n_times - 1) / sf
    rem_segments = _sanitize_segments(segments, t_end)

    # Logs utiles
    print(f"Fichiers détectés : {fif_path} + {annot_path}")
    print(f"Durée fichier FIF : {t_end:.2f} secondes")
    if rem_segments:
        t0, t1 = rem_segments[0]
        print(f"Premier segment REM : {t0:.2f} s -> {t1:.2f} s ({(t1 - t0):.1f} s)")
    else:
        print("[INFO] Aucun segment REM détecté dans l'annotation.")

    return raw, rem_segments
