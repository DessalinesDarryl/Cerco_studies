import mne
from segment_rem import extract_rem_segments

def load_signals_and_annotations(fif_path, annot_path):
    raw = mne.io.read_raw_fif(fif_path, preload=True)
    rem_segments = extract_rem_segments(annot_path)

    print(f"Fichiers détectés : {fif_path} + {annot_path}")
    print(f"Durée fichier FIF : {raw.times[-1]:.2f} secondes")
    if rem_segments:
        t0, t1 = rem_segments[0]
        print(f"Premier segment REM : {t0:.2f} s → {t1:.2f} s ({(t1 - t0):.1f} s)")
    else:
        print("[INFO] Aucun segment REM détecté dans l'annotation.")
        
    raw.set_channel_types({
        "EOGG": "eog",
        "EOGD": "eog"
    })

    return raw, rem_segments
