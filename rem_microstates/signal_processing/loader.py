import mne
from segment_rem import extract_rem_segments

def load_signals_and_annotations(edf_path, annot_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    rem_segments = extract_rem_segments(annot_path)
    print(f"Fichiers détectés : {edf_path} + {annot_path}")
    print(f"Durée fichier EDF : {raw.times[-1]:.2f} secondes")
    print(f"Premier segment REM à t={rem_segments[0][0]:.2f} secondes")

    raw.set_channel_types({
        "EOGG": "eog",
        "EOGD": "eog"
    })

    return raw, rem_segments
