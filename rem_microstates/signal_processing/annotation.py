from mne import Annotations

def annotate_microstates(raw, windows, labels, window_sec):
    """
    Ajoute des annotations 'phasic' / 'tonic' à l'objet raw.

    Args:
        raw (mne.io.Raw): signal EEG original
        windows (list of mne.io.Raw): fenêtres découpées
        labels (list of str): liste de labels ('phasic' / 'tonic') pour chaque fenêtre
        window_sec (float): durée de chaque fenêtre
    """
    onset = [win.first_time for win in windows]
    duration = [window_sec] * len(windows)
    description = labels

    annotations = Annotations(onset=onset, duration=duration, description=description)
    raw.set_annotations(annotations)
    print(f"[INFO] {len(annotations)} annotations 'tonic/phasic' ajoutées à raw.")
