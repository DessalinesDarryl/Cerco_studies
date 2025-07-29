import mne

def annotate_microstates_to_raw(raw, windows, labels, win_len_sec=4.0):
    """Ajoute les labels tonic/phasic en tant qu'annotations dans l'objet raw"""
    onsets = []
    durations = []
    descriptions = []

    for i, (win, label) in enumerate(zip(windows, labels)):
        onset = win.times[0]  # début de la fenêtre en secondes
        duration = win_len_sec
        onsets.append(onset)
        durations.append(duration)
        descriptions.append(label)

    annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    raw.set_annotations(annotations)
    return raw

