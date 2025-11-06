# src/segmentation/rem.py
import os, mne, numpy as np
def segment_rem(raw_fif: str, annot_txt: str, epoch_len_s: float, out_epochs_fif: str):
    raw = mne.io.read_raw_fif(raw_fif, preload=True, verbose=False)
    # Hypnogramme simple: on suppose un fichier .txt avec colonnes "start_s,end_s,stage"
    rem_intervals = []
    with open(annot_txt) as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and parts[2].upper()=="REM":
                rem_intervals.append((float(parts[0]), float(parts[1])))
    if not rem_intervals: return None
    events, event_id = [], {"REM":1}
    for (s,e) in rem_intervals:
        t = s
        while t + epoch_len_s <= e:
            events.append([int((t)*raw.info["sfreq"]), 0, 1]); t += epoch_len_s
    events = np.array(events); picks = mne.pick_types(raw.info, eeg=True, eog=True, emg=True, ecg=True)
    epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=epoch_len_s, picks=picks, baseline=None, preload=True, verbose=False)
    os.makedirs(os.path.dirname(out_epochs_fif), exist_ok=True); epochs.save(out_epochs_fif, overwrite=True)
    return out_epochs_fif
