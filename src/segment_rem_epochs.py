import os
import glob
import mne
import numpy as np
import pandas as pd

from tqdm import tqdm

def detect_rem_type(epoch, eog_ch_name):
    """
    Détermine si une époque est REM phasique, tonique ou indéterminée
    selon l'activité EOG.
    """
    eog_data = epoch.copy().pick_channels([eog_ch_name]).get_data()[0, 0]
    peak_thresh_phasic = 25e-6  # 25 µV en V
    peak_thresh_tonic = 100e-6 # 100 µV en V

    # détection de pics simples (abs > seuil)
    if np.any(np.abs(eog_data) >= peak_thresh_tonic):
        return "tonique"
    elif np.any(np.abs(eog_data) >= peak_thresh_phasic):
        return "phasique"
    else:
        return "indet"

def epoch_and_label_rem_segments(rem_dir, save_dir, eog_ch_name="EOG"):
    os.makedirs(save_dir, exist_ok=True)
    rem_files = glob.glob(os.path.join(rem_dir, "*_REM_raw.fif"))

    for rem_path in tqdm(rem_files):
        base = os.path.basename(rem_path).replace("_REM_raw.fif", "")
        raw = mne.io.read_raw_fif(rem_path, preload=True)

        if eog_ch_name not in raw.ch_names:
            print(f"{base} : Canal EOG non trouvé.")
            continue

        # découpage en époques de 4s
        epochs = mne.make_fixed_length_epochs(raw, duration=4.0, preload=True)

        # classification tonique/phasique/indéterminé
        labels = [detect_rem_type(epoch, eog_ch_name) for epoch in epochs]

        # assignation dans metadata
        epochs.metadata = pd.DataFrame({"rem_type": labels})

        # sauvegarde
        out_path = os.path.join(save_dir, f"{base}_REM-epo.fif")
        epochs.save(out_path, overwrite=True)
        print(f"{base} : {len(epochs)} époques sauvegardées.")

if __name__ == "__main__":
    epoch_and_label_rem_segments(
        rem_dir="../data/rem_segments",
        save_dir="../data/epochs_rem",
        eog_ch_name="EOG"  # à adapter selon le nom exact
    )
