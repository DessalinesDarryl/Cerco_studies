import mne
raw = mne.io.read_raw_edf("../CERCO_STUDIES/data/AE129_231121C-A.edf", preload=True) 

eeg_channels = [ch for ch in raw.ch_names if ch.startswith('EEG ') or ch.startswith('EOG')]
raw_eeg = raw.copy().pick_channels(eeg_channels)

# Plot
raw_eeg.plot(n_channels=len(raw_eeg.ch_names),start= 50, duration=10, scalings='auto', show=False)
