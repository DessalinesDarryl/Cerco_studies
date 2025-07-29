import numpy as np

def detect_eog_bursts(epoch, threshold=25):
    eog_data = epoch.copy().pick_channels(['EOGG', 'EOGD']).get_data()
    amp = np.ptp(eog_data, axis=-1)  # peak-to-peak
    return float((amp > threshold).sum()) / eog_data.shape[0]  # score ∈ [0,1]
