# src/filters.py

def get_standard_bands():
    return {
        "Delta (0.5–4 Hz)": (0.5, 4),
        "Theta (4–8 Hz)": (4, 8),
        "Alpha (8–13 Hz)": (8, 13),
        "Beta (13–30 Hz)": (13, 30),
        "Gamma (30–45 Hz)": (30, 45)
    }

def filter_band(raw, l_freq, h_freq):
    return raw.copy().filter(l_freq=l_freq, h_freq=h_freq)

def filter_all_bands(raw):
    bands = get_standard_bands()
    return {name: filter_band(raw, l, h) for name, (l, h) in bands.items()}
