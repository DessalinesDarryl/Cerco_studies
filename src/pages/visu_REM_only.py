import streamlit as st
import mne
import os
import matplotlib.pyplot as plt
import platform
from pathlib import Path
from filters import get_standard_bands, filter_band
from annotations import get_rem_annotations
from utils import list_fif_files  # version récursive mise à jour ci-dessous

st.set_page_config(page_title="REM EEG", layout="wide")
st.title("Visualisation des segments REM - EEG monopolaire")

# Détection système
system = platform.system()
if system == "Darwin":
    disque = "/Volumes/Crucial X6"
elif system == "Windows":
    disque = "D:"
else:
    raise RuntimeError("Système non supporté.")

# Répertoire REM-only
base_dir = Path(f"{disque}/EEG/preprocessed/monopolaire/rem_only")
fif_dir = st.sidebar.text_input("Dossier des segments REM", base_dir)

if not os.path.isdir(fif_dir):
    st.error("Le dossier spécifié n'existe pas.")
    st.stop()

# Recherche récursive dans les sous-dossiers
fif_files = list_fif_files(fif_dir)
print(fif_files)
if not fif_files:
    st.warning("Aucun fichier .fif trouvé dans ce dossier.")
    st.stop()

# Sélection et chargement du fichier .fif
selected_file = st.sidebar.selectbox("Sélectionner un segment REM :", fif_files)
raw_path = os.path.join(fif_dir, selected_file)
base_name = Path(selected_file).stem  # nom du fichier sans extension

raw = mne.io.read_raw_fif(raw_path, preload=True)

st.markdown(f"**Patient :** `{base_name}`")
st.markdown(f"**Durée totale du segment :** `{raw.times[-1]:.1f}` s")

# Sélection des canaux
channel_selection = st.multiselect(
    "Canaux à visualiser (laisser vide pour tous)",
    options=raw.ch_names,
    default=[]
)
raw_display = raw.copy() if len(channel_selection) == 0 else raw.copy().pick_channels(channel_selection)

# Bande EEG
bands = get_standard_bands()
band_name = st.selectbox("Filtrer dans une bande EEG :", ["Aucune"] + list(bands.keys()))
if band_name != "Aucune":
    l_freq, h_freq = bands[band_name]
    raw_display = filter_band(raw_display, l_freq, h_freq)
    st.markdown(f"Filtrage appliqué : **{band_name}**")

# Affichage
amplitude = st.number_input("Amplitude (µV)", min_value=1.0, max_value=50000000.0, value=25.0, step=1.0)
duration = st.slider("Durée affichée (secondes)", 5, 60, 20)
start_time = st.slider("Début (secondes)", 0, int(raw.times[-1] - duration), 0)

plt.close('all')
fig = raw_display.plot(
    start=start_time,
    duration=duration,
    scalings=dict(eeg=amplitude * 1e-6),
    remove_dc=True,
    show=False
)
st.pyplot(fig=fig, clear_figure=True)
