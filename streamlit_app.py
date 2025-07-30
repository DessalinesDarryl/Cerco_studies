import streamlit as st
import mne
import os
import matplotlib.pyplot as plt
import platform
from pathlib import Path
from rem_microstates.signal_processing.filters import get_standard_bands, filter_band
from rem_microstates.signal_processing.utils import list_fif_files, get_base_name

st.set_page_config(page_title="Visualisation REM tonic/phasic", layout="wide")
st.title("Visualisation des micro-états REM (tonic/phasic)")

# Détection automatique du système
system = platform.system()
if system == "Darwin":
    disque = "/Volumes/Crucial X6"
elif system == "Windows":
    disque = "D:"
elif system == "Linux":
    disque = "/media/darryld/Crucial X6"
else:
    raise RuntimeError("Système non supporté.")

# Dossier des fichiers FIF annotés
base_dir = Path(f"{disque}/EEG/raw") 
fif_dir = st.sidebar.text_input("Dossier des fichiers .fif annotés", base_dir)

if not os.path.isdir(fif_dir):
    st.error("Le dossier spécifié n'existe pas.")
    st.stop()

# Lister les fichiers
fif_files = list_fif_files(fif_dir)
if not fif_files:
    st.warning("Aucun fichier .fif trouvé dans ce dossier.")
    st.stop()

# Sélection fichier
selected_file = st.sidebar.selectbox("Sélectionner un patient :", fif_files)
raw_path = os.path.join(fif_dir, selected_file)
base_name = get_base_name(selected_file)

# Chargement du signal
raw = mne.io.read_raw_fif(raw_path, preload=True)

# Infos générales
st.markdown(f"**Patient sélectionné :** `{base_name}`")
st.markdown(f"**Canaux détectés :** {len(raw.ch_names)}")

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

# Affichage des annotations tonic/phasic déjà présentes dans le raw
if st.checkbox("Afficher les annotations tonic/phasic (détectées automatiquement)"):
    if raw.annotations and any(a in ["tonic", "phasic"] for a in raw.annotations.description):
        st.success(f"{len(raw.annotations)} annotations trouvées.")
    else:
        st.warning("Aucune annotation 'tonic' ou 'phasic' trouvée dans ce fichier.")

# Échelle amplitude µV
amplitude = st.number_input("Amplitude (µV)", min_value=1.0, max_value=5000000.0, value=25.0, step=1.0)

# Affichage temporel
duration = st.slider("Durée affichée (secondes)", 5, 60, 20)
start_time = st.slider("Début (secondes)", 0, int(raw.times[-1] - duration), 0)

# Affichage du tracé EEG
plt.close('all')
fig = raw_display.plot(
    start=start_time,
    duration=duration,
    scalings=dict(eeg=amplitude * 1e-6),
    remove_dc=True,
    show=False,
    title=f"{base_name} - Annotations micro-états REM"
)

st.pyplot(fig=fig, clear_figure=True)
st.success("Affichage terminé.")
