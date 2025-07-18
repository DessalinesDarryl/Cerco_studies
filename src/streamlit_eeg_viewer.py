import streamlit as st
import mne
import os
import matplotlib.pyplot as plt
from filters import get_standard_bands, filter_band
from annotations import get_rem_annotations
from utils import list_fif_files, get_base_name

st.set_page_config(page_title="Visualisation EEG", layout="wide")
st.title("Visualisation des signaux EEG")

# Dossier ICA / RPF
base_dir = r"D:\\EEG\\preprocessed\\monopolaire\\full" 
fif_dir = st.sidebar.text_input("Dossier des fichiers .fif", base_dir)

if not os.path.isdir(fif_dir):
    st.error("Le dossier spécifié n'existe pas.")
    st.stop()

# Lister les fichiers fif disponibles
fif_files = list_fif_files(fif_dir)
if not fif_files:
    st.warning("Aucun fichier .fif trouvé dans ce dossier.")
    st.stop()

# Sélection du fichier
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

# Annotations REM
if st.checkbox("Afficher les périodes REM sur le signal"):
    annot_root = st.sidebar.text_input("Dossier des annotations (.txt)", "D:/EEG/raw")
    rem_annotations = get_rem_annotations(base_name, annot_root)

    if rem_annotations:
        raw_display.set_annotations(rem_annotations)
        st.success(f"{len(rem_annotations)} segments REM ajoutés.")
    else:
        st.warning("Aucune période REM trouvée pour ce patient.")

# Échelle amplitude µV
amplitude = st.number_input("Amplitude (µV)", min_value=1.0, max_value=500000.0, value=25.0, step=1.0)


# Plage d’affichage
duration = st.slider("Durée affichée (secondes)", 5, 60, 20)
start_time = st.slider("Début (secondes)", 0, int(raw.times[-1] - duration), 0)

# Affichage du tracé EEG
plt.close('all')
fig = raw_display.plot(
    start=start_time,
    duration=duration,
    scalings = dict(eeg=amplitude * 1e-6),
    remove_dc=True,
    show=False
)
st.pyplot(fig=fig, clear_figure=True)

st.success("Affichage terminé.")
