import streamlit as st
import mne
import os
from filters import get_standard_bands, filter_band
from annotations import get_rem_annotations
from utils import list_fif_files, get_base_name

st.set_page_config(page_title="Visualisation EEG", layout="wide")
st.title("Visualisation des signaux EEG.")

# Sélection du dossier contenant les fichiers fif
fif_dir = st.sidebar.text_input("Dossier contenant les fichiers .fif", "data/preprocessed")

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
raw.pick_types(eeg=True)

# Infos générales
st.markdown(f"**Patient sélectionné :** `{base_name}`")
st.markdown(f"**Canaux EEG détectés :** {len(raw.ch_names)}")

# Choix du canal
channel_options = ["Tous les canaux"] + raw.ch_names
selected_channel = st.selectbox("Canal EEG à visualiser :", channel_options)
raw_display = raw.copy() if selected_channel == "Tous les canaux" else raw.copy().pick_channels([selected_channel])

# Bande EEG
bands = get_standard_bands()
band_name = st.selectbox("Filtrer dans une bande EEG :", ["Aucune"] + list(bands.keys()))
if band_name != "Aucune":
    l_freq, h_freq = bands[band_name]
    raw_display = filter_band(raw_display, l_freq, h_freq)
    st.markdown(f" Filtrage appliqué : **{band_name}**")

# Ajout d’annotations REM
if st.checkbox("Afficher les périodes REM sur le signal"):
    annot_root = st.sidebar.text_input("Dossier contenant les fichiers d'annotations (.txt)", "D:/EEG/raw")
    rem_annotations = get_rem_annotations(base_name, annot_root)

    if rem_annotations:
        raw_display.set_annotations(rem_annotations)
        st.success(f"{len(rem_annotations)} segments REM ajoutés au signal.")
    else:
        st.warning("Aucune période REM trouvée pour ce patient.")

# Durée et plage temporelle
duration = st.slider("Durée affichée (secondes) :", 5, 60, 20)
start_time = st.slider("Début du segment (secondes) :", 0, int(raw.times[-1] - duration), 0)

# Affichage
fig = raw_display.plot(start=start_time, duration=duration, show=False)
st.pyplot(fig=fig, clear_figure=True)

st.success("Affichage terminé.")
