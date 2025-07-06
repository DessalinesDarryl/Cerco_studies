import streamlit as st
import mne
import os
from utils import get_standard_bands, filter_band
from segment_rem import load_annotation_file

st.set_page_config(page_title="Visualisation EEG", layout="wide")
st.title("Visualisation des signaux EEG.")

# Sélection du dossier contenant les fichiers fif
fif_dir = st.sidebar.text_input("Dossier contenant les fichiers .fif", "data/preprocessed")

if not os.path.isdir(fif_dir):
    st.error("Le dossier spécifié n'existe pas.")
    st.stop()

# Lister les fichiers fif disponibles
fif_files = sorted([f for f in os.listdir(fif_dir) if f.endswith(".fif")])
if not fif_files:
    st.warning("Aucun fichier .fif trouvé dans ce dossier.")
    st.stop()

# Sélection du fichier
selected_file = st.sidebar.selectbox("Sélectionner un patient :", fif_files)
base_name = selected_file.replace("_eeg_cleaned_raw.fif", "")
raw_path = os.path.join(fif_dir, selected_file)

# Charger le fichier
raw = mne.io.read_raw_fif(raw_path, preload=True)
raw.pick_types(eeg=True)

# Affichage des infos de base
st.markdown(f"**Patient sélectionné :** `{selected_file}`")
st.markdown(f"**Canaux EEG détectés :** {len(raw.ch_names)}")

# Ajout d'une option pour tous les canaux
channel_options = ["Tous les canaux"] + raw.ch_names
selected_channel = st.selectbox("Canal EEG à visualiser :", channel_options)

# Copie selon le choix
if selected_channel == "Tous les canaux":
    raw_display = raw.copy()
else:
    raw_display = raw.copy().pick_channels([selected_channel])

# Bande de fréquence
bands = get_standard_bands()
band_name = st.selectbox("Filtrer dans une bande EEG :", ["Aucune"] + list(bands.keys()))
if band_name != "Aucune":
    l_freq, h_freq = bands[band_name]
    raw_display = filter_band(raw_display, l_freq, h_freq)
    st.markdown(f"Filtrage appliqué : **{band_name}**")

# Segment temporel
duration = st.slider("Durée affichée (secondes) :", 5, 60, 20)
start_time = st.slider("Début du segment (secondes) :", 0, int(raw.times[-1] - duration), 0)

# Option d’affichage des périodes REM
show_rem = st.checkbox("Afficher les périodes REM")

if show_rem:
    annot_dir = st.text_input("Dossier des fichiers d'annotations (.txt)", "D:/EEG/raw")
    txt_path = None

    if annot_dir:
        import glob
        txt_candidates = glob.glob(os.path.join(annot_dir, "**", f"{base_name}hypnoEXP.txt"), recursive=True)
        if txt_candidates:
            txt_path = txt_candidates[0]

    if txt_path and os.path.exists(txt_path):
        try:
            rem_intervals = load_annotation_file(txt_path)
            annotations = mne.Annotations(
                onset=[start for start, _ in rem_intervals],
                duration=[dur for _, dur in rem_intervals],
                description=["REM"] * len(rem_intervals),
            )
            raw_display.set_annotations(annotations)
            st.success("Annotations REM chargées et affichées.")
        except Exception as e:
            st.warning(f"Erreur de chargement des REM : {e}")
    else:
        st.warning(f"Aucune annotation trouvée pour {base_name}")

# Affichage du tracé
fig = raw_display.plot(start=start_time, duration=duration, show=False)
st.pyplot(fig=fig, clear_figure=True)

st.success("Affichage terminé. Vous pouvez changer les options à gauche pour explorer d'autres segments.")
