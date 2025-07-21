import streamlit as st

st.set_page_config(page_title="EEG Toolbox", layout="wide")
st.title(" EEG Visualisation Toolbox")

st.markdown("""
Bienvenue dans l'interface EEG interactive.

Utilisez la barre latérale pour naviguer entre les différentes fonctionnalités :
- **Visualisation EEG** : exploration complète des signaux EEG prétraités.
- **Visualisation REM** : visualisation des segments correspondant aux périodes de sommeil paradoxal (REM) extraits.

---

###  Informations
Cette application vous permet de :
- Visualiser les signaux EEG monopolaire prétraités (.fif),
- Appliquer un filtrage par bande de fréquences EEG,
- Afficher les annotations REM si disponibles,
- Explorer les segments REM uniquement via une page dédiée.

---

**Projet en cours de développement — version 1.0**
""")
