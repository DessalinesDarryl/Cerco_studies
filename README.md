# Cerco_studies
Ce repo concerne mes travaux réalisés dans le cadre de mon stage de §mois réalisé au Cerco-CNRS pour l'analyse de signaux neurophysioloqigues dans le Trouble du Comportement en Sommeil Paradoxal (TCSP).

Structure : 

CERCO_STUDIES/
├── data/
│   ├── raw/                         # Fichiers .edf bruts
│   ├── preprocessed/               # Fichiers nettoyés ICA (.fif)
│   ├── rem_segments/               # Segments EEG/EMG/ECG extraits pendant REM
│   ├── features/                   # Fichiers .csv ou .npz contenant les features par sujet
│   └── nv_patientsRBD_identifiants.xlsx  # Données patient
│
├── notebooks/
│   ├── eeg_exploring.ipynb         # Explorations manuelles
│   ├── feature_analysis.ipynb      # Visualisation et stats
│   └── sleep_staging.ipynb         # Détection automatique des phases REM (si besoin)
│
├── src/                            # Code source
│   ├── __init__.py
│   ├── preprocessing.py            # ICA, filtres, nettoyage
│   ├── segment_rem.py              # Extraction REM
│   ├── feature_extraction.py       # EEG/EMG/ECG features
│   ├── stats_analysis.py           # Stats inter-patho
│   └── utils.py                    # Fonctions utiles (ex: loader, synchroniseur, etc.)
│
├── results/
│   ├── figures/
│   └── tables/
│
├── README.md
└── requirements.txt
