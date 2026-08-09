"""
results_io.py
--------------
Convention UNIQUE de sortie pour tout le pipeline RSWA : tous les
fichiers generes (CSV, rapports) sont ecrits sous RESULTS_ROOT
(results/ a la racine du projet), quel que soit le script qui les
produit (main_pipeline, statistics_pipeline, multicollinearity_pca,
scripts de demonstration test_*.py).

Ceci evite la dispersion de sorties observee dans l'ancien pipeline
(data/, results/, visualisation2/, visualisation3/, visu3/ selon les
scripts 01-09) : un seul dossier racine, sous-dossiers par etape.

Pour changer l'emplacement de sortie de TOUT le pipeline, modifier
UNIQUEMENT RESULTS_ROOT ci-dessous (ou definir la variable
d'environnement RSWA_RESULTS_ROOT, prioritaire).
"""

import os
from pathlib import Path

import pandas as pd

# Racine des resultats. Priorite : variable d'environnement
# RSWA_RESULTS_ROOT, sinon results/ AU MEME NIVEAU que le dossier
# scripts/ (donc results/ et scripts/ cote a cote a la racine du
# projet, PAS results/ a l'interieur de scripts/) -- fonctionne quel
# que soit le dossier depuis lequel les scripts sont lances.
# Ce fichier vit dans <projet>/scripts/results_io.py, donc :
#   parent      = <projet>/scripts
#   parent.parent = <projet>          <- racine attendue de results/
RESULTS_ROOT = Path(
    os.environ.get("RSWA_RESULTS_ROOT", Path(__file__).resolve().parent.parent / "results")
)


def results_path(*parts) -> Path:
    """
    Construit un chemin sous RESULTS_ROOT et cree les dossiers parents
    manquants. Exemple :
        results_path("cohort", "dataset_final.csv")
        -> <projet>/results/cohort/dataset_final.csv
    """
    p = RESULTS_ROOT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_dataframe(df: pd.DataFrame, *parts, index: bool = False) -> Path:
    """Sauvegarde un DataFrame en CSV sous RESULTS_ROOT/*parts."""
    path = results_path(*parts)
    df.to_csv(path, index=index)
    return path


def save_text(text: str, *parts) -> Path:
    """Sauvegarde un texte brut (ex. resume, log) sous RESULTS_ROOT/*parts."""
    path = results_path(*parts)
    path.write_text(text, encoding="utf-8")
    return path