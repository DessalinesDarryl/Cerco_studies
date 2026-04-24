#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_features.py
===================

But
---
Construire un **dataset de features au niveau patient** à partir :
1) de fichiers MNE `*_REM-epo.fif` contenant des **époques REM** (typiquement 4 secondes),
2) optionnellement d’un CSV issu de `emg_rbd.py` (ou équivalent) contenant des métriques EMG/EOG par epoch.

Le script parcourt récursivement `in_root`, calcule des features EEG par epoch, agrège ces features
au niveau patient (moyenne + écart-type), puis joint (merge) des agrégats EMG (déjà au niveau patient).

Entrées
-------
- `in_root` (dans cfg YAML) :
    Dossier racine contenant des fichiers `*_REM-epo.fif`, par patient.
- `rbd_emg_csv` (optionnel dans cfg YAML) :
    CSV contenant des mesures par epoch REM (type == "REM_EPOCH_4S"), par canal EMG, par patient.
- `n_workers` :
    Nombre de workers (ProcessPoolExecutor).

Sortie
------
- `out_csv` :
    Un CSV final (1 ligne = 1 patient) contenant :
      - EEG : agrégats des features temporelles et spectrales
      - EMG/EOG : agrégats patient-level (rswa fraction, ratios, tonic eog, corrélation etc.)

Structure des features EEG
--------------------------
Le script produit deux familles de features EEG :

1) **Features temporelles** (compute_time_eeg_features)
   Pour chaque epoch et chaque canal EEG :
   - mean, std, var
   - rms (root mean square)
   - zc (zero crossing count)
   - skew, kurtosis
   - ptp (peak-to-peak amplitude)
   - line length (somme des |diff|)

   Puis on concatène les canaux -> un vecteur par epoch.

2) **Features spectrales** (compute_spectral_eeg_features)
   Importées depuis `src.features.spectral_eeg`.
   (Le détail dépend de ton implémentation : PSD par bande, ratios, etc.)

Agrégation patient-level
------------------------
Pour un patient, on a une matrice X : (n_epochs, n_features).
On agrège chaque feature sur les epochs :
- moyenne : `<feature>_mean`
- écart-type : `<feature>_std`

EMG/EOG patient-level
---------------------
Le CSV EMG attendu contient une colonne `type` et des lignes `REM_EPOCH_4S`.
On agrège d’abord par canal (patient_id, channel), puis on agrège par patient.

Attention :  Points d’attention / hypothèses
----------------------------------
- On suppose que le `patient_id` peut être récupéré via `fpath.stem.split("_")[0]`.
  Exemple : `AN166_REM-epo.fif` => patient_id = `AN166`.
- On suppose que les types MNE des canaux EEG sont correctement définis (`eeg=True`).
- Le CSV EMG peut ne pas contenir `eye_emg_corr` selon ton pipeline : dans ce cas il faut
  soit l’ajouter côté `emg_rbd.py`, soit gérer l’absence ici (actuellement on suppose la colonne existe).

Exécution
---------
Le script utilise la config YAML via `--config` (argument géré par add_common_args).

Exemple :
python scripts/extract_features.py --config configs/extract_features.yaml --n_workers 15
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse
from typing import Dict, List, Optional

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import mne

from scipy.stats import skew, kurtosis

from src.utils.config import load_yaml, add_common_args
from src.utils.logging import get_logger
from src.features.spectral_eeg import compute_spectral_eeg_features


# ======================================================
# Extraction des features EEG par epoch
# ======================================================
def compute_time_eeg_features(epochs: mne.Epochs) -> tuple[Optional[np.ndarray], List[str]]:
    """
    Calcule des **features temporelles EEG par epoch**.

    Cette fonction extrait les canaux EEG (selon les types MNE), récupère la donnée
    sous forme (n_epochs, n_ch, n_times), puis calcule des statistiques temporelles
    simples sur l’axe temps (axis=-1).

    Features calculées (par epoch, par canal)
    ----------------------------------------
    - mean : moyenne sur le temps
    - std  : écart-type sur le temps
    - var  : variance sur le temps
    - rms  : RMS = sqrt(mean(x^2))
    - zc   : nombre de changements de signe (zero-crossings)
    - skew : asymétrie (scipy.stats.skew)
    - kurt : kurtosis (scipy.stats.kurtosis)
    - ptp  : peak-to-peak = max-min
    - linelen : somme des |diff| successifs (proxy de complexité)

    Paramètres
    ----------
    epochs : mne.Epochs
        Objet epochs déjà chargé (préload recommandé si usage intensif).

    Retours
    -------
    feats : np.ndarray | None
        Matrice (n_epochs, n_features_total), où n_features_total = n_ch * n_features_par_canal.
        Retourne None si aucun canal EEG n’est disponible.
    names : list[str]
        Noms des colonnes correspondant à feats (ordre aligné avec feats).

    Notes
    -----
    - Les unités dépendent de tes données MNE (souvent Volts). Si tu veux µV, il faut convertir.
    - zc ici est un *count* (pas normalisé par durée).
    """
    picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    if len(picks) == 0:
        return None, []

    # Structure des données : (n_epochs, n_channels, n_times)
    data = epochs.get_data()[:, picks, :]

    # 1) Statistiques temporelles de base
    mean = data.mean(axis=-1)
    std = data.std(axis=-1)
    var = data.var(axis=-1)
    rms = np.sqrt((data ** 2).mean(axis=-1))

    # 2) Nombre de changements de signe (zero-crossings)
    zc = ((np.diff(np.sign(data), axis=-1) != 0)).sum(axis=-1)

    # 3) Asymétrie et kurtosis sur l’axe temporel
    skw = skew(data, axis=-1, nan_policy="omit")
    krt = kurtosis(data, axis=-1, nan_policy="omit")

    # 4) Amplitude peak-to-peak
    ptp = np.ptp(data, axis=-1)

    # 5) Line length (somme des variations absolues)
    ll = np.sum(np.abs(np.diff(data, axis=-1)), axis=-1)

    # 6) Empilement final : (n_epochs, n_channels, n_features)
    feats = np.stack([mean, std, var, rms, zc, skw, krt, ptp, ll], axis=2)

    n_epochs, n_ch, n_f = feats.shape
    feats = feats.reshape(n_epochs, n_ch * n_f)

    # 7) Construction des noms de features, alignés avec l’ordre des canaux EEG
    names: List[str] = []
    for ch_idx in picks:
        ch = epochs.ch_names[ch_idx]
        names.extend([
            f"eeg_{ch}_mean",
            f"eeg_{ch}_std",
            f"eeg_{ch}_var",
            f"eeg_{ch}_rms",
            f"eeg_{ch}_zc",
            f"eeg_{ch}_skew",
            f"eeg_{ch}_kurt",
            f"eeg_{ch}_ptp",
            f"eeg_{ch}_linelen",
        ])

    return feats, names


def aggregate_patient_features(X: np.ndarray, names: List[str]) -> Dict[str, float]:
    """
    Agrège des features au niveau patient à partir de features epoch-level.

    Contexte
    --------
    On calcule d’abord des features par epoch : X est donc typiquement une matrice
    de taille (n_epochs, n_features). Pour obtenir **1 ligne par patient**, on
    agrège chaque feature sur l’ensemble des epochs du patient.

    Agrégation implémentée
    ----------------------
    - mean sur les epochs  -> `<feature>_mean`
    - std sur les epochs   -> `<feature>_std`

    Paramètres
    ----------
    X : np.ndarray
        Matrice (n_epochs, n_features).
    names : list[str]
        Liste des noms des features correspondant à l'axe colonnes de X.

    Retours
    -------
    row : dict[str, float]
        Dictionnaire prêt à être inséré dans une ligne DataFrame.

    Notes
    -----
    - Si tu veux des statistiques robustes (médiane/IQR), tu peux les ajouter ici.
    - X peut contenir des NaN : ici, on utilise mean/std numpy classiques (NaN propagés).
      Si tu veux ignorer NaN : np.nanmean / np.nanstd.
    """
    row: Dict[str, float] = {}

    # Ici, on laisse la propagation des NaN (choix explicite).
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    for i, name in enumerate(names):
        row[f"{name}_mean"] = float(mean[i])
        row[f"{name}_std"] = float(std[i])

    return row


# ======================================================
# EMG / EOG FEATURES (PATIENT LEVEL)
# ======================================================
def load_rbd_emg_features_patient_level(rbd_csv: Optional[Path], log) -> Optional[pd.DataFrame]:
    """
    Charge et agrège les features EMG/EOG issues du pipeline RBD (CSV).

    Le CSV attendu contient des lignes au format "long" :
    - une colonne `type` indiquant la nature de la ligne :
        - "REM_EPOCH_4S" pour les résumés par epoch de 4s
        - (éventuellement) "PHASIC" pour événements (ignorés ici)
    - au minimum : patient_id, channel
    - mesures typiques : rswa, phasic_ratio, tonic_ratio, tonic_eog, eye_emg_corr, etc.

    Pipeline d’agrégation
    ---------------------
    1) Filtre `type == "REM_EPOCH_4S"`.
    2) Agrégation par canal : groupby(["patient_id","channel"])
       -> stats par canal (moyennes, fractions)
    3) Agrégation patient : groupby("patient_id")
       -> moyenne sur les canaux

    Paramètres
    ----------
    rbd_csv : Path | None
        Chemin vers le CSV EMG. Si None/inexistant -> retourne None.
    log : logger
        Logger du projet.

    Retours
    -------
    agg_pat : pd.DataFrame | None
        DataFrame avec 1 ligne par patient_id et colonnes agrégées EMG.
        Retourne None si CSV absent ou invalide.

    Notes / robustesse
    ------------------
    - Si certaines colonnes attendues manquent (ex: eye_emg_corr), le groupby échouera.
      Dans ce cas, soit tu ajoutes la colonne au pipeline EMG, soit tu adaptes ici.
    """
    if not rbd_csv or not rbd_csv.exists():
        log.warning("CSV RBD EMG introuvable >>> EMG ignoré")
        return None

    df = pd.read_csv(rbd_csv)

    if "type" not in df.columns:
        log.warning("CSV EMG sans colonne 'type' >>> EMG ignoré")
        return None

    df = df[df["type"] == "REM_EPOCH_4S"].copy()
    if df.empty:
        log.warning("CSV EMG sans lignes REM_EPOCH_4S >>> EMG ignoré")
        return None

    # Sécurités : cast bool / numeric si nécessaire
    # Ici on tente d’être tolérant si "rswa" ou "tonic_eog" sont en str.
    for bool_col in ("rswa", "tonic_eog"):
        if bool_col in df.columns and df[bool_col].dtype != bool:
            df[bool_col] = (
                df[bool_col].astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes", "y"])
            )

    # -------- agrégation par canal --------
    # Attention :  eye_emg_corr peut ne pas exister selon tes scripts -> on gère ça proprement.
    has_eye_corr = "eye_emg_corr" in df.columns

    agg_dict = dict(
        rswa_fraction=("rswa", "mean") if "rswa" in df.columns else None,
        phasic_ratio_mean=("phasic_ratio", "mean") if "phasic_ratio" in df.columns else None,
        tonic_ratio_mean=("tonic_ratio", "mean") if "tonic_ratio" in df.columns else None,
        tonic_eog_fraction=("tonic_eog", "mean") if "tonic_eog" in df.columns else None,
    )

    if has_eye_corr:
        agg_dict.update(
            eye_emg_corr_mean=("eye_emg_corr", "mean"),
            eye_emg_corr_abs_max=("eye_emg_corr", lambda x: np.nanmax(np.abs(x))),
        )

    # Retire les entrées None si colonne manquante
    agg_dict = {k: v for k, v in agg_dict.items() if v is not None}

    required_cols = {"patient_id", "channel"}
    if not required_cols.issubset(set(df.columns)):
        log.warning("CSV EMG sans colonnes patient_id/channel >>> EMG ignoré")
        return None

    agg_ch = df.groupby(["patient_id", "channel"]).agg(**agg_dict).reset_index()

    # -------- agrégation patient --------
    # On moyenne les métriques canal-level sur les canaux disponibles.
    pat_agg_dict = {}
    if "rswa_fraction" in agg_ch.columns:
        pat_agg_dict["emg_rswa_fraction_mean"] = ("rswa_fraction", "mean")
    if "phasic_ratio_mean" in agg_ch.columns:
        pat_agg_dict["emg_phasic_ratio_mean"] = ("phasic_ratio_mean", "mean")
    if "tonic_ratio_mean" in agg_ch.columns:
        pat_agg_dict["emg_tonic_ratio_mean"] = ("tonic_ratio_mean", "mean")
    if "tonic_eog_fraction" in agg_ch.columns:
        pat_agg_dict["emg_tonic_eog_fraction_mean"] = ("tonic_eog_fraction", "mean")

    if has_eye_corr and "eye_emg_corr_mean" in agg_ch.columns:
        pat_agg_dict["emg_eye_corr_mean"] = ("eye_emg_corr_mean", "mean")
    if has_eye_corr and "eye_emg_corr_abs_max" in agg_ch.columns:
        pat_agg_dict["emg_eye_corr_abs_max"] = ("eye_emg_corr_abs_max", "max")

    agg_pat = agg_ch.groupby("patient_id").agg(**pat_agg_dict).reset_index()
    return agg_pat


# ======================================================
# PER FILE PROCESSING
# ======================================================
def _process_one_feature_file(fpath: Path) -> Optional[Dict[str, float]]:
    """
    Traite un seul fichier `*_REM-epo.fif` et renvoie une ligne patient-level.

    Étapes
    ------
    1) Déduit patient_id depuis le nom de fichier :
         patient_id = fpath.stem.split("_")[0]
       Exemple : "AN166_REM-epo" -> "AN166"
    2) Charge les epochs : mne.read_epochs(...)
    3) Calcule :
       - features temporelles EEG : compute_time_eeg_features
       - features spectrales EEG  : compute_spectral_eeg_features
    4) Concatène toutes les features epoch-level disponibles
    5) Agrège au niveau patient : aggregate_patient_features
    6) Retourne un dict (ligne) prêt pour DataFrame

    Paramètres
    ----------
    fpath : Path
        Chemin vers un fichier MNE Epochs : `*_REM-epo.fif`.

    Retours
    -------
    row : dict[str, float] | None
        - dict si features calculées avec succès
        - None si aucun canal EEG exploitable / features vides

    Notes
    -----
    - Cette fonction est exécutée en multiprocess => éviter les objets non picklables.
    - Les logs sont minimaux pour ne pas saturer la sortie multi-process.
    """
    log = get_logger("extract_features")

    patient_id = fpath.stem.split("_")[0]
    log.info(f"EEG features >>> {patient_id}")

    epochs = mne.read_epochs(fpath, preload=True, verbose=False)

    # Features EEG temps
    X_time, names_time = compute_time_eeg_features(epochs)

    # Features EEG spectrales (définies ailleurs dans ton projet)
    X_spec, names_spec = compute_spectral_eeg_features(epochs)

    parts: List[np.ndarray] = []
    names: List[str] = []

    if X_time is not None and len(names_time) > 0:
        parts.append(X_time)
        names += names_time

    if X_spec is not None and len(names_spec) > 0:
        parts.append(X_spec)
        names += names_spec

    if not parts:
        # Aucun EEG exploitable
        return None

    # Concat features epoch-level
    X = np.concatenate(parts, axis=1)

    # Agrégation patient-level
    row: Dict[str, float] = {"patient_id": patient_id}
    row.update(aggregate_patient_features(X, names))
    return row


# ======================================================
# MAIN
# ======================================================
def main(cfg: Dict):
    """
    Point d’entrée principal (batch).

    Paramètres attendus dans cfg (YAML)
    ----------------------------------
    cfg["in_root"] : str
        Dossier contenant les fichiers `*_REM-epo.fif` (récursif).
    cfg["out_csv"] : str
        Chemin du CSV final.
    cfg.get("rbd_emg_csv") : str | None
        Chemin du CSV EMG/EOG (optionnel).
    cfg.get("n_workers", 15) : int
        Nombre de processus.

    Pipeline
    --------
    1) Liste tous les `*_REM-epo.fif` sous in_root
    2) Multi-process : _process_one_feature_file pour chaque fichier
    3) Concatène les lignes patient EEG => df_eeg (1 ligne/patient)
    4) Charge agrégats EMG patient-level (si CSV fourni)
    5) Merge (left join) EEG + EMG sur patient_id
    6) Écrit out_csv
    """
    log = get_logger("extract_features")

    in_root = Path(cfg["in_root"])
    out_csv = Path(cfg["out_csv"])
    rbd_emg_csv = Path(cfg["rbd_emg_csv"]) if cfg.get("rbd_emg_csv") else None
    n_workers = int(cfg.get("n_workers", 15))

    fif_files = sorted(in_root.glob("**/*_REM-epo.fif"))
    if not fif_files:
        log.warning("Aucun fichier *_REM-epo.fif trouvé")
        return

    log.info(f"{len(fif_files)} fichiers *_REM-epo.fif trouvés. n_workers={n_workers}")

    rows: List[Dict[str, float]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_process_one_feature_file, f): f for f in fif_files}
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except Exception as e:
                f = futures[fut]
                log.error(f"[ERREUR] sur {f}: {e}")
                continue

            if row is not None:
                rows.append(row)

    df_eeg = pd.DataFrame(rows).drop_duplicates("patient_id")
    log.info(f"EEG features : {df_eeg.shape}")

    df_final = df_eeg

    # Ajout EMG patient-level
    df_emg = load_rbd_emg_features_patient_level(rbd_emg_csv, log)
    if df_emg is not None and not df_emg.empty:
        df_final = df_final.merge(df_emg, on="patient_id", how="left")
        log.info(f"EEG + EMG features : {df_final.shape}")
    else:
        log.info("Aucune feature EMG ajoutée (CSV absent/vide/invalide).")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_csv, index=False)
    log.info(f"Features écrites >>> {out_csv}")


if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--n_workers", type=int, default=15, help="Nombre de workers en parallèle.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg["n_workers"] = args.n_workers

    main(cfg)
