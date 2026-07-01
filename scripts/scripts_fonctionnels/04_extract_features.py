from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
04_extract_features.py  —  Extraction des features EEG et fusion EMG
=============================================================================

Ce script est AUTONOME : il ne dépend d'aucun dossier src/, ni de fichier YAML.

Ce qu'il fait :
  Pour chaque fichier *_REM-epo.fif dans REM_EPO_ROOT :
    1. Lecture des époques REM
    2. Calcul des features temporelles EEG (mean, std, rms, zero-crossings, etc.)
    3. Calcul des features spectrales EEG (puissances par bande, ratios, entropie...)
    4. Agrégation au niveau patient (moyenne + écart-type sur les epochs)
  Puis fusion avec les features EMG/EOG issues de 03_segment_rswa.py (optionnel).

Sortie :
  - OUTPUT_CSV : un CSV (1 ligne par patient) avec toutes les features EEG + EMG

Comment utiliser ce script :
  1. Modifier les chemins dans la section "PARAMÈTRES" ci-dessous
  2. Lancer :  python scripts/scripts_fonctionnels/04_extract_features.py 2>&1 | tee logfiles/log_04_extract_features.txt

Dépendances requises :
  pip install mne numpy pandas scipy

=============================================================================

INPUTS
------
REM_EPO_ROOT : fichiers *_REM-epo.fif (sortie de 02_segment_rem.py)
               Exemple : data/preprocessed/rem_epo/AN166_raw_REM-epo.fif
 
RBD_EMG_CSV  : CSV EMG/EOG (sortie de 03_segment_rswa.py) — optionnel
               Exemple : results/emg/rbd_emg_events_and_summary_4s_per_channel.csv
               (None = features EEG seules)
 
OUTPUTS
-------
OUTPUT_CSV : CSV features patient-level (1 ligne par patient)
             Chemin  : results/eeg/eeg_features.csv
             Colonnes :
               patient_id                — identifiant
               eeg_{ch}_{band}_bp_abs_mean/_std  — puissance absolue
               eeg_{ch}_{band}_bp_rel_mean/_std  — puissance relative
               eeg_{ch}_spec_centroid_mean/_std  — centroïde spectral
               eeg_{ch}_mean_mean/_std           — features temporelles
               emg_rswa_fraction_mean            — fraction RSWA (si EMG fourni)
               emg_phasic_ratio_mean             — ratio phasique moyen
               emg_tonic_ratio_mean              — ratio tonique moyen
               ... (une colonne _mean + _std par feature epoch-level)
 
DÉPENDANCES PIPELINE
--------------------
Ce script est l'ÉTAPE 4. Il requiert les sorties de 02 et (optionnellement) 03.
Sa sortie alimente :
  > 05_build_dataset.py   (fusion avec labels)
  > NB_A_eeg_spectral.ipynb (visualisation EEG)
"""

# ============================================================================
# PARAMÈTRES  <<<  À MODIFIER SELON CONFIGURATION
# ============================================================================
 
REM_EPO_ROOT = r"c:\dev\Cerco_studies\data\rem_epo"
RBD_EMG_CSV = r"c:\dev\Cerco_studies\data\rbd_emg_events_and_summary_4s_per_channel.csv"
OUTPUT_CSV = r"c:\dev\Cerco_studies\data\eeg_features.csv"
N_WORKERS = 4
 
# ============================================================================
# IMPORTS
# ============================================================================
 
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional
 
import mne
import numpy as np
import pandas as pd
from scipy.stats import entropy, kurtosis, skew
 
# ============================================================================
# LOGGER
# ============================================================================
 
def _get_logger(name: str = "extract_features") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger
 
 
# ============================================================================
# FEATURES TEMPORELLES EEG (inchangé — pas de bug ici, données et noms
# étaient déjà construits dans le même ordre : channel-major, feature-minor)
# ============================================================================
 
def _compute_temporal_eeg_features(epochs: mne.Epochs):
    picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    if len(picks) == 0:
        return None, []
 
    data = epochs.get_data()[:, picks, :]
 
    mean  = data.mean(axis=-1)
    std   = data.std(axis=-1)
    var   = data.var(axis=-1)
    rms   = np.sqrt((data ** 2).mean(axis=-1))
    zc    = (np.diff(np.sign(data), axis=-1) != 0).sum(axis=-1)
    skw   = skew(data, axis=-1, nan_policy="omit")
    krt   = kurtosis(data, axis=-1, nan_policy="omit")
    ptp   = np.ptp(data, axis=-1)
    ll    = np.sum(np.abs(np.diff(data, axis=-1)), axis=-1)
 
    feats = np.stack([mean, std, var, rms, zc, skw, krt, ptp, ll], axis=2)
    n_epochs, n_ch, n_f = feats.shape
    feats = feats.reshape(n_epochs, n_ch * n_f)
 
    names = []
    for ch_idx in picks:
        ch = epochs.ch_names[ch_idx]
        names.extend([f"eeg_{ch}_mean", f"eeg_{ch}_std", f"eeg_{ch}_var",
                      f"eeg_{ch}_rms", f"eeg_{ch}_zc", f"eeg_{ch}_skew",
                      f"eeg_{ch}_kurt", f"eeg_{ch}_ptp", f"eeg_{ch}_linelen"])
 
    return feats, names
 
 
# ============================================================================
# FEATURES SPECTRALES EEG — *** FONCTION CORRIGÉE ***
# ============================================================================
 
DEFAULT_BANDS = {
    "delta":      (0.5,  4.0),
    "theta":      (4.0,  8.0),
    "alpha":      (8.0,  12.0),
    "beta":       (12.0, 30.0),
    "gamma_bas":  (30.0, 50.0),
    "gamma_haut": (50.0, 80.0),
}
 
 
def _compute_spectral_eeg_features(epochs: mne.Epochs):
    """
    Calcule des features spectrales EEG par epoch et par canal.
 
    CORRECTIF : construction colonne par colonne (données + nom ajoutés
    ensemble, dans la même itération), pour garantir que chaque nom de
    colonne correspond bien à sa donnée — plus de risque de désynchronisation
    par arithmétique d'indices entre deux ordres de boucle différents.
    """
    picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    if len(picks) == 0:
        return None, []
 
    psd = epochs.compute_psd(method="welch", fmin=0.5, fmax=80.0, picks=picks,
                              n_fft=None, n_overlap=0, average="mean", verbose=False)
    psds  = psd.get_data()   # (n_epochs, n_ch, n_freqs)
    freqs = psd.freqs
 
    psd_sum = psds.sum(axis=-1) + 1e-20   # (n_epochs, n_ch)
    n_epochs, n_ch, _ = psds.shape
    band_names = list(DEFAULT_BANDS.keys())
 
    # --- Puissance abs/rel par bande : dict {band_name: array (n_epochs, n_ch)} ---
    band_powers_abs: Dict[str, np.ndarray] = {}
    band_powers_rel: Dict[str, np.ndarray] = {}
    for band_name, (fmin, fmax) in DEFAULT_BANDS.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        bp = psds[:, :, mask].sum(axis=-1)          # (n_epochs, n_ch)
        band_powers_abs[band_name] = bp
        band_powers_rel[band_name] = bp / psd_sum
 
    # --- Descripteurs spectraux : chacun (n_epochs, n_ch) ---
    p_norm       = psds / psd_sum[:, :, np.newaxis]
    spec_entropy = entropy(p_norm + 1e-20, base=2, axis=-1)
    centroid     = (p_norm * freqs[np.newaxis, np.newaxis, :]).sum(axis=-1)
    peak_freq    = freqs[psds.argmax(axis=-1)]
    var_f        = (p_norm * (freqs[np.newaxis, np.newaxis, :] - centroid[..., np.newaxis]) ** 2).sum(axis=-1)
    bandwidth    = np.sqrt(var_f)
 
    # --- Construction colonne par colonne : channel-major, band-minor ---
    # (données ET noms ajoutés ensemble à chaque itération -> alignement garanti)
    cols: List[np.ndarray] = []
    names: List[str] = []
 
    for ci, ch_idx in enumerate(picks):
        ch = epochs.ch_names[ch_idx]
        for b in band_names:
            cols.append(band_powers_abs[b][:, ci])
            names.append(f"eeg_{ch}_{b}_bp_abs")
        for b in band_names:
            cols.append(band_powers_rel[b][:, ci])
            names.append(f"eeg_{ch}_{b}_bp_rel")
        cols.append(centroid[:, ci]);     names.append(f"eeg_{ch}_spec_centroid")
        cols.append(spec_entropy[:, ci]); names.append(f"eeg_{ch}_spec_entropy")
        cols.append(peak_freq[:, ci]);    names.append(f"eeg_{ch}_peak_freq")
        cols.append(bandwidth[:, ci]);    names.append(f"eeg_{ch}_bandwidth")
 
    # --- Ratios inter-bandes sur la moyenne des canaux (déjà correct dans
    #     la version originale : 1 valeur scalaire par epoch, pas de canal) ---
    band_means = {b: band_powers_abs[b].mean(axis=1) for b in band_names}
 
    def _ratio(num, den):
        r = band_means[num] / (band_means[den] + 1e-20)
        cols.append(r)
        names.append(f"eeg_ratio_{num}_over_{den}")
 
    if "alpha" in band_means and "beta" in band_means:
        _ratio("alpha", "beta")
    if "theta" in band_means and "alpha" in band_means:
        _ratio("theta", "alpha")
    if "theta" in band_means and "beta" in band_means:
        _ratio("theta", "beta")
    if "gamma_bas" in band_means and "beta" in band_means:
        _ratio("gamma_bas", "beta")
 
    X = np.stack(cols, axis=1)   # (n_epochs, n_features)
    return X, names
 
 
# ============================================================================
# AGRÉGATION PATIENT-LEVEL (inchangé)
# ============================================================================
 
def _aggregate_patient(X: np.ndarray, names: List[str]) -> Dict[str, float]:
    row: Dict[str, float] = {}
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    for i, name in enumerate(names):
        row[f"{name}_mean"] = float(mean[i])
        row[f"{name}_std"]  = float(std[i])
    return row
 
 
# ============================================================================
# FEATURES EMG PATIENT-LEVEL (inchangé)
# ============================================================================
 
def _load_emg_features_patient_level(rbd_csv: Optional[Path], log) -> Optional[pd.DataFrame]:
    if rbd_csv is None or not rbd_csv.exists():
        log.warning("CSV EMG introuvable > EMG ignoré.")
        return None
 
    df = pd.read_csv(rbd_csv)
 
    if "type" not in df.columns:
        log.warning("CSV EMG sans colonne 'type' > EMG ignoré.")
        return None
 
    df = df[df["type"] == "REM_EPOCH_4S"].copy()
    if df.empty:
        log.warning("CSV EMG sans lignes REM_EPOCH_4S > EMG ignoré.")
        return None
 
    for bool_col in ("rswa", "tonic_eog"):
        if bool_col in df.columns and df[bool_col].dtype != bool:
            df[bool_col] = (df[bool_col].astype(str).str.strip().str.lower()
                            .isin(["1", "true", "t", "yes", "y"]))
 
    if not {"patient_id", "channel"}.issubset(set(df.columns)):
        log.warning("CSV EMG sans colonnes patient_id/channel > EMG ignoré.")
        return None
 
    agg_dict = {}
    for col, agg, out in [("rswa", "mean", "rswa_fraction"),
                           ("phasic_ratio", "mean", "phasic_ratio_mean"),
                           ("tonic_ratio", "mean", "tonic_ratio_mean"),
                           ("tonic_eog", "mean", "tonic_eog_fraction")]:
        if col in df.columns:
            agg_dict[out] = (col, agg)
 
    agg_ch = df.groupby(["patient_id", "channel"]).agg(**agg_dict).reset_index()
 
    pat_dict = {}
    for src, dst in [("rswa_fraction", "emg_rswa_fraction_mean"),
                     ("phasic_ratio_mean", "emg_phasic_ratio_mean"),
                     ("tonic_ratio_mean", "emg_tonic_ratio_mean"),
                     ("tonic_eog_fraction", "emg_tonic_eog_fraction_mean")]:
        if src in agg_ch.columns:
            pat_dict[dst] = (src, "mean")
 
    return agg_ch.groupby("patient_id").agg(**pat_dict).reset_index()
 
 
# ============================================================================
# TRAITEMENT D'UN FICHIER *_REM-epo.fif (inchangé)
# ============================================================================
 
def _process_one_file(fpath: Path) -> Optional[Dict[str, float]]:
    log = _get_logger()
    patient_id = fpath.stem.split("_")[0]
    log.info(f"Features EEG >>> {patient_id}")
 
    epochs = mne.read_epochs(fpath, preload=True, verbose=False)
 
    X_time, names_time = _compute_temporal_eeg_features(epochs)
    X_spec, names_spec = _compute_spectral_eeg_features(epochs)
 
    parts, names = [], []
    if X_time is not None:
        parts.append(X_time)
        names += names_time
    if X_spec is not None:
        parts.append(X_spec)
        names += names_spec
 
    if not parts:
        return None
 
    X   = np.concatenate(parts, axis=1)
    row = {"patient_id": patient_id}
    row.update(_aggregate_patient(X, names))
    return row
 
 
# ============================================================================
# POINT D'ENTRÉE PRINCIPAL (inchangé)
# ============================================================================
 
def main() -> None:
    log = _get_logger()
 
    in_root = Path(REM_EPO_ROOT)
    out_csv = Path(OUTPUT_CSV)
    rbd_csv = Path(RBD_EMG_CSV) if RBD_EMG_CSV is not None else None
 
    fif_files = sorted(in_root.glob("**/*_REM-epo.fif"))
    if not fif_files:
        log.warning(f"Aucun fichier *_REM-epo.fif trouvé dans {in_root}")
        return
 
    log.info(f"{len(fif_files)} fichiers trouvés. Lancement avec {N_WORKERS} worker(s).")
 
    rows: List[Dict] = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_process_one_file, f): f for f in fif_files}
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except Exception as e:
                log.error(f"[ERREUR] {futures[fut].name} : {e}")
                continue
            if row is not None:
                rows.append(row)
 
    df_eeg = pd.DataFrame(rows).drop_duplicates("patient_id")
    log.info(f"EEG features : {df_eeg.shape}")
 
    df_emg = _load_emg_features_patient_level(rbd_csv, log)
    if df_emg is not None and not df_emg.empty:
        df_final = df_eeg.merge(df_emg, on="patient_id", how="left")
        log.info(f"EEG + EMG features : {df_final.shape}")
    else:
        df_final = df_eeg
        log.info("Aucune feature EMG ajoutée.")
 
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_csv, index=False)
    log.info(f"Features écrites > {out_csv}")
 
 
if __name__ == "__main__":
    main()
