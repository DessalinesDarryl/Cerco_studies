from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
B_psd_by_group.py - PSD moyen par canal EEG et par catégorie de patient
=============================================================================

Ce script est AUTONOME : il ne dépend d'aucun dossier src/, ni de fichier YAML.

Ce qu'il fait :
  Pour chaque fichier *_REM-epo.fif dans REM_EPO_ROOT :
    1. Calcule le PSD (Welch, 0.5-80 Hz) sur chaque canal EEG bipolaire
    2. Moyenne le PSD sur toutes les époques REM du patient
  Puis, par canal :
    3. Regroupe les patients par catégorie clinique (SYN/Narco/TCSPi/EAI)
    4. Trace 1 courbe = moyenne inter-patients ± écart-type (SD), par groupe
    5. Sauvegarde 1 figure par canal (12 figures au total)

Choix retenus (cf. échange) :
  - 1 figure par canal (12 figures séparées)
  - Échelle linéaire en µV²/Hz
  - Bande d'incertitude = moyenne ± écart-type (SD) inter-patients

Comment utiliser ce script :
  1. Modifier les chemins dans la section "PARAMÈTRES" ci-dessous
  2. Lancer :  python B_psd_by_group.py

Dépendances requises :
  pip install mne numpy pandas scipy matplotlib seaborn
=============================================================================

INPUTS
------
REM_EPO_ROOT : fichiers *_REM-epo.fif (sortie de 02_segment_rem.py)
LABELS_TXT   : fichier de labels patients (mêmes colonnes/format que A_eeg_spectral.ipynb)

OUTPUTS
-------
OUTPUT_FIGS : 1 figure PSD par canal EEG bipolaire (FIG_FORMAT, DPI réglables)
"""

# ============================================================================
# PARAMÈTRES  <<<  À MODIFIER SELON VOTRE CONFIGURATION
# ============================================================================
 
REM_EPO_ROOT = r"c:\dev\Cerco_studies\data\rem_epo"
LABELS_TXT   = r"c:\dev\Cerco_studies\data\patients_label.txt"
OUTPUT_FIGS  = r"c:\dev\Cerco_studies\data\visualisation\eeg\psd_by_group"
 
SAVE_FIGS  = True
FIG_FORMAT = "pdf"   # "pdf" ou "svg" (vectoriel) ; "png" pour un rendu plus rapide
DPI        = 300
 
FMIN, FMAX = 0.5, 80.0     # bande affichée (Hz)
PSD_WIN_SEC = 4.0          # durée de fenêtre Welch (s) - cf. script de référence
PSD_OVERLAP = 0.50         # recouvrement Welch (fraction) - cf. script de référence
 
COMMON_NFREQ = 400                                   # grille de fréquences commune
COMMON_FREQS = None                                  # calculée dans main() une fois FMIN/FMAX fixés
 
NOTCH_FREQS = [50.0]        # fréquence(s) de bruit secteur à masquer (ajouter 100.0 si harmonique visible)
NOTCH_HALFWIDTH_HZ = 1.0    # demi-largeur de la bande masquée autour de chaque fréquence (Hz)

N_WORKERS = 4   # calcul du PSD par patient en parallèle
 
PALETTE = {
    "EAI":   "#C9D175",
    "Narco": "#F15854",
    "SYN":   "#44AA99",
    "TCSPi": "#BEBEBE",
}
GROUP_ORDER = ["SYN", "Narco", "TCSPi", "EAI"]
 
BIPOLAR_CHANNELS = [
    "Fp1-T3", "Fp1-C3", "T3-O1",
    "Fp2-T4", "Fp2-C4", "T4-O2",
    "Fp1-A1", "Fp2-A1", "T3-A1",
    "C3-A1",  "T4-A1",  "C4-A1",
]
 
# ============================================================================
# IMPORTS
# ============================================================================
 
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
 
import mne
import numpy as np
import pandas as pd
 
warnings.filterwarnings("ignore")
 
 
def _get_logger(name: str = "psd_by_group") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger
 
 
# ============================================================================
# CHARGEMENT DES LABELS (identique aux scripts précédents)
# ============================================================================
 
def _load_labels() -> pd.DataFrame:
    try:
        lbl = pd.read_csv(LABELS_TXT)
        cols = {c.lower(): c for c in lbl.columns}
        id_col  = cols.get("patient_id") or cols.get("identifiant")
        lbl_col = cols.get("label_str") or cols.get("diagnostic") or cols.get("label")
        lbl = lbl[[id_col, lbl_col]].rename(columns={id_col: "patient_id", lbl_col: "group"})
    except Exception:
        lbl = pd.read_csv(LABELS_TXT, header=None, names=["patient_id", "group"])
 
    lbl["patient_id"] = lbl["patient_id"].astype(str).str.strip()
    lbl["group"]      = lbl["group"].astype(str).str.strip()
 
    def to_macro(s: str) -> str:
        u = s.upper()
        if any(k in u for k in ("PARK", "MPI", "AMS", "DCL", "DLB", "PAF")):
            return "SYN"
        if "NARCO" in u:
            return "Narco"
        if "TCSP" in u or "RBDI" in u:
            return "TCSPi"
        if "EAI" in u or "ENCEPHALITE" in u:
            return "EAI"
        return s
 
    lbl["group"] = lbl["group"].map(to_macro)
    return lbl
 
 
# ============================================================================
# CALCUL DU PSD (vrai Welch multi-fenêtres, signal REM concaténé)
# ============================================================================
 
def _welch_psd_array(data_uv: np.ndarray, sfreq: float, fmin: float, fmax: float,
                      win_sec: float, overlap: float):
    """
    data_uv : (n_ch, n_times) en µV, déjà concaténé sur toutes les époques REM.
    Retourne (freqs, psd) avec psd en µV²/Hz, shape (n_ch, n_freqs).
    """
    from mne.time_frequency import psd_array_welch
 
    n_times = int(data_uv.shape[1])
    n_per_seg = int(round(sfreq * win_sec))
    n_per_seg = max(8, min(n_per_seg, n_times))
 
    n_overlap = int(round(overlap * n_per_seg))
    if n_overlap >= n_per_seg:
        n_overlap = max(0, n_per_seg - 1)
 
    psd, freqs = psd_array_welch(
        data_uv, sfreq,
        fmin=fmin, fmax=fmax,
        n_fft=n_per_seg, n_per_seg=n_per_seg, n_overlap=n_overlap,
        average="mean", verbose="ERROR",
    )
    return np.asarray(freqs, dtype=float), np.asarray(psd, dtype=float)
 
 
def _epochs_to_concat_uv(epochs: mne.Epochs, ch_present: List[str]) -> np.ndarray:
    """Concatène les époques REM bout à bout -> (n_ch, n_epochs*n_times), en µV."""
    ep = epochs.copy().pick(ch_present)
    data = ep.get_data()                      # (n_epochs, n_ch, n_times), en V
    data_uv = data * 1e6                       # V -> µV AVANT le calcul du PSD
    n_ep, n_ch, n_times = data_uv.shape
    return np.transpose(data_uv, (1, 0, 2)).reshape(n_ch, n_ep * n_times)
 
 
def _interp_to_common_grid(freqs: np.ndarray, psd_row: np.ndarray, grid: np.ndarray) -> np.ndarray:
    m = np.isfinite(freqs) & np.isfinite(psd_row)
    f, y = freqs[m], psd_row[m]
    if f.size < 2:
        return np.full_like(grid, np.nan, dtype=float)
    return np.interp(grid, f, y, left=y[0], right=y[-1])
 
 
def _process_one_patient(fpath_str: str, common_freqs: np.ndarray):
    """
    Calcule le PSD (vrai Welch, µV²/Hz) pour les canaux bipolaires EEG présents
    chez ce patient, interpolé sur common_freqs.
    Retourne (patient_id, psd_common (n_ch_present, n_common_freq), ch_present, err).
    """
    fpath = Path(fpath_str)
    patient_id = fpath.stem.split("_")[0]
 
    epochs = mne.read_epochs(fpath, preload=True, verbose=False)
 
    ch_present = [ch for ch in BIPOLAR_CHANNELS if ch in epochs.ch_names]
    if not ch_present:
        return patient_id, None, None, "Aucun canal bipolaire EEG trouvé."
 
    sfreq = float(epochs.info["sfreq"])
    data_uv = _epochs_to_concat_uv(epochs, ch_present)   # (n_ch, n_times_concat), µV
 
    freqs, psd_lin = _welch_psd_array(data_uv, sfreq, FMIN, FMAX, PSD_WIN_SEC, PSD_OVERLAP)
 
    psd_common = np.stack([_interp_to_common_grid(freqs, psd_lin[i], common_freqs)
                            for i in range(psd_lin.shape[0])])
    return patient_id, psd_common, ch_present, None
 
 
# ============================================================================
# PLOT - 1 figure par canal, en dB / axe X log (cf. script de référence)
# ============================================================================
def _interp_nan_gap(freqs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Comble les trous NaN par interpolation linéaire - affichage uniquement,
    le calcul stats reste sur les données masquées."""
    y = y.copy()
    nan_mask = np.isnan(y)
    if nan_mask.any() and (~nan_mask).sum() >= 2:
        y[nan_mask] = np.interp(freqs[nan_mask], freqs[~nan_mask], y[~nan_mask])
    return y

def _plot_channel(channel: str, freqs: np.ndarray,
                   group_curves_db: Dict[str, np.ndarray], group_n: Dict[str, int]):
    """
    group_curves_db[group] : array (n_patients_du_groupe, n_freqs), déjà en dB.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter, FixedLocator, LogLocator
 
    fig, ax = plt.subplots(figsize=(9, 5.5))
 
    for group in GROUP_ORDER:
        if group not in group_curves_db:
            continue
        data_db = group_curves_db[group]         # (n_patients, n_freqs), en dB
        mean_curve = np.nanmean(data_db, axis=0)
        sd_curve   = np.nanstd(data_db, axis=0)
 
        mean_curve_plot = _interp_nan_gap(freqs, mean_curve)
        lower_plot = _interp_nan_gap(freqs, mean_curve - sd_curve)
        upper_plot = _interp_nan_gap(freqs, mean_curve + sd_curve)

        color = PALETTE.get(group, None)
        n = group_n.get(group, data_db.shape[0])
        ax.plot(freqs, mean_curve_plot, color=color, linewidth=1.8, label=f"{group} (n={n})")
        ax.fill_between(freqs, lower_plot, upper_plot,
                         color=color, alpha=0.15, linewidth=0)
 
    major_ticks = [0.5, 1, 2, 4, 8, 13, 30, 50, 80]
    ax.set_xscale("log")
    ax.set_xlim(FMIN, FMAX)
    ax.xaxis.set_major_locator(FixedLocator(major_ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 5, 7)))
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.07)
 
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("PSD (dB re µV²/Hz)")
    ax.set_title(f"PSD moyen par groupe - {channel}\n(moyenne ± écart-type)")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
 
    if SAVE_FIGS:
        save_path = Path(OUTPUT_FIGS) / f"psd_{channel.replace('-', '')}.{FIG_FORMAT}"
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
 
 
# ============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================================
 
def main() -> None:
    global COMMON_FREQS
    log = _get_logger()
    Path(OUTPUT_FIGS).mkdir(parents=True, exist_ok=True)
 
    COMMON_FREQS = np.linspace(FMIN, FMAX, COMMON_NFREQ)
 
    in_root = Path(REM_EPO_ROOT)
    fif_files = sorted(in_root.glob("**/*_REM-epo.fif"))
    if not fif_files:
        log.warning(f"Aucun fichier *_REM-epo.fif trouvé dans {in_root}")
        return
    log.info(f"{len(fif_files)} fichiers trouvés. Calcul du PSD (Welch {PSD_WIN_SEC:.0f}s/"
             f"{int(PSD_OVERLAP*100)}%) avec {N_WORKERS} worker(s).")
 
    # --- Calcul du PSD par patient, en parallèle ---
    per_patient: Dict[str, Dict] = {}
    errors: List[Tuple[str, str]] = []
 
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_process_one_patient, str(f), COMMON_FREQS): f for f in fif_files}
        for fut in as_completed(futures):
            fpath = futures[fut]
            try:
                patient_id, psd_common, ch_present, err = fut.result()
            except Exception as e:
                errors.append((fpath.name, str(e)))
                log.error(f"[ERREUR] {fpath.name} : {e}")
                continue
 
            if err is not None:
                errors.append((patient_id, err))
                log.warning(f"[{patient_id}] {err}")
                continue
 
            per_patient[patient_id] = {"psd": psd_common, "channels": ch_present}
 
    log.info(f"PSD calculé pour {len(per_patient)} patients ({len(errors)} erreurs).")
    if errors:
        log.warning("Patients en erreur :")
        for pid, msg in errors:
            log.warning(f"  - {pid} : {msg}")
 
    if not per_patient:
        log.error("Aucun PSD calculé - arrêt.")
        return
 
    # --- Masquage du bruit secteur (50 Hz, etc.) : NaN sur la bande concernée ---
    if NOTCH_FREQS:
        notch_mask = np.zeros(COMMON_FREQS.shape, dtype=bool)
        for nf in NOTCH_FREQS:
            notch_mask |= (np.abs(COMMON_FREQS - nf) <= NOTCH_HALFWIDTH_HZ)
        if notch_mask.any():
            log.info(f"Masquage notch : {NOTCH_FREQS} Hz (± {NOTCH_HALFWIDTH_HZ} Hz) "
                     f"-> {notch_mask.sum()} points masqués sur {len(COMMON_FREQS)}")
            for info in per_patient.values():
                info["psd"][:, notch_mask] = np.nan
 
    # --- Association aux groupes cliniques ---
    lbl = _load_labels()
    lbl_map = dict(zip(lbl["patient_id"], lbl["group"]))
 
    n_no_label, n_bad_group = 0, 0
    for pid in list(per_patient.keys()):
        group = lbl_map.get(pid)
        if group is None:
            n_no_label += 1
            per_patient[pid]["group"] = None
        elif group not in GROUP_ORDER:
            n_bad_group += 1
            per_patient[pid]["group"] = None
        else:
            per_patient[pid]["group"] = group
 
    if n_no_label:
        log.warning(f"{n_no_label} patients sans label dans {LABELS_TXT} (exclus des plots).")
    if n_bad_group:
        log.warning(f"{n_bad_group} patients avec un groupe non reconnu (exclus des plots).")
 
    # --- Construction des courbes par canal x groupe (en dB), puis plot ---
    eps = np.finfo(float).tiny
 
    for channel in BIPOLAR_CHANNELS:
        group_curves: Dict[str, List[np.ndarray]] = {g: [] for g in GROUP_ORDER}
 
        for pid, info in per_patient.items():
            group = info["group"]
            if group is None or channel not in info["channels"]:
                continue
            ch_idx = info["channels"].index(channel)
            group_curves[group].append(info["psd"][ch_idx])   # linéaire, µV²/Hz
 
        group_curves_db: Dict[str, np.ndarray] = {}
        group_n: Dict[str, int] = {}
        for g, v in group_curves.items():
            group_n[g] = len(v)
            if len(v) >= 2:
                lin = np.stack(v)
                group_curves_db[g] = 10.0 * np.log10(np.maximum(lin, eps))
 
        if not group_curves_db:
            log.warning(f"[{channel}] Pas assez de patients par groupe (<2) - figure ignorée.")
            continue
 
        log.info(f"[{channel}] Patients par groupe : {group_n}")
        _plot_channel(channel, COMMON_FREQS, group_curves_db, group_n)
 
    log.info(f"Figures PSD écrites dans {OUTPUT_FIGS}")
    log.info("Terminé.")
 
 
if __name__ == "__main__":
    main()
