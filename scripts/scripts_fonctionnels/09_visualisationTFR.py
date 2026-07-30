from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
D_tfr_group_maps.py - Cartes temps-fréquence (TFR) par groupe + cartes relatives
=============================================================================

Ce script est AUTONOME : il ne dépend d'aucun dossier src/, ni de fichier YAML.
Méthodologie reprise de vos scripts de référence (TFR Morlet par patient,
comparaisons pairwise en triptyque abs/rel), adaptée aux conventions du
pipeline actuel (patients_label.txt, GROUP_ORDER, DISPLAY_LABELS, PALETTE).

Ce qu'il fait :
  1. Pour chaque fichier *_REM-epo.fif (REM_EPO_ROOT) :
       - Sélectionne les 12 canaux EEG bipolaires
       - Calcule le TFR (ondelettes de Morlet) par canal, moyenné sur les
         époques REM de ce patient -> 1 carte (n_freqs x n_times) en dB
       - Étape parallélisée sur les patients (la plus coûteuse)
  2. Associe chaque patient à son groupe clinique (patients_label.txt)
  3. Moyenne les cartes patient-level en cartes groupe-level, par canal
     (1 patient = 1 carte dans la moyenne, indépendamment du nombre d'époques)
  4. CARTES TEMPS-FRÉQUENCE : sauvegarde 1 figure par groupe x canal
  5. CARTES RELATIVES : pour chaque paire de groupes et chaque canal commun,
     triptyque (carte A | carte B | différence) en deux versions :
       - différence absolue (dB)
       - différence relative (%), échelle fixe ±100 %

Limites connues (héritées de la simplification par rapport au script de
référence) :
  - Seul le stade REM est traité ici (le script de référence gérait aussi
    N2/N3 via un mapping timeline originale -> timeline nettoyée, qui
    n'a pas d'équivalent dans ce pipeline pour l'instant). STAGE_LABEL est
    laissé en paramètre pour faciliter une extension future.
  - Si la durée des époques REM diffère légèrement entre patients, les cartes
    sont rognées à la durée minimale commune avant moyenne (comme
    align_min() dans le script de référence).
  - La différence relative (%) est calculée directement sur les valeurs en
    dB déjà moyennées (comme dans le script de référence), pas sur la
    puissance linéaire - c'est un choix de reprise à l'identique de la
    méthodologie fournie, gardez-le en tête dans l'interprétation.

Comment utiliser ce script :
  1. Modifier les chemins dans la section "PARAMÈTRES" ci-dessous
  2. Lancer :  python D_tfr_group_maps.py

Dépendances requises :
  pip install mne numpy pandas matplotlib
=============================================================================
"""

# ============================================================================
# PARAMÈTRES  <<<  À MODIFIER SELON VOTRE CONFIGURATION
# ============================================================================

REM_EPO_ROOT = r"c:\dev\Cerco_studies\data\rem_epo"
LABELS_TXT   = r"c:\dev\Cerco_studies\data\patients_label.txt"
OUTPUT_ROOT  = r"c:\dev\Cerco_studies\data\visualisation2\eeg\tfr"

STAGE_LABEL = "REM"   # pour l'arborescence de sortie et les titres (extensible à N2/N3 plus tard)

SAVE_FIGS  = True
FIG_FORMAT = "pdf"    # "pdf" ou "svg" (vectoriel) ; "png" pour un rendu plus rapide
DPI        = 300      # monter à 600-1200 pour de la qualité poster (cf. script de référence)

# --- Paramètres du TFR (ondelettes de Morlet) - repris du script de référence ---
FREQ_MIN    = 1.0
FREQ_MAX    = 40.0
N_FREQS     = 30
CYCLES_MULT = 0.5
DECIM       = 2

CMAP_TFR      = "jet"      # cartes temps-fréquence par groupe
CMAP_DIFF     = "RdBu_r"   # cartes de différence
AUTO_PCT      = (5, 95)    # percentiles pour l'échelle auto des cartes par groupe

# Différence relative (%) : échelle automatique (percentiles robustes, comme la diff absolue)
# ou fixe (+/- REL_DIFF_LIM). "auto" recommandé sauf besoin explicite de comparer plusieurs
# figures entre elles sur une échelle strictement identique.
REL_DIFF_SCALE = "fixed"    # "auto" ou "fixed"
REL_DIFF_LIM   = 50.0     # utilisé seulement si REL_DIFF_SCALE == "fixed"

MIN_N_FOR_GROUP = 2   # un groupe avec moins de patients que ce seuil (pour un canal donné)
                       # est exclu des cartes et des comparaisons

N_WORKERS = 4   # calcul du TFR par patient en parallèle (étape coûteuse)

PALETTE = {
    "EAI":   "#C9D175",
    "Narco": "#F15854",
    "SYN":   "#44AA99",
    "TCSPi": "#BEBEBE",
}
# Ordre d'affichage voulu : EAI > Narco > SYN > TCSPi
GROUP_ORDER = ["EAI", "Narco", "SYN", "TCSPi"]

# Libellés affichés (titres, noms de fichiers) - les codes internes ci-dessus
# restent utilisés pour la fusion avec patients_label.txt et le filtrage.
DISPLAY_LABELS = {
    "SYN":   "Syn",
    "Narco": "Narco",
    "TCSPi": "iRBD",
    "EAI":   "AI",
}

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
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _get_logger(name: str = "tfr_group_maps") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "", name)


# ============================================================================
# CHARGEMENT DES LABELS (identique aux scripts précédents)
# ============================================================================

def _load_labels() -> pd.DataFrame:
    # patients_label.txt n'a pas de ligne d'en-tête : 2 colonnes, patient_id puis group.
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
# CALCUL DU TFR PAR PATIENT (worker parallèle)
# ============================================================================

def _compute_patient_tfr(fpath_str: str):
    """
    Calcule le TFR (Morlet, dB) pour les canaux bipolaires EEG présents chez ce
    patient, moyenné sur ses époques REM.
    Retourne (patient_id, {"channels", "freqs", "times", "Z"} | None, err | None).
    Z : array (n_ch, n_freqs, n_times) en dB.
    """
    from mne.time_frequency import tfr_morlet

    fpath = Path(fpath_str)
    patient_id = fpath.stem.split("_")[0]

    epochs = mne.read_epochs(fpath, preload=True, verbose=False)

    ch_present = [ch for ch in BIPOLAR_CHANNELS if ch in epochs.ch_names]
    if not ch_present:
        return patient_id, None, "Aucun canal bipolaire EEG trouvé."

    ep = epochs.copy().pick(ch_present)
    ep.apply_function(lambda x: x * 1e6, channel_wise=True)  # V -> µV avant le TFR

    freqs = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQS)
    n_cycles = freqs * CYCLES_MULT

    power = tfr_morlet(ep, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=False, average=True, decim=DECIM, verbose=False)

    Z = 10.0 * np.log10(np.maximum(power.data, np.finfo(float).tiny))  # (n_ch, n_freqs, n_times)

    return patient_id, {"channels": list(power.ch_names),
                        "freqs": power.freqs, "times": power.times, "Z": Z}, None


# ============================================================================
# PLOT - carte temps-fréquence pour UN groupe x UN canal
# ============================================================================

def _plot_tfr_map(group: str, channel: str, Z: np.ndarray,
                   freqs: np.ndarray, times: np.ndarray, n: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display_name = DISPLAY_LABELS.get(group, group)

    fig, ax = plt.subplots(figsize=(7, 5))
    vmin, vmax = np.percentile(Z, list(AUTO_PCT))
    im = ax.imshow(Z, aspect="auto", origin="lower",
                    extent=[times[0], times[-1], freqs[0], freqs[-1]],
                    cmap=CMAP_TFR, vmin=vmin, vmax=vmax)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Fréquence (Hz)")
    ax.set_title(f"{display_name} - {channel} ({STAGE_LABEL}, n={n})\nPuissance moyenne")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Power (dB)")
    plt.tight_layout()

    if SAVE_FIGS:
        out_dir = Path(OUTPUT_ROOT) / "group_maps" / STAGE_LABEL / _sanitize(channel)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"tfr_{_sanitize(display_name)}_{_sanitize(channel)}.{FIG_FORMAT}"
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# PLOT - triptyque de comparaison (groupe A | groupe B | différence)
# ============================================================================

def _extent(times: np.ndarray, freqs: np.ndarray):
    return [times[0], times[-1], freqs[0], freqs[-1]]


def _robust_limits(a: np.ndarray, p_lo: float = 5, p_hi: float = 95):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(a, [p_lo, p_hi])
    if lo == hi:
        return lo - 1e-6, hi + 1e-6
    return float(lo), float(hi)


def _plot_comparison(group_a: str, group_b: str, channel: str,
                      za: np.ndarray, zb: np.ndarray, n_a: int, n_b: int,
                      freqs: np.ndarray, times: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    disp_a = DISPLAY_LABELS.get(group_a, group_a)
    disp_b = DISPLAY_LABELS.get(group_b, group_b)

    diff_abs = za - zb
    diff_rel = 100.0 * (za - zb) / np.maximum(np.abs(zb), 1e-12)

    out_dir = Path(OUTPUT_ROOT) / "group_comparisons" / STAGE_LABEL / _sanitize(channel)
    out_dir.mkdir(parents=True, exist_ok=True)

    if REL_DIFF_SCALE == "fixed":
        vmin_rel, vmax_rel = -REL_DIFF_LIM, REL_DIFF_LIM
        ticks_rel = [-REL_DIFF_LIM, -REL_DIFF_LIM / 2, 0, REL_DIFF_LIM / 2, REL_DIFF_LIM]
    else:
        vmin_rel, vmax_rel = _robust_limits_symmetric(diff_rel)
        ticks_rel = None

    for diff, label, fname_suffix, vmin_d, vmax_d, ticks in [
        (diff_abs, "Delta Power (dB)", "abs", *_robust_limits_symmetric(diff_abs), None),
        (diff_rel, "Delta Power (%)", "rel", vmin_rel, vmax_rel, ticks_rel),
    ]:
        vmin_a, vmax_a = _robust_limits(za)
        vmin_b, vmax_b = _robust_limits(zb)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

        im1 = axes[0].imshow(za, aspect="auto", origin="lower", extent=_extent(times, freqs),
                             cmap=CMAP_TFR, vmin=vmin_a, vmax=vmax_a)
        axes[0].set_title(f"{disp_a} - {channel} (n={n_a})")
        axes[0].set_xlabel("Temps (s)"); axes[0].set_ylabel("Fréquence (Hz)")
        plt.colorbar(im1, ax=axes[0]).set_label("Power (dB)")

        im2 = axes[1].imshow(zb, aspect="auto", origin="lower", extent=_extent(times, freqs),
                             cmap=CMAP_TFR, vmin=vmin_b, vmax=vmax_b)
        axes[1].set_title(f"{disp_b} - {channel} (n={n_b})")
        axes[1].set_xlabel("Temps (s)")
        plt.colorbar(im2, ax=axes[1]).set_label("Power (dB)")

        im3 = axes[2].imshow(diff, aspect="auto", origin="lower", extent=_extent(times, freqs),
                             cmap=CMAP_DIFF, vmin=vmin_d, vmax=vmax_d)
        axes[2].set_title(f"Diff ({fname_suffix}) - {disp_a} vs {disp_b}")
        axes[2].set_xlabel("Temps (s)")
        cbar3 = plt.colorbar(im3, ax=axes[2], ticks=ticks) if ticks else plt.colorbar(im3, ax=axes[2])
        cbar3.set_label(label)

        fig.suptitle(f"Comparaison TFR - {STAGE_LABEL} - {channel}", y=1.02, fontsize=12)
        fig.tight_layout()

        if SAVE_FIGS:
            save_path = out_dir / (f"compare_{_sanitize(disp_a)}_vs_{_sanitize(disp_b)}"
                                    f"_{fname_suffix}.{FIG_FORMAT}")
            fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)


def _robust_limits_symmetric(a: np.ndarray, p_lo: float = 5, p_hi: float = 95):
    lo, hi = _robust_limits(a, p_lo, p_hi)
    v = max(abs(lo), abs(hi)) or 1e-6
    return -v, v


# ============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================================

def main() -> None:
    log = _get_logger()
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    in_root = Path(REM_EPO_ROOT)
    fif_files = sorted(in_root.glob("**/*_REM-epo.fif"))
    if not fif_files:
        log.warning(f"Aucun fichier *_REM-epo.fif trouvé dans {in_root}")
        return

    log.info(f"{len(fif_files)} fichiers trouvés. Calcul TFR (Morlet, {N_FREQS} freqs "
             f"{FREQ_MIN}-{FREQ_MAX} Hz) avec {N_WORKERS} worker(s) - étape la plus longue.")

    # --- Calcul du TFR par patient, en parallèle ---
    per_patient: Dict[str, Dict] = {}
    errors: List[Tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_compute_patient_tfr, str(f)): f for f in fif_files}
        for fut in as_completed(futures):
            fpath = futures[fut]
            try:
                patient_id, data, err = fut.result()
            except Exception as e:
                errors.append((fpath.name, str(e)))
                log.error(f"[ERREUR] {fpath.name} : {e}")
                continue

            if err is not None:
                errors.append((patient_id, err))
                log.warning(f"[{patient_id}] {err}")
                continue

            per_patient[patient_id] = data

    log.info(f"TFR calculé pour {len(per_patient)} patients ({len(errors)} erreurs).")
    if errors:
        log.warning("Patients en erreur :")
        for pid, msg in errors:
            log.warning(f"  - {pid} : {msg}")

    if not per_patient:
        log.error("Aucun TFR calculé - arrêt.")
        return

    # --- Association aux groupes cliniques ---
    lbl = _load_labels()
    lbl_map = dict(zip(lbl["patient_id"], lbl["group"]))

    n_no_label, n_bad_group = 0, 0
    for pid, data in per_patient.items():
        group = lbl_map.get(pid)
        if group is None:
            n_no_label += 1
            data["group"] = None
        elif group not in GROUP_ORDER:
            n_bad_group += 1
            data["group"] = None
        else:
            data["group"] = group

    if n_no_label:
        log.warning(f"{n_no_label} patients sans label dans {LABELS_TXT} (exclus).")
    if n_bad_group:
        log.warning(f"{n_bad_group} patients avec un groupe non reconnu (exclus).")

    # --- Vérification de la grille de fréquences (déterministe, doit toujours matcher) ---
    ref_freqs = next(iter(per_patient.values()))["freqs"]
    for pid, data in per_patient.items():
        if not np.allclose(data["freqs"], ref_freqs):
            log.warning(f"[{pid}] grille de fréquences différente de la référence - vérifier la config MNE.")

    # --- Alignement temporel : rogne à la durée minimale commune (cf. align_min du script de référence) ---
    min_n_times = min(data["Z"].shape[-1] for data in per_patient.values())
    ref_times = None
    for data in per_patient.values():
        data["Z"] = data["Z"][:, :, :min_n_times]
        if ref_times is None:
            ref_times = data["times"][:min_n_times]
    log.info(f"Grille temporelle commune : {min_n_times} points ({ref_times[0]:.2f}s à {ref_times[-1]:.2f}s).")

    # --- Moyenne des cartes patient-level en cartes groupe-level, par canal ---
    # (1 patient = 1 carte dans la moyenne, indépendamment de son nombre d'époques)
    group_curves: Dict[str, Dict[str, List[np.ndarray]]] = {g: {} for g in GROUP_ORDER}
    for data in per_patient.values():
        group = data["group"]
        if group is None:
            continue
        for ci, ch in enumerate(data["channels"]):
            group_curves[group].setdefault(ch, []).append(data["Z"][ci])

    group_avg: Dict[str, Dict[str, Dict]] = {g: {} for g in GROUP_ORDER}
    for g in GROUP_ORDER:
        for ch, arrs in group_curves[g].items():
            if len(arrs) >= MIN_N_FOR_GROUP:
                group_avg[g][ch] = {"Z": np.mean(np.stack(arrs), axis=0), "n": len(arrs)}
        n_per_ch = {ch: v["n"] for ch, v in group_avg[g].items()}
        log.info(f"[{DISPLAY_LABELS.get(g, g)}] patients par canal : {n_per_ch}")

    # --- CARTES TEMPS-FRÉQUENCE : 1 figure par groupe x canal ---
    log.info("Génération des cartes temps-fréquence par groupe...")
    for g in GROUP_ORDER:
        for ch, info in group_avg[g].items():
            _plot_tfr_map(g, ch, info["Z"], ref_freqs, ref_times, info["n"])

    # --- CARTES RELATIVES : comparaisons pairwise entre groupes ---
    log.info("Génération des cartes de comparaison (relatives)...")
    for group_a, group_b in combinations(GROUP_ORDER, 2):
        common_channels = sorted(set(group_avg[group_a]) & set(group_avg[group_b]))
        if not common_channels:
            log.info(f"[{DISPLAY_LABELS.get(group_a, group_a)} vs {DISPLAY_LABELS.get(group_b, group_b)}] "
                     f"aucun canal commun avec assez de patients - ignoré.")
            continue
        for ch in common_channels:
            info_a = group_avg[group_a][ch]
            info_b = group_avg[group_b][ch]
            _plot_comparison(group_a, group_b, ch, info_a["Z"], info_b["Z"],
                              info_a["n"], info_b["n"], ref_freqs, ref_times)
        log.info(f"[{DISPLAY_LABELS.get(group_a, group_a)} vs {DISPLAY_LABELS.get(group_b, group_b)}] "
                 f"{len(common_channels)} canaux comparés.")

    log.info(f"Cartes écrites dans {OUTPUT_ROOT}")
    log.info("Terminé.")


if __name__ == "__main__":
    main()