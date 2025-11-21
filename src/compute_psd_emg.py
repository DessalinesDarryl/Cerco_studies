#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_psd_emg.py

Objectifs :
  1) Calculer la puissance EMG absolue (µV²) sur 30-100 Hz
     pendant les périodes RSWA uniquement (détectées via ton CSV).
  2) Calculer la PSD EMG (30-100 Hz) pendant :
        - RSWA
        - REM_sans_RSWA
     et construire, pour chaque patient, un spectre moyen sur une grille de fréquences commune.
  3) Tracer des spectres moyens par catégorie de patients (groupes issus de l'Excel),
     et comparer RSWA vs REM_sans_RSWA.

Entrées :
  - Dossiers gp2 : /.../gp2/{PATIENT}/*.fif (prétraités)
  - CSV RSWA : rbd_emg_events_and_summary_4s_per_channel.csv
  - Excel des catégories : nv_patientsRBD_identifiants.xlsx (via cohort.load_patient_groups)

Sorties :
  - out_root/{PAT}/{PAT}_emg_psd_rswa_full_30_100.png        (PSD moyenne tous canaux EMG, RSWA)
  - out_root/{PAT}/{PAT}_emg_psd_rswa_{CANAL}.png            (PSD canal EMG, RSWA)
  - out_root/emg_rswa_patient_power_30_100.csv               (1 ligne par patient, puissance 30-100 Hz RSWA)
  - out_root/emg_rswa_channel_power_30_100.csv               (long : patient × canal, puissance 30-100 Hz RSWA)
  - out_root/group_emg_psd_rswa_30_100.png                   (spectre moyen par catégorie, RSWA)
  - out_root/group_emg_psd_nonrswa_30_100.png                (spectre moyen par catégorie, REM_sans_RSWA)
  - out_root/global_emg_psd_rswa_vs_nonrswa_30_100.png       (RSWA vs nonRSWA, tous patients)
"""

import os, sys, argparse
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator

import mne
mne.set_config('MNE_MEMMAP_MIN_SIZE', '1M', set_env=True)
mne.set_log_level("WARNING")

# ---------- chemins par défaut ----------
GP2_ROOT_DEFAULT = "/home/darryld/documents/EEG/preprocessed/bipolaire/1_noArtefacts/gp2/"
RSWA_CSV_DEFAULT = "/home/darryld/Cerco_studies/data/rbd_emg_events_and_summary_4s_per_channel.csv"
OUT_ROOT_DEFAULT  = "/home/darryld/documents/EEG/preprocessed/bipolaire/3_results_analysis/gp2_PSD_RBD/c-peyron_P95/"
EXCEL_DEFAULT     = "/home/darryld/documents/nv_patientsRBD_identifiants.xlsx"
SHEET_INDEX_DEFAULT = 2  

# ---------- paramètres PSD ----------
FMIN, FMAX = 30.0, 100.0    # bande unique
WIN_SEC    = 4.0
OVERLAP    = 0.5

COMMON_NFREQ  = 200
COMMON_FREQS  = np.linspace(FMIN, FMAX, COMMON_NFREQ)

# ---------- palette groupes (identique à ton script EEG) ----------
PALETTE = {
    "autoimmune encephalitides": "#C9D175",
    "narcolepsy":                "#F15854",
    "synucleopathy":             "#44AA99",
    "tcspi":                     "#BEBEBE",
    "unknown":                   "#000000",
}
def color_for_group(label: str) -> str:
    return PALETTE.get(str(label).strip().lower(), PALETTE["unknown"])
def short_label(label: str) -> str:
    lab = str(label).strip().lower()
    return {
        "autoimmune encephalitides": "AI",
        "narcolepsy":                "Narco",
        "synucleopathy":             "Syn",
        "tcspi":                     "TCSP",
        "unknown":                   "Unknown",
    }.get(lab, label)


# ---------- utils généraux ----------
def normalize_id(x: str) -> str:
    return "".join(ch for ch in str(x).strip().upper() if ch.isalnum())

def sanitize_name(x: str) -> str:
    return "".join(c for c in str(x) if c.isalnum() or c in ("_", "-")).strip("_-").lower()

def list_patients(gp2_root: Path):
    subs = [p for p in gp2_root.iterdir() if p.is_dir() and not p.name.startswith("._")]
    return sorted([normalize_id(s.name) for s in subs])

def find_fif_for_patient(base: str, gp2_root: Path) -> Path | None:
    pdir = gp2_root / base
    if not pdir.exists():
        return None
    for pat in ("*_art_annotated.fif", "*.fif"):
        files = sorted(pdir.glob(pat))
        if files:
            return files[0]
    return None

def interp_to_common_grid(freqs: np.ndarray, psd_lin: np.ndarray,
                          grid: np.ndarray = COMMON_FREQS) -> np.ndarray:
    """Interpole une PSD linéaire sur la grille commune."""
    m = np.isfinite(freqs) & np.isfinite(psd_lin)
    f = freqs[m]; y = psd_lin[m]
    if f.size < 2:
        return np.full_like(grid, np.nan, dtype=float)
    return np.interp(grid, f, y, left=y[0], right=y[-1])


# --------- Chargement des catégories (même fonction que dans le script EEG) ---------
def load_groups_from_excel(xlsx_path: Path, sheet_index: int | None):
    from cohort import load_patient_groups as _lpg
    df, group_map, _demo = _lpg(str(xlsx_path), sheet_index=sheet_index)
    # normalisation des clés/valeurs comme dans l'autre script
    return {str(k).strip().upper(): str(v).strip().lower() for k, v in group_map.items()}


# ---------- PSD helpers ----------
def _welch_psd_array(data, sfreq, fmin, fmax, win_sec=WIN_SEC, overlap=OVERLAP):
    """
    Welch PSD sur (n_chan, n_times) en µV.
    Retour : freqs, psd_lin (µV²/Hz).
    """
    from mne.time_frequency import psd_array_welch

    n_times = int(data.shape[1])
    n_per_seg = int(round(float(sfreq) * float(win_sec)))
    n_per_seg = max(8, min(n_per_seg, n_times))

    n_overlap = int(round(float(overlap) * n_per_seg))
    if n_overlap >= n_per_seg:
        n_overlap = max(0, n_per_seg - 1)

    psd, freqs = psd_array_welch(
        data, sfreq,
        fmin=float(fmin), fmax=float(fmax),
        n_fft=n_per_seg,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        average="mean",
        verbose="ERROR",
    )
    return np.asarray(freqs, float), np.asarray(psd, float)


def total_power_30_100(freqs: np.ndarray, psd_lin: np.ndarray) -> np.ndarray:
    """
    Intègre la PSD (lin) sur 30-100 Hz.
    psd_lin : (n_chan, n_freq)
    Retour : array (n_chan,) en µV².
    """
    idx = np.where((freqs >= FMIN) & (freqs <= FMAX))[0]
    if idx.size == 0:
        return np.zeros(psd_lin.shape[0], float)
    power = np.trapz(psd_lin[:, idx], freqs[idx], axis=1).astype(float)
    return power


# ---------- Figures individuelles ----------
def make_patient_full(base, freqs, psd_lin, out_png):
    """PSD EMG moyenne (tous canaux) sur 30-100 Hz, en dB."""
    eps = np.finfo(float).tiny
    psd_db = 10.0 * np.log10(np.maximum(psd_lin, eps))
    m   = np.nanmean(psd_db, axis=0)
    p10 = np.nanpercentile(psd_db, 10, axis=0)
    p90 = np.nanpercentile(psd_db, 90, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4))
    (line,) = ax.plot(freqs, m, lw=2)
    ax.fill_between(freqs, p10, p90, alpha=0.08, color=line.get_color())
    ax.set_xscale("log"); ax.set_xlim(FMIN, FMAX)
    ticks = list(range(int(FMIN), int(FMAX) + 1, 10))
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("PSD (dB re µV²/Hz)")
    ax.set_title(f"{base} - EMG RSWA (30-100 Hz) moy ± p10-p90")
    ax.grid(True, alpha=0.2)
    plt.tight_layout(); fig.savefig(out_png, dpi=230); plt.close(fig)


def make_patient_fig_channel(base, ch_label, freqs, psd_lin_channel, out_png):
    """PSD EMG d'un canal pendant RSWA (30-100 Hz)."""
    eps = np.finfo(float).tiny
    psd_db = 10.0 * np.log10(np.maximum(psd_lin_channel[np.newaxis, :], eps))[0]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(freqs, psd_db, lw=1.8)
    ax.set_xscale("log"); ax.set_xlim(FMIN, FMAX)
    ticks = list(range(int(FMIN), int(FMAX) + 1, 10))
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("PSD (dB re µV²/Hz)")
    ax.set_title(f"{base} - {ch_label} (RSWA 30-100 Hz)")
    ax.grid(True, alpha=0.2)
    plt.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)


# ---------- logique principale par patient ----------
def process_patient(base: str, gp2_root: Path, rswa_df: pd.DataFrame, out_root: Path):
    """
    Retourne un dict :
      {
        "base": base_n,
        "ok": True/False,
        "row_patient": {...},        # puissance moyenne RSWA
        "rows_chan": [...],          # puissance 30-100 par canal RSWA
        "rswa_spec": [...],          # PSD moyenne (lin) RSWA sur COMMON_FREQS
        "nonrswa_spec": [...],       # PSD moyenne (lin) REM_sans_RSWA sur COMMON_FREQS
      }
    """
    # --- NEW: stockage des PSD par canal (RSWA / nonRSWA) ---
    chan_specs_rswa = []   # liste de dicts {base, channel, spec}
    chan_specs_non  = []   # idem pour REM sans RSWA

    base_n = normalize_id(base)
    fif = find_fif_for_patient(base_n, gp2_root)
    if fif is None:
        print(f"[{base_n}] pas de .fif trouvé")
        return {"base": base_n, "ok": False, "reason": "no_fif"}

    df_all = rswa_df[(rswa_df["patient_id"] == base_n) &
                     (rswa_df["type"] == "REM_EPOCH_4S")].copy()
    if df_all.empty:
        print(f"[{base_n}] aucune époque REM_EPOCH_4S dans le CSV")
        return {"base": base_n, "ok": False, "reason": "no_epochs"}

    df_rswa    = df_all[df_all["rswa"].astype(bool)]
    df_nonrswa = df_all[~df_all["rswa"].astype(bool)]

    raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    first_t = float(raw.first_time)

    emg_picks = mne.pick_types(raw.info, meg=False, eeg=False, eog=False, emg=True, stim=False, misc=False)
    if emg_picks.size == 0:
        print(f"[{base_n}] aucun canal EMG")
        return {"base": base_n, "ok": False, "reason": "no_emg"}

    out_dir = out_root / base_n
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pour la PSD RSWA (figures + puissances)
    all_freqs_rswa = None
    psd_list_rswa  = []
    ch_names_rswa  = []
    chan_powers    = []  # pour CSV par canal (RSWA)
    chan_durs      = []  # durée totale RSWA par canal

    # Pour la PSD non-RSWA (spectre moyen par patient)
    all_freqs_non  = None
    psd_list_non   = []

    # --- boucle sur canaux EMG ---
    for idx in emg_picks:
        ch_name = raw.ch_names[idx]

        # RSWA
        dch_rswa = df_rswa[df_rswa["channel"] == ch_name]
        # nonRSWA
        dch_non  = df_nonrswa[df_nonrswa["channel"] == ch_name]

        # ---------- RSWA : concat des segments ----------
        segments_rswa = []
        total_dur_rswa = 0.0
        for _, row in dch_rswa.iterrows():
            t0 = float(row["epoch_start_sec"])
            t1 = float(row["epoch_end_sec"])
            s0 = int(round((t0 - first_t) * sfreq))
            s1 = int(round((t1 - first_t) * sfreq))
            s0 = max(0, min(s0, raw.n_times - 1))
            s1 = max(s0 + 1, min(s1, raw.n_times))
            seg = raw.get_data(picks=[idx], start=s0, stop=s1) * 1e6  # µV
            segments_rswa.append(seg)
            total_dur_rswa += (s1 - s0) / sfreq

        # ---------- non-RSWA : concat des segments ----------
        segments_non = []
        for _, row in dch_non.iterrows():
            t0 = float(row["epoch_start_sec"])
            t1 = float(row["epoch_end_sec"])
            s0 = int(round((t0 - first_t) * sfreq))
            s1 = int(round((t1 - first_t) * sfreq))
            s0 = max(0, min(s0, raw.n_times - 1))
            s1 = max(s0 + 1, min(s1, raw.n_times))
            seg = raw.get_data(picks=[idx], start=s0, stop=s1) * 1e6  # µV
            segments_non.append(seg)

        has_rswa = len(segments_rswa) > 0
        has_non  = len(segments_non)  > 0

        if not has_rswa and not has_non:
            continue

        # --- RSWA : PSD + figures + puissances ---
        if has_rswa:
            data_cat_rswa = np.concatenate(segments_rswa, axis=1)   # (1, n_times_concat)
            freqs_r, psd_ch_rswa = _welch_psd_array(data_cat_rswa, sfreq, FMIN, FMAX)

            if all_freqs_rswa is None:
                all_freqs_rswa = freqs_r
            else:
                if not np.allclose(all_freqs_rswa, freqs_r):
                    psd_ch_rswa = np.interp(
                        all_freqs_rswa, freqs_r, psd_ch_rswa[0, :]
                    )[np.newaxis, :]
            if all_freqs_rswa is None:
                all_freqs_rswa = freqs_r

            # --- NEW: PSD de ce canal interpolée sur COMMON_FREQS (RSWA) ---
            spec_rswa_chan_common = interp_to_common_grid(
                all_freqs_rswa, psd_ch_rswa[0, :], COMMON_FREQS
            )
            chan_specs_rswa.append({
                "base": base_n,
                "channel": ch_name,
                "spec": spec_rswa_chan_common.tolist(),
            })

            psd_list_rswa.append(psd_ch_rswa[0, :])
            ch_names_rswa.append(ch_name)
            chan_durs.append(total_dur_rswa)

            power = total_power_30_100(all_freqs_rswa, psd_ch_rswa)[0]
            chan_powers.append(power)

            # figure canal RSWA
            out_png_ch = out_dir / f"{base_n}_emg_psd_rswa_{sanitize_name(ch_name)}.png"
            make_patient_fig_channel(base_n, ch_name, all_freqs_rswa, psd_ch_rswa[0, :], out_png_ch)


        # --- non-RSWA : PSD pour le spectre moyen ---
        if has_non:
            data_cat_non = np.concatenate(segments_non, axis=1)   # (1, n_times_concat)
            freqs_n, psd_ch_non = _welch_psd_array(data_cat_non, sfreq, FMIN, FMAX)

            if all_freqs_non is None:
                all_freqs_non = freqs_n
            else:
                if not np.allclose(all_freqs_non, freqs_n):
                    psd_ch_non = np.interp(
                        all_freqs_non, freqs_n, psd_ch_non[0, :]
                    )[np.newaxis, :]
            if all_freqs_non is None:
                all_freqs_non = freqs_n

            # --- NEW: PSD de ce canal interpolée sur COMMON_FREQS (nonRSWA) ---
            spec_non_chan_common = interp_to_common_grid(
                all_freqs_non, psd_ch_non[0, :], COMMON_FREQS
            )
            chan_specs_non.append({
                "base": base_n,
                "channel": ch_name,
                "spec": spec_non_chan_common.tolist(),
            })

            psd_list_non.append(psd_ch_non[0, :])


    # ---------- post-traitement RSWA ----------
    rswa_spec_common = None
    nonrswa_spec_common = None
    row_patient = None
    rows_chan = []

    if psd_list_rswa and all_freqs_rswa is not None:
        psd_lin_rswa = np.vstack(psd_list_rswa)  # (n_chan_RSWA, n_freq)
        # figure globale patient (RSWA)
        out_png_full = out_dir / f"{base_n}_emg_psd_rswa_full_30_100.png"
        make_patient_full(base_n, all_freqs_rswa, psd_lin_rswa, out_png_full)

        # puissance moyenne patient (moyenne des canaux EMG)
        powers_arr = np.asarray(chan_powers, float)
        mean_power = float(np.nanmean(powers_arr))

        row_patient = {
            "base": base_n,
            "mean_abs_power_30_100": mean_power,
        }

        for ch, pwr, dur in zip(ch_names_rswa, chan_powers, chan_durs):
            rows_chan.append({
                "base": base_n,
                "channel": ch,
                "abs_power_30_100": float(pwr),
                "rswa_duration_sec": float(dur),
            })

        # spectre moyen RSWA (moyenne canaux) sur COMMON_FREQS
        spec_rswa_mean = np.nanmean(psd_lin_rswa, axis=0)
        rswa_spec_common = interp_to_common_grid(all_freqs_rswa, spec_rswa_mean, COMMON_FREQS)

    # ---------- post-traitement nonRSWA ----------
    if psd_list_non and all_freqs_non is not None:
        psd_lin_non = np.vstack(psd_list_non)
        spec_non_mean = np.nanmean(psd_lin_non, axis=0)
        nonrswa_spec_common = interp_to_common_grid(all_freqs_non, spec_non_mean, COMMON_FREQS)

    ok = (rswa_spec_common is not None) or (nonrswa_spec_common is not None)
    if not ok and row_patient is None:
        print(f"[{base_n}] aucun canal EMG avec RSWA/nonRSWA exploitable")
        return {"base": base_n, "ok": False, "reason": "no_valid_psd"}

    return {
        "base": base_n,
        "ok": ok,
        "row_patient": row_patient,
        "rows_chan": rows_chan,
        "rswa_spec": rswa_spec_common.tolist() if rswa_spec_common is not None else None,
        "nonrswa_spec": nonrswa_spec_common.tolist() if nonrswa_spec_common is not None else None,
        "rswa_spec_by_channel": chan_specs_rswa,
        "nonrswa_spec_by_channel": chan_specs_non,
    }


# ---------- Figures groupales ----------
def plot_group_condition_spectra(df_info: pd.DataFrame,
                                 spec_map: dict,
                                 condition_label: str,
                                 out_path: Path):
    """
    df_info : DataFrame index=base, colonne 'group'
    spec_map : base -> np.ndarray (lin) sur COMMON_FREQS
    condition_label : 'RSWA' ou 'REM_sans_RSWA'
    """

    print(
        f"[DEBUG] plot_group_condition_spectra: condition={condition_label}, "
        f"n_patients={len(spec_map)}, n_df_info={len(df_info)}"
    )

    bases_ok = [b for b in df_info.index if b in spec_map]
    if not bases_ok:
        print(f"[WARN] Aucun spectre disponible pour {condition_label}")
        return

    # Empilement des spectres (lin -> dB)
    SPEC = np.vstack([spec_map[b] for b in bases_ok])  # (n_pat, n_freq)
    eps = np.finfo(float).tiny
    SPEC_DB = 10.0 * np.log10(np.maximum(SPEC, eps))

    # Masque pour enlever la zone du notch 50 Hz
    mask = (COMMON_FREQS < 48) | (COMMON_FREQS > 52)
    freqs_plot = COMMON_FREQS[mask]
    SPEC_DB = SPEC_DB[:, mask]   # on ne garde que les fréquences non notchées

    DF_ALIGNED = df_info.loc[bases_ok].copy()

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    for grp, idx_labels in DF_ALIGNED.groupby("group").groups.items():
        if len(idx_labels) == 0:
            continue
        idx_int = np.array([DF_ALIGNED.index.get_loc(b) for b in idx_labels], dtype=int)
        sub = SPEC_DB[idx_int, :]   # dB, déjà masqué 48–52 Hz
        if sub.size == 0:
            continue
        m   = np.nanmean(sub, axis=0)
        p10 = np.nanpercentile(sub, 10, axis=0)
        p90 = np.nanpercentile(sub, 90, axis=0)
        c = color_for_group(grp)
        ax.plot(freqs_plot, m, lw=2, label=short_label(grp), color=c)
        ax.fill_between(freqs_plot, p10, p90, alpha=0.08, color=c)

    ax.set_xscale("log")
    ax.set_xlim(FMIN, FMAX)
    ticks = list(range(int(FMIN), int(FMAX) + 1, 10))
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("PSD (dB re µV²/Hz)")
    ax.set_title(f"EMG {condition_label} (30-100 Hz) - moy ± p10-p90 par catégorie")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=4, frameon=False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=230)
    plt.close(fig)
    print(">>>", out_path)

def plot_group_condition_spectra_per_channel(
    channel: str,
    df_info: pd.DataFrame,
    spec_map: dict,
    condition_label: str,
    out_path: Path,
):
    """
    Figure pour **un canal donné** :
      - x : fréquence 30–100 Hz (log)
      - y : PSD (dB)
      - une courbe par catégorie de patients (+ p10–p90)

    df_info  : DataFrame index=base, colonne 'group'
    spec_map : dict {base -> np.ndarray (PSD linéaire sur COMMON_FREQS)}
    """
    print(
        f"[DEBUG] plot_group_condition_spectra_per_channel: canal={channel}, "
        f"condition={condition_label}, n_patients={len(spec_map)}"
    )

    # Patients pour lesquels on a un spectre pour ce canal
    bases_ok = [b for b in df_info.index if b in spec_map]
    if not bases_ok:
        print(f"[WARN] Aucun spectre disponible pour {condition_label} canal {channel}")
        return

    SPEC = np.vstack([spec_map[b] for b in bases_ok])  # (n_pat, n_freq)
    eps = np.finfo(float).tiny
    SPEC_DB = 10.0 * np.log10(np.maximum(SPEC, eps))

    # Masque pour supprimer la zone du notch 50 Hz
    mask = (COMMON_FREQS < 48) | (COMMON_FREQS > 52)
    freqs_plot = COMMON_FREQS[mask]
    SPEC_DB = SPEC_DB[:, mask]

    DF_ALIGNED = df_info.loc[bases_ok].copy()

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for grp, idx_labels in DF_ALIGNED.groupby("group").groups.items():
        if len(idx_labels) == 0:
            continue
        idx_int = np.array([DF_ALIGNED.index.get_loc(b) for b in idx_labels], dtype=int)
        sub = SPEC_DB[idx_int, :]
        if sub.size == 0:
            continue

        m   = np.nanmean(sub, axis=0)
        p10 = np.nanpercentile(sub, 10, axis=0)
        p90 = np.nanpercentile(sub, 90, axis=0)
        c   = color_for_group(grp)

        ax.plot(freqs_plot, m, lw=2, label=short_label(grp), color=c)
        ax.fill_between(freqs_plot, p10, p90, alpha=0.08, color=c)

    ax.set_xscale("log")
    ax.set_xlim(FMIN, FMAX)
    ticks = list(range(int(FMIN), int(FMAX) + 1, 10))
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("PSD (dB re µV²/Hz)")
    ax.set_title(f"EMG {condition_label} - canal {channel} (30–100 Hz) moy ± p10-p90 par catégorie")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, frameon=False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=230)
    plt.close(fig)
    print(">>>", out_path)


def plot_global_rswa_vs_nonrswa(spec_map_rswa: dict, spec_map_non: dict, out_path: Path):
    """
    Spectres globaux (tous patients confondus) RSWA vs REM_sans_RSWA.
    On masque la zone 48–52 Hz pour éviter le trou du notch.
    """
    def stack_from_map(spec_map):
        if not spec_map:
            return None
        arrs = [np.asarray(spec_map[b], float) for b in spec_map.keys()]
        return np.vstack(arrs)

    SPEC_RSWA = stack_from_map(spec_map_rswa)
    SPEC_NON  = stack_from_map(spec_map_non)

    if SPEC_RSWA is None and SPEC_NON is None:
        print("[WARN] Aucun spectre pour RSWA ou nonRSWA (global)")
        return

    eps = np.finfo(float).tiny
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    # masque pour enlever le notch 50 Hz
    mask = (COMMON_FREQS < 48) | (COMMON_FREQS > 52)
    freqs_plot = COMMON_FREQS[mask]

    if SPEC_RSWA is not None:
        SPEC_RSWA_DB = 10.0 * np.log10(np.maximum(SPEC_RSWA, eps))
        SPEC_RSWA_DB = SPEC_RSWA_DB[:, mask]          # on applique le masque
        m   = np.nanmean(SPEC_RSWA_DB, axis=0)
        p10 = np.nanpercentile(SPEC_RSWA_DB, 10, axis=0)
        p90 = np.nanpercentile(SPEC_RSWA_DB, 90, axis=0)
        ax.plot(freqs_plot, m, lw=2, label="RSWA", color="#D62728")
        ax.fill_between(freqs_plot, p10, p90, alpha=0.08, color="#D62728")

    if SPEC_NON is not None:
        SPEC_NON_DB = 10.0 * np.log10(np.maximum(SPEC_NON, eps))
        SPEC_NON_DB = SPEC_NON_DB[:, mask]            # on applique le masque
        m   = np.nanmean(SPEC_NON_DB, axis=0)
        p10 = np.nanpercentile(SPEC_NON_DB, 10, axis=0)
        p90 = np.nanpercentile(SPEC_NON_DB, 90, axis=0)
        ax.plot(freqs_plot, m, lw=2, label="REM sans RSWA", color="#1F77B4")
        ax.fill_between(freqs_plot, p10, p90, alpha=0.08, color="#1F77B4")

    ax.set_xscale("log")
    ax.set_xlim(FMIN, FMAX)
    ticks = list(range(int(FMIN), int(FMAX) + 1, 10))
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("PSD (dB re µV²/Hz)")
    ax.set_title("EMG RSWA vs REM sans RSWA (30-100 Hz) - moy ± p10-p90 (tous patients)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=230)
    plt.close(fig)
    print(">>>", out_path)


# ---------- CLI / main ----------
def parse_args():
    p = argparse.ArgumentParser(description="PSD EMG (30-100 Hz) pendant RSWA et REM_sans_RSWA.")
    p.add_argument("--gp2-root", type=str, default=GP2_ROOT_DEFAULT,
                   help="Dossier gp2 contenant les dossiers patients.")
    p.add_argument("--rswa-csv", type=str, default=RSWA_CSV_DEFAULT,
                   help="CSV des événements RSWA (sortie de ton script EMG).")
    p.add_argument("--out-root", type=str, default=OUT_ROOT_DEFAULT,
                   help="Dossier de sortie pour les figures et CSV.")
    p.add_argument("--patients", type=str, nargs="*", default=None,
                   help="Liste explicite de patients (sinon détection auto dans gp2-root).")
    p.add_argument("--workers", type=int, default=20,
                   help="Nombre de workers parallèles (0 => CPU-1).")
    p.add_argument("--excel", type=str, default=EXCEL_DEFAULT,
                   help="Fichier Excel des catégories de patients.")
    p.add_argument("--sheet-index", type=int, default=SHEET_INDEX_DEFAULT,
                   help="Index de feuille pour load_patient_groups.")
    return p.parse_args()


def main():
    args = parse_args()
    gp2_root = Path(args.gp2_root)
    out_root = Path(args.out_root)
    rswa_csv = Path(args.rswa_csv)
    xlsx     = Path(args.excel)

    if not gp2_root.exists():
        raise SystemExit(f"[CONFIG] gp2-root introuvable : {gp2_root}")
    if not rswa_csv.exists():
        raise SystemExit(f"[CONFIG] RSWA CSV introuvable : {rswa_csv}")
    out_root.mkdir(parents=True, exist_ok=True)

    rswa_df = pd.read_csv(rswa_csv)

    if args.patients:
        bases = [normalize_id(b) for b in args.patients]
    else:
        bases = list_patients(gp2_root)

    if not bases:
        raise SystemExit("Aucun patient détecté.")

    print(f"Patients ({len(bases)}) : {bases}")

    # Groupes (labels) depuis le même Excel que le script EEG
    group_map = {}
    if xlsx.exists():
        try:
            group_map = load_groups_from_excel(xlsx, args.sheet_index)
        except Exception as e:
            print(f"[WARN] Excel lu mais erreur: {e}")
    else:
        print(f"[WARN] Excel non trouvé: {xlsx}")

    # -------- parallélisation par patient --------
    cpu = os.cpu_count() or 1
    n_workers = (cpu - 1) if args.workers in (0, None) else args.workers
    n_workers = max(1, min(n_workers, len(bases)))
    print(f"[INFO] Parallèle avec {n_workers} worker(s) (spawn, maxtasksperchild=1)")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, maxtasksperchild=1) as pool:
        results = pool.starmap(
            process_patient,
            [(b, gp2_root, rswa_df, out_root) for b in bases]
        )

    # -------- compilation des résultats --------
    rows_pat = []
    rows_chan_all = []
    spec_map_rswa = {}
    spec_map_non  = {}
    chan_rswa_records = []   # NEW
    chan_non_records  = []   # NEW

    for r in results:
        if r is None or not isinstance(r, dict):
            continue

        # --- NEW: PSD par canal ---
        chan_rswa_records.extend(r.get("rswa_spec_by_channel", []))
        chan_non_records.extend(r.get("nonrswa_spec_by_channel", []))

        if not r.get("ok"):
            print(f"[{r.get('base')}] SKIP: {r.get('reason')}")
            continue

        base = r["base"]

        # patient-level power row
        if r.get("row_patient") is not None:
            rows_pat.append(r["row_patient"])

        # channel-level rows
        rows_chan_all.extend(r.get("rows_chan", []))

        # spectra
        if r.get("rswa_spec") is not None:
            spec_map_rswa[base] = np.asarray(r["rswa_spec"], float)
        if r.get("nonrswa_spec") is not None:
            spec_map_non[base] = np.asarray(r["nonrswa_spec"], float)

    # ---------- CSV : puissances RSWA ----------
    if rows_pat:
        df_pat = pd.DataFrame(rows_pat)
        df_pat["group"] = df_pat["base"].map(lambda b: group_map.get(b, "unknown"))
        df_pat = df_pat.set_index("base").sort_index()
        df_pat.to_csv(out_root / "emg_rswa_patient_power_30_100.csv", float_format="%.8e")
        print(">>>", out_root / "emg_rswa_patient_power_30_100.csv")

    else:
        print("Aucun résultat de puissance RSWA à sauvegarder (emg_rswa_patient_power_30_100.csv vide).")

    if rows_chan_all:
        df_ch = pd.DataFrame(rows_chan_all)

        # Ajout de la catégorie du patient
        df_ch["group"] = df_ch["base"].map(lambda b: group_map.get(b, "unknown"))

        csv_path = out_root / "emg_rswa_channel_power_30_100.csv"
        xlsx_path = out_root / "emg_rswa_channel_power_30_100.xlsx"

        df_ch.to_csv(csv_path, index=False, float_format="%.8e")
        df_ch.to_excel(xlsx_path, index=False)  # pour ton notebook/Excel

        print(">>>", csv_path)
        print(">>>", xlsx_path)


    print(f"[DEBUG] len(rows_pat)      = {len(rows_pat)}")
    print(f"[DEBUG] len(spec_map_rswa) = {len(spec_map_rswa)}")
    print(f"[DEBUG] len(spec_map_non)  = {len(spec_map_non)}")

    # ---------- Figures groupales (spectres) ----------
    print("[DEBUG] préparation des figures groupales...")

    # DataFrame info : base + group
    all_bases = sorted(set(spec_map_rswa) | set(spec_map_non))
    if not all_bases:
        print("[WARN] Aucun spectre RSWA/nonRSWA pour les figures groupales.")
        return

    df_info = pd.DataFrame({"base": all_bases})
    df_info["group"] = df_info["base"].map(lambda b: group_map.get(b, "unknown"))
    df_info = df_info.set_index("base")

    print(f"[DEBUG] df_info (bases pour figures groupales) = {len(df_info)}")
    print(f"[DEBUG] groups présents = {df_info['group'].value_counts().to_dict()}")

    # --- NEW: préparation des spectres par canal (channel -> base -> spec) ---
    from collections import defaultdict

    chan_map_rswa = defaultdict(dict)   # {channel: {base: np.array}}
    chan_map_non  = defaultdict(dict)

    for rec in chan_rswa_records:
        base = rec["base"]
        ch   = rec["channel"]
        spec = np.asarray(rec["spec"], float)
        chan_map_rswa[ch][base] = spec

    for rec in chan_non_records:
        base = rec["base"]
        ch   = rec["channel"]
        spec = np.asarray(rec["spec"], float)
        chan_map_non[ch][base] = spec

    # RSWA par groupe
    if spec_map_rswa:
        out_png_rswa = out_root / "group_emg_psd_rswa_30_100.png"
        print("Plot 'group_emg_psd_rswa_30_100' en cours...")
        plot_group_condition_spectra(df_info, spec_map_rswa, "RSWA", out_png_rswa)

    # nonRSWA par groupe
    if spec_map_non:
        out_png_non = out_root / "group_emg_psd_nonrswa_30_100.png"
        print("Plot 'group_emg_psd_nonrswa_30_100' en cours...")
        plot_group_condition_spectra(df_info, spec_map_non, "REM sans RSWA", out_png_non)

    # Figure globale RSWA vs REM_sans_RSWA
    if spec_map_rswa or spec_map_non:
        out_png_global = out_root / "global_emg_psd_rswa_vs_nonrswa_30_100.png"
        print("Plot 'global_emg_psd_rswa_vs_nonrswa_30_100' en cours...")
        plot_global_rswa_vs_nonrswa(spec_map_rswa, spec_map_non, out_png_global)

    # --- NEW: figures par canal et par catégorie ---
    per_channel_dir = out_root / "per_channel_group_psd"
    per_channel_dir.mkdir(parents=True, exist_ok=True)

    # RSWA par canal
    for ch, spec_map_ch in chan_map_rswa.items():
        out_png = per_channel_dir / f"emg_psd_rswa_{sanitize_name(ch)}_by_group_30_100.png"
        print(f"[INFO] Plot RSWA par canal pour {ch} ...")
        plot_group_condition_spectra_per_channel(
            channel=ch,
            df_info=df_info,
            spec_map=spec_map_ch,
            condition_label="RSWA",
            out_path=out_png,
        )

    # REM sans RSWA par canal
    for ch, spec_map_ch in chan_map_non.items():
        out_png = per_channel_dir / f"emg_psd_nonrswa_{sanitize_name(ch)}_by_group_30_100.png"
        print(f"[INFO] Plot nonRSWA par canal pour {ch} ...")
        plot_group_condition_spectra_per_channel(
            channel=ch,
            df_info=df_info,
            spec_map=spec_map_ch,
            condition_label="REM sans RSWA",
            out_path=out_png,
        )

if __name__ == "__main__":
    main()
