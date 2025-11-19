#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_psd_emg.py

Objectif :
  - Calculer la puissance EMG absolue (µV²) sur 30–100 Hz
    pendant les périodes RSWA uniquement (détectées via ton CSV).

Entrées :
  - Dossiers gp2 : /.../gp2/{PATIENT}/*.fif (prétraités)
  - CSV RSWA : rbd_emg_events_and_summary_4s_per_channel.csv

Sorties :
  - out_root/{PAT}/{PAT}_emg_psd_rswa_full_30_100.png   (PSD moyenne tous canaux EMG)
  - out_root/{PAT}/{PAT}_emg_psd_rswa_{CANAL}.png       (PSD canal EMG)
  - out_root/emg_rswa_patient_power_30_100.csv          (1 ligne par patient)
  - out_root/emg_rswa_channel_power_30_100.csv          (long : patient × canal)
"""

import os, sys, argparse
from pathlib import Path
import multiprocessing as mp  # <- pour la parallélisation

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator

import mne
mne.set_config('MNE_MEMMAP_MIN_SIZE', '1M', set_env=True)
mne.set_log_level("WARNING")

# === chemins par défaut ===
GP2_ROOT_DEFAULT = "/home/darryld/documents/EEG/preprocessed/bipolaire/1_noArtefacts/gp2/"
RSWA_CSV_DEFAULT = "/home/darryld/Cerco_studies/data/rbd_emg_events_and_summary_4s_per_channel.csv"
OUT_ROOT_DEFAULT  = "/home/darryld/documents/EEG/preprocessed/bipolaire/3_results_analysis/gp2_PSD_RBD/c-peyron_P95/"

# === paramètres PSD ===
FMIN, FMAX = 30.0, 100.0     # bande unique
WIN_SEC    = 4.0
OVERLAP    = 0.5


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
    Intègre la PSD (lin) sur 30–100 Hz.
    psd_lin : (n_chan, n_freq)
    Retour : array (n_chan,) en µV².
    """
    idx = np.where((freqs >= FMIN) & (freqs <= FMAX))[0]
    if idx.size == 0:
        return np.zeros(psd_lin.shape[0], float)
    power = np.trapz(psd_lin[:, idx], freqs[idx], axis=1).astype(float)
    return power


# ---------- Figures ----------
def make_patient_full(base, freqs, psd_lin, out_png):
    """PSD EMG moyenne (tous canaux) sur 30–100 Hz, en dB."""
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
    ax.set_title(f"{base} - EMG RSWA (30–100 Hz) moy ± p10–p90")
    ax.grid(True, alpha=0.2)
    plt.tight_layout(); fig.savefig(out_png, dpi=230); plt.close(fig)


def make_patient_fig_channel(base, ch_label, freqs, psd_lin_channel, out_png):
    """PSD EMG d'un canal pendant RSWA (30–100 Hz)."""
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
    ax.set_title(f"{base} - {ch_label} (RSWA 30–100 Hz)")
    ax.grid(True, alpha=0.2)
    plt.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)


# ---------- RSWA helpers ----------
def extract_rswa_epochs_for_patient(rswa_df: pd.DataFrame, patient_id: str) -> pd.DataFrame:
    dfp = rswa_df[(rswa_df["patient_id"] == patient_id) &
                  (rswa_df["type"] == "REM_EPOCH_4S") &
                  (rswa_df["rswa"].astype(bool))]
    return dfp.copy()


# ---------- logique principale par patient ----------
def process_patient(base: str, gp2_root: Path, rswa_df: pd.DataFrame, out_root: Path):
    base_n = normalize_id(base)
    fif = find_fif_for_patient(base_n, gp2_root)
    if fif is None:
        print(f"[{base_n}] pas de .fif trouvé")
        return None

    dfp = extract_rswa_epochs_for_patient(rswa_df, base_n)
    if dfp.empty:
        print(f"[{base_n}] aucune époque RSWA dans le CSV")
        return None

    raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    first_t = float(raw.first_time)

    emg_picks = mne.pick_types(raw.info, meg=False, eeg=False, eog=False, emg=True, stim=False, misc=False)
    if emg_picks.size == 0:
        print(f"[{base_n}] aucun canal EMG")
        return None

    out_dir = out_root / base_n
    out_dir.mkdir(parents=True, exist_ok=True)

    all_freqs = None
    psd_list  = []
    ch_names  = []
    chan_powers = []  # pour CSV par canal
    chan_durs   = []  # durée totale RSWA par canal

    # --- boucle sur canaux EMG ---
    for idx in emg_picks:
        ch_name = raw.ch_names[idx]
        dch = dfp[dfp["channel"] == ch_name]
        if dch.empty:
            continue

        segments = []
        total_dur = 0.0
        for _, row in dch.iterrows():
            t0 = float(row["epoch_start_sec"])
            t1 = float(row["epoch_end_sec"])
            s0 = int(round((t0 - first_t) * sfreq))
            s1 = int(round((t1 - first_t) * sfreq))
            s0 = max(0, min(s0, raw.n_times - 1))
            s1 = max(s0 + 1, min(s1, raw.n_times))
            seg = raw.get_data(picks=[idx], start=s0, stop=s1) * 1e6  # µV
            segments.append(seg)
            total_dur += (s1 - s0) / sfreq

        if not segments:
            continue

        data_cat = np.concatenate(segments, axis=1)   # (1, n_times_concat)
        freqs, psd_ch = _welch_psd_array(data_cat, sfreq, FMIN, FMAX)

        if all_freqs is None:
            all_freqs = freqs
        else:
            # en pratique identiques ; si petite diff -> on interpole
            if not np.allclose(all_freqs, freqs):
                psd_ch = np.interp(all_freqs, freqs, psd_ch[0, :])[np.newaxis, :]

        psd_list.append(psd_ch[0, :])
        ch_names.append(ch_name)
        chan_durs.append(total_dur)

        # puissance absolue 30–100 Hz pour ce canal
        power = total_power_30_100(all_freqs, psd_ch)[0]
        chan_powers.append(power)

        # figure canal
        out_png_ch = out_dir / f"{base_n}_emg_psd_rswa_{sanitize_name(ch_name)}.png"
        make_patient_fig_channel(base_n, ch_name, all_freqs, psd_ch[0, :], out_png_ch)

    if not psd_list:
        print(f"[{base_n}] aucun canal EMG avec RSWA exploitable")
        return None

    psd_lin = np.vstack(psd_list)  # (n_chan, n_freq)

    # figure globale patient
    out_png_full = out_dir / f"{base_n}_emg_psd_rswa_full_30_100.png"
    make_patient_full(base_n, all_freqs, psd_lin, out_png_full)

    # puissance moyenne patient (moyenne des canaux EMG)
    powers_arr = np.asarray(chan_powers, float)
    mean_power = float(np.nanmean(powers_arr))

    # ligne patient
    row_patient = {
        "base": base_n,
        "mean_abs_power_30_100": mean_power,
    }

    # lignes par canal
    rows_chan = []
    for ch, pwr, dur in zip(ch_names, chan_powers, chan_durs):
        rows_chan.append({
            "base": base_n,
            "channel": ch,
            "abs_power_30_100": float(pwr),
            "rswa_duration_sec": float(dur),
        })

    return row_patient, rows_chan


# ---------- CLI / main ----------
def parse_args():
    p = argparse.ArgumentParser(description="PSD EMG (30–100 Hz) pendant RSWA uniquement.")
    p.add_argument("--gp2-root", type=str, default=GP2_ROOT_DEFAULT,
                   help="Dossier gp2 contenant les dossiers patients.")
    p.add_argument("--rswa-csv", type=str, default=RSWA_CSV_DEFAULT,
                   help="CSV des événements RSWA (sortie de ton script EMG).")
    p.add_argument("--out-root", type=str, default=OUT_ROOT_DEFAULT,
                   help="Dossier de sortie pour les figures et CSV.")
    p.add_argument("--patients", type=str, nargs="*", default=None,
                   help="Liste explicite de patients (sinon détection auto dans gp2-root).")
    p.add_argument("--workers", type=int, default=8,
                   help="Nombre de workers parallèles (0 => CPU-1).")
    return p.parse_args()


def main():
    args = parse_args()
    gp2_root = Path(args.gp2_root)
    out_root = Path(args.out_root)
    rswa_csv = Path(args.rswa_csv)

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

    rows_pat = []
    rows_chan_all = []
    for res in results:
        if res is None:
            continue
        row_patient, rows_chan = res
        rows_pat.append(row_patient)
        rows_chan_all.extend(rows_chan)

    if not rows_pat:
        print("Aucun résultat à sauvegarder.")
        return

    df_pat = pd.DataFrame(rows_pat).set_index("base").sort_index()
    df_pat.to_csv(out_root / "emg_rswa_patient_power_30_100.csv", float_format="%.8e")
    print(">>>", out_root / "emg_rswa_patient_power_30_100.csv")

    if rows_chan_all:
        df_ch = pd.DataFrame(rows_chan_all)
        df_ch.to_csv(out_root / "emg_rswa_channel_power_30_100.csv",
                     index=False, float_format="%.8e")
        print(">>>", out_root / "emg_rswa_channel_power_30_100.csv")


if __name__ == "__main__":
    main()
