#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSD (REM ou RSWA-concat) - global + par canal + figures et agrégats + barplots inter-catégories.

Mode REM :
  - lit {BASE}_REM_concat.fif

Mode RSWA :
  - lit {BASE}_RSWA_concat.fif (fichier déjà construit en amont en concaténant les fenêtres RSWA)

Sorties (identiques quel que soit le mode) :
  - out_root/{BASE}/{BASE}_psd_bands.png
  - out_root/{BASE}/{BASE}_psd_full_0p5_80.png
  - out_root/{BASE}/{BASE}_psd_bands_{CANAL}.png
  - out_root/all_band_powers.csv
  - out_root/per_channel_band_powers.csv
  - out_root/<groupe>_band_power_abs.png
  - out_root/group_band_power_rel.png
  - out_root/perband_abs_chan_{CANAL}_{BAND}.png
  - out_root/perband_rel_chan_{CANAL}_{BAND}.png
  - out_root/group_psd_bands.png
  - out_root/group_psd_full_0p5_80.png
  - out_root/group_means_abs.csv / group_means_rel.csv
  - out_root/patients_unknown_category.txt
"""

from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
import multiprocessing as mp
import faulthandler; faulthandler.enable()

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator, LogLocator

import mne
mne.set_config("MNE_MEMMAP_MIN_SIZE", "1M", set_env=True)

# ---------- Paramètres par défaut ----------
DATA_DIR_DEFAULT  = "/home/darryld/documents/EEG/preprocessed/bipolaire/1bis_RBD/method_95percentile"
OUT_ROOT_DEFAULT  = "/home/darryld/documents/EEG/results"

BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta":  (13.0, 30.0),
    "Gamma_Bas": (30.0, 50.0),
    "Gamma_Haut": (50.0, 80.0),
}
FMIN, FMAX = 0.5, 100.0
COMMON_NFREQ = 400
COMMON_FREQS = np.linspace(FMIN, FMAX, COMMON_NFREQ)

MIN_PATIENTS_PER_CHANNEL = 5

# ---------- Fonctions utilitaires ----------
def normalize_id(x: str) -> str:
    return "".join(ch for ch in str(x).strip().upper() if ch.isalnum())

def sanitize_name(x: str) -> str:
    return "".join(c for c in str(x) if c.isalnum() or c in ("_", "-")).strip("_-").lower()

def mode_suffix(mode: str) -> str:
    if mode == "rswa":
        return "_RSWA_concat.fif"
    if mode == "rswa_epo":
        return "_RSWA-epo.fif"
    return "_REM_concat.fif"


def list_patients(data_dir: Path, mode: str) -> list[str]:
    bases = set()
    suf = mode_suffix(mode)

    # sous-dossiers
    for child in data_dir.iterdir():
        if child.is_dir() and not child.name.startswith("._"):
            bases.add(normalize_id(child.name))

    # fichiers concat
    for p in data_dir.rglob(f"*{suf}"):
        if p.is_file() and not p.name.startswith("._"):
            bases.add(normalize_id(p.stem.split("_")[0]))

    return sorted(bases)


def find_concat_fif(base: str, data_dir: Path, mode: str) -> Path | None:
    suf = mode_suffix(mode)
    candidates = [
        data_dir / base / f"{base}{suf}",
        data_dir / f"{base}{suf}",
    ]
    return next((p for p in candidates if p.exists()), None)

# --------- Chargement des catégories patients ---------
def load_groups_from_excel(xlsx_path: Path, sheet_index: int | None):
    from cohort import load_patient_groups as _lpg
    _df, group_map, _demo = _lpg(str(xlsx_path), sheet_index=sheet_index)
    return {str(k).strip().upper(): str(v).strip().lower() for k, v in group_map.items()}

# ---------- Fonctions de calcul PSD ----------
def _welch_psd_array(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    win_sec: float = 4.0,
    overlap: float = 0.50
):
    from mne.time_frequency import psd_array_welch

    n_times = int(data.shape[1])
    n_per_seg = int(round(float(sfreq) * float(win_sec)))
    n_per_seg = max(8, min(n_per_seg, n_times))

    n_overlap = int(round(float(overlap) * n_per_seg))
    if n_overlap >= n_per_seg:
        n_overlap = max(0, n_per_seg - 1)

    print(f"[PSD] sf={sfreq:.3f}Hz | n_times={n_times} | n_per_seg={n_per_seg} | n_overlap={n_overlap}")

    psd, freqs = psd_array_welch(
        data, sfreq,
        fmin=float(fmin), fmax=float(fmax),
        n_fft=n_per_seg,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        average="mean",
        verbose="ERROR",
    )
    return np.asarray(freqs, dtype=float), np.asarray(psd, dtype=float)

def compute_psd(
    raw: mne.io.BaseRaw,
    fmin=FMIN,
    fmax=FMAX,
    include_emg: bool = False,
    emg_band=(30.0, 100.0),
):
    all_freqs = None
    all_psd, all_names = [], []

    # EEG
    eeg_picks = mne.pick_types(
        raw.info, meg=False, eeg=True, eog=False, ecg=False, emg=False,
        stim=False, misc=False, resp=False, seeg=False, ecog=False, fnirs=False, exclude=()
    )
    if eeg_picks.size > 0:
        data = raw.get_data(picks=eeg_picks) * 1e6  # µV
        sf = float(raw.info["sfreq"])
        freqs_eeg, psd_eeg = _welch_psd_array(data, sf, fmin, fmax)
        all_freqs = freqs_eeg
        all_psd.append(psd_eeg)
        all_names.extend([raw.ch_names[i] for i in eeg_picks])

    # EMG optionnel
    if include_emg:
        emg_picks = mne.pick_types(
            raw.info, meg=False, eeg=False, eog=False, ecg=False, emg=True,
            stim=False, misc=False, resp=False, seeg=False, ecog=False, fnirs=False, exclude=()
        )
        if emg_picks.size > 0:
            emg_inst = raw.copy().pick(emg_picks)
            lo, hi = float(emg_band[0]), float(emg_band[1])
            emg_inst.filter(l_freq=lo, h_freq=hi, picks="all", method="fir", verbose="ERROR")
            data = emg_inst.get_data(picks="all") * 1e6
            sf = float(emg_inst.info["sfreq"])
            freqs_emg, psd_emg = _welch_psd_array(data, sf, fmin, fmax)

            if all_freqs is None:
                all_freqs = freqs_emg
            elif not np.allclose(all_freqs, freqs_emg, rtol=0, atol=1e-12):
                psd_emg = np.vstack([
                    np.interp(all_freqs, freqs_emg, row, left=row[0], right=row[-1])
                    for row in psd_emg
                ])

            all_psd.append(psd_emg)
            all_names.extend(emg_inst.ch_names)

    if not all_psd:
        raise RuntimeError("Aucun canal EEG/EMG disponible pour la PSD (après sélection).")

    psd_lin = np.vstack(all_psd)
    return all_freqs, psd_lin, all_names

def integrate_band_powers(freqs, psd_lin, bands: dict[str, tuple[float, float]]):
    idx_all = np.where((freqs >= FMIN) & (freqs <= FMAX))[0]
    total_abs = np.trapz(psd_lin[:, idx_all], freqs[idx_all], axis=1).astype(float)
    total_abs[total_abs == 0] = np.finfo(float).eps

    abs_dict, rel_dict = {}, {}
    for name, (lo, hi) in bands.items():
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        val = (np.trapz(psd_lin[:, idx], freqs[idx], axis=1).astype(float)
               if idx.size else np.zeros(psd_lin.shape[0]))
        abs_dict[name] = val
        rel_dict[name] = val / total_abs
    return total_abs, abs_dict, rel_dict

def make_patient_fig_global(base: str, freqs, psd_lin, out_png: Path, bands=BANDS):
    eps = np.finfo(float).tiny
    psd_db = 10.0 * np.log10(np.maximum(psd_lin, eps))
    band_list = list(bands.items())
    fig, axes = plt.subplots(3, 2, figsize=(10, 9)); axes = axes.ravel()
    fig.suptitle(f"{base} - PSD (tous canaux) - moy ± p10-p90", fontsize=14, y=0.98)
    for i, (name, (lo, hi)) in enumerate(band_list):
        ax = axes[i]
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if idx.size == 0:
            ax.set_visible(False); continue
        band_f = freqs[idx]; band_db = psd_db[:, idx]
        m = np.nanmean(band_db, axis=0)
        p10 = np.nanpercentile(band_db, 10, axis=0)
        p90 = np.nanpercentile(band_db, 90, axis=0)
        ax.plot(band_f, m, lw=2)
        ax.fill_between(band_f, p10, p90, alpha=0.08)
        ax.set_xscale("log")
        lo_i = max(1, int(np.ceil(lo))); hi_i = int(np.floor(hi))
        ticks = list(range(lo_i, hi_i + 1))
        if ticks:
            ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.minorticks_off()
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel("Fréquence (Hz)")
        ax.set_ylabel("PSD (dB re µV²/Hz)")
        ax.set_title(f"{name}  [{lo:.1f}-{hi:.1f}] Hz")
        ax.grid(True, alpha=0.2)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_png, dpi=220)
    plt.close()

def make_patient_fig_channel(base: str, ch_label: str, freqs, psd_lin_channel, out_png: Path, bands=BANDS):
    eps = np.finfo(float).tiny
    psd_db = 10.0 * np.log10(np.maximum(psd_lin_channel[np.newaxis, :], eps))
    band_list = list(bands.items())
    fig, axes = plt.subplots(3, 2, figsize=(9, 8)); axes = axes.ravel()
    fig.suptitle(f"{base} - PSD - canal {ch_label}", fontsize=14, y=0.98)
    for i, (name, (lo, hi)) in enumerate(band_list):
        ax = axes[i]
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if idx.size == 0:
            ax.set_visible(False); continue
        band_f = freqs[idx]; band_db = psd_db[0, idx]
        ax.plot(band_f, band_db, lw=1.8)
        ax.set_xscale("log")
        lo_i = max(1, int(np.ceil(lo))); hi_i = int(np.floor(hi))
        ticks = list(range(lo_i, hi_i + 1))
        if ticks:
            ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.minorticks_off()
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel("Fréquence (Hz)")
        ax.set_ylabel("PSD (dB re µV²/Hz)")
        ax.set_title(f"{name}  [{lo:.1f}-{hi:.1f}] Hz")
        ax.grid(True, alpha=0.2)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_png, dpi=210)
    plt.close()

def make_patient_full_spectrum(base: str, freqs, psd_lin, out_png: Path):
    eps = np.finfo(float).tiny
    psd_db = 10.0 * np.log10(np.maximum(psd_lin, eps))
    m   = np.nanmean(psd_db, axis=0)
    p10 = np.nanpercentile(psd_db, 10, axis=0)
    p90 = np.nanpercentile(psd_db, 90, axis=0)

    major = [10, 13, 30, 50, 80, 100]
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.8))
    (line,) = ax.plot(freqs, m, lw=2)
    ax.fill_between(freqs, p10, p90, alpha=0.08, color=line.get_color(), linewidth=0)
    ax.set_xscale("log")
    ax.set_xlim(10, 100)
    ax.xaxis.set_major_locator(FixedLocator(major))
    ax.minorticks_off()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("PSD (dB re µV²/Hz)")
    ax.set_title(f"{base} - PSD (10-100 Hz) - moy ± p10-p90")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_png, dpi=230)
    plt.close(fig)

def interp_to_common_grid(freqs: np.ndarray, psd_lin_mean: np.ndarray, grid: np.ndarray = COMMON_FREQS) -> np.ndarray:
    m = np.isfinite(freqs) & np.isfinite(psd_lin_mean)
    f = freqs[m]; y = psd_lin_mean[m]
    if f.size < 2:
        return np.full_like(grid, np.nan, dtype=float)
    return np.interp(grid, f, y, left=y[0], right=y[-1])

def band_indices(grid: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.where((grid >= lo) & (grid < hi))[0]

def epochs_to_raw_concat(epochs: mne.Epochs) -> mne.io.Raw:
    """
    Concatène des epochs (n_epochs, n_ch, n_times) en un Raw continu (n_ch, n_epochs*n_times).
    Utile pour réutiliser la même compute_psd(raw).
    """
    data = epochs.get_data()  # shape: (n_epochs, n_ch, n_times)
    if data.size == 0:
        raise RuntimeError("Epochs vide (0 epochs).")

    # concat dans le temps
    data2 = np.transpose(data, (1, 0, 2)).reshape(data.shape[1], -1)  # (n_ch, n_epochs*n_times)

    info = epochs.info.copy()
    raw = mne.io.RawArray(data2, info, verbose="ERROR")

    # optionnel : conserver les noms/types (RawArray reprend info)
    return raw

# ---------- Worker ----------
def process_one(base: str, data_dir: Path, out_root: Path, mode: str, include_emg: bool, emg_band: tuple):
    import os as _os
    base_n = normalize_id(base)
    out_dir = out_root / base_n
    out_dir.mkdir(parents=True, exist_ok=True)

    fif = find_concat_fif(base_n, data_dir, mode)
    if fif is None:
        return {"base": base_n, "ok": False, "reason": f"{mode} fif not found"}

    try:
        if mode == "rswa_epo":
            epochs = mne.read_epochs(fif, preload=True, verbose="ERROR")
            raw = epochs_to_raw_concat(epochs)
        else:
            raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")

        print(f"[{base_n}] PID={_os.getpid()} | mode={mode} | file={fif}")
        print(f"[{base_n}] types:", pd.Series(raw.get_channel_types()).value_counts().to_dict())
    except Exception as e:
        return {"base": base_n, "ok": False, "reason": f"read_error: {e}"}

    try:
        freqs, psd_lin, ch_names = compute_psd(raw, fmin=FMIN, fmax=FMAX, include_emg=include_emg, emg_band=emg_band)
        total_abs_ch, band_abs_ch, band_rel_ch = integrate_band_powers(freqs, psd_lin, BANDS)
        psd_lin_mean = np.nanmean(psd_lin, axis=0)
        spec_lin_common = interp_to_common_grid(freqs, psd_lin_mean, COMMON_FREQS)
    except Exception as e:
        return {"base": base_n, "ok": False, "reason": f"psd_error: {e}"}

    abs_mean_global = {k: float(np.nanmean(v)) for k, v in band_abs_ch.items()}
    rel_mean_global = {k: float(np.nanmean(v)) for k, v in band_rel_ch.items()}
    total_mean_global = float(np.nanmean(total_abs_ch))

    try:
        make_patient_fig_global(base_n, freqs, psd_lin, out_dir / f"{base_n}_psd_bands.png", bands=BANDS)
    except Exception as e:
        print(f"[{base_n}] figure globale erreur: {e}")

    try:
        make_patient_full_spectrum(base_n, freqs, psd_lin, out_dir / f"{base_n}_psd_full_0p5_80.png")
    except Exception as e:
        print(f"[{base_n}] figure plein spectre erreur: {e}")

    per_channel_rows = []
    spec_chan = []
    for i, ch in enumerate(ch_names):
        try:
            out_png = out_dir / f"{base_n}_psd_bands_{sanitize_name(ch)}.png"
            make_patient_fig_channel(base_n, ch, freqs, psd_lin[i, :], out_png, bands=BANDS)
        except Exception as e:
            print(f"[{base_n}] figure canal '{ch}' erreur: {e}")

        for b in BANDS.keys():
            per_channel_rows.append({
                "base": base_n,
                "channel": ch,
                "band": b,
                "abs": float(band_abs_ch[b][i]),
                "rel": float(band_rel_ch[b][i]),
                "total_abs_channel": float(total_abs_ch[i]),
            })

        try:
            spec_lin_common_chan = interp_to_common_grid(freqs, psd_lin[i, :], COMMON_FREQS)
        except Exception:
            spec_lin_common_chan = np.full_like(COMMON_FREQS, np.nan, dtype=float)
        spec_chan.append({"channel": ch, "spec_lin_common": spec_lin_common_chan.tolist()})

    row = {"base": base_n, "total_abs": total_mean_global}
    row.update({f"abs_{b}": abs_mean_global[b] for b in BANDS})
    row.update({f"rel_{b}": rel_mean_global[b] for b in BANDS})

    return {
        "base": base_n,
        "ok": True,
        "row": row,
        "spec_lin_common": spec_lin_common.tolist(),
        "per_channel_rows": per_channel_rows,
        "spec_chan": spec_chan,
    }

# ---------- Plots groupes ----------
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
    return {"autoimmune encephalitides":"AI","narcolepsy":"Narco","synucleopathy":"Syn","tcspi":"TCSP","unknown":"Unknown"}.get(lab, label)

def grouped_barplot(df: pd.DataFrame, out_path: Path, title: str, ylabel: str):
    cats = list(df.index); bands = list(df.columns)
    x = np.arange(len(cats)); width = 0.12 if len(bands) > 6 else 0.15
    fig = plt.figure(figsize=(max(8, 1.6*len(cats)), 5))
    for i, b in enumerate(bands):
        plt.bar(x + i*width, df[b].values, width=width, label=b)
    plt.xticks(x + width*(len(bands)-1)/2, cats, rotation=45, ha="right")
    plt.ylabel(ylabel); plt.title(title); plt.legend(ncol=2, fontsize=8)
    plt.tight_layout(); fig.savefig(out_path, dpi=230); plt.close(fig)

def per_channel_barplots_long(df_ch: pd.DataFrame, out_root: Path, bands=BANDS, kind: str = "abs",
                              min_patients: int = MIN_PATIENTS_PER_CHANNEL):
    assert kind in ("abs","rel")
    value_col = {"abs":"abs","rel":"rel"}[kind]
    cnt_by_chan = df_ch.groupby("channel")["base"].nunique()
    eligible = sorted([c for c, n in cnt_by_chan.items() if n >= min_patients])
    if not eligible:
        print("[WARN] Aucun canal avec effectif suffisant pour barplots par canal."); return

    for chan_full in eligible:
        sub_c = df_ch[df_ch["channel"] == chan_full]
        for band in bands.keys():
            sub = sub_c[sub_c["band"] == band]
            if sub.empty: 
                continue
            grp = sub.groupby("group")[value_col]
            mean = grp.mean().sort_index()
            cnt  = grp.count().reindex(mean.index)
            std  = grp.std(ddof=1).reindex(mean.index)
            sem  = (std / np.sqrt(cnt.clip(lower=1))).fillna(0.0)

            colors = [color_for_group(g) for g in mean.index]
            xticks = [f"{short_label(g)} (n={int(cnt[g])})" for g in mean.index]
            x = np.arange(len(mean))

            fig = plt.figure(figsize=(max(7, 1.6*len(mean)), 4.6))
            plt.bar(x, mean.values, yerr=sem.values, capsize=3, color=colors)
            plt.xticks(x, xticks, rotation=30, ha="right")
            plt.ylabel("PSD absolue (µV²)" if kind=="abs" else "PSD relative")
            plt.title(f"{band} - canal {chan_full} (moyenne par catégorie)")
            plt.tight_layout()
            out = out_root / f"perband_{kind}_chan_{sanitize_name(chan_full)}_{band}.png"
            fig.savefig(out, dpi=230)
            plt.close(fig)

# ---------- Main ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",  type=str, default=DATA_DIR_DEFAULT,
                   help="Dossier contenant les fichiers concat (REM ou RSWA).")
    p.add_argument("--out-root",  type=str, default=OUT_ROOT_DEFAULT)
    p.add_argument("--mode", type=str, default="rem", choices=["rem","rswa","rswa_epo"],
                   help="rem: lit *_REM_concat.fif ; rswa: lit *_RSWA_concat.fif ; rswa_epo: lit *_RSWA-epo.fif")
    p.add_argument("--workers",   type=int, default=15, help="0 => CPU-1")
    p.add_argument("--patients",  type=str, nargs="*", default=None,
                   help="liste explicite (sinon auto depuis data-dir)")
    p.add_argument("--excel", type=str, default="/home/darryld/documents/nv_patientsRBD_identifiants.xlsx")
    p.add_argument("--sheet-index", type=int, default=2)
    p.add_argument("--min-patients-per-channel", type=int, default=MIN_PATIENTS_PER_CHANNEL)
    p.add_argument("--include-emg", action="store_true",
                   help="Inclure les canaux EMG dans les PSD et figures (filtre 30-100 Hz).")
    p.add_argument("--emg-band", type=float, nargs=2, metavar=("LO","HI"),
                   default=(30.0, 100.0), help="Bande EMG pour le filtrage avant PSD.")
    return p.parse_args()

def main():
    mne.set_log_level("WARNING")
    args = parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    if not data_dir.exists():
        raise SystemExit(f"[CONFIG] Dossier introuvable: {data_dir}")
    out_root.mkdir(parents=True, exist_ok=True)

    # Patients
    if args.patients:
        bases = sorted({normalize_id(b.strip().strip(",")) for b in args.patients if b.strip()})
    else:
        bases = list_patients(data_dir, args.mode)
    if not bases:
        raise SystemExit("Aucun patient détecté.")
    print(f"[INFO] mode={args.mode} | patients={len(bases)}")

    # Groupes
    group_map = {}
    xlsx = Path(args.excel)
    if xlsx.exists():
        try:
            group_map = load_groups_from_excel(xlsx, args.sheet_index)
        except Exception as e:
            print(f"[WARN] Excel lu mais erreur: {e}")
    else:
        print(f"[WARN] Excel non trouvé: {xlsx}")

    # Pool
    cpu = os.cpu_count() or 1
    n_workers = (cpu - 1 if args.workers in (0, None) else args.workers)
    n_workers = max(1, min(n_workers, len(bases)))
    print(f"[INFO] Parallèle avec {n_workers} worker(s) (spawn, maxtasksperchild=1)")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, maxtasksperchild=1) as pool:
        results = pool.starmap(
            process_one,
            [(b, data_dir, out_root, args.mode, args.include_emg, tuple(args.emg_band)) for b in bases]
        )

    # Compile
    rows, per_channel_all = [], []
    for r in results:
        if not r.get("ok"):
            print(f"[{r['base']}] SKIP: {r.get('reason')}")
            continue
        base = r["base"]
        label = group_map.get(base, "unknown")
        row = {"base": base, "group": label, **r["row"]}
        rows.append(row)
        per_channel_all.extend(r.get("per_channel_rows", []))

    if not rows:
        print("Aucun résultat exploitable."); return

    df = pd.DataFrame(rows).set_index("base").sort_index()
    (out_root / "all_band_powers.csv").write_text("")  # touch for clarity in logs (optional)
    csv_all = out_root / "all_band_powers.csv"
    df.to_csv(csv_all, float_format="%.8e")
    print(f">>> {csv_all}")

    if per_channel_all:
        df_ch = pd.DataFrame(per_channel_all)
        df_ch["group"] = df_ch["base"].map(lambda b: group_map.get(b, "unknown"))
        csv_ch = out_root / "per_channel_band_powers.csv"
        df_ch.to_csv(csv_ch, index=False, float_format="%.8e")
        print(f">>> {csv_ch}")
    else:
        df_ch = pd.DataFrame(columns=["base","channel","band","abs","rel","total_abs_channel","group"])

    # Figures inter-groupes (globale)
    spec_map = {r["base"]: np.array(r["spec_lin_common"], dtype=float)
                for r in results if r.get("ok") and "spec_lin_common" in r}
    bases_ok = [b for b in df.index if b in spec_map]
    if bases_ok:
        SPEC = np.vstack([spec_map[b] for b in bases_ok])
        DF_ALIGNED = df.loc[bases_ok].copy()
        eps = np.finfo(float).tiny
        SPEC_DB = 10.0 * np.log10(np.maximum(SPEC, eps))

        fig_bands, axes_bands = plt.subplots(3, 2, figsize=(11, 9))
        axes = axes_bands.ravel()
        fig_bands.suptitle(f"PSD ({args.mode}) - courbes par bande, par groupe (moy ± p10-p90)", fontsize=14, y=0.98)

        band_list = list(BANDS.items())
        for i, (name, (lo, hi)) in enumerate(band_list):
            ax = axes[i]
            idx = band_indices(COMMON_FREQS, lo, hi)
            if idx.size == 0:
                ax.set_visible(False); 
                continue

            for grp, idx_labels in DF_ALIGNED.groupby("group").groups.items():
                idx_int = np.array([DF_ALIGNED.index.get_loc(b) for b in idx_labels], dtype=int)
                sub = SPEC_DB[idx_int][:, idx]
                if sub.size == 0:
                    continue
                m   = np.nanmean(sub, axis=0)
                p10 = np.nanpercentile(sub, 10, axis=0)
                p90 = np.nanpercentile(sub, 90, axis=0)
                c = color_for_group(grp)
                ax.plot(COMMON_FREQS[idx], m, lw=2, label=f"{short_label(grp)}", color=c)
                ax.fill_between(COMMON_FREQS[idx], p10, p90, alpha=0.08, color=c)

            ax.set_xscale("log")
            lo_i = max(1, int(np.ceil(lo))); hi_i = int(np.floor(hi))
            ticks = list(range(lo_i, hi_i + 1))
            if ticks:
                ax.xaxis.set_major_locator(FixedLocator(ticks))
                ax.minorticks_off()
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_xlabel("Fréquence (Hz)")
            ax.set_ylabel("PSD (dB re µV²/Hz)")
            ax.set_title(f"{name}  [{lo:.1f}-{hi:.1f}] Hz")
            ax.grid(True, alpha=0.2)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig_bands.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        out_png = out_root / "group_psd_bands.png"
        fig_bands.savefig(out_png, dpi=230)
        plt.close(fig_bands)
        print(f">>> {out_png}")

        fig_full, ax_full = plt.subplots(1, 1, figsize=(9, 5))
        for grp, idx_labels in DF_ALIGNED.groupby("group").groups.items():
            idx_int = np.array([DF_ALIGNED.index.get_loc(b) for b in idx_labels], dtype=int)
            sub = SPEC_DB[idx_int]
            if sub.size == 0:
                continue
            m   = np.nanmean(sub, axis=0)
            p10 = np.nanpercentile(sub, 10, axis=0)
            p90 = np.nanpercentile(sub, 90, axis=0)
            c = color_for_group(grp)
            ax_full.plot(COMMON_FREQS, m, lw=2, label=short_label(grp), color=c)
            ax_full.fill_between(COMMON_FREQS, p10, p90, alpha=0.08, color=c)

        major_ticks = [0.5, 1, 2, 4, 8, 13, 30, 50, 80]
        ax_full.set_xscale("log")
        ax_full.set_xlim(0.5, 80)
        ax_full.xaxis.set_major_locator(FixedLocator(major_ticks))
        ax_full.xaxis.set_major_formatter(ScalarFormatter())
        ax_full.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 5, 7)))
        ax_full.grid(True, which="major", alpha=0.25)
        ax_full.grid(True, which="minor", alpha=0.07)
        ax_full.set_xlabel("Fréquence (Hz)")
        ax_full.set_ylabel("PSD (dB re µV²/Hz)")
        ax_full.set_title(f"PSD ({args.mode}) - plein spectre par groupe (moy ± p10-p90)")
        ax_full.legend(ncol=5, frameon=False)

        plt.tight_layout()
        out_png = out_root / "group_psd_full_0p5_80.png"
        fig_full.savefig(out_png, dpi=230)
        plt.close(fig_full)
        print(f">>> {out_png}")
    else:
        print("[WARN] Aucune PSD interpolée récupérée, skip figures inter-groupes.")

    # Agrégats par groupe
    band_names = list(BANDS.keys())
    abs_cols = [f"abs_{b}" for b in band_names]
    rel_cols = [f"rel_{b}" for b in band_names]

    g_abs = df.groupby("group")[abs_cols].mean()
    g_rel = df.groupby("group")[rel_cols].mean()
    (out_root / "group_means_abs.csv").write_text("")  # touch optional
    (out_root / "group_means_rel.csv").write_text("")  # touch optional
    g_abs.to_csv(out_root / "group_means_abs.csv", float_format="%.8e")
    g_rel.to_csv(out_root / "group_means_rel.csv", float_format="%.8e")

    # Barplots par canal (inter-catégories)
    if not df_ch.empty:
        per_channel_barplots_long(df_ch, out_root, bands=BANDS, kind="abs", min_patients=args.min_patients_per_channel)
        per_channel_barplots_long(df_ch, out_root, bands=BANDS, kind="rel", min_patients=args.min_patients_per_channel)

    # Patients unknown
    missing = [b for b in df.index if group_map.get(b, "unknown") == "unknown"]
    if missing:
        with open(out_root / "patients_unknown_category.txt", "w") as f:
            for b in sorted(set(missing)):
                f.write(f"{b}\n")
        print(f"[INFO] {len(set(missing))} patient(s) sans catégorie -> patients_unknown_category.txt")

if __name__ == "__main__":
    main()
