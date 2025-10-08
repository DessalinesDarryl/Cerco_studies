#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSD REM - global + **par canal** (figures et agrégats) + barplots inter-catégories.

Entrées :
  - REM :   /home/darryld/documents/EEG/preprocessed/bipolaire/2_rem_only/{BASE}/{BASE}_REM_concat.fif
  - Excel : --excel <path> (Feuil3 par défaut ; recherche tolérante des colonnes)

Sorties :
  - out_root/{BASE}/{BASE}_psd_bands.png                 (global, inchangé)
  - out_root/{BASE}/{BASE}_psd_bands_{CANAL}.png         (une figure par **canal complet**)
  - out_root/all_band_powers.csv                         (agrégats globaux patient)
  - out_root/per_channel_band_powers.csv                 (long : base, channel, band, abs, rel, total_abs_channel)
  - out_root/<groupe>_band_power_abs.png                 (global absolu par groupe)
  - out_root/group_band_power_rel.png                    (global relatif par groupe)
  - out_root/perband_abs_chan_{CANAL}_{BAND}.png         (inter-catégories par **canal complet**)
  - out_root/perband_rel_chan_{CANAL}_{BAND}.png         (inter-catégories par **canal complet**)
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

import multiprocessing as mp
import faulthandler; faulthandler.enable()

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator

import mne
mne.set_config('MNE_MEMMAP_MIN_SIZE', '1M', set_env=True)

# ---------- Defaults ----------
REM_DIR_DEFAULT  = "/home/darryld/documents/EEG/preprocessed/bipolaire/2_rem_only/gp2"
OUT_ROOT_DEFAULT = "/home/darryld/documents/EEG/preprocessed/bipolaire/3_results_analysis/gp2"

BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta":  (13.0, 30.0),
    "Gamma_Bas": (30.0, 50.0),
    "Gamma_Haut": (50.0, 80.0),
}
FMIN, FMAX = 0.5, 80.0
COMMON_NFREQ = 400
COMMON_FREQS = np.linspace(FMIN, FMAX, COMMON_NFREQ)

# Combien de patients minimum doivent avoir un canal pour tracer les barplots inter-catégories
MIN_PATIENTS_PER_CHANNEL = 5

# ---------- Utils ----------
def normalize_id(x: str) -> str:
    return "".join(ch for ch in str(x).strip().upper() if ch.isalnum())

def sanitize_name(x: str) -> str:
    # pour les noms de fichiers : on supprime les caractères problématiques
    return "".join(c for c in str(x) if c.isalnum() or c in ("_", "-")).strip("_-").lower()

def list_patients(rem_dir: Path) -> list[str]:
    bases = set()
    for child in rem_dir.iterdir():
        if child.is_dir() and not child.name.startswith("._"):
            bases.add(normalize_id(child.name))
    for p in rem_dir.glob("*_REM_concat.fif"):
        if p.is_file() and not p.name.startswith("._"):
            bases.add(normalize_id(p.stem.split("_")[0]))
    return sorted(bases)

def find_rem_fif(base: str, rem_dir: Path) -> Path | None:
    p = rem_dir / base / f"{base}_REM_concat.fif"
    if p.exists(): return p
    p2 = rem_dir / f"{base}_REM_concat.fif"
    return p2 if p2.exists() else None

# --------- Chargement des catégories ---------
def load_groups_from_excel(xlsx_path: Path, sheet_index: int | None):
    from cohort import load_patient_groups as _lpg
    df, group_map, _demo = _lpg(str(xlsx_path), sheet_index=sheet_index)
    return {str(k).strip().upper(): str(v).strip().lower() for k, v in group_map.items()}

# ---------- PSD helpers ----------
def compute_psd(raw: mne.io.BaseRaw, fmin=FMIN, fmax=FMAX):
    inst = raw.copy().pick_types(meg=False, eeg=True, eog=False, ecg=False, emg=False,
                                 stim=False, misc=False, resp=False, seeg=False, ecog=False, fnirs=False)
    if len(inst.ch_names) == 0:
        raise RuntimeError("Aucun canal EEG")
    inst.load_data()
    picks = mne.pick_types(inst.info, eeg=True, meg=False, eog=False, ecg=False, emg=False)
    inst.apply_function(lambda x: x * 1e6, picks=picks, channel_wise=True)  # µV
    try: inst.set_unit("eeg", "uV")
    except Exception: pass

    sf = float(inst.info["sfreq"])
    n_times = int(inst.n_times)
    n_per_seg = max(2, min(int(sf*4), n_times))   # ~4 s
    n_overlap = max(0, min(n_per_seg//2, n_per_seg-1))

    psd = inst.compute_psd(method="welch", fmin=float(fmin), fmax=float(fmax),
                           n_per_seg=n_per_seg, n_overlap=n_overlap, verbose="ERROR")
    freqs = psd.freqs
    data  = psd.get_data()  # (n_chan, n_freq) en µV²/Hz
    return freqs, data, inst.ch_names

def integrate_band_powers(freqs, psd_lin, bands: dict[str, tuple[float,float]]):
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
    fig.suptitle(f"{base} - PSD REM (tous canaux) - moy ± p10–p90", fontsize=14, y=0.98)
    for i, (name, (lo, hi)) in enumerate(band_list):
        ax = axes[i]
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if idx.size == 0: ax.set_visible(False); continue
        band_f = freqs[idx]; band_db = psd_db[:, idx]
        m = np.nanmean(band_db, axis=0)
        p10 = np.nanpercentile(band_db, 10, axis=0)
        p90 = np.nanpercentile(band_db, 90, axis=0)
        ax.plot(band_f, m, lw=2); ax.fill_between(band_f, p10, p90, alpha=0.25)
        ax.set_xscale("log")
        lo_i = max(1, int(np.ceil(lo))); hi_i = int(np.floor(hi))
        ticks = list(range(lo_i, hi_i + 1))
        if ticks: ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.minorticks_off()
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel("Fréquence (Hz)"); ax.set_ylabel("PSD (dB re µV²/Hz)")
        ax.set_title(f"{name}  [{lo:.1f}-{hi:.1f}] Hz"); ax.grid(True, alpha=0.2)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(out_png, dpi=220); plt.close()

def make_patient_fig_channel(base: str, ch_label: str, freqs, psd_lin_channel, out_png: Path, bands=BANDS):
    """
    Figure 3x2 par **canal** : un seul canal -> pas d'IC ; on trace la PSD du canal par sous-bande.
    """
    eps = np.finfo(float).tiny
    psd_db = 10.0 * np.log10(np.maximum(psd_lin_channel[np.newaxis, :], eps))
    band_list = list(bands.items())
    fig, axes = plt.subplots(3, 2, figsize=(9, 8)); axes = axes.ravel()
    fig.suptitle(f"{base} - PSD REM - canal {ch_label}", fontsize=14, y=0.98)
    for i, (name, (lo, hi)) in enumerate(band_list):
        ax = axes[i]
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if idx.size == 0: ax.set_visible(False); continue
        band_f = freqs[idx]; band_db = psd_db[0, idx]
        ax.plot(band_f, band_db, lw=1.8)
        ax.set_xscale("log")
        lo_i = max(1, int(np.ceil(lo))); hi_i = int(np.floor(hi))
        ticks = list(range(lo_i, hi_i + 1))
        if ticks: ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.minorticks_off()
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel("Fréquence (Hz)"); ax.set_ylabel("PSD (dB re µV²/Hz)")
        ax.set_title(f"{name}  [{lo:.1f}-{hi:.1f}] Hz"); ax.grid(True, alpha=0.2)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(out_png, dpi=210); plt.close()

def interp_to_common_grid(freqs: np.ndarray, psd_lin_mean: np.ndarray,
                          grid: np.ndarray = COMMON_FREQS) -> np.ndarray:
    m = np.isfinite(freqs) & np.isfinite(psd_lin_mean)
    f = freqs[m]; y = psd_lin_mean[m]
    if f.size < 2: return np.full_like(grid, np.nan, dtype=float)
    return np.interp(grid, f, y, left=y[0], right=y[-1])

def band_indices(grid: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.where((grid >= lo) & (grid < hi))[0]

# ---------- Worker ----------
def process_one(base: str, rem_dir: Path, out_root: Path):
    base_n = normalize_id(base)
    out_dir = out_root / base_n
    out_dir.mkdir(parents=True, exist_ok=True)

    fif = find_rem_fif(base_n, rem_dir)
    if fif is None:
        return {"base": base_n, "ok": False, "reason": "REM fif not found"}

    try:
        raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
    except Exception as e:
        return {"base": base_n, "ok": False, "reason": f"read_error: {e}"}

    try:
        freqs, psd_lin, ch_names = compute_psd(raw, fmin=FMIN, fmax=FMAX)   # psd_lin: (n_chan, n_freq)
        total_abs_ch, band_abs_ch, band_rel_ch = integrate_band_powers(freqs, psd_lin, BANDS)
        psd_lin_mean = np.nanmean(psd_lin, axis=0)
        spec_lin_common = interp_to_common_grid(freqs, psd_lin_mean, COMMON_FREQS)
    except Exception as e:
        return {"base": base_n, "ok": False, "reason": f"psd_error: {e}"}

    # --- agrégats globaux patient
    abs_mean_global = {k: float(np.nanmean(v)) for k, v in band_abs_ch.items()}
    rel_mean_global = {k: float(np.nanmean(v)) for k, v in band_rel_ch.items()}
    total_mean_global = float(np.nanmean(total_abs_ch))

    # --- figure globale patient
    try:
        make_patient_fig_global(base_n, freqs, psd_lin, out_dir / f"{base_n}_psd_bands.png", bands=BANDS)
    except Exception as e:
        print(f"[{base_n}] figure globale erreur: {e}")

    # --- figures PAR CANAL + table longue + PSD interp par canal
    per_channel_rows = []
    spec_chan = []  # NEW: PSD interpolée (lin) par canal pour les agrégats inter-patients

    for i, ch in enumerate(ch_names):
        # figure par canal (patient) :
        try:
            out_png = out_dir / f"{base_n}_psd_bands_{sanitize_name(ch)}.png"
            make_patient_fig_channel(base_n, ch, freqs, psd_lin[i, :], out_png, bands=BANDS)
        except Exception as e:
            print(f"[{base_n}] figure canal '{ch}' erreur: {e}")

        # lignes CSV "long" (par bande, par canal) :
        for b in BANDS.keys():
            per_channel_rows.append({
                "base": base_n,
                "channel": ch,
                "band": b,
                "abs": float(band_abs_ch[b][i]),
                "rel": float(band_rel_ch[b][i]),
                "total_abs_channel": float(total_abs_ch[i]),
            })

        # PSD interpolée par canal sur la grille commune
        try:
            spec_lin_common_chan = interp_to_common_grid(freqs, psd_lin[i, :], COMMON_FREQS)
        except Exception:
            spec_lin_common_chan = np.full_like(COMMON_FREQS, np.nan, dtype=float)
        spec_chan.append({"channel": ch, "spec_lin_common": spec_lin_common_chan.tolist()})


    # --- ligne CSV patient (agrégats globaux)
    row = {"base": base_n, "total_abs": total_mean_global}
    row.update({f"abs_{b}": abs_mean_global[b] for b in BANDS})
    row.update({f"rel_{b}": rel_mean_global[b] for b in BANDS})

    return {
        "base": base_n, "ok": True,
        "row": row,
        "spec_lin_common": spec_lin_common.tolist(),   # global (moyenne canaux)
        "per_channel_rows": per_channel_rows,          # long CSV
        "spec_chan": spec_chan,                        # PSD interp par canal
    }

# ---------- Plots groupes (globaux) ----------
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

def per_band_barplots(df: pd.DataFrame, out_root: Path, bands=BANDS, kind: str = "abs"):
    assert kind in ("abs","rel")
    ylabel = "PSD absolue (µV²)" if kind == "abs" else "PSD relative"
    for band in bands.keys():
        col = f"{kind}_{band}"
        if col not in df.columns: continue
        grp = df.groupby("group")[col]
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
        plt.ylabel(ylabel); plt.title(f"{band} - moyenne par catégorie")
        plt.tight_layout()
        out = out_root / f"perband_{kind}_{band}.png"
        fig.savefig(out, dpi=230); plt.close(fig)

# ---------- NOUVEAU : barplots inter-catégories **par canal complet** ----------
def per_channel_barplots_long(df_ch: pd.DataFrame, out_root: Path, bands=BANDS, kind: str = "abs",
                              min_patients: int = MIN_PATIENTS_PER_CHANNEL):
    """
    df_ch : colonnes = base, channel, band, abs, rel, total_abs_channel, + group (merge en amont)
    Produit perband_{kind}_chan_{CANAL}_{BAND}.png pour les canaux présents chez >= min_patients.
    """
    assert kind in ("abs","rel")
    value_col = {"abs":"abs","rel":"rel"}[kind]
    # canaux éligibles (par **nom complet**)
    cnt_by_chan = df_ch.groupby("channel")["base"].nunique()
    eligible = sorted([c for c, n in cnt_by_chan.items() if n >= min_patients])
    if not eligible:
        print("[WARN] Aucun canal avec effectif suffisant pour barplots par canal."); return
    for chan_full in eligible:
        sub_c = df_ch[df_ch["channel"] == chan_full]
        print(f"[INFO] Traitement du canal: {chan_full}")
        for band in bands.keys():
            sub = sub_c[sub_c["band"] == band]
            if sub.empty: continue
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
            fig.savefig(out, dpi=230); plt.close(fig)

# ---------- Main ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rem-dir",   type=str, default=REM_DIR_DEFAULT)
    p.add_argument("--out-root",  type=str, default=OUT_ROOT_DEFAULT)
    p.add_argument("--workers",   type=int, default=8, help="0 => CPU-1")
    p.add_argument("--patients",  type=str, nargs="*", default=None,
                   help="liste explicite (sinon auto depuis rem-dir)")
    p.add_argument("--excel", type=str, default="/home/darryld/documents/nv_patientsRBD_identifiants.xlsx")
    p.add_argument("--sheet-index", type=int, default=2)
    p.add_argument("--min-patients-per-channel", type=int, default=MIN_PATIENTS_PER_CHANNEL,
                   help="min sujets/chan pour barplots inter-catégories par canal")
    return p.parse_args()

def main():
    mne.set_log_level("WARNING")
    args = parse_args()

    rem_dir  = Path(args.rem_dir)
    out_root = Path(args.out_root)
    if not rem_dir.exists():
        raise SystemExit(f"[CONFIG] Dossier introuvable: {rem_dir}")
    out_root.mkdir(parents=True, exist_ok=True)

    # Patients
    if args.patients:
        bases = sorted({normalize_id(b.strip().strip(",")) for b in args.patients if b.strip()})
    else:
        bases = list_patients(rem_dir)
    if not bases:
        raise SystemExit("Aucun patient détecté.")
    print(f"Patients ({len(bases)}): {bases}")

    # Groupes (labels)
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
    n_workers = (cpu-1 if args.workers in (0, None) else args.workers)
    n_workers = max(1, min(n_workers, len(bases)))
    print(f"[INFO] Parallèle avec {n_workers} worker(s) (spawn, maxtasksperchild=1)")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, maxtasksperchild=1) as pool:
        results = pool.starmap(process_one, [(b, rem_dir, out_root) for b in bases])

    # Compile
    rows, missing, per_channel_all = [], [], []
    for r in results:
        if not r.get("ok"):
            print(f"[{r['base']}] SKIP: {r.get('reason')}")
            continue
        base = r["base"]
        label = group_map.get(base, "unknown")
        if label == "unknown":
            missing.append(base)
        row = {"base": base, "group": label, **r["row"]}
        rows.append(row)
        per_channel_all.extend(r.get("per_channel_rows", []))

    if not rows:
        print("Aucun résultat exploitable."); return

    # CSV principal (agrégats globaux patient)
    df = pd.DataFrame(rows).set_index("base").sort_index()
    csv_all = out_root / "all_band_powers.csv"
    df.to_csv(csv_all, float_format="%.8e")
    print(f">>> {csv_all}")

    # CSV long par canal (nom complet)
    if per_channel_all:
        df_ch = pd.DataFrame(per_channel_all)
        # Merge des groupes
        df_ch["group"] = df_ch["base"].map(lambda b: group_map.get(b, "unknown"))
        csv_ch = out_root / "per_channel_band_powers.csv"
        df_ch.to_csv(csv_ch, index=False, float_format="%.8e")
        print(f">>> {csv_ch}")
    else:
        df_ch = pd.DataFrame(columns=["base","channel","band","abs","rel","total_abs_channel","group"])

    # PSD interpolées pour figure inter-groupes (globale) - inchangé
    spec_map = {r["base"]: np.array(r["spec_lin_common"], dtype=float)
                for r in results if r.get("ok") and "spec_lin_common" in r}
    bases_ok = [b for b in df.index if b in spec_map]
    if bases_ok:
        SPEC = np.vstack([spec_map[b] for b in bases_ok])
        DF_ALIGNED = df.loc[bases_ok].copy()
        eps = np.finfo(float).tiny
        SPEC_DB = 10.0 * np.log10(np.maximum(SPEC, eps))

        fig, axes = plt.subplots(3, 2, figsize=(11, 9)); axes = axes.ravel()
        fig.suptitle("PSD REM - courbes par bande, superposées par groupe (moy ± p10–p90)", fontsize=14, y=0.98)

        band_list = list(BANDS.items())
        for i, (name, (lo, hi)) in enumerate(band_list):
            ax = axes[i]
            idx = band_indices(COMMON_FREQS, lo, hi)
            if idx.size == 0: ax.set_visible(False); continue
            for grp, idx_labels in DF_ALIGNED.groupby("group").groups.items():
                if len(idx_labels) == 0: continue
                idx_int = np.array([DF_ALIGNED.index.get_loc(b) for b in idx_labels], dtype=int)
                if idx_int.size == 0: continue
                sub = SPEC_DB[idx_int][:, idx]
                if sub.size == 0: continue
                m   = np.nanmean(sub, axis=0)
                p10 = np.nanpercentile(sub, 10, axis=0)
                p90 = np.nanpercentile(sub, 90, axis=0)
                color = color_for_group(grp)
                ax.plot(COMMON_FREQS[idx], m, lw=2, label=f"{short_label(grp)}", color=color)
                ax.fill_between(COMMON_FREQS[idx], p10, p90, alpha=0.12, color=color)

            ax.set_xscale("log")
            lo_i = max(1, int(np.ceil(lo))); hi_i = int(np.floor(hi))
            ticks = list(range(lo_i, hi_i + 1))
            if ticks: ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.minorticks_off()
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_xlabel("Fréquence (Hz)")
            ax.set_ylabel("PSD (dB re µV²/Hz)")
            ax.set_title(f"{name}  [{lo:.1f}-{hi:.1f}] Hz")
            ax.grid(True, alpha=0.2)

        if len(band_list) < len(axes): axes[-1].axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles: plt.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        out_png = out_root / "group_psd_bands.png"
        plt.savefig(out_png, dpi=230); plt.close(); print(f">>> {out_png}")
    else:
        print("[WARN] Aucune PSD interpolée récupérée, skip figure inter-groupes.")

    # Agrégats par label (global)
    band_names = list(BANDS.keys())
    abs_cols =[f"abs_{b}" for b in band_names]
    rel_cols = [f"rel_{b}" for b in band_names]
    g_abs = df.groupby("group")[abs_cols].mean().loc[:, abs_cols]
    g_rel = df.groupby("group")[rel_cols].mean().loc[:, rel_cols]
    g_abs.to_csv(out_root / "group_means_abs.csv", float_format="%.8e")
    g_rel.to_csv(out_root / "group_means_rel.csv", float_format="%.8e")

    # Figures globales existantes
    group_counts = df.groupby("group").size().to_dict()
    for grp in g_abs.index:
        vals = [g_abs.loc[grp, f"abs_{b}"] for b in band_names]
        names = band_names; n = int(group_counts.get(grp, 0))
        fig = plt.figure(figsize=(7, 4)); plt.bar(names, vals)
        plt.ylabel("PSD absolue (µV²)"); plt.title(f"Band power (absolute, µV²) - {grp} (n={n})")
        plt.tight_layout(); out_png = out_root / f"{sanitize_name(grp)}_band_power_abs.png"
        fig.savefig(out_png, dpi=230); plt.close(fig)

    g_rel_plot = g_rel.copy(); g_rel_plot.columns = band_names
    grouped_barplot(g_rel_plot, out_root / "group_band_power_rel.png",
                    "Band power (relative) - mean per category", ylabel="PSD relative")

    # --- Barplots inter-catégories **par canal complet**
    if not df_ch.empty:
        per_channel_barplots_long(df_ch, out_root, bands=BANDS, kind="abs",
                                  min_patients=args.min_patients_per_channel)
        per_channel_barplots_long(df_ch, out_root, bands=BANDS, kind="rel",
                                  min_patients=args.min_patients_per_channel)

    if missing:
        with open(out_root / "patients_unknown_category.txt", "w") as f:
            for b in sorted(set(missing)): f.write(f"{b}\n")
        print(f"[INFO] {len(set(missing))} patient(s) sans catégorie connue -> patients_unknown_category.txt")

    # ---------- Figures inter-groupes **par canal** (moy ± p10–p90), une image par canal ----------
    # Collecte: channel -> { base -> spec_lin_common (lin) }
    chan_map = {}  # dict[str, dict[base:str, np.ndarray]]
    for r in results:
        if not r.get("ok"): 
            continue
        base = r["base"]
        # sauter les patients absents du df (sécurité)
        if base not in df.index:
            continue
        for item in r.get("spec_chan", []):
            ch = item.get("channel")
            spec_list = item.get("spec_lin_common")
            if ch is None or spec_list is None:
                continue
            arr = np.asarray(spec_list, dtype=float)
            if arr.shape[0] != COMMON_FREQS.shape[0]:
                continue
            chan_map.setdefault(ch, {})[base] = arr

    # Filtrer canaux avec effectif suffisant
    min_pat = args.min_patients_per_channel
    eligible_channels = sorted([ch for ch, d in chan_map.items() if len(d) >= min_pat])
    if not eligible_channels:
        print("[WARN] Aucun canal avec effectif suffisant pour figures inter-groupes par canal.")
    else:
        print(f"[INFO] Figures inter-groupes par canal pour {len(eligible_channels)} canaux (min {min_pat} patients).")

    # Helper d'index pour corriger l'erreur d'indexation vue précédemment
    # (on mappe base -> rang dans SPEC_CH pour chaque canal)
    for ch in eligible_channels:
        base_to_spec = chan_map[ch]  # dict base -> spec_lin (lin)
        # Aligner sur les bases disponibles dans df
        bases_ch = [b for b in df.index if b in base_to_spec]
        if not bases_ch:
            continue

        SPEC_CH = np.vstack([base_to_spec[b] for b in bases_ch])  # (n_patients_ch, n_freq)
        eps = np.finfo(float).tiny
        SPEC_CH_DB = 10.0 * np.log10(np.maximum(SPEC_CH, eps))

        DF_CH = df.loc[bases_ch].copy()  # pour les groupes
        row_index = {b: i for i, b in enumerate(bases_ch)}  # base -> ligne SPEC_CH

        fig, axes = plt.subplots(3, 2, figsize=(18, 10)); axes = axes.ravel()
        fig.suptitle(f"PSD REM - canal {ch} - courbes par bande, superposées par groupe (moy ± p10–p90)", fontsize=14, y=0.98)

        band_list = list(BANDS.items())
        for i, (name, (lo, hi)) in enumerate(band_list):
            ax = axes[i]
            idx = band_indices(COMMON_FREQS, lo, hi)
            if idx.size == 0:
                ax.set_visible(False); 
                continue

            # Parcours des groupes
            for grp, idx_labels in DF_CH.groupby("group").groups.items():
                if len(idx_labels) == 0:
                    continue
                # indices entiers robustes (corrige l'IndexError des labels string)
                idx_int = np.array([row_index[b] for b in idx_labels if b in row_index], dtype=int)
                if idx_int.size == 0:
                    continue

                sub = SPEC_CH_DB[idx_int][:, idx]  # (n_grp, n_freq_band)
                if sub.size == 0:
                    continue

                m   = np.nanmean(sub, axis=0)
                p10 = np.nanpercentile(sub, 10, axis=0)
                p90 = np.nanpercentile(sub, 90, axis=0)

                color = color_for_group(grp)
                ax.plot(COMMON_FREQS[idx], m, lw=2, label=f"{short_label(grp)}", color=color)
                ax.fill_between(COMMON_FREQS[idx], p10, p90, alpha=0.12, color=color)

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

        if len(band_list) < len(axes):
            axes[-1].axis("off")

        # Légende globale (si dispo)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)

        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        out_png = out_root / f"group_psd_bands_chan_{sanitize_name(ch)}.png"
        fig.savefig(out_png, dpi=230)
        plt.close(fig)
        print(f">>> {out_png}")


if __name__ == "__main__":
    main()
