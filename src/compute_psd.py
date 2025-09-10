#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSD par patient (REM) + figures 'moy ± p10–p90' par bande + barplots d'agrégats par label (parallélisé).

Entrées :
  - REM :   /home/darryld/documents/EEG/preprocessed/bipolaire/2_rem_only/{BASE}/{BASE}_REM_concat.fif
  - Excel : --excel <path> (Feuil3 par défaut ; recherche tolérante des colonnes)

Sorties :
  - out_root/{BASE}/{BASE}_psd_bands.png  (figure 5 sous-plots Delta..Gamma, en dB, moy ± p10–p90)
  - out_root/all_band_powers.csv          (abs & rel par bande + total par patient)
  - out_root/<groupe>_band_power_abs.png  (barplot par label : bandes + total, en µV², une figure par groupe)
  - out_root/group_band_power_rel.png     (barplot global : bandes relatives)
  - out_root/group_means_abs.csv / group_means_rel.csv
  - out_root/patients_unknown_category.txt (si des bases n’ont pas de label)
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
from matplotlib.ticker import ScalarFormatter, FuncFormatter, FixedLocator

import mne
mne.set_config('MNE_MEMMAP_MIN_SIZE', '1M', set_env=True)

# ---------- Defaults ----------
REM_DIR_DEFAULT  = "/home/darryld/documents/EEG/preprocessed/bipolaire/2_rem_only"
OUT_ROOT_DEFAULT = "/home/darryld/documents/EEG/preprocessed/bipolaire/3_results_analysis"

BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta":  (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}
FMIN, FMAX = 0.5, 45.0   # bande globale (total)

# ---------- Utils ----------
def normalize_id(x: str) -> str:
    return "".join(ch for ch in str(x).strip().upper() if ch.isalnum())

def sanitize_name(x: str) -> str:
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

def make_patient_fig(base: str, freqs, psd_lin, out_png: Path, bands=BANDS):
    eps = np.finfo(float).tiny
    psd_db = 10.0 * np.log10(np.maximum(psd_lin, eps))  # dB re µV²/Hz

    band_list = list(bands.items())
    nrows, ncols = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 9))
    axes = axes.ravel()
    fig.suptitle(f"{base} — PSD REM — moy ± p10–p90 par bande", fontsize=14, y=0.98)

    for i, (name, (lo, hi)) in enumerate(band_list):
        ax = axes[i]
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if idx.size == 0:
            ax.set_visible(False); continue
        band_f = freqs[idx]
        band_db = psd_db[:, idx]
        m = np.nanmean(band_db, axis=0)
        p10 = np.nanpercentile(band_db, 10, axis=0)
        p90 = np.nanpercentile(band_db, 90, axis=0)

        ax.plot(band_f, m, lw=2)
        ax.fill_between(band_f, p10, p90, alpha=0.25)
        ax.set_xscale("log")

        # ticks entiers uniquement, sans 10^
        lo_i = max(1, int(np.ceil(lo)))            # pas d'entier < 1 sur axe log
        hi_i = int(np.floor(hi))
        ticks = list(range(lo_i, hi_i + 1))
        if ticks:
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.minorticks_off()

        ax.set_xlabel("Fréquence (Hz)")
        ax.set_ylabel("PSD (dB re µV²/Hz)")
        ax.set_title(f"{name}  [{lo:.1f}-{hi:.1f}] Hz")
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_formatter(ScalarFormatter())

    if len(band_list) < len(axes):
        axes[-1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

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
        freqs, psd_lin, ch_names = compute_psd(raw, fmin=FMIN, fmax=FMAX)
        total_abs, band_abs, band_rel = integrate_band_powers(freqs, psd_lin, BANDS)
    except Exception as e:
        return {"base": base_n, "ok": False, "reason": f"psd_error: {e}"}

    abs_mean = {k: float(np.nanmean(v)) for k, v in band_abs.items()}
    rel_mean = {k: float(np.nanmean(v)) for k, v in band_rel.items()}
    total_mean = float(np.nanmean(total_abs))

    try:
        make_patient_fig(base_n, freqs, psd_lin, out_dir / f"{base_n}_psd_bands.png", bands=BANDS)
    except Exception as e:
        print(f"[{base_n}] figure erreur: {e}")

    row = {
        "base": base_n,
        "total_abs": total_mean,
        **{f"abs_{b}": abs_mean[b] for b in BANDS},
        **{f"rel_{b}": rel_mean[b] for b in BANDS},
    }
    return {"base": base_n, "ok": True, "row": row}

# ---------- Plots groupes ----------
    # --- Couleurs fixes par catégorie (hex) ---
PALETTE = {
    "autoimmune encephalitides": "#C9D175",  # AI  (jaune-vert)
    "narcolepsy":                "#F15854",  # Narco (rouge)
    "synucleopathy":             "#44AA99",  # Syn (bleu-vert)
    "tcspi":                     "#BEBEBE",  # TCSP (gris)
    "unknown":                   "#000000",  # fallback
}

def color_for_group(label: str) -> str:
    return PALETTE.get(str(label).strip().lower(), PALETTE["unknown"])

def short_label(label: str) -> str:
    lab = str(label).strip().lower()
    return {
        "autoimmune encephalitides": "AI",
        "narcolepsy": "Narco",
        "synucleopathy": "Syn",
        "tcspi": "TCSP",
        "unknown": "Unknown",
    }.get(lab, label)


def grouped_barplot(df: pd.DataFrame, out_path: Path, title: str, ylabel: str):
    cats = list(df.index); bands = list(df.columns)
    x = np.arange(len(cats))
    width = 0.12 if len(bands) > 6 else 0.15
    fig = plt.figure(figsize=(max(8, 1.6*len(cats)), 5))
    for i, b in enumerate(bands):
        plt.bar(x + i*width, df[b].values, width=width, label=b)
    plt.xticks(x + width*(len(bands)-1)/2, cats, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=230)
    plt.close(fig)

def per_band_barplots(df_patients: pd.DataFrame, out_root: Path,
                      bands=BANDS, kind: str = "abs"):
    """
    Une figure par bande : comparaison inter-catégorie.
    Couleurs = palette fixe par groupe. Barres d'erreur = SEM.
    """
    assert kind in ("abs", "rel")
    ylabel = "PSD intégrée (µV²)" if kind == "abs" else "PSD relative"

    for band in bands.keys():
        col = f"{kind}_{band}"
        if col not in df_patients.columns:
            continue

        grp = df_patients.groupby("group")[col]
        mean = grp.mean().sort_index()
        cnt  = grp.count().reindex(mean.index)
        std  = grp.std(ddof=1).reindex(mean.index)
        sem  = (std / np.sqrt(cnt.clip(lower=1))).fillna(0.0)

        # couleurs par groupe
        colors = [color_for_group(g) for g in mean.index]
        xticks = [f"{short_label(g)} (n={int(cnt[g])})" for g in mean.index]

        x = np.arange(len(mean))
        fig = plt.figure(figsize=(max(7, 1.6*len(mean)), 4.6))
        plt.bar(x, mean.values, yerr=sem.values, capsize=3, color=colors)
        plt.xticks(x, xticks, rotation=30, ha="right")
        plt.ylabel(ylabel)
        plt.title(f"{band} - moyenne par catégorie")
        plt.tight_layout()
        out = out_root / f"perband_{kind}_{band}.png"
        fig.savefig(out, dpi=230)
        plt.close(fig)


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

    # Compile CSV
    rows, missing = [], []
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

    if not rows:
        print("Aucun résultat exploitable."); return

    df = pd.DataFrame(rows).set_index("base").sort_index()
    csv_all = out_root / "all_band_powers.csv"
    df.to_csv(csv_all, float_format="%.8e")
    print(f"→ {csv_all}")

    # Agrégats par label
    band_names = list(BANDS.keys())
    abs_cols =[f"abs_{b}" for b in band_names]
    rel_cols = [f"rel_{b}" for b in band_names]

    g_abs = df.groupby("group")[abs_cols].mean().loc[:, abs_cols]
    g_rel = df.groupby("group")[rel_cols].mean().loc[:, rel_cols]
    g_abs.to_csv(out_root / "group_means_abs.csv", float_format="%.8e")
    g_rel.to_csv(out_root / "group_means_rel.csv", float_format="%.8e")

    # --- une figure par groupe pour l'ABSOLU ---
    group_counts = df.groupby("group").size().to_dict()
    for grp in g_abs.index:
        vals = [g_abs.loc[grp, f"abs_{b}"] for b in band_names]
        names = band_names
        n = int(group_counts.get(grp, 0))
        fig = plt.figure(figsize=(7, 4))
        plt.bar(names, vals)
        plt.ylabel("PSD absolue (µV²)")
        plt.title(f"Band power (absolute, µV²) - {grp} (n={n})")
        plt.tight_layout()
        out_png = out_root / f"{sanitize_name(grp)}_band_power_abs.png"
        fig.savefig(out_png, dpi=230)
        plt.close(fig)

    # On conserve le barplot global pour les RELATIVES
    g_rel_plot = g_rel.copy(); g_rel_plot.columns = band_names
    grouped_barplot(g_rel_plot, out_root / "group_band_power_rel.png",
                    "Band power (relative) - mean per category",
                    ylabel="PSD relative")
    
    # --- barplots *par bande* (comparaison inter-catégorie) ---
    per_band_barplots(df, out_root, bands=BANDS, kind="abs")  # fichiers perband_abs_<Band>.png
    per_band_barplots(df, out_root, bands=BANDS, kind="rel")  # fichiers perband_rel_<Band>.png


    if missing:
        with open(out_root / "patients_unknown_category.txt", "w") as f:
            for b in sorted(set(missing)): f.write(f"{b}\n")
        print(f"[INFO] {len(set(missing))} patient(s) sans catégorie connue -> patients_unknown_category.txt")

if __name__ == "__main__":
    main()
