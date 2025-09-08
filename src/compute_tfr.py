#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrées:
  - REM:  /home/darryld/documents/EEG/preprocessed/bipolaire/2_rem_only
  - N2/N3 (full clean concat): /home/darryld/documents/EEG/preprocessed/bipolaire/1_noArtefacts
  - Hypnogrammes .txt/.csv (timeline ORIGINE): /home/darryld/documents/EEG/raw
  - (optionnel) CSV artéfacts pour mapping (timeline ORIGINE -> timeline CLEAN):
        {base}_artifact_windows.csv (colonnes: start_time_s,end_time_s)

Sorties:
  - out_root/{base}/{base}_{canal}_tfr_REM.png
  - out_root/{base}/{base}_{canal}_tfr_N2.png
  - out_root/{base}/{base}_{canal}_tfr_N3.png
"""

from __future__ import annotations

# --------- Limites BLAS & multiprocess sûrs ---------
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import multiprocessing as mp
import faulthandler; faulthandler.enable()

# --------- Imports ----------
from pathlib import Path
import argparse
import tempfile
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import mne
mne.set_config('MNE_MEMMAP_MIN_SIZE', '1M', set_env=True)

# --------- Dossiers par défaut ----------
DEFAULT_REM_DIR   = "/home/darryld/documents/EEG/preprocessed/bipolaire/2_rem_only"
DEFAULT_CLEAN_DIR = "/home/darryld/documents/EEG/preprocessed/bipolaire/1_noArtefacts"
DEFAULT_ANNOT_DIR = "/home/darryld/documents/EEG/raw"
DEFAULT_OUT_ROOT  = "/home/darryld/documents/EEG/preprocessed/bipolaire/3_results_analysis"

# --------- TFR / Epochs ----------
FREQ_MIN    = 1.0
FREQ_MAX    = 40.0
N_FREQS     = 30
CYCLES_MULT = 0.5
EPOCH_DUR   = 4.0
DECIM       = 2
CMAP        = "jet"
VMIN, VMAX  = None, None          # autoscale si None
AUTO_PCT    = (5, 95)

# Seuils de durée pour N2/N3
MIN_SEG_S   = EPOCH_DUR
MIN_TOTAL_S = 2 * EPOCH_DUR

# --------- Globaux dans workers ----------
REM_DIR: Path | None   = None
CLEAN_DIR: Path | None = None
ANNOT_DIR: Path | None = None
OUT_ROOT: Path | None  = None

# ======================== Hypnogrammes (txt/csv) ========================
def _read_hypnogram_any(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".txt":
        df = pd.read_csv(path, sep="\t",
                         names=["start", "time", "stage", "index"],
                         engine="python")
        df = df.dropna(subset=["start", "stage"])
        df["start"] = df["start"].astype(float)
        stage = df["stage"]
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, sep=";", engine="python")
        cols = {c.lower().strip(): c for c in df.columns}
        if "position (epoch)" in cols:
            df["start"] = (df[cols["position (epoch)"]].astype(int) - 1) * 30.0
        elif "epoch" in cols:
            df["start"] = (df[cols["epoch"]].astype(int) - 1) * 30.0
        elif "absolute position (hh:mm:ss.ms)" in cols:
            t0 = pd.to_timedelta(df[cols["absolute position (hh:mm:ss.ms)"]].iloc[0])
            df["start"] = (pd.to_timedelta(df[cols["absolute position (hh:mm:ss.ms)"]]) - t0).dt.total_seconds()
        else:
            raise ValueError("CSV hypnogram: colonne epoch/time manquante.")
        stage_col = (cols.get('default staging set ("stage")')
                     or cols.get("stage") or cols.get("stade"))
        stage = df[stage_col]
    else:
        raise ValueError(f"Extension non gérée: {path.suffix}")

    def norm(s: str) -> str:
        s = (str(s) or "").strip().upper()
        map_ = {
            "SP": "REM", "R": "REM", "REM": "REM",
            "V": "W", "WAKE": "W", "W": "W",
            "S2": "N2", "STAGE2": "N2", "NREM2": "N2",
            "S3": "N3", "STAGE3": "N3", "NREM3": "N3",
        }
        return map_.get(s, s)

    df = df.assign(stage=stage.map(norm)).sort_values("start")
    df["duration"] = df["start"].shift(-1) - df["start"]
    df = df.iloc[:-1]
    return df[["start", "duration", "stage"]]

def get_stage_annotations(base_name: str, annot_dir: Path, stages=("N2", "N3")) -> mne.Annotations | None:
    want = {s.upper() for s in stages}
    pdir = annot_dir / base_name.split("_")[0]
    for p in list(pdir.glob("*.txt")) + list(pdir.glob("*.csv")):
        try:
            df = _read_hypnogram_any(p)
            keep = df[df["stage"].isin(want)]
            if len(keep) == 0:
                continue
            return mne.Annotations(
                onset=keep["start"].astype(float).tolist(),
                duration=keep["duration"].astype(float).tolist(),
                description=keep["stage"].tolist(),
            )
        except Exception:
            continue
    return None

# ======================== Découverte des patients ========================
def _bases_from_dir(d: Path) -> list[str]:
    bases = set()
    # sous-dossiers
    for child in d.iterdir():
        if child.is_dir() and not child.name.startswith("._"):
            bases.add(child.name)
    # fichiers à la racine
    for p in d.glob("*.fif"):
        if p.is_file() and not p.name.startswith("._"):
            bases.add(p.stem.split("_")[0])
    return sorted(bases)

def discover_patients(rem_dir: Path, clean_dir: Path) -> list[str]:
    return sorted(set(_bases_from_dir(rem_dir)) | set(_bases_from_dir(clean_dir)))

def find_rem_fifs(base: str, rem_dir: Path) -> list[Path]:
    """Retourne tous les FIF REM trouvés pour concat éventuelle."""
    files = []
    pdir = rem_dir / base
    if pdir.exists():
        files += [p for p in pdir.glob("*.fif") if p.is_file() and not p.name.startswith("._")]
    files += [p for p in rem_dir.glob(f"{base}*REM*.fif") if p.is_file() and not p.name.startswith("._")]
    # fallback: tout .fif commençant par base
    if not files:
        files += [p for p in rem_dir.glob(f"{base}*.fif") if p.is_file() and not p.name.startswith("._")]
    return sorted(files)

def find_clean_fif(base: str, clean_dir: Path) -> Path | None:
    """Trouve le FIF 'full clean' concaténé (sans artéfacts) pour base."""
    cand = []
    pdir = clean_dir / base
    if pdir.exists():
        cand += [p for p in pdir.glob("*.fif") if p.is_file() and not p.name.startswith("._")]
    if not cand:
        cand += [p for p in clean_dir.glob(f"{base}*.fif") if p.is_file() and not p.name.startswith("._")]
    return max(cand, key=lambda p: p.stat().st_size) if cand else None

# ======================== Mapping ORIG -> CLEAN via CSV artéfacts ========================
def _merge_intervals(intervals):
    if not intervals:
        return []
    ints = sorted(intervals, key=lambda x: x[0])
    merged = [ints[0]]
    for s, e in ints[1:]:
        s0, e0 = merged[-1]
        if s <= e0: merged[-1] = (s0, max(e0, e))
        else:       merged.append((s, e))
    return merged

def _load_artifact_windows_csv(base: str, start_dir: Path) -> list[tuple[float, float]]:
    """
    Cherche un CSV *artifact_windows.csv* dans start_dir/{base}/ puis récursivement.
    Format attendu: colonnes start_time_s,end_time_s.
    """
    candidates = []
    pdir = start_dir / base
    if pdir.exists():
        candidates += list(pdir.glob(f"*artifact_windows.csv"))
    candidates += list(start_dir.glob(f"{base}*artifact_windows.csv"))
    if not candidates:
        candidates = list(start_dir.rglob(f"{base}*artifact_windows.csv"))
    for csvp in candidates:
        try:
            df = pd.read_csv(csvp)
            if {"start_time_s", "end_time_s"} <= set(map(str.lower, df.columns.str.lower())):
                cols = {c.lower(): c for c in df.columns}
                st = df[cols["start_time_s"]].astype(float).to_numpy()
                en = df[cols["end_time_s"]].astype(float).to_numpy()
            else:
                st = df.iloc[:, 0].astype(float).to_numpy()
                en = df.iloc[:, 1].astype(float).to_numpy()
            intervals = [(float(s), float(e)) for s, e in zip(st, en) if e > s]
            return _merge_intervals(intervals)
        except Exception:
            continue
    return []

def _orig2clean_mapping_from_artifacts(bad_orig: list[tuple[float, float]], t_end_orig: float):
    """Construit la table de mapping (gs, ge, c0) à partir des intervalles BAD (timeline ORIGINE)."""
    if not bad_orig:
        return [(0.0, float(t_end_orig), 0.0)]
    bad = _merge_intervals(bad_orig)
    good = []
    t0 = 0.0
    for (bs, be) in bad:
        if bs > t0: good.append((t0, bs))
        t0 = max(t0, be)
    if t_end_orig > t0: good.append((t0, float(t_end_orig)))
    mapping, c0 = [], 0.0
    for (s, e) in good:
        mapping.append((float(s), float(e), float(c0)))
        c0 += (e - s)
    return mapping

def map_intervals_to_clean(intervals, mapping):
    """Mappe (s,e) ORIG -> CLEAN avec éventuel morcellement."""
    out = []
    for (s, e) in intervals:
        if e <= s: continue
        for (gs, ge, c0) in mapping:
            a = max(s, gs); b = min(e, ge)
            if b > a:
                out.append((c0 + (a - gs), c0 + (b - gs)))
    return out

# ======================== TFR utils ========================
def _format_time_axes(figs):
    for f in figs:
        for ax in f.axes:
            xlabel = (ax.get_xlabel() or "").lower()
            if "time" in xlabel:
                ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
                ax.ticklabel_format(axis="x", style="plain", useOffset=False)
                ax.set_xlabel("Time (s)")

def _sanitize(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in ("_", "-")).replace(" ", "")

def _plot_and_save_power(power, ch, base, stage, out_png, vmin_eff, vmax_eff, n_epochs):
    try:
        fig = power.plot(picks=[ch], dB=True, cmap=str(CMAP),
                         vmin=vmin_eff, vmax=vmax_eff,
                         baseline=None, show=False)
    except TypeError:
        fig = power.plot(picks=[ch], dB=True, cmap=str(CMAP),
                         baseline=None, show=False)
        figs_tmp = fig if isinstance(fig, (list, tuple)) else [fig]
        for f in figs_tmp:
            for ax in f.axes:
                artists = list(ax.images) + [c for c in ax.collections if hasattr(c, "set_clim")]
                for art in artists:
                    try: art.set_clim(vmin_eff, vmax_eff)
                    except Exception: pass

    figs = fig if isinstance(fig, (list, tuple)) else [fig]
    _format_time_axes(figs)

    title = f"{base} — {ch} — {stage}  (n_epochs={n_epochs})"
    for f in figs:
        try: f.set_constrained_layout(False)
        except Exception: pass
        f.suptitle(title, y=1.02)
        try: f.tight_layout(rect=[0, 0, 1, 0.96])
        except Exception: f.subplots_adjust(top=0.90)

    ax0 = figs[0].axes[0] if figs and figs[0].axes else None
    is_blank = (ax0 is None) or (len(ax0.images) == 0 and len(ax0.collections) == 0)
    if is_blank:
        print(f"[{base}:{stage}:{ch}] figure vide -> skip")
        try:
            for f in figs: plt.close(f)
        except Exception: pass
        return False

    if isinstance(fig, (list, tuple)):
        fig = fig[0]
    try:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[{base}:{stage}:{ch}] [ok] {out_png.name}")
    except Exception as e:
        print(f"[{base}:{stage}:{ch}] Save figure erreur: {e}")
    finally:
        plt.close(fig)
    return True

# ======================== Pipelines ========================
def pipeline_rem(base: str, rem_dir: Path, out_root: Path):
    rem_files = find_rem_fifs(base, rem_dir)
    if not rem_files:
        print(f"[{base}] REM: aucun FIF -> skip"); return

    raws = []
    for p in rem_files:
        try:
            raws.append(mne.io.read_raw_fif(p, preload=True, verbose="ERROR"))
        except Exception as e:
            print(f"[{base}] REM: lecture {p.name} échouée ({e})")
    if not raws:
        print(f"[{base}] REM: aucune donnée chargeable -> skip"); return

    raw = raws[0] if len(raws) == 1 else mne.concatenate_raws(raws, verbose="ERROR")

    # EEG uniquement
    try:
        raw.pick_types(meg=False, eeg=True, eog=False, ecg=False, emg=False, stim=False,
                       misc=False, resp=False, seeg=False, ecog=False, fnirs=False)
        if len(raw.ch_names) == 0:
            print(f"[{base}] REM: aucun canal EEG -> skip"); return
    except Exception as e:
        print(f"[{base}] REM: pick_types EEG échoué ({e})"); return

    # µV
    try:
        raw.load_data()
        eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, emg=False)
        raw.apply_function(lambda x: x * 1e6, picks=eeg_picks, channel_wise=True)
        if hasattr(raw, "set_unit"):
            try: raw.set_unit("eeg", "uV")
            except Exception: pass
    except Exception as e:
        print(f"[{base}] REM: scaling µV échoué ({e})"); return

    # Epochs
    try:
        epochs = mne.make_fixed_length_epochs(raw, duration=float(EPOCH_DUR),
                                              overlap=0.0, preload=True, verbose="ERROR")
        epochs.pick_types(meg=False, eeg=True, eog=False, ecg=False, emg=False, stim=False, misc=False)
    except Exception as e:
        print(f"[{base}] REM: création epochs échouée ({e})"); return

    freqs = np.linspace(float(FREQ_MIN), float(FREQ_MAX), int(N_FREQS))
    n_cycles = freqs * float(CYCLES_MULT)
    n_epochs = len(epochs)

    out_dir = out_root / base
    out_dir.mkdir(parents=True, exist_ok=True)

    for ch in epochs.ch_names:
        out_png = out_dir / f"{base}_{_sanitize(ch)}_tfr_REM.png"
        # --- SKIP si déjà produit ---
        if out_png.exists():
            print(f"[{base}:REM:{ch}] déjà présent -> skip")
            continue
        try:
            power = mne.time_frequency.tfr_morlet(
                epochs, freqs=freqs, n_cycles=n_cycles,
                use_fft=True, return_itc=False, average=True,
                picks=[ch], decim=int(DECIM), verbose="ERROR"
            )
            Z = 10.0 * np.log10(np.maximum(power.data[0], np.finfo(float).tiny))
            if VMIN is None or VMAX is None:
                lo, hi = np.percentile(Z, list(AUTO_PCT))
                vmin_eff, vmax_eff = float(lo), float(hi)
            else:
                vmin_eff, vmax_eff = float(VMIN), float(VMAX)
            _plot_and_save_power(power, ch, base, "REM", out_png, vmin_eff, vmax_eff, n_epochs)
        except Exception as e:
            print(f"[{base}:REM:{ch}] TFR erreur: {e}")

def pipeline_n2n3(base: str, clean_dir: Path, annot_dir: Path, out_root: Path):
    fif = find_clean_fif(base, clean_dir)
    if fif is None:
        print(f"[{base}] N2/N3: clean FIF introuvable -> skip"); return
    try:
        raw_full = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
    except Exception as e:
        print(f"[{base}] N2/N3: lecture échouée ({e})"); return

    # Hypnogramme (timeline ORIG)
    stage_ann = get_stage_annotations(base, annot_dir, stages=("N2", "N3"))
    if stage_ann is None or len(stage_ann) == 0:
        print(f"[{base}] N2/N3: aucune annotation -> skip"); return

    # Mapping ORIG -> CLEAN via CSV artéfacts si dispo
    bad_orig = _load_artifact_windows_csv(base, clean_dir)
    if bad_orig:
        mapping = _orig2clean_mapping_from_artifacts(bad_orig, t_end_orig=float(bad_orig[-1][1]) + 1e9)  # t_end approx
    else:
        mapping = None
        print(f"[{base}] WARNING: pas de CSV *_artifact_windows.csv* trouvé — "
              f"crop direct sur timeline CLEAN (supposée non-concat), sinon résultats faux.")

    sf = float(raw_full.info["sfreq"])
    t_end = float(raw_full.times[-1])
    eps   = 1.0 / sf

    by_stage = {"N2": [], "N3": []}
    for onset, dur, desc in zip(stage_ann.onset, stage_ann.duration, stage_ann.description):
        s0 = float(onset); e0 = s0 + float(dur)
        tag = (str(desc) or "").upper()
        if e0 > s0 and tag in by_stage:
            by_stage[tag].append((s0, e0))

    out_dir = out_root / base
    out_dir.mkdir(parents=True, exist_ok=True)

    for stage in ("N2", "N3"):
        intervals_orig = by_stage.get(stage, [])
        if not intervals_orig:
            print(f"[{base}:{stage}] aucun intervalle -> skip"); continue

        if mapping:
            intervals_clean = map_intervals_to_clean(intervals_orig, mapping)
        else:
            # fallback naïf: clip sur la timeline clean
            intervals_clean = [(max(0.0, s), min(e, t_end)) for (s, e) in intervals_orig if min(e, t_end) > max(0.0, s)]

        # filtre durées
        kept = [(s, e) for (s, e) in intervals_clean if (e - s) >= MIN_SEG_S]
        if not kept or sum(e - s for (s, e) in kept) < MIN_TOTAL_S:
            print(f"[{base}:{stage}] durée insuffisante (< {MIN_TOTAL_S:.2f}s) -> skip"); continue

        # Concat depuis le clean
        parts = []
        for (s, e) in kept:
            try:
                seg = raw_full.copy().crop(tmin=s, tmax=e - eps, verbose="ERROR")
                parts.append(seg)
            except Exception as ex:
                print(f"[{base}:{stage}] crop {s:.2f}-{e:.2f} échoué: {ex}")
        if not parts:
            print(f"[{base}:{stage}] rien à concaténer -> skip"); continue
        stage_raw = mne.concatenate_raws(parts, verbose="ERROR")

        # EEG uniquement + µV
        try:
            stage_raw.pick_types(meg=False, eeg=True, eog=False, ecg=False, emg=False, stim=False,
                                 misc=False, resp=False, seeg=False, ecog=False, fnirs=False)
            if len(stage_raw.ch_names) == 0:
                print(f"[{base}:{stage}] aucun EEG -> skip"); continue
        except Exception as e:
            print(f"[{base}:{stage}] pick EEG échoué ({e})"); continue

        try:
            stage_raw.load_data()
            eeg_picks = mne.pick_types(stage_raw.info, eeg=True, meg=False, eog=False, ecg=False, emg=False)
            stage_raw.apply_function(lambda x: x * 1e6, picks=eeg_picks, channel_wise=True)
            if hasattr(stage_raw, "set_unit"):
                try: stage_raw.set_unit("eeg", "uV")
                except Exception: pass
        except Exception as e:
            print(f"[{base}:{stage}] scaling µV échoué ({e})"); continue

        # Epochs + TFR
        try:
            epochs = mne.make_fixed_length_epochs(stage_raw, duration=float(EPOCH_DUR),
                                                  overlap=0.0, preload=True, verbose="ERROR")
            epochs.pick_types(meg=False, eeg=True, eog=False, ecg=False, emg=False, stim=False, misc=False)
        except Exception as e:
            print(f"[{base}:{stage}] epochs échoué ({e})"); continue

        freqs = np.linspace(float(FREQ_MIN), float(FREQ_MAX), int(N_FREQS))
        n_cycles = freqs * float(CYCLES_MULT)
        n_epochs = len(epochs)

        for ch in epochs.ch_names:
            out_png = out_dir / f"{base}_{_sanitize(ch)}_tfr_{stage}.png"
            # --- SKIP si déjà produit ---
            if out_png.exists():
                print(f"[{base}:{stage}:{ch}] déjà présent -> skip")
                continue
            try:
                power = mne.time_frequency.tfr_morlet(
                    epochs, freqs=freqs, n_cycles=n_cycles,
                    use_fft=True, return_itc=False, average=True,
                    picks=[ch], decim=int(DECIM), verbose="ERROR"
                )
                Z = 10.0 * np.log10(np.maximum(power.data[0], np.finfo(float).tiny))
                if VMIN is None or VMAX is None:
                    lo, hi = np.percentile(Z, list(AUTO_PCT))
                    vmin_eff, vmax_eff = float(lo), float(hi)
                else:
                    vmin_eff, vmax_eff = float(VMIN), float(VMAX)
                _plot_and_save_power(power, ch, base, stage, out_png, vmin_eff, vmax_eff, n_epochs)
            except Exception as e:
                print(f"[{base}:{stage}:{ch}] TFR erreur: {e}")

# ======================== Pool / Main ========================
def _worker_wrapper(base: str):
    # cache Matplotlib isolé
    try:
        mpl_cache = os.path.join(os.getenv("TMPDIR") or "/tmp", f"mplcache_{os.getpid()}")
        os.environ["MPLCONFIGDIR"] = mpl_cache
        os.makedirs(mpl_cache, exist_ok=True)
    except Exception:
        pass
    try:
        # REM
        pipeline_rem(base, REM_DIR, OUT_ROOT)
        # N2 / N3
        pipeline_n2n3(base, CLEAN_DIR, ANNOT_DIR, OUT_ROOT)
    except Exception as e:
        import traceback
        print(f"[{base}] Worker error: {e}\n{traceback.format_exc()}")

def _init_worker(gl_rem_dir: str, gl_clean_dir: str, gl_annot_dir: str, gl_out_root: str):
    global REM_DIR, CLEAN_DIR, ANNOT_DIR, OUT_ROOT
    REM_DIR   = Path(gl_rem_dir)
    CLEAN_DIR = Path(gl_clean_dir)
    ANNOT_DIR = Path(gl_annot_dir)
    OUT_ROOT  = Path(gl_out_root)
    # isole le cache MPL
    try:
        mpl_cache = os.path.join(tempfile.gettempdir(), f"mplcache_{os.getpid()}")
        os.environ["MPLCONFIGDIR"] = mpl_cache
        os.makedirs(mpl_cache, exist_ok=True)
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rem-dir",   type=str, default=DEFAULT_REM_DIR)
    parser.add_argument("--clean-dir", type=str, default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--annot-dir", type=str, default=DEFAULT_ANNOT_DIR)
    parser.add_argument("--out-root",  type=str, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--workers",   type=int, default=8, help="0 => CPU-1")
    parser.add_argument("--patients",  type=str, nargs="*", default=None, help="Liste de bases (sinon auto-discovery).")
    args = parser.parse_args()

    rem_dir   = Path(args.rem_dir)
    clean_dir = Path(args.clean_dir)
    annot_dir = Path(args.annot_dir)
    out_root  = Path(args.out_root)
    for d in [rem_dir, clean_dir, annot_dir]:
        if not d.exists():
            raise SystemExit(f"[CONFIG] Dossier introuvable: {d}")
    out_root.mkdir(parents=True, exist_ok=True)

    # patients
    if args.patients:
        bases = sorted(set(args.patients))
    else:
        bases = discover_patients(rem_dir, clean_dir)
    if not bases:
        raise SystemExit("Aucun patient détecté.")
    print(f"Patients ({len(bases)}): {bases}")

    # pool
    cpu = os.cpu_count() or 1
    n_workers = (cpu-1 if args.workers in (0, None) else args.workers)
    n_workers = max(1, min(n_workers, len(bases)))
    print(f"[INFO] Lancement en parallèle avec {n_workers} worker(s) (spawn, maxtasksperchild=1)")

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=n_workers,
        maxtasksperchild=1,
        initializer=_init_worker,
        initargs=(str(rem_dir), str(clean_dir), str(annot_dir), str(out_root)),
    ) as pool:
        for _ in pool.imap_unordered(_worker_wrapper, bases, chunksize=1):
            pass

if __name__ == "__main__":
    main()
