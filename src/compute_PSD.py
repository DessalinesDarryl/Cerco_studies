#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calcul de la puissance spectrale par patient (REM-focus par défaut, bandes depuis src/filters.py)
---------------------------------------------------------
- Recherche récursive des .fif prétraités (un par patient).
- Mode full/rem/auto (REM = concat des segments rem_only/<patient>/...).
- PSD (Welch) + puissances par bandes + TFR Morlet par canal.
- Sauvegardes par patient dans results/{patient}/.
- Faible RAM : PSD memmap sur disque (--tmp-dir), TFR canal-par-canal + décimation.
- Les signaux EEG sont **toujours** convertis en microvolts (µV). Les PSD sont donc en µV²/Hz.

Exemple:
  python compute_spectral_power.py --mode rem \
    --fif-dir "/home/darryld/documents/EEG/preprocessed/bipolaire/full" \
    --rem-dir "/home/darryld/documents/EEG/preprocessed/bipolaire/rem_only" \
    --results-dir "/home/darryld/documents/EEG/preprocessed/bipolaire/results" \
    --suffix bip --n-jobs 4 --tmp-dir /var/tmp/eeg_tmp
"""
from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mne

# Matplotlib (backend non interactif => OK en multiprocessing/headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shutil

# -------------------- Utilitaires maison (si présents) --------------------
try:
    from utils import list_fif_files as utils_list_fif_files, get_base_name as utils_get_base_name  # src/utils.py
    HAVE_UTILS = True
except Exception:
    HAVE_UTILS = False

# -------------------- Config par défaut --------------------
DEFAULT_FIF_DIR = "/home/darryld/documents/EEG/preprocessed/bipolaire/full"
DEFAULT_REM_DIR = "/home/darryld/documents/EEG/preprocessed/bipolaire/rem_only"
DEFAULT_RESULTS_DIR = "/home/darryld/documents/EEG/preprocessed/bipolaire/results"
DEFAULT_TMP_DIR = "/var/tmp/eeg_tmp"  # sur /dev/sda2

# Bandes: on essaie d'utiliser src/filters.get_standard_bands() si disponible
try:
    from filters import get_standard_bands as _get_bands_from_filters  # src/filters.py
    _BANDS_TMP = _get_bands_from_filters()
    STANDARD_BANDS: Dict[str, Tuple[float, float]] = {
        str(k): (float(v[0]), float(v[1])) for k, v in _BANDS_TMP.items()
    }
except Exception:
    STANDARD_BANDS: Dict[str, Tuple[float, float]] = {
        "Delta": (0.5, 4.0),
        "Theta": (4.0, 8.0),
        "Alpha": (8.0, 13.0),
        "Beta": (13.0, 30.0),
        "Gamma": (30.0, 45.0),
    }

# -------------------- Utilitaires fichiers --------------------
def list_patients_fifs(fif_dir: Path, suffix: str) -> List[Path]:
    """Recherche les fichiers patients *_preprocessed_<suffix>.fif (récursif)."""
    if HAVE_UTILS:
        rels = utils_list_fif_files(str(fif_dir))  # chemins relatifs
        selected = []
        for rel in rels:
            name = os.path.basename(rel)
            if name.endswith(f"_preprocessed_{suffix}.fif") and not name.startswith("._"):
                selected.append(fif_dir / rel)
        return sorted(map(Path, selected))
    else:
        return sorted(
            p for p in fif_dir.rglob(f"*_preprocessed_{suffix}.fif")
            if p.suffix == ".fif" and not p.name.startswith("._")
        )

def base_from_fif(fif_path: Path, suffix: str) -> str:
    """Nom patient à partir du fichier *_preprocessed_<suffix>.fif."""
    if HAVE_UTILS:
        return utils_get_base_name(fif_path.name, suffix=f"_preprocessed_{suffix}.fif")
    return fif_path.stem.split('_')[0]

def find_rem_segments(rem_root: Path, base_name: str) -> List[Path]:
    """Liste des segments REM .fif d'un patient (dossier rem_only/<base>/)."""
    cand_dir = rem_root / base_name
    if not cand_dir.exists():
        return []
    return sorted([p for p in cand_dir.glob(f"{base_name}_REM_*.fif") if p.is_file()])

# -------------------- Band helpers --------------------
def adapt_bands_to_passband(bands: Dict[str, Tuple[float, float]], highpass: float, lowpass: float) -> Dict[str, Tuple[float, float]]:
    """Intersection des bandes EEG avec la passband effective [highpass, lowpass]."""
    out = {}
    hp = max(0.0, float(highpass or 0.0))
    lp = float(lowpass or np.inf)
    for name, (f1, f2) in bands.items():
        lo = max(f1, hp)
        hi = min(f2, lp)
        if hi > lo:
            out[name] = (lo, hi)
    return out

def compute_band_powers(freqs: np.ndarray, psd: np.ndarray, bands: Dict[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
    """Puissance absolue par bande pour chaque canal à partir d'une PSD (µV²/Hz)."""
    band_powers = {}
    for name, (fmin, fmax) in bands.items():
        idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
        if idx.size == 0:
            band_powers[name] = np.zeros(psd.shape[0])
            continue
        band_powers[name] = np.trapz(psd[:, idx], freqs[idx], axis=1)
    return band_powers

# -------------------- Traitement patient --------------------
@dataclass
class PatientResult:
    base: str
    mode_used: str
    csv_path: Path
    fig_band_path: Path
    fig_psd_path: Optional[Path]

def load_raw_for_mode(fif_path: Path, mode: str, rem_root: Path, suffix: str) -> Tuple[mne.io.BaseRaw, str]:
    """Charge le Raw selon le mode ('full' ou 'rem'). Retourne (raw, mode_effectif)."""
    base = base_from_fif(fif_path, suffix)
    if mode == "rem":
        segs = find_rem_segments(rem_root, base)
        if not segs:
            raise FileNotFoundError(f"Aucun segment REM trouvé pour {base} dans {rem_root}")
        raws = [mne.io.read_raw_fif(p, preload=True, verbose="ERROR") for p in segs]
        raw = mne.concatenate_raws(raws, verbose=False)
        return raw, "rem"
    elif mode == "auto":
        segs = find_rem_segments(rem_root, base)
        if segs:
            raws = [mne.io.read_raw_fif(p, preload=True, verbose="ERROR") for p in segs]
            raw = mne.concatenate_raws(raws, verbose=False)
            return raw, "rem"
        else:
            raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
            return raw, "full"
    else:
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
        return raw, "full"

def process_one_patient(
    fif_path: Path,
    rem_root: Path,
    results_dir: Path,
    bands: Dict[str, Tuple[float, float]],
    fmin: float,
    fmax: float,
    make_psd_plot: bool = True,
    mode: str = "rem",
    suffix: str = "bip",
    # TFR params
    tfr_enabled: bool = True,
    tfr_fmin: Optional[float] = None,
    tfr_fmax: Optional[float] = None,
    tfr_n_freqs: int = 60,
    tfr_cycles_mult: float = 0.3,
    tfr_vmin: Optional[float] = None,
    tfr_vmax: Optional[float] = None,
    tfr_cmap: str = "jet",
    tfr_decim: int = 1,
    overwrite_figs: bool = False,
    # tmp / memmap
    tmp_dir: Optional[Path] = None,
) -> PatientResult:
    base = base_from_fif(fif_path, suffix)

    # 1) Charger les données selon le mode
    raw, mode_used = load_raw_for_mode(fif_path, mode, rem_root, suffix)

    # 2) Ne garder que l'EEG
    raw.pick_types(eeg=True, exclude="bads")

    # --- Conversion systématique en µV ---
    raw.apply_function(lambda x: x * 1e6, picks="eeg", channel_wise=True)
    try:
        raw.set_unit("eeg", "uV")  # selon version MNE
    except Exception:
        pass

    # 3) Bande passante réelle & adaptation des bandes
    hp = raw.info.get("highpass", 0.0)
    lp = raw.info.get("lowpass", np.inf)
    fmin_eff = max(fmin, hp or 0.0)
    fmax_eff = min(fmax, lp or np.inf)
    if not math.isfinite(fmax_eff):
        fmax_eff = fmax
    bands_eff = adapt_bands_to_passband(bands, highpass=fmin_eff, lowpass=fmax_eff)
    if not bands_eff:
        raise RuntimeError(f"Aucune bande valide après adaptation (hp={hp}, lp={lp}, fmin={fmin}, fmax={fmax}).")

    # Répertoires de sortie
    patient_dir = results_dir / base
    patient_dir.mkdir(parents=True, exist_ok=True)

    # 4) PSD (Welch) — bas RAM via memmap disque + sécurisation n_per_seg/n_overlap
    sfreq = float(raw.info["sfreq"])
    n_times = int(raw.n_times)
    n_per_seg = max(2, min(int(sfreq * 4), n_times))         # ~4 s mais borné
    n_overlap = max(0, min(n_per_seg // 2, n_per_seg - 1))   # < n_per_seg

    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    ch_names = [raw.info["ch_names"][pi] for pi in picks]

    # PSD du premier canal pour récupérer la grille de fréquences
    psd0 = raw.compute_psd(
        fmin=fmin_eff, fmax=fmax_eff, method="welch",
        n_fft=None, n_overlap=n_overlap, n_per_seg=n_per_seg,
        picks=[picks[0]], verbose="ERROR"
    )
    freqs = psd0.freqs
    n_freqs = len(freqs)

    # Memmap PSD (n_channels, n_freqs)
    tmp_store = tmp_dir or patient_dir
    mm_path = tmp_store / f"{base}_psd_mm.dat"
    psd_mm = np.memmap(mm_path, dtype="float32", mode="w+", shape=(len(picks), n_freqs))
    psd_mm[0, :] = psd0.get_data()[0].astype("float32")

    for i, pi in enumerate(picks[1:], start=1):
        psd_i = raw.compute_psd(
            fmin=fmin_eff, fmax=fmax_eff, method="welch",
            n_fft=None, n_overlap=n_overlap, n_per_seg=n_per_seg,
            picks=[pi], verbose="ERROR"
        ).get_data()[0]
        psd_mm[i, :] = psd_i.astype("float32")
    psd_mm.flush()
    psd = np.asarray(psd_mm)  # vue memmap (pas de copie pleine RAM)

    # 5) Puissances par bande (absolues, µV²)
    band_powers_abs = compute_band_powers(freqs, psd, bands_eff)

    # 6) Puissance totale (pour relatives)
    idx_all = np.where((freqs >= fmin_eff) & (freqs <= fmax_eff))[0]
    total_power = np.trapz(psd[:, idx_all], freqs[idx_all], axis=1).astype(float)
    total_power[total_power == 0] = np.finfo(float).eps

    # 7) Relatives
    band_powers_rel = {k: v / total_power for k, v in band_powers_abs.items()}

    # 8) DataFrame par canal
    df_abs = pd.DataFrame({f"abs_{k}": v for k, v in band_powers_abs.items()}, index=ch_names)
    df_rel = pd.DataFrame({f"rel_{k}": v for k, v in band_powers_rel.items()}, index=ch_names)
    df = pd.concat([df_abs, df_rel], axis=1)
    df["total_power"] = total_power
    df.index.name = "channel"

    # 9) Moyennes patient
    means = pd.DataFrame(df.mean(axis=0)).T
    means.index = ["mean_over_channels"]
    df_out = pd.concat([df, means])

    # 10) Sauvegardes CSV / figures bande & PSD
    csv_path = patient_dir / f"{base}_band_power_{mode_used}.csv"
    df_out.to_csv(csv_path, float_format="%.8e")

    rel_means = {k: float(np.mean(band_powers_rel[k])) for k in bands_eff.keys()}
    fig1 = plt.figure(figsize=(7, 4))
    plt.bar(list(rel_means.keys()), list(rel_means.values()))
    plt.ylabel("Puissance relative")
    plt.xlabel("Bande (Hz)")
    plt.title(f"{base} — {mode_used.upper()} — Puissance relative par bande")
    plt.tight_layout()
    fig_band_path = patient_dir / f"{base}_band_power_{mode_used}.png"
    fig1.savefig(fig_band_path, dpi=200)
    plt.close(fig1)

    fig_psd_path = None
    if make_psd_plot:
        psd_mean = psd.mean(axis=0)
        fig2 = plt.figure(figsize=(7, 4))
        plt.loglog(freqs, psd_mean)
        plt.xlabel("Fréquence (Hz)")
        plt.ylabel("PSD moyenne (µV²/Hz)")
        plt.title(f"{base} — {mode_used.upper()} — PSD moyenne (log-log)")
        plt.tight_layout()
        fig_psd_path = patient_dir / f"{base}_psd_{mode_used}.png"
        fig2.savefig(fig_psd_path, dpi=200)
        plt.close(fig2)

    # 11) TFR par canal (Morlet) — canal par canal + décimation (faible RAM)
    if tfr_enabled:
        dur = (raw.n_times - 1) / sfreq  # une longue époque
        epochs = mne.make_fixed_length_epochs(raw, duration=dur, overlap=0.0, preload=False, verbose="ERROR")

        # Si l'utilisateur ne donne pas de fmin/fmax TFR -> on colle à la passband effective
        tfr_fmin_eff = float(tfr_fmin) if tfr_fmin is not None else float(fmin_eff)
        tfr_fmax_eff = float(tfr_fmax) if tfr_fmax is not None else float(fmax_eff)
        freqs_tfr = np.linspace(tfr_fmin_eff, tfr_fmax_eff, int(tfr_n_freqs))
        n_cycles = freqs_tfr * float(tfr_cycles_mult)

        for ch_name in ch_names:
            out_png = patient_dir / f"{base}_{ch_name}_tfr_{mode_used}.png"
            if out_png.exists() and not overwrite_figs:
                continue

            try:
                power = mne.time_frequency.tfr_morlet(
                    epochs, freqs=freqs_tfr, n_cycles=n_cycles,
                    use_fft=True, return_itc=False, average=True,
                    picks=[ch_name], decim=max(1, int(tfr_decim)), verbose="ERROR"
                )
            except Exception as e:
                print(f"[{base}:{ch_name}] TFR skipped: {e}")
                continue

            # --- affichage TFR avec vmin/vmax optionnels ---
            kwargs = dict(picks=[ch_name], dB=True, cmap=str(tfr_cmap),
                          baseline=None, show=False)

            # si l'utilisateur a fourni vmin/vmax -> on les utilise; sinon autoscale
            if tfr_vmin is not None and tfr_vmax is not None:
                kwargs_with_limits = {**kwargs, "vmin": float(tfr_vmin), "vmax": float(tfr_vmax)}
            else:
                kwargs_with_limits = kwargs

            try:
                fig = power.plot(**kwargs_with_limits)
            except TypeError:
                # anciennes versions : replot sans vmin/vmax
                fig = power.plot(**kwargs)
                # si des limites ont été demandées, on les applique a posteriori
                if tfr_vmin is not None and tfr_vmax is not None:
                    figs_tmp = fig if isinstance(fig, (list, tuple)) else [fig]
                    for f in figs_tmp:
                        for ax in f.axes:
                            artists = list(ax.images) + [c for c in ax.collections if hasattr(c, "set_clim")]
                            for art in artists:
                                art.set_clim(float(tfr_vmin), float(tfr_vmax))

            # --- détection figure vide -> log & skip (rien n'est sauvegardé) ---
            figs = fig if isinstance(fig, (list, tuple)) else [fig]
            is_blank = True
            for f in figs:
                for ax in f.axes:
                    if ax.images or ax.collections:
                        is_blank = False
                        break
                if not is_blank:
                    break

            if is_blank:
                print(f"[{base}:{ch_name}] TFR ignorée : figure vide (aucune image/collection rendue).")
                # fermeture propre et passage au canal suivant
                for f in figs:
                    plt.close(f)
                del power
                continue

            # --- figure non vide -> on enregistre ---
            fig_to_save = figs[0]
            fig_to_save.savefig(out_png, dpi=200)
            plt.close(fig_to_save)
            del power

    # (optionnel) supprimer le memmap de ce patient si stocké dans tmp_dir
    try:
        if tmp_dir is not None and (tmp_store / f"{base}_psd_mm.dat").exists():
            os.remove(tmp_store / f"{base}_psd_mm.dat")
    except Exception:
        pass

    return PatientResult(base=base, mode_used=mode_used, csv_path=csv_path, fig_band_path=fig_band_path, fig_psd_path=fig_psd_path)

# -------------------- Main --------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calcul de la puissance spectrale par patient (Full/REM) + TFR par canal")
    p.add_argument("--fif-dir", type=str, default=DEFAULT_FIF_DIR, help="Dossier racine des .fif prétraités (recherche récursive)")
    p.add_argument("--rem-dir", type=str, default=DEFAULT_REM_DIR, help="Racine des segments REM (rem_only/<patient>/)")
    p.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR, help="Dossier de sortie des résultats")
    p.add_argument("--suffix", type=str, default="bip", help="Suffixe des fichiers patients (bip|monop)")
    p.add_argument("--mode", choices=["auto", "full", "rem"], default="rem", help="Source des données : rem (par défaut), full ou auto")
    p.add_argument("--fmin", type=float, default=1.0, help="Fréquence min pour le calcul PSD")
    p.add_argument("--fmax", type=float, default=80.0, help="Fréquence max pour le calcul PSD")
    p.add_argument("--n-jobs", type=int, default=1, help="Nombre de processus en parallèle")
    p.add_argument("--no-psd-plot", action="store_true", help="Ne pas sauvegarder la figure PSD moyenne")

    # Options TFR (Morlet)
    p.add_argument("--tfr", dest="tfr", action="store_true", help="Calculer et sauvegarder les TFR par canal")
    p.add_argument("--no-tfr", dest="tfr", action="store_false", help="Désactiver la sauvegarde des TFR")
    p.set_defaults(tfr=True)
    p.add_argument("--tfr-fmin", type=float, default=None, help="Fréquence min TFR (auto=passband)")
    p.add_argument("--tfr-fmax", type=float, default=None, help="Fréquence max TFR (auto=passband)")
    p.add_argument("--tfr-n-freqs", type=int, default=60, help="Nombre de points de fréquence pour la TFR")
    p.add_argument("--tfr-cycles-mult", type=float, default=0.3, help="Multiplicateur n_cycles = freqs * mult")
    p.add_argument("--tfr-vmin", type=float, default=None, help="vmin dB pour la TFR (autoscale si omis)")
    p.add_argument("--tfr-vmax", type=float, default=None, help="vmax dB pour la TFR (autoscale si omis)")
    p.add_argument("--tfr-cmap", type=str, default="jet", help="Colormap TFR")
    p.add_argument("--tfr-decim", type=int, default=1, help="Décimation TFR (>=1) pour réduire RAM/CPU")
    p.add_argument("--overwrite-figs", action="store_true", help="Écraser les figures TFR existantes")

    # Temp files sur /dev/sda2
    p.add_argument("--tmp-dir", type=str, default=DEFAULT_TMP_DIR, help="Répertoire temporaire (memmap) — idéalement sur /dev/sda2")
    p.add_argument("--keep-tmp", action="store_true", help="Conserver le répertoire temporaire (debug)")

    return p.parse_args()

def main():
    mne.set_log_level("WARNING")
    args = parse_args()

    fif_dir = Path(args.fif_dir)
    rem_root = Path(args.rem_dir) if args.rem_dir else Path(args.fif_dir).parent / "rem_only"
    results_dir = Path(args.results_dir)

    # Prépare le répertoire temporaire sur /dev/sda2
    tmp_root = Path(args.tmp_dir)
    tmp_root.mkdir(parents=True, exist_ok=True)
    print(f"Tmp dir: {tmp_root}")

    fif_files = list_patients_fifs(fif_dir, suffix=args.suffix)
    if not fif_files:
        print(f"Aucun fichier *_preprocessed_{args.suffix}.fif trouvé dans {fif_dir} (recherche récursive)")
        return

    print(f"Patients trouvés : {len(fif_files)} | mode={args.mode} | résultats→ {results_dir}")

    results: List[PatientResult] = []
    with ProcessPoolExecutor(max_workers=max(1, args.n_jobs)) as ex:
        fut2path = {
            ex.submit(
                process_one_patient,
                fif_path=fp,
                rem_root=rem_root,
                results_dir=results_dir,
                bands=STANDARD_BANDS,
                fmin=args.fmin,
                fmax=args.fmax,
                make_psd_plot=not args.no_psd_plot,
                mode=args.mode,
                suffix=args.suffix,
                tfr_enabled=args.tfr,
                tfr_fmin=args.tfr_fmin,
                tfr_fmax=args.tfr_fmax,
                tfr_n_freqs=args.tfr_n_freqs,
                tfr_cycles_mult=args.tfr_cycles_mult,
                tfr_vmin=args.tfr_vmin,
                tfr_vmax=args.tfr_vmax,
                tfr_cmap=args.tfr_cmap,
                tfr_decim=args.tfr_decim,
                overwrite_figs=args.overwrite_figs,
                tmp_dir=tmp_root,
            ): fp
            for fp in fif_files
        }

        for fut in as_completed(fut2path):
            fp = fut2path[fut]
            base = base_from_fif(fp, args.suffix)
            try:
                res = fut.result()
                results.append(res)
                extra = f" + {res.fig_psd_path.name}" if res.fig_psd_path else ""
                print(f">>> {base}: CSV={res.csv_path.name} | FIG={res.fig_band_path.name}{extra}")
            except Exception as e:
                print(f" {base}: {e}")

    # Nettoyage du tmp si demandé
    if not args.keep_tmp:
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception as e:
            print(f"Warn: impossible de supprimer {tmp_root}: {e}")

    # Récapitulatif global
    ok = sum(1 for _ in results)
    print(f"Terminé. Patients traités: {ok}/{len(fif_files)}. Résultats dans {results_dir}")

if __name__ == "__main__":
    main()
