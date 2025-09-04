#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch REM extraction (FIF + hypnogram .txt) + bipolar montage + per-channel Morlet TFR with autoscale.
+ sauvegarde d'un FULL "clean" (sans artéfacts) et traitement identique pour N2 et N3.

- Cherche automatiquement les patients dans fif_root: {base}/*.fif ou directement *.fif à la racine
  (ou utilise une liste PATIENTS définie à la main).
- Charge le .fif, lit les annotations .txt via get_rem_annotations(base, annot_root)
  pour REM, et get_stage_annotations(..., stages=("N2","N3")) pour N2/N3.  # --> (A MODIFIER C'EST LE MEME FICHIER POUR LES 3 !!)

- Construit un FULL "clean" = concat de tous les intervalles qui ne chevauchent pas des annotations
  contenant "BAD" (insensible à la casse), et le SAUVEGARDE : {base}_FULL_clean_noBAD.fif.
- REM : Extrait/concatène les segments REM (print durées, skip si trop court), applique un montage
  bipolaire (MYMONTAGE_BIP), met l'EEG en µV, calcule TFR (Morlet) par canal avec autoscale robuste,
  formatage de l'axe temps en secondes simples, et titre incluant n_epochs.
- N2/N3 : Même pipeline que REM mais en partant du FULL "clean" (mapping temporel).
- Sauvegardes:
    - out_root/{base}/{base}_FULL_clean_noBAD.fif
    - out_root/{base}/{base}_REM_concat_uV.fif
    - out_root/{base}/{base}_{canal}_tfr_REM.png
    - out_root/{base}/{base}_{canal}_tfr_N2.png
    - out_root/{base}/{base}_{canal}_tfr_N3.png
"""

# ==========================
#   Parallélisation & CPU
# ==========================
# À définir AVANT d'importer numpy/scipy/mne
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # macOS/Accelerate

import multiprocessing as mp
import faulthandler; faulthandler.enable()  # log des crashes natifs

# ==========================
#   Imports standard
# ==========================
import platform
from pathlib import Path
from collections import Counter
import argparse
import numpy as np
import pandas as pd
import tempfile


# Matplotlib non-interactif sûr en multiprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Projet / scientifique
import mne
mne.set_config('MNE_MEMMAP_MIN_SIZE', '1M', set_env=True)  # favorise memmap

# ===================== PARAMÈTRES GLOBAUX =====================
# Ces variables seront fixées dynamiquement dans main() après détection du système
fif_root: Path | None = None
annot_root: Path | None = None
out_root: Path | None = None

# Laisser à None pour auto-découverte, ou donner une liste:
# - soit ["AE129","BJ138",...]
# - soit [("AE129", "/chemin/vers/fichier.fif"), ...] pour pointer un .fif précis
PATIENTS = None

# TFR
freq_min    = 1.0
freq_max    = 40.0
n_freqs     = 30
cycles_mult = 0.5
epoch_dur   = 4.0
decim       = 2
cmap        = "jet"
vmin, vmax  = None, None       # None => autoscale via percentiles ci-dessous
auto_pct    = (5, 95)

# I/O options
save_rem_fif   = True
overwrite_figs = True

# Seuils de durée pour garder les segments (REM/N2/N3)
MIN_SEG_S   = epoch_dur          # au moins 1 epoch
MIN_TOTAL_S = 2 * epoch_dur      # durée totale minimale après filtrage

# Montage bipolaire — on conserve la même définition que ton script REM
EEG_BIP  = ["Fp2-C4", "C4-O2", "T4-O2", "Cz-Pz", "Fp1-C3", "C3-O1", "Fp1-T3", "T3-O1"]
EOG_BIP  = ["EOGD-A1", "EOGG-A1"]
KEEP_RAW = ["Menton", "JAMBG", "JAMBD", "RONF", "EMG1", "EMG2", "ECG"]

# Après re-référencement bipolaire, on ne filtre PAS par nom "EEG"
RESTRICT_TO_NAME_WITH_EEG = False
# ===================== /PARAMS =====================


# ---- Import de la fonction d'annotations (fallback si besoin) ----

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
        # Colonnes flexibles
        cols = {c.lower().strip(): c for c in df.columns}
        # start en secondes : priorité à l'epoch (30s), sinon delta sur l'heure absolue
        if "position (epoch)" in cols:
            df["start"] = (df[cols["position (epoch)"]].astype(int) - 1) * 30.0
        elif "epoch" in cols:
            df["start"] = (df[cols["epoch"]].astype(int) - 1) * 30.0
        elif "absolute position (hh:mm:ss.ms)" in cols:
            t0 = pd.to_timedelta(df[cols["absolute position (hh:mm:ss.ms)"]].iloc[0])
            df["start"] = (pd.to_timedelta(df[cols["absolute position (hh:mm:ss.ms)"]]) - t0).dt.total_seconds()
        else:
            raise ValueError("CSV hypnogram: colonne epoch/time manquante.")
        # colonne de stade
        stage_col = (cols.get('default staging set ("stage")')
                     or cols.get("stage") or cols.get("stade"))
        stage = df[stage_col]
    else:
        raise ValueError(f"Extension non gérée: {path.suffix}")

    # Normalisation des stades
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
    df = df.iloc[:-1]  # on retire la dernière ligne (durée inconnue)
    return df[["start", "duration", "stage"]]

def _ensure_get_rem_annotations():
    try:
        from src.annotations import get_rem_annotations
        return get_rem_annotations
    except Exception:
        def get_rem_annotations(base_name, annot_dir):
            patient_code = base_name.split("_")[0]
            pdir = Path(annot_dir) / patient_code
            candidates = list(pdir.glob("*.txt")) + list(pdir.glob("*.csv"))
            for p in candidates:
                try:
                    df = _read_hypnogram_any(p)
                    rem = df[df["stage"] == "REM"]
                    if len(rem) > 0:
                        return mne.Annotations(
                            onset=rem["start"].astype(float).tolist(),
                            duration=rem["duration"].astype(float).tolist(),
                            description=["REM"] * len(rem),
                        )
                except Exception:
                    continue
            return None
        return get_rem_annotations

get_rem_annotations = _ensure_get_rem_annotations()

def _ensure_get_stage_annotations():
    def get_stage_annotations(base_name: str, annot_dir: str, stages=("N2", "N3")):
        want = {s.upper() for s in stages}
        patient_code = base_name.split("_")[0]
        pdir = Path(annot_dir) / patient_code
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
    return get_stage_annotations

get_stage_annotations = _ensure_get_stage_annotations()



def _sanitize(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in ("_", "-")).replace(" ", "")


def discover_patients(fif_root: Path):
    """Retourne une liste [(base, fif_path), ...] en cherchant des .fif.
    - Priorité: sous-dossiers {base}/*.fif (prend le .fif le plus gros)
    - Fallback: fichiers *.fif directement dans fif_root (base = préfixe avant le premier "_")
    """
    mapping = {}
    # 1) sous-dossiers {base}/*.fif
    for child in sorted(fif_root.iterdir()):
        if not child.is_dir():
            continue
        base = child.name
        fif_candidates = [p for p in child.glob("*.fif") if p.is_file() and not p.name.startswith("._")]
        if not fif_candidates:
            continue
        fif_path = max(fif_candidates, key=lambda p: p.stat().st_size)
        mapping[base] = fif_path

    # 2) fichiers *.fif à la racine
    root_fifs = [p for p in fif_root.glob("*.fif") if p.is_file() and not p.name.startswith("._")]
    for p in root_fifs:
        base = p.stem.split("_")[0]
        cur = mapping.get(base)
        if cur is None or p.stat().st_size > cur.stat().st_size:
            mapping[base] = p

    items = sorted(mapping.items())  # [(base, path), ...]
    return items


# ===================== BIPOLAIRE =====================

def _safe_bipolar(inst: mne.io.BaseRaw, anode: str, cathode: str, new_name: str, base: str) -> bool:
    """Crée un canal bipolaire anode-cathode ssi les deux existent."""
    if anode in inst.ch_names and cathode in inst.ch_names:
        try:
            mne.set_bipolar_reference(
                inst, anode=anode, cathode=cathode, ch_name=new_name,
                drop_refs=False, copy=False, verbose="ERROR"
            )
            return True
        except Exception as e:
            print(f"[{base}] Bipolaire {new_name} échec: {e}")
    else:
        missing = [x for x in (anode, cathode) if x not in inst.ch_names]
        print(f"[{base}] Bipolaire {new_name} ignoré (manque {missing})")
    return False


def _apply_bipolar_montage(inst: mne.io.BaseRaw, base: str) -> None:
    """Applique le montage bipolaire MYMONTAGE_BIP + typage des canaux + réduction au set voulu."""
    pairs = [
        ("EOGD", "A1",  "EOGD-A1"),
        ("EOGG", "A1",  "EOGG-A1"),
        ("Fp2",  "C4",  "Fp2-C4"),
        ("C4",   "O2",  "C4-O2"),
        ("T4",   "O2",  "T4-O2"),
        ("Cz",   "Pz",  "Cz-Pz"),
        ("Fp1",  "C3",  "Fp1-C3"),
        ("C3",   "O1",  "C3-O1"),
        ("Fp1",  "T3",  "Fp1-T3"),
        ("T3",   "O1",  "T3-O1"),
    ]

    created = []
    for a, c, n in pairs:
        if _safe_bipolar(inst, a, c, n, base):
            created.append(n)

    # Typage: EEG pour paires EEG, EOG pour EOG*, EMG/ECG pour les capteurs conservés
    eeg_bip = EEG_BIP
    eog_bip = EOG_BIP

    type_map = {}
    for ch in eeg_bip:
        if ch in inst.ch_names:
            type_map[ch] = "eeg"
    for ch in eog_bip:
        if ch in inst.ch_names:
            type_map[ch] = "eog"
    for ch in ("Menton", "EMG1", "EMG2", "JAMBG", "JAMBD"):
        if ch in inst.ch_names:
            type_map[ch] = "emg"
    if "ECG" in inst.ch_names:
        type_map["ECG"] = "ecg"
    if "RONF" in inst.ch_names:
        type_map["RONF"] = "misc"

    if type_map:
        try:
            inst.set_channel_types(type_map)
        except Exception as e:
            print(f"[{base}] set_channel_types après bipolaire: {e}")

    # Réduire strictement aux canaux voulus (ceux créés + capteurs conservés)
    desired = eog_bip + eeg_bip + KEEP_RAW
    present = [ch for ch in desired if ch in inst.ch_names]
    if not present:
        print(f"[{base}] Aucun canal bipolaire/utile présent après montage → rien à faire")
        return
    inst.pick(present)

    print(f"[{base}] Montage bipolaire créé. Canaux conservés ({len(inst.ch_names)}): {inst.ch_names}")


# ===================== OUTILS FULL-CLEAN (BAD*) =====================

def _merge_intervals(intervals):
    """Fusionne des intervalles (start, end) éventuellement chevauchants."""
    if not intervals:
        return []
    ints = sorted(intervals, key=lambda x: x[0])
    merged = [ints[0]]
    for s, e in ints[1:]:
        s0, e0 = merged[-1]
        if s <= e0:
            merged[-1] = (s0, max(e0, e))
        else:
            merged.append((s, e))
    return merged


def build_full_clean(raw_full: mne.io.BaseRaw, base: str):  # vérifier l'origine du raw_full
    """Construit un enregistrement concaténé sans intervalles dont la description contient 'BAD'."""
    sfreq = float(raw_full.info["sfreq"])
    t_end = raw_full.times[-1]
    eps   = 1.0 / sfreq

    bad_intervals = []
    for onset, dur, desc in zip(raw_full.annotations.onset,
                                raw_full.annotations.duration,
                                raw_full.annotations.description):
        if "BAD_ARTIFACT" in (str(desc) or "").upper() and float(dur) > 0:
            s = float(onset)
            e = min(float(onset) + float(dur), t_end)
            if e > s:
                bad_intervals.append((s, e))
    bad_intervals = _merge_intervals(bad_intervals)

    if not bad_intervals:
        print(f"[{base}] Aucun intervalle BAD* détecté (full conservé tel quel).")
        return raw_full.copy(), [(0.0, t_end, 0.0)]

    good = []
    t0 = 0.0
    for (bs, be) in bad_intervals:
        if bs > t0:
            good.append((t0, bs))
        t0 = max(t0, be)
    if t_end > t0:
        good.append((t0, t_end))
    if not good:
        print(f"[{base}] Attention: tout est BAD* → rien à garder.")
        return None, []

    parts, mapping, t_clean = [], [], 0.0
    for (s, e) in good:
        try:
            seg = raw_full.copy().crop(tmin=s, tmax=e - eps, verbose="ERROR")
            parts.append(seg)
            mapping.append((s, e, t_clean))
            t_clean += (e - s)
        except Exception as ex:
            print(f"[{base}] Crop good {s:.2f}-{e:.2f} échoué: {ex}")
    if not parts:
        return None, []

    raw_clean = mne.concatenate_raws(parts, verbose="ERROR")
    print(f"[{base}] Full nettoyé construit: {len(good)} segments bons, durée={t_clean:.2f}s")
    return raw_clean, mapping


def map_intervals_to_clean(intervals, mapping):
    """Mappe des intervalles (s, e) de la timeline ORIGINE vers la timeline NETTOYÉE."""
    out = []
    for (s, e) in intervals:
        if e <= s:
            continue
        for (gs, ge, c0) in mapping:
            a = max(s, gs)
            b = min(e, ge)
            if b > a:
                out.append((c0 + (a - gs), c0 + (b - gs)))
    return out


# ===================== /BIPOLAIRE & FULL-CLEAN =====================


def _find_fif_for_base(base: str) -> Path | None:
    """Cherche un .fif pour `base` dans fif_root/{base}/*.fif, sinon à la racine par préfixe du nom."""
    global fif_root
    assert fif_root is not None
    # 1) dossier du patient
    child = fif_root / base
    if child.is_dir():
        cand = [p for p in child.glob("*.fif") if p.is_file() and not p.name.startswith("._")]
        if cand:
            return max(cand, key=lambda p: p.stat().st_size)
    # 2) racine, par préfixe stem
    cand = [p for p in fif_root.glob(f"{base}*.fif") if p.is_file() and not p.name.startswith("._")]
    if cand:
        return max(cand, key=lambda p: p.stat().st_size)
    return None


def _format_time_axes(figs):
    """Applique l’affichage 'Time (s)' sans puissances de 10 sur les axes temps."""
    for f in figs:
        for ax in f.axes:
            xlabel = (ax.get_xlabel() or "").lower()
            if "time" in xlabel:
                ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
                ax.ticklabel_format(axis="x", style="plain", useOffset=False)
                ax.set_xlabel("Time (s)")


def _plot_and_save_power(power, ch, base, stage, out_png, vmin_eff, vmax_eff, n_epochs):
    """Plot TFR, formate l’axe temps, ajoute le titre avec n_epochs, et sauvegarde."""
    try:
        fig = power.plot(
            picks=[ch], dB=True, cmap=str(cmap),
            vmin=vmin_eff, vmax=vmax_eff,
            baseline=None, show=False
        )
    except TypeError:
        fig = power.plot(picks=[ch], dB=True, cmap=str(cmap),
                         baseline=None, show=False)
        figs_tmp = fig if isinstance(fig, (list, tuple)) else [fig]
        for f in figs_tmp:
            for ax in f.axes:
                artists = list(ax.images) + [c for c in ax.collections if hasattr(c, "set_clim")]
                for art in artists:
                    try:
                        art.set_clim(vmin_eff, vmax_eff)
                    except Exception:
                        pass

    figs = fig if isinstance(fig, (list, tuple)) else [fig]
    _format_time_axes(figs)

    # ---- Titre AU-DESSUS de l'image ----
    title = f"{base} — {ch} — {stage}  (n_epochs={n_epochs})"
    for f in figs:
        # Évite les conflits entre constrained_layout et tight_layout
        try:
            f.set_constrained_layout(False)
        except Exception:
            pass
        # titre au-dessus des axes
        f.suptitle(title, y=1.02)             # >1.00 le met au-dessus de l'axe/colorbar
        # espace pour le titre
        try:
            f.tight_layout(rect=[0, 0, 1, 0.96])  # garde 4% en haut pour le titre
        except Exception:
            f.subplots_adjust(top=0.90)           # fallback

    # Garde-fou "figure blanche"
    ax0 = figs[0].axes[0] if figs and figs[0].axes else None
    is_blank = (ax0 is None) or (len(ax0.images) == 0 and len(ax0.collections) == 0)
    if is_blank:
        print(f"[{base}:{stage}:{ch}] figure vide -> skip (rien sauvegardé)")
        try:
            for f in figs:
                plt.close(f)
        except Exception:
            pass
        return False

    # Sauvegarde
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



def process_one_patient(item):
    global out_root, annot_root
    if isinstance(item, tuple):
        base, fif_path = item
        fif_path = Path(fif_path)
    else:
        base = str(item)
        fif_path = _find_fif_for_base(base)

    if fif_path is None or not Path(fif_path).exists():
        print(f"[{base}] FIF introuvable -> skip")
        return

    print(f"\n=== {base} ===")
    try:
        raw_full = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
    except Exception as e:
        print(f"[{base}] Erreur lecture FIF: {e} -> skip")
        return
    print(raw_full)

    # =================== FULL CLEAN (sans BAD*) + SAUVEGARDE ===================
    out_dir = out_root / base
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_fif_path = out_dir / f"{base}_FULL_clean_noBAD.fif"

    raw_clean, mapping = build_full_clean(raw_full, base)
    if raw_clean is None:
        print(f"[{base}] Rien à garder après retrait des BAD* -> skip")
        return
    try:
        raw_clean.save(clean_fif_path, overwrite=True)
        print(f"[{base}] Sauvé: {clean_fif_path.name}")
    except Exception as e:
        print(f"[{base}] Save FULL_clean échoué: {e}")

    # =================== PIPELINE REM (inchangé) ===================
    # Annotations REM depuis .txt 
    rem_annots = get_rem_annotations(base, annot_dir=str(annot_root))
    if rem_annots is None or len(rem_annots) == 0:
        print(f"[{base}] Aucune annotation REM -> skip REM")
    else:
        sfreq = float(raw_full.info["sfreq"])
        t_end = raw_full.times[-1]
        eps   = 1.0 / sfreq

        # Filtrage des segments valides
        valid = []
        for onset, dur, desc in zip(rem_annots.onset, rem_annots.duration, rem_annots.description):
            if str(desc).upper() != "REM":
                continue
            if dur is None or dur <= 0:
                continue
            tmin = max(0.0, float(onset))
            tmax = min(tmin + float(dur), t_end) - eps
            if tmax <= tmin:
                continue
            valid.append((tmin, tmax))

        if not valid:
            print(f"[{base}] Aucun segment REM valide -> skip REM")
        else:
            durations = [tmax - tmin for (tmin, tmax) in valid]
            total_dur = float(np.sum(durations))
            print(f"[{base}] REM: Segments valides={len(valid)} | total={total_dur:.2f}s | "
                  f"min={np.min(durations):.2f}s | median={np.median(durations):.2f}s | max={np.max(durations):.2f}s")

            # Filtrage "trop court" selon MIN_SEG_S / MIN_TOTAL_S
            kept = []
            for (tmin, tmax) in valid:
                dur = tmax - tmin
                if dur < MIN_SEG_S:
                    print(f"[{base}]  - REM skip {tmin:.2f}-{tmax:.2f}s (durée {dur:.2f}s < {MIN_SEG_S:.2f}s)")
                else:
                    kept.append((tmin, tmax))
            if not kept or sum(tmax - tmin for (tmin, tmax) in kept) < MIN_TOTAL_S:
                print(f"[{base}] REM: durée après filtrage insuffisante (< {MIN_TOTAL_S:.2f}s) -> skip REM")
            else:
                # Concaténation REM à partir du full original (inchangé)
                rem_raws = []
                for (tmin, tmax) in kept:
                    try:
                        seg = raw_full.copy().crop(tmin=tmin, tmax=tmax, verbose="ERROR")
                        rem_raws.append(seg)
                    except Exception as e:
                        print(f"[{base}] Crop REM {tmin:.2f}-{tmax:.2f}s échoué: {e}")

                if not rem_raws:
                    print(f"[{base}] REM: Aucun segment ajouté -> skip")
                else:
                    rem_raw = mne.concatenate_raws(rem_raws, verbose="ERROR")
                    print(rem_raw)

                    # Montage bipolaire puis typage (identique)
                    _apply_bipolar_montage(rem_raw, base)

                    # --- Ne garder QUE l'EEG par TYPE (les 8 bipolaires listés) ---
                    try:
                        rem_raw.pick_types(
                            meg=False, eeg=True, eog=False, ecg=False, emg=False, stim=False,
                            misc=False, resp=False, seeg=False, ecog=False, fnirs=False
                        )
                        if RESTRICT_TO_NAME_WITH_EEG:
                            eeg_names = [ch for ch in rem_raw.ch_names if "EEG" in ch.upper()]
                            if len(eeg_names) == 0:
                                print(f"[{base}] Aucun canal avec 'EEG' dans le nom; conservez tous les EEG typés.")
                            else:
                                rem_raw.pick(eeg_names)
                        print(f"[{base}] Canaux EEG (REM) retenus ({len(rem_raw.ch_names)}): {rem_raw.ch_names}")
                    except Exception as e:
                        print(f"[{base}] Échec du filtrage EEG-only (REM): {e} -> skip REM")
                    else:
                        # Scaling µV (toujours)
                        try:
                            rem_raw.load_data()
                            eeg_picks = mne.pick_types(rem_raw.info, eeg=True, meg=False, eog=False, ecg=False, emg=False)
                            rem_raw.apply_function(lambda x: x * 1e6, picks=eeg_picks, channel_wise=True)
                            if hasattr(rem_raw, "set_unit"):
                                try:
                                    rem_raw.set_unit("eeg", "uV")
                                except Exception:
                                    pass
                        except Exception as e:
                            print(f"[{base}] Échec scaling µV (REM): {e} -> skip REM")
                        else:
                            # I/O
                            if save_rem_fif:
                                try:
                                    out_fif = out_dir / f"{base}_REM_concat_uV.fif"
                                    rem_raw.save(out_fif, overwrite=True)
                                    print(f"[{base}] Sauvé: {out_fif}")
                                except Exception as e:
                                    print(f"[{base}] Échec save FIF (REM): {e}")

                            # TFR (Morlet) — epochs fixes
                            try:
                                epochs = mne.make_fixed_length_epochs(
                                    rem_raw, duration=float(epoch_dur), overlap=0.0, preload=True, verbose="ERROR"
                                )
                                # Ne garder que l'EEG dans epochs
                                epochs.pick_types(meg=False, eeg=True, eog=False, ecg=False, emg=False, stim=False, misc=False)
                            except Exception as e:
                                print(f"[{base}] Échec création/filtrage epochs (REM): {e} -> skip TFR REM")
                            else:
                                freqs = np.linspace(float(freq_min), float(freq_max), int(n_freqs))
                                n_cycles = freqs * float(cycles_mult)
                                ch_names = epochs.ch_names
                                n_epochs = len(epochs)
                                print(f"[{base}] Canaux EEG pour TFR REM ({len(ch_names)}): {ch_names}")

                                for ch in ch_names:
                                    out_png = out_dir / f"{base}_{_sanitize(ch)}_tfr_REM.png"
                                    if out_png.exists() and not overwrite_figs:
                                        print(f"[{base}:REM:{ch}] [skip] {out_png.name} existe déjà.")
                                        continue

                                    try:
                                        power = mne.time_frequency.tfr_morlet(
                                            epochs, freqs=freqs, n_cycles=n_cycles,
                                            use_fft=True, return_itc=False, average=True,
                                            picks=[ch], decim=int(decim), verbose="ERROR"
                                        )
                                    except Exception as e:
                                        print(f"[{base}:REM:{ch}] TFR erreur: {e} -> skip")
                                        continue

                                    # Autoscale robuste en dB
                                    try:
                                        Z = 10.0 * np.log10(np.maximum(power.data[0], np.finfo(float).tiny))
                                        if vmin is None or vmax is None:
                                            lo, hi = np.percentile(Z, list(auto_pct))
                                            vmin_eff, vmax_eff = float(lo), float(hi)
                                        else:
                                            vmin_eff, vmax_eff = float(vmin), float(vmax)
                                    except Exception as e:
                                        print(f"[{base}:REM:{ch}] Autoscale erreur: {e} -> skip")
                                        del power
                                        continue

                                    # Plot/format/titre + save
                                    _plot_and_save_power(power, ch, base, "REM", out_png, vmin_eff, vmax_eff, n_epochs)
                                    del power

    # =================== PIPELINE N2 / N3 (à partir du FULL clean) ===================
    stage_ann = get_stage_annotations(base, annot_dir=str(annot_root), stages=("N2", "N3"))
    if stage_ann is None or len(stage_ann) == 0:
        print(f"[{base}] Aucune annotation N2/N3 -> skip N2/N3")
    else:
        # Intervalles N2/N3 sur la timeline ORIGINE (full)
        sfreq_full = float(raw_full.info["sfreq"])
        t_end_full = raw_full.times[-1]
        eps_full = 1.0 / sfreq_full

        by_stage = {"N2": [], "N3": []}
        for onset, dur, desc in zip(stage_ann.onset, stage_ann.duration, stage_ann.description):
            s = max(0.0, float(onset))
            e = min(s + float(dur), t_end_full) - eps_full
            tag = (str(desc) or "").upper()
            if e > s and tag in by_stage:
                by_stage[tag].append((s, e))

        # Pour chaque stade, on mappe vers la timeline NETTOYÉE et on applique le pipeline identique
        for stage in ("N2", "N3"):
            intervals_orig = by_stage.get(stage, [])
            if not intervals_orig:
                print(f"[{base}:{stage}] Aucun intervalle -> skip")
                continue

            intervals_clean = map_intervals_to_clean(intervals_orig, mapping)
            if not intervals_clean:
                print(f"[{base}:{stage}] Intersections avec 'bons' = 0 -> skip")
                continue

            durs = [e - s for (s, e) in intervals_clean if e > s]
            total = float(np.sum(durs)) if durs else 0.0
            print(f"[{base}:{stage}] Segments valides: {len(durs)} | "
                  f"total={total:.2f}s | min={np.min(durs):.2f}s | "
                  f"median={np.median(durs):.2f}s | max={np.max(durs):.2f}s")

            kept = []
            for (s, e) in intervals_clean:
                dur = e - s
                if dur < MIN_SEG_S:
                    print(f"[{base}:{stage}]  - skip seg {s:.2f}-{e:.2f}s (durée {dur:.2f}s < {MIN_SEG_S:.2f}s)")
                else:
                    kept.append((s, e))
            if not kept or sum(e - s for (s, e) in kept) < MIN_TOTAL_S:
                print(f"[{base}:{stage}] Durée après filtrage insuffisante (< {MIN_TOTAL_S:.2f}s) -> skip")
                continue

            # Extraire depuis le FULL clean (et non le full original)
            sfreq_clean = float(raw_clean.info["sfreq"])
            eps_clean   = 1.0 / sfreq_clean
            parts = []
            for (s, e) in kept:
                try:
                    seg = raw_clean.copy().crop(tmin=s, tmax=e - eps_clean, verbose="ERROR")
                    parts.append(seg)
                except Exception as ex:
                    print(f"[{base}:{stage}] Crop {s:.2f}-{e:.2f} échoué: {ex}")
            if not parts:
                print(f"[{base}:{stage}] Rien à concaténer -> skip")
                continue
            stage_raw = mne.concatenate_raws(parts, verbose="ERROR")

            # Montage bipolaire + EEG only + µV (identique à REM)
            _apply_bipolar_montage(stage_raw, base)
            try:
                stage_raw.pick_types(meg=False, eeg=True, eog=False, ecg=False, emg=False, stim=False,
                                     misc=False, resp=False, seeg=False, ecog=False, fnirs=False)
                if RESTRICT_TO_NAME_WITH_EEG:
                    eeg_names = [ch for ch in stage_raw.ch_names if "EEG" in ch.upper()]
                    if len(eeg_names) > 0:
                        stage_raw.pick(eeg_names)
                if len(stage_raw.ch_names) == 0:
                    print(f"[{base}:{stage}] Aucun canal EEG bipolaire -> skip")
                    continue
            except Exception as e:
                print(f"[{base}:{stage}] pick_types EEG échoué: {e} -> skip")
                continue

            try:
                stage_raw.load_data()
                eeg_picks = mne.pick_types(stage_raw.info, eeg=True, meg=False, eog=False, ecg=False, emg=False)
                stage_raw.apply_function(lambda x: x * 1e6, picks=eeg_picks, channel_wise=True)
                if hasattr(stage_raw, "set_unit"):
                    try:
                        stage_raw.set_unit("eeg", "uV")
                    except Exception:
                        pass
            except Exception as e:
                print(f"[{base}:{stage}] Échec scaling µV: {e} -> skip")
                continue

            # Epochs + Morlet + autoscale + plots (mêmes réglages que REM)
            try:
                epochs = mne.make_fixed_length_epochs(
                    stage_raw, duration=float(epoch_dur), overlap=0.0, preload=True, verbose="ERROR"
                )
                epochs.pick_types(meg=False, eeg=True, eog=False, ecg=False, emg=False, stim=False, misc=False)
            except Exception as e:
                print(f"[{base}:{stage}] Échec création/filtrage epochs: {e} -> skip TFR")
                continue

            n_epochs = len(epochs)
            freqs = np.linspace(float(freq_min), float(freq_max), int(n_freqs))
            n_cycles = freqs * float(cycles_mult)

            for ch in epochs.ch_names:
                out_png = out_dir / f"{base}_{_sanitize(ch)}_tfr_{stage}.png"
                if out_png.exists() and not overwrite_figs:
                    print(f"[{base}:{stage}:{ch}] [skip] {out_png.name} existe déjà.")
                    continue

                try:
                    power = mne.time_frequency.tfr_morlet(
                        epochs, freqs=freqs, n_cycles=n_cycles,
                        use_fft=True, return_itc=False, average=True,
                        picks=[ch], decim=int(decim), verbose="ERROR"
                    )
                except Exception as e:
                    print(f"[{base}:{stage}:{ch}] TFR erreur: {e} -> skip")
                    continue

                try:
                    Z = 10.0 * np.log10(np.maximum(power.data[0], np.finfo(float).tiny))
                    if vmin is None or vmax is None:
                        lo, hi = np.percentile(Z, list(auto_pct))
                        vmin_eff, vmax_eff = float(lo), float(hi)
                    else:
                        vmin_eff, vmax_eff = float(vmin), float(vmax)
                except Exception as e:
                    print(f"[{base}:{stage}:{ch}] Autoscale erreur: {e} -> skip")
                    del power
                    continue

                _plot_and_save_power(power, ch, base, stage, out_png, vmin_eff, vmax_eff, n_epochs)
                del power

    print(f"[{base}] Terminé.")


# ===================== DÉTECTION DU SYSTÈME & I/O ROOTS =====================

def detect_disque(explicit: str | None = None) -> str:
    """Détecte le disque/chemin racine selon l'OS, avec possibilité d'override."""
    if explicit:
        return explicit
    system = platform.system()
    if system == "Darwin":
        return "/Volumes/Crucial X6"
    elif system == "Windows":
        return "D:"
    elif system == "Linux":
        # Adapter si besoin; fallback générique
        user = os.getenv("USER") or os.getenv("USERNAME") or ""
        return f"/media/{user}/Crucial X6" if user else "/media/Crucial X6"
    else:
        raise RuntimeError("Système non supporté pour la détection de disque.")


def _warmup_matplotlib():
    """Évite les races sur la création du cache Matplotlib en multiprocess."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    matplotlib.get_cachedir()
    fm.findfont('DejaVu Sans', rebuild_if_missing=True)
    fig = plt.figure()
    plt.plot([0, 1], [0, 1])
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)


# ===================== LANCEMENT BATCH (PARALLÈLE) =====================

def _worker_wrapper(item):
    """Wrapper top-level picklable pour spawn: isole le cache MPL par PID et exécute un patient."""
    try:
        mpl_cache = os.path.join(os.getenv("TMPDIR") or "/tmp", f"mplcache_{os.getpid()}")
        os.environ["MPLCONFIGDIR"] = mpl_cache
        os.makedirs(mpl_cache, exist_ok=True)
    except Exception:
        pass
    try:
        return process_one_patient(item)
    except Exception as e:
        import traceback
        raise RuntimeError(f"Worker error on {item}: {e}\n{traceback.format_exc()}") from e

def _init_worker(gl_out_root: str, gl_annot_root: str, gl_fif_root: str):
    """Initialise les globaux dans chaque worker (spawn)."""
    global out_root, annot_root, fif_root
    out_root = Path(gl_out_root)
    annot_root = Path(gl_annot_root)
    fif_root = Path(gl_fif_root)

    # isole le cache Matplotlib par PID (évite les races)
    try:
        mpl_cache = os.path.join(tempfile.gettempdir(), f"mplcache_{os.getpid()}")
        os.environ["MPLCONFIGDIR"] = mpl_cache
        os.makedirs(mpl_cache, exist_ok=True)
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4,
                        help="Nb de processus en parallèle (0 => CPU-1)")
    parser.add_argument("--disk", type=str, default=None,
                        help="Chemin racine du disque (override). Si omis: auto-détection par OS.")
    parser.add_argument("--patients", type=str, nargs="*", default=None,
                        help="Liste de bases patients à traiter (sinon auto-discovery).")
    args = parser.parse_args()

    # Détection de disque selon OS (override CLI possible)
    #disque = detect_disque(args.disk)
    disque = "/home/darryld/documents" # connexion ssh

    # Fixe les roots selon le disque détecté
    fif_root    = Path(f"{disque}/EEG/preprocessed/bipolaire/full")
    annot_root  = Path(f"{disque}/EEG/raw")
    out_root    = Path(f"{disque}/EEG/preprocessed/bipolaire/PWD")

    # Vérif d'existence + création
    for p in [fif_root, annot_root]:
        if not p.exists():
            raise SystemExit(f"[CONFIG] Dossier introuvable: {p}")
    out_root.mkdir(parents=True, exist_ok=True)

    # Patients
    if args.patients:
        # Mode liste de bases (strings)
        PATIENTS = [(b, _find_fif_for_base(b)) for b in args.patients]
    if PATIENTS is None:
        PATIENTS = discover_patients(fif_root)
        bases_preview = [b for b, _ in PATIENTS]
        print(f"Patients détectés ({len(PATIENTS)}): {bases_preview}")

    if not PATIENTS:
        raise SystemExit("Aucun patient à traiter.")

    _warmup_matplotlib()

    # Calcul du nombre de workers
    cpu = os.cpu_count() or 1
    max_workers = (cpu - 1) if args.workers in (0, None) else max(1, args.workers)
    max_workers = min(max_workers, len(PATIENTS))
    print(f"[INFO] Lancement en multiprocess avec {max_workers} worker(s) (CPU={cpu}) (maxtasksperchild=1)")

    # Pool spawn, une tâche par patient, recycle les workers
    ctx = mp.get_context("spawn")
    try:
        with ctx.Pool(
            processes=max_workers,
            maxtasksperchild=1,
            initializer=_init_worker,
            initargs=(str(out_root), str(annot_root), str(fif_root)),
        ) as pool:
            for _ in pool.imap_unordered(_worker_wrapper, PATIENTS, chunksize=1):
                pass
    except Exception as e:
        import traceback
        print(f"[POOL ERROR] {e}\n{traceback.format_exc()}")

