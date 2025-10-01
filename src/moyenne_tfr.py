#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agrégation TFR par CATEGORIE DE PATIENT à partir d'un Excel de métadonnées.

Entrées:
  - REM:  /home/darryld/documents/EEG/preprocessed/bipolaire/2_rem_only
  - N2/N3 (full clean concat): /home/darryld/documents/EEG/preprocessed/bipolaire/1_noArtefacts
  - Hypnogrammes .txt/.csv (timeline ORIG): /home/darryld/documents/EEG/raw
  - (opt) CSV artéfacts ORIG->CLEAN: {base}_artifact_windows.csv (start_time_s,end_time_s)
  - META Excel (--meta-xlsx): colonnes au minimum: Identifiant, Catégorie de patient
      * fallback auto vers Dx avant PSG si la colonne de catégorie est non-informative *

Sorties (par catégorie patient):
  out_root/_by_patient_category/{category}/{stage}/
    - group_{canal}_tfr.png
    - group_{canal}_power.npy
    - times.npy
    - freqs.npy
    - group_{canal}_spectrum.csv
"""

from __future__ import annotations

# --------- Limites BLAS & env sûrs ---------
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import mne
mne.set_config('MNE_MEMMAP_MIN_SIZE', '1M', set_env=True)

# Cache MNE pour joblib memmap (économies RAM/IO)
try:
    cache_base = "/dev/shm/mne_cache" if os.path.isdir("/dev/shm") else "/tmp/mne_cache"
    Path(cache_base).mkdir(parents=True, exist_ok=True)
    mne.set_cache_dir(cache_base) 
    mne.set_config("MNE_CACHE_DIR", cache_base, set_env=True)
except Exception:
    pass


from concurrent.futures import ProcessPoolExecutor, as_completed

# --------- Dossiers par défaut ----------
DEFAULT_REM_DIR   = "/home/darryld/documents/EEG/preprocessed/bipolaire/2_rem_only/gp2"
DEFAULT_CLEAN_DIR = "/home/darryld/documents/EEG/preprocessed/bipolaire/1_noArtefacts/gp2"
DEFAULT_ANNOT_DIR = "/home/darryld/documents/EEG/raw"
DEFAULT_OUT_ROOT  = "/home/darryld/documents/EEG/preprocessed/bipolaire/3_results_analysis/gp2"

# --------- TFR / Epochs ----------
FREQ_MIN     = 1.0
FREQ_MAX     = 80.0
N_FREQS      = 30
CYCLES_MULT  = 0.5
EPOCH_DUR    = 4.0
DECIM        = 2
TARGET_SFREQ = 200.0   # resample pour aligner n_times entre sujets
CMAP         = "jet"
AUTO_PCT     = (5, 95)
VMIN, VMAX   = None, None   # autoscale si None

# Seuils de durée pour N2/N3
MIN_SEG_S    = EPOCH_DUR
MIN_TOTAL_S  = 2 * EPOCH_DUR

# --------- Utils ---------
def _sanitize(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in ("_", "-")).replace(" ", "")

def _format_time_axes(ax):
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

def _bases_from_dir(d: Path) -> list[str]:
    bases = set()
    if not d.exists():
        return []
    for child in d.iterdir():
        if child.is_dir() and not child.name.startswith("._"):
            bases.add(child.name)
    for p in d.glob("*.fif"):
        if p.is_file() and not p.name.startswith("._"):
            bases.add(p.stem.split("_")[0])
    return sorted(bases)

def discover_patients(rem_dir: Path, clean_dir: Path) -> list[str]:
    return sorted(set(_bases_from_dir(rem_dir)) | set(_bases_from_dir(clean_dir)))

def find_rem_fifs(base: str, rem_dir: Path) -> list[Path]:
    files = []
    pdir = rem_dir / base
    if pdir.exists():
        files += [p for p in pdir.glob("*.fif") if p.is_file() and not p.name.startswith("._")]
    files += [p for p in rem_dir.glob(f"{base}*REM*.fif") if p.is_file() and not p.name.startswith("._")]
    if not files:
        files += [p for p in rem_dir.glob(f"{base}*.fif") if p.is_file() and not p.name.startswith("._")]
    return sorted(files)

def find_clean_fif(base: str, clean_dir: Path) -> Path | None:
    cand = []
    pdir = clean_dir / base
    if pdir.exists():
        cand += [p for p in pdir.glob("*.fif") if p.is_file() and not p.name.startswith("._")]
    if not cand:
        cand += [p for p in clean_dir.glob(f"{base}*.fif") if p.is_file() and not p.name.startswith("._")]
    return max(cand, key=lambda p: p.stat().st_size) if cand else None

# --------- Hypnogrammes ---------
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
        map_ = {"SP":"REM","R":"REM","REM":"REM","V":"W","WAKE":"W","W":"W",
                "S2":"N2","STAGE2":"N2","NREM2":"N2","S3":"N3","STAGE3":"N3","NREM3":"N3"}
        return map_.get(s, s)

    df = df.assign(stage=stage.map(norm)).sort_values("start")
    df["duration"] = df["start"].shift(-1) - df["start"]
    df = df.iloc[:-1]
    return df[["start", "duration", "stage"]]

def get_stage_annotations(base_name: str, annot_dir: Path, stages=("N2","N3")) -> mne.Annotations | None:
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

# --------- Mapping artéfacts ORIG -> CLEAN ---------
def _merge_intervals(intervals):
    if not intervals: return []
    ints = sorted(intervals, key=lambda x: x[0])
    merged = [ints[0]]
    for s, e in ints[1:]:
        s0, e0 = merged[-1]
        if s <= e0: merged[-1] = (s0, max(e0, e))
        else: merged.append((s, e))
    return merged

def _load_artifact_windows_csv(base: str, start_dir: Path) -> list[tuple[float,float]]:
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
            lc = {c.lower(): c for c in df.columns}
            if {"start_time_s","end_time_s"} <= set(lc):
                st = df[lc["start_time_s"]].astype(float).to_numpy()
                en = df[lc["end_time_s"]].astype(float).to_numpy()
            else:
                st = df.iloc[:,0].astype(float).to_numpy()
                en = df.iloc[:,1].astype(float).to_numpy()
            intervals = [(float(s), float(e)) for s,e in zip(st,en) if e>s]
            return _merge_intervals(intervals)
        except Exception:
            continue
    return []

def _orig2clean_mapping_from_artifacts(bad_orig: list[tuple[float,float]], t_end_orig: float):
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
    for (s,e) in good:
        mapping.append((float(s), float(e), float(c0)))
        c0 += (e - s)
    return mapping

def map_intervals_to_clean(intervals, mapping):
    out = []
    for (s,e) in intervals:
        if e <= s: continue
        for (gs,ge,c0) in mapping:
            a = max(s, gs); b = min(e, ge)
            if b > a: out.append((c0 + (a - gs), c0 + (b - gs)))
    return out

# --------- Extraction epochs par stage ---------
def _epochs_from_rem(base: str, rem_dir: Path) -> mne.Epochs | None:
    rem_files = find_rem_fifs(base, rem_dir)
    if not rem_files: return None
    raws = []
    for p in rem_files:
        try:
            raws.append(mne.io.read_raw_fif(p, preload=True, verbose="ERROR"))
        except Exception:
            pass
    if not raws: return None
    raw = raws[0] if len(raws)==1 else mne.concatenate_raws(raws, verbose="ERROR")
    try:
        raw.pick("eeg")
        if len(raw.ch_names)==0: return None
        raw.load_data()
        raw.apply_function(lambda x: x*1e6, channel_wise=True)
        if hasattr(raw, "set_unit"):
            try: raw.set_unit("eeg","uV")
            except Exception: pass
        epochs = mne.make_fixed_length_epochs(raw, duration=float(EPOCH_DUR),
                                              overlap=0.0, preload=True, verbose="ERROR")
        epochs.pick("eeg")
        return epochs
    except Exception:
        return None

def _epochs_from_stage(base: str, clean_dir: Path, annot_dir: Path, stage: str) -> mne.Epochs | None:
    fif = find_clean_fif(base, clean_dir)
    if fif is None: return None
    try:
        raw_full = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
    except Exception:
        return None

    stage_ann = get_stage_annotations(base, annot_dir, stages=(stage,))
    if stage_ann is None or len(stage_ann)==0: return None

    bad_orig = _load_artifact_windows_csv(base, clean_dir)
    t_end_approx = float(raw_full.times[-1]) if not bad_orig else (bad_orig[-1][1] + 1e9)
    mapping = _orig2clean_mapping_from_artifacts(bad_orig, t_end_orig=t_end_approx)

    by_stage = []
    for onset, dur, desc in zip(stage_ann.onset, stage_ann.duration, stage_ann.description):
        if str(desc).upper()==stage:
            by_stage.append((float(onset), float(onset)+float(dur)))
    if not by_stage: return None

    sf = float(raw_full.info["sfreq"])
    t_end = float(raw_full.times[-1]); eps = 1.0/sf

    intervals_clean = map_intervals_to_clean(by_stage, mapping) if mapping else by_stage
    kept = [(max(0.0,s), min(e,t_end)) for (s,e) in intervals_clean if (min(e,t_end)-max(0.0,s))>=MIN_SEG_S]
    if not kept or sum(e-s for (s,e) in kept) < MIN_TOTAL_S:
        return None

    parts = []
    for (s,e) in kept:
        try:
            parts.append(raw_full.copy().crop(tmin=s, tmax=e-eps, verbose="ERROR"))
        except Exception:
            pass
    if not parts: return None

    stage_raw = mne.concatenate_raws(parts, verbose="ERROR")
    try:
        stage_raw.pick("eeg")
        if len(stage_raw.ch_names)==0: return None
        stage_raw.load_data()
        stage_raw.apply_function(lambda x: x*1e6, channel_wise=True)
        if hasattr(stage_raw, "set_unit"):
            try: stage_raw.set_unit("eeg","uV")
            except Exception: pass
        epochs = mne.make_fixed_length_epochs(stage_raw, duration=float(EPOCH_DUR),
                                              overlap=0.0, preload=True, verbose="ERROR")
        epochs.pick("eeg")
        return epochs
    except Exception:
        return None

# --------- TFR par patient & aligment ---------
def compute_tfr_by_channel(epochs: mne.Epochs) -> dict[str, dict]:
    if epochs is None or len(epochs) == 0:
        return {}

    try:
        epochs_res = epochs.copy().resample(TARGET_SFREQ)
    except Exception:
        epochs_res = epochs

    freqs = np.linspace(float(FREQ_MIN), float(FREQ_MAX), int(N_FREQS))
    n_cycles = freqs * float(CYCLES_MULT)

    out = {}
    for ch in epochs_res.ch_names:
        try:
            power = epochs_res.compute_tfr(
                method="morlet",
                freqs=freqs,
                n_cycles=n_cycles,
                use_fft=True,
                return_itc=False,
                average=True,         # moyenne des epochs (comme avant)
                picks=[ch],
                decim=int(DECIM),
                output="power",       # retourne un AverageTFR
                verbose="ERROR",
            )
            # power.data shape: (n_channels=1, n_freqs, n_times)
            Z = 10.0 * np.log10(np.maximum(power.data[0], np.finfo(float).tiny))
            out[ch] = {"Z": Z, "times": power.times.copy(), "freqs": power.freqs.copy()}
        except Exception:
            continue
    return out


def aggregate_category(channel_arrays: list[np.ndarray]) -> np.ndarray | None:
    if not channel_arrays: return None
    min_f = min(a.shape[0] for a in channel_arrays)
    min_t = min(a.shape[1] for a in channel_arrays)
    stack = np.stack([a[:min_f, :min_t] for a in channel_arrays], axis=0)  # (n_subj, n_freq, n_time)
    return np.nanmean(stack, axis=0)

# --------- Plot & save ---------
def save_group_outputs(out_dir: Path, stage: str, ch: str,
                       Zmean: np.ndarray, times: np.ndarray, freqs: np.ndarray, cat: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "times.npy", times)
    np.save(out_dir / "freqs.npy", freqs)
    np.save(out_dir / f"group_{_sanitize(ch)}_power.npy", Zmean)
    spectrum = Zmean.mean(axis=1)
    pd.DataFrame({"freq_hz": freqs[:len(spectrum)], "power_db": spectrum}).to_csv(
        out_dir / f"group_{_sanitize(ch)}_spectrum.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(Zmean, aspect="auto", origin="lower",
                   extent=[times[0], times[len(Zmean[0])-1], freqs[0], freqs[len(Zmean)-1]],
                   cmap=str(CMAP))
    _format_time_axes(ax)
    cbar = plt.colorbar(im, ax=ax); cbar.set_label("Power (dB)")
    ax.set_title(f"{stage} - {ch} ({cat})")
    if VMIN is None or VMAX is None:
        lo, hi = np.percentile(Zmean, list(AUTO_PCT))
        im.set_clim(float(lo), float(hi))
    else:
        im.set_clim(float(VMIN), float(VMAX))
    fig.tight_layout()
    fig.savefig(out_dir / f"group_{_sanitize(ch)}_tfr.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

# --------- Lecture Excel (auto-détection feuille/colonnes) ---------
def load_meta_from_excel(xlsx_path: Path,
                         id_col_hint="Identifiant",
                         category_col_hint="Catégorie de patient",
                         fallback_category_candidates=("Dx avant PSG", "Dx", "Groupe", "Groupe patient")) -> pd.DataFrame:
    xlsx_path = Path(xlsx_path)
    xl = pd.read_excel(xlsx_path, sheet_name=None)  # toutes les feuilles
    chosen = None
    for name, df in xl.items():
        cols = {str(c).strip(): c for c in df.columns}
        if id_col_hint in cols:
            chosen = df
            break
    if chosen is None:
        # sinon, prend la 1ère feuille et tentera de repérer les colonnes par similarité simple
        first_name = next(iter(xl))
        chosen = xl[first_name]

    df = chosen.copy()
    cols = {str(c).strip(): c for c in df.columns}
    # identifiant
    if id_col_hint not in cols:
        # petite recherche tolérante (ex: "Identifiant " ou casse)
        key = next((k for k in cols if k.lower().startswith("identifiant")), None)
        if key: id_col = cols[key]
        else: raise SystemExit(f"Colonne 'Identifiant' introuvable dans l'Excel {xlsx_path.name}")
    else:
        id_col = cols[id_col_hint]

    # catégorie
    category_col = cols.get(category_col_hint, None)
    if category_col is None:
        # fallback direct
        for cand in fallback_category_candidates:
            if cand in cols:
                category_col = cols[cand]; break
        if category_col is None:
            # recherche tolérante
            key = next((k for k in cols if "catég" in k.lower() or "dx" in k.lower() or "groupe" in k.lower()), None)
            if key: category_col = cols[key]
    df_out = pd.DataFrame({
        "base": df[id_col].astype(str).str.strip(),
        "category": df[category_col].astype(str).str.strip() if category_col is not None else pd.Series(["NA"]*len(df))
    })

    # Si catégorie trop peu informative (1 seule valeur non-nan), on tente fallback
    def is_uninformative(series: pd.Series) -> bool:
        vals = set([v for v in series.dropna().astype(str) if v.strip() != "" and v.lower() != "nan"])
        return (len(vals) <= 1)

    if (category_col is None) or is_uninformative(df_out["category"]):
        for cand in fallback_category_candidates:
            if cand in cols:
                tmp = df[cols[cand]].astype(str).str.strip()
                if not is_uninformative(tmp):
                    df_out["category"] = tmp
                    break

    # Nettoyage NA
    df_out["category"] = df_out["category"].replace({"nan":"NA","NaN":"NA","None":"NA","": "NA"}).fillna("NA")
    return df_out.dropna(subset=["base"]).reset_index(drop=True)

# --------- Worker (processus) ---------
def _worker_one_patient(base: str, cat: str, rem_dir: str, clean_dir: str, annot_dir: str, stages: list[str]):
    """Traite un patient et renvoie un dict sérialisable: (base, cat, res)."""
    rem_dir_p   = Path(rem_dir)
    clean_dir_p = Path(clean_dir)
    annot_dir_p = Path(annot_dir)

    res = {"_ref_axes": {}}

    # REM
    if "REM" in stages:
        ep_rem = _epochs_from_rem(base, rem_dir_p)
        tfr_rem = compute_tfr_by_channel(ep_rem) if ep_rem is not None else {}
        if tfr_rem:
            res.setdefault("REM", {})
            for ch, d in tfr_rem.items():
                res["REM"].setdefault(ch, []).append(d["Z"])
            res["_ref_axes"]["REM"] = {"times": next(iter(tfr_rem.values()))["times"],
                                       "freqs": next(iter(tfr_rem.values()))["freqs"]}

    # N2 / N3
    for st in ["N2", "N3"]:
        if st not in stages:
            continue
        ep = _epochs_from_stage(base, clean_dir_p, annot_dir_p, stage=st)
        tfr = compute_tfr_by_channel(ep) if ep is not None else {}
        if tfr:
            res.setdefault(st, {})
            for ch, d in tfr.items():
                res[st].setdefault(ch, []).append(d["Z"])
            res["_ref_axes"][st] = {"times": next(iter(tfr.values()))["times"],
                                    "freqs": next(iter(tfr.values()))["freqs"]}
    return base, cat, res

# --------- Main pipeline ---------
def main():
    parser = argparse.ArgumentParser(description="Moyennes TFR par catégorie de patient (par canal, par stage) depuis un Excel.")
    parser.add_argument("--rem-dir",   type=str, default=DEFAULT_REM_DIR)
    parser.add_argument("--clean-dir", type=str, default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--annot-dir", type=str, default=DEFAULT_ANNOT_DIR)
    parser.add_argument("--out-root",  type=str, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--meta-xlsx", type=str, required=True, help="Chemin de l'Excel (métadonnées)")
    parser.add_argument("--category-col", type=str, default="Catégorie de patient",
                        help="Nom exact de la colonne catégorie à privilégier (fallback auto si non informatif).")
    parser.add_argument("--patients",  type=str, nargs="*", default=None,
                        help="Si fourni, restreint aux bases listées.")
    parser.add_argument("--stages",    type=str, nargs="*", default=["REM","N2","N3"],
                        help="Stages à traiter (par défaut: REM N2 N3)")
    parser.add_argument("--workers",   type=int, default=10,
                        help="Nombre de processus pour paralléliser par patient (>=1).")
    args = parser.parse_args()

    rem_dir   = Path(args.rem_dir)
    clean_dir = Path(args.clean_dir)
    annot_dir = Path(args.annot_dir)
    out_root  = Path(args.out_root)
    for d in [rem_dir, clean_dir, annot_dir]:
        if not d.exists():
            raise SystemExit(f"[CONFIG] Dossier introuvable: {d}")
    out_root.mkdir(parents=True, exist_ok=True)

    # patients détectés sur disque
    all_bases = discover_patients(rem_dir, clean_dir)
    if args.patients:
        sel = sorted(set(args.patients))
    else:
        sel = all_bases
    if not sel:
        raise SystemExit("Aucun patient détecté.")

    # --- META via Excel ---
    meta = load_meta_from_excel(Path(args.meta_xlsx),
                                id_col_hint="Identifiant",
                                category_col_hint=args.category_col)
    # ne garder que ceux présents sur disque (sécurité)
    meta = meta[meta["base"].isin(sel)]
    if meta.empty:
        raise SystemExit("Aucun des 'Identifiant' de l'Excel ne correspond aux dossiers trouvés.")

    # Normalisation légère (minuscule, accents simples) pour rendre le mapping robuste
    import unicodedata
    def _norm_txt(s: str) -> str:
        s = str(s or "").strip().lower()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        return s

    # Dictionnaire de correspondance demandé
    cat_mapping = {
        "syn_patients": "syn",
        "narco_patients": "narco",
        "tcsp_patients": "tcspi",
        "autres_patients": "eai",
    }

    # Appliquer le mapping (en conservant la valeur d'origine si non trouvée)
    meta["category"] = meta["category"].apply(lambda x: cat_mapping.get(_norm_txt(x), x))

    # (optionnel) Avertir s'il reste des catégories non mappées
    unmapped = sorted({c for c in meta["category"].unique() if c not in cat_mapping.values()})
    if unmapped:
        print(f"[INFO] Catégories non mappées (conservées telles quelles) : {unmapped}")

    base2cat = dict(zip(meta["base"], meta["category"]))
    categories = sorted(meta["category"].dropna().unique().tolist())

    # conteneur: category -> stage -> channel -> list[np.ndarray]
    store: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {}

    # --- Traitement patients: série ou parallèle ---
    workers = max(1, int(args.workers))
    if workers == 1:
        # boucle originale en série
        for base in sel:
            cat = base2cat.get(base)
            if not cat:
                print(f"[WARN] {base}: pas de catégorie Excel -> skip")
                continue

            # REM
            if "REM" in args.stages:
                ep_rem = _epochs_from_rem(base, rem_dir)
                tfr_rem = compute_tfr_by_channel(ep_rem) if ep_rem is not None else {}
                for ch, d in tfr_rem.items():
                    store.setdefault(cat, {}).setdefault("REM", {}).setdefault(ch, []).append(d["Z"])
                if tfr_rem:
                    store.setdefault(cat, {}).setdefault("_ref_axes", {})["REM"] = {
                        "times": next(iter(tfr_rem.values()))["times"],
                        "freqs": next(iter(tfr_rem.values()))["freqs"],
                    }

            # N2 / N3
            for st in ["N2","N3"]:
                if st not in args.stages: 
                    continue
                ep = _epochs_from_stage(base, clean_dir, annot_dir, stage=st)
                tfr = compute_tfr_by_channel(ep) if ep is not None else {}
                for ch, d in tfr.items():
                    store.setdefault(cat, {}).setdefault(st, {}).setdefault(ch, []).append(d["Z"])
                if tfr:
                    store.setdefault(cat, {}).setdefault("_ref_axes", {})[st] = {
                        "times": next(iter(tfr.values()))["times"],
                        "freqs": next(iter(tfr.values()))["freqs"],
                    }
    else:
        # parallèle par patient
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = []
            for base in sel:
                cat = base2cat.get(base)
                if not cat:
                    print(f"[WARN] {base}: pas de catégorie Excel -> skip")
                    continue
                futures.append(ex.submit(
                    _worker_one_patient,
                    base, cat,
                    str(rem_dir), str(clean_dir), str(annot_dir),
                    list(args.stages)
                ))

            for fut in as_completed(futures):
                try:
                    base, cat, res = fut.result()
                except Exception as e:
                    print(f"[ERROR] worker a échoué: {e}")
                    continue

                # fusion des résultats
                if "_ref_axes" in res:
                    for st2, ax in res["_ref_axes"].items():
                        store.setdefault(cat, {}).setdefault("_ref_axes", {})[st2] = ax
                for st in ["REM", "N2", "N3"]:
                    if st in res:
                        for ch, mats in res[st].items():
                            store.setdefault(cat, {}).setdefault(st, {}).setdefault(ch, []).extend(mats)

    # --- Agrégation & sorties ---
    root_group = out_root / "_by_patient_category"
    n_groups = 0
    for cat in categories:
        cat_dict = store.get(cat, {})
        if not cat_dict: 
            continue
        for st in args.stages:
            ch_dict = cat_dict.get(st, {})
            if not ch_dict:
                print(f"[INFO] Cat '{cat}' Stage '{st}': aucun canal à agréger.")
                continue
            axes = cat_dict.get("_ref_axes", {}).get(st, None)
            if not axes:
                # fallback: prend tailles de la 1ère matrice
                any_ch = next(iter(ch_dict.keys()))
                example = ch_dict[any_ch][0]
                times = np.linspace(0, EPOCH_DUR, example.shape[1], endpoint=False)
                freqs = np.linspace(FREQ_MIN, FREQ_MAX, example.shape[0])
            else:
                times, freqs = axes["times"], axes["freqs"]

            stage_out = root_group / _sanitize(cat) / st
            for ch, mats in ch_dict.items():
                Zmean = aggregate_category(mats)
                if Zmean is None:
                    continue
                save_group_outputs(stage_out, st, ch, Zmean, times, freqs, cat)
                n_groups += 1

    print(f"[DONE] Group plots écrits: {n_groups}. Racine: {root_group}")

if __name__ == "__main__":
    main()
