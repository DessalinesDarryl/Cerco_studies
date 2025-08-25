# utils/compare_mat.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import gc
import os
import tempfile

import numpy as np
import pandas as pd
import scipy.io as sio
import unicodedata


# -------------------------
# Helpers généraux
# -------------------------
def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def _norm_label(x) -> str:
    """Normalise un label: lower, sans accents, '_'/'-' -> espace, trim, espaces compressés."""
    if isinstance(x, np.ndarray):
        try:
            x = x.item()
        except Exception:
            x = str(x)
    s = str(x)
    s = _strip_accents(s).lower().replace("_", " ").replace("-", " ").strip()
    s = " ".join(s.split())
    return s

def _merge_intervals(intervals: List[Tuple[float, float]], tol: float = 0.0) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    ints = []
    for s, e in intervals:
        s = float(s); e = float(e)
        if e < s:
            s, e = e, s
        ints.append((s, e))
    ints.sort(key=lambda x: x[0])

    out: List[Tuple[float, float]] = []
    cs, ce = ints[0]
    for s, e in ints[1:]:
        if s <= ce + tol:
            ce = max(ce, e)
        else:
            out.append((cs, ce))
            cs, ce = s, e
    out.append((cs, ce))
    return out

def _is_inside(start: float, end: float, intervals: List[Tuple[float, float]], tol: float = 0.5) -> bool:
    s = float(start); e = float(end)
    for a, b in intervals:
        if (a - tol) <= s and e <= (b + tol):
            return True
    return False


# -------------------------
# Extraction “times” robuste
# -------------------------
def _extract_interval(times, pad_sec: float = 2.0) -> Optional[Tuple[float, float]]:
    """
    Retourne (start, end) en s à partir d'un grand nombre de formats possibles :
      - scalaire / array numériques (1 valeur => centré ±pad_sec ; >=2 => [t0, tN])
      - struct MATLAB (attributs) avec champs {position,duration} ou {tmin,tmax} etc.
      - dict-like avec clés {position,duration} ou {onset,offset} ou {start,end}
    Renvoie None si rien d'exploitable.
    """
    if times is None:
        return None

    # 1) Tentative directe: vecteur numérique
    try:
        t = np.asarray(times, dtype=float).ravel()
        if t.size == 1:
            c = float(t[0]);  return (c - pad_sec, c + pad_sec)
        if t.size >= 2:
            return (float(t[0]), float(t[-1]))
    except Exception:
        pass

    # 2) Struct MATLAB (objet avec attributs)
    try:
        # Cas (position, duration)
        if hasattr(times, "position"):
            pos = np.asarray(getattr(times, "position")).astype(float).ravel()
            if pos.size:
                start = float(np.min(pos))
                dur = getattr(times, "duration", None)
                if dur is not None:
                    dur = np.asarray(dur).astype(float).ravel()
                    end = float(np.max(pos + (dur[0] if dur.size else 0.0)))
                else:
                    end = float(np.max(pos))
                if end < start:
                    start, end = end, start
                return (start, end)
        # Cas (tmin, tmax) / (onset, offset) / (start, end)
        for a, b in (("tmin", "tmax"), ("onset", "offset"), ("start", "end")):
            if hasattr(times, a) and hasattr(times, b):
                s = float(np.asarray(getattr(times, a)).ravel()[0])
                e = float(np.asarray(getattr(times, b)).ravel()[0])
                if e < s:
                    s, e = e, s
                return (s, e)
    except Exception:
        pass

    # 3) Dict-like
    if isinstance(times, dict):
        try:
            # position/duration
            if "position" in times:
                pos = np.asarray(times["position"], dtype=float).ravel()
                if pos.size:
                    start = float(np.min(pos))
                    if "duration" in times and times["duration"] is not None:
                        dur = np.asarray(times["duration"], dtype=float).ravel()
                        end = float(np.max(pos + (dur[0] if dur.size else 0.0)))
                    else:
                        end = float(np.max(pos))
                    if end < start:
                        start, end = end, start
                    return (start, end)
            # onset/offset ou start/end
            for a, b in (("onset", "offset"), ("start", "end"), ("tmin", "tmax")):
                if a in times and b in times:
                    s = float(np.asarray(times[a]).ravel()[0])
                    e = float(np.asarray(times[b]).ravel()[0])
                    if e < s:
                        s, e = e, s
                    return (s, e)
        except Exception:
            pass

    return None


# -------------------------
# Chargement des intervalles REM phasiques
# -------------------------
def _load_rem_phasic_intervals(mat_path: Path, pad_sec: float = 2.0) -> List[Tuple[float, float]]:
    """
    Lit un .mat et extrait les intervalles taggés 'rem phasic/phasique'.
    Retourne une liste d'intervalles [(start, end), ...] fusionnés/triés.
    """
    m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    # Tenter plusieurs clés possibles
    events = None
    for k in ("events", "Events", "annotations", "Annotations"):
        if k in m:
            events = m[k]
            break
    if events is None:
        # rien à extraire
        del m; gc.collect()
        return []

    # Libère le gros dict au plus tôt
    del m; gc.collect()

    # Normaliser l'itération
    try:
        it = np.atleast_1d(events).flat
    except Exception:
        it = [events]

    rem_phasic: List[Tuple[float, float]] = []
    for ev in it:
        # Essayer plusieurs champs label
        label = None
        for fld in ("label", "type", "name", "description"):
            if hasattr(ev, fld):
                label = getattr(ev, fld)
                break
            if isinstance(ev, dict) and fld in ev:
                label = ev[fld]
                break
        if label is None:
            continue

        lab = _norm_label(label)
        # tolérant aux variantes: "rem phasic", "rem phasique", "rem_phasic", etc.
        if ("rem" in lab) and ("phasic" in lab or "phasique" in lab):
            # Essayer plusieurs sources de temps
            times = None
            for fld in ("times", "time", "timestamp"):
                if hasattr(ev, fld):
                    times = getattr(ev, fld); break
                if isinstance(ev, dict) and fld in ev:
                    times = ev[fld]; break
            # Si pas de champ, on tentera directement les couples tmin/tmax etc. dans _extract_interval
            interval = _extract_interval(times, pad_sec=pad_sec) if times is not None else None
            if interval is None:
                # Essayer directement sur la structure de ev
                interval = _extract_interval(ev, pad_sec=pad_sec)
            if interval is None:
                continue
            start, end = interval
            rem_phasic.append((start, end))

    return _merge_intervals(rem_phasic, tol=0.0)


# -------------------------
# Comparaison principale
# -------------------------
def compare_with_mat(
    df_windows: pd.DataFrame,
    raw_dir: Path,
    patient_id: str,
    out_patient_dir: Path,
    *,
    glob_pattern: str = "events_{pid}*.mat",
    inside_tol_sec: float = 0.5,
    pad_sec_if_single_time: float = 2.0,
    errors_xlsx_suffix: str = "_microstates_errors.xlsx",
    verbose: bool = True,
) -> Optional[pandas.DataFrame]:
    """
    Compare les labels auto (df_windows['label']) aux annotations manuelles (.mat).
    - df_windows attend les colonnes: 'tmin', 'tmax', 'label' (label auto).
    - Cherche les .mat dans: raw_dir/patient_id/glob_pattern
    - Sauvegarde les erreurs dans un Excel si divergences (écriture atomique).

    Retourne le DataFrame des erreurs (ou None si pas de .mat). S'il n'y a pas
    d'erreur, renvoie par convention un DF des "bons".
    """
    mat_folder = raw_dir / patient_id
    candidates = sorted(mat_folder.glob(glob_pattern.format(pid=patient_id)))
    if not candidates:
        if verbose:
            print(f"[{patient_id}] Pas d’événements .mat pour comparaison manuelle.")
        return None

    # plus récent
    mat_path = max(candidates, key=lambda p: p.stat().st_mtime)
    rem_phasic = _load_rem_phasic_intervals(mat_path, pad_sec=pad_sec_if_single_time)

    # Sanity: forcer numérique
    df = df_windows.copy()
    df["tmin"] = pd.to_numeric(df["tmin"], errors="coerce")
    df["tmax"] = pd.to_numeric(df["tmax"], errors="coerce")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["tmin", "tmax"]).reset_index(drop=True)

    erreurs, bons = [], []
    for _, row in df.iterrows():
        start, end, pred = float(row["tmin"]), float(row["tmax"]), str(row["label"])
        true = "phasic" if _is_inside(start, end, rem_phasic, tol=inside_tol_sec) else "tonic"
        result = {"tmin": start, "tmax": end, "auto_label": pred, "true_label": true}
        (bons if pred == true else erreurs).append(result)

    if erreurs:
        df_err = pd.DataFrame(erreurs)
        out_path = out_patient_dir / f"{patient_id}{errors_xlsx_suffix}"
        # écriture atomique
        with tempfile.TemporaryDirectory(prefix=f"{patient_id}_", dir=out_patient_dir) as td:
            tmp_xlsx = Path(td) / out_path.name
            df_err.to_excel(tmp_xlsx, index=False, engine="xlsxwriter")
            os.replace(tmp_xlsx, out_path)
        if verbose:
            print(f"[{patient_id}] {len(erreurs)} erreurs sauvegardées ({out_path.name}).")
        return df_err
    else:
        if verbose:
            print(f"[{patient_id}] 0 erreurs, pas de fichier d'erreurs généré.")
        return pd.DataFrame(bons)
