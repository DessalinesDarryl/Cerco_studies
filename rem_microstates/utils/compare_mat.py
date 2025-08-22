# compare_mat.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import unicodedata


# -------------------------
# Helpers généraux
# -------------------------
def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def _norm_label(x) -> str:
    """Normalise un label: lower, sans accents, remplace '_' et '-' par espace, trim."""
    if isinstance(x, np.ndarray):
        try:
            x = x.item()
        except Exception:
            x = str(x)
    s = str(x)
    s = _strip_accents(s).lower().replace("_", " ").replace("-", " ").strip()
    # compress espaces
    s = " ".join(s.split())
    return s

def _as_1d_float_array(x) -> np.ndarray:
    try:
        a = np.asarray(x, dtype=float).ravel()
        return a if a.size else np.array([], dtype=float)
    except Exception:
        return np.array([], dtype=float)

def _merge_intervals(intervals: List[Tuple[float, float]], tol: float = 0.0) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    ints = sorted((float(s), float(e)) if float(s) <= float(e) else (float(e), float(s))
                  for s, e in intervals)
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
    Retourne (start, end) en s à partir d'une variété de formats:
      - scalaire / array numériques (1 valeur => centré ±pad_sec ; >=2 => [t0, tN])
      - struct MATLAB (attributs) avec champs 'position' et éventuellement 'duration'
      - dict-like avec clés 'position'/'duration'
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
        if hasattr(times, "position"):
            pos = np.asarray(getattr(times, "position")).astype(float).ravel()
            dur = getattr(times, "duration", None)
            if pos.size:
                start = float(np.min(pos))
                if dur is not None:
                    dur = np.asarray(dur).astype(float).ravel()
                    end = float(np.max(pos + (dur[0] if dur.size else 0.0)))
                else:
                    end = float(np.max(pos))
                if end < start:
                    start, end = end, start
                return (start, end)
    except Exception:
        pass

    # 3) Dict-like (si table importée comme dict)
    if isinstance(times, dict):
        try:
            pos = times.get("position", None)
            if pos is not None:
                pos = np.asarray(pos, dtype=float).ravel()
                if pos.size:
                    start = float(np.min(pos))
                    dur = times.get("duration", None)
                    if dur is not None:
                        dur = np.asarray(dur, dtype=float).ravel()
                        end = float(np.max(pos + (dur[0] if dur.size else 0.0)))
                    else:
                        end = float(np.max(pos))
                    if end < start:
                        start, end = end, start
                    return (start, end)
        except Exception:
            pass

    return None


# -------------------------
# Chargement des intervalles REM phasique
# -------------------------
def _load_rem_phasic_intervals(mat_path: Path, pad_sec: float = 2.0) -> List[Tuple[float, float]]:
    try:
        from scipy.io import loadmat
    except ImportError as e:
        raise RuntimeError("scipy n'est pas installé : pip install scipy") from e

    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    events = np.atleast_1d(mat.get("events", []))

    # gestion des cas degens (events scalaire/objet)
    try:
        it = events.flat
    except Exception:
        it = [events]

    rem_phasic: List[Tuple[float, float]] = []
    for ev in it:
        label = _norm_label(getattr(ev, "label", ""))
        # tolérant aux variantes: "rem phasique", "rem phasic", "rem_phasic", etc.
        if ("rem" in label) and ("phasic" in label or "phasique" in label):
            interval = _extract_interval(getattr(ev, "times", None), pad_sec=pad_sec)
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
) -> Optional[pd.DataFrame]:
    """
    Compare les labels auto (df_windows['label']) aux annotations manuelles (.mat).
    - df_windows attend les colonnes: 'tmin', 'tmax', 'label' (label auto).
    - Cherche les .mat dans: raw_dir/patient_id/glob_pattern
    - Sauvegarde les erreurs dans un Excel si divergences.

    Retourne le DataFrame des erreurs (ou None si pas de .mat). S'il n'y a
    aucune erreur, renvoie par convention un DF des "bons" (pour exploitation
    éventuelle), comme dans ta version précédente.
    """
    mat_folder = raw_dir / patient_id
    candidates = sorted(mat_folder.glob(glob_pattern.format(pid=patient_id)))
    if not candidates:
        if verbose:
            print(f"[{patient_id}] Pas d’événements .mat pour comparaison manuelle.")
        return None

    # on prend le plus récent s’il y en a plusieurs
    mat_path = max(candidates, key=lambda p: p.stat().st_mtime)

    rem_phasic = _load_rem_phasic_intervals(mat_path, pad_sec=pad_sec_if_single_time)

    erreurs = []
    bons = []
    for _, row in df_windows.iterrows():
        start, end, pred = float(row["tmin"]), float(row["tmax"]), str(row["label"]).strip().lower()
        true = "phasic" if _is_inside(start, end, rem_phasic, tol=inside_tol_sec) else "tonic"
        result = {"tmin": start, "tmax": end, "auto_label": pred, "true_label": true}
        (bons if pred == true else erreurs).append(result)

    if erreurs:
        df_err = pd.DataFrame(erreurs)
        out_path = out_patient_dir / f"{patient_id}{errors_xlsx_suffix}"
        df_err.to_excel(out_path, index=False)
        if verbose:
            print(f"[{patient_id}] {len(erreurs)} erreurs sauvegardées ({out_path.name}).")
        return df_err
    else:
        if verbose:
            print(f"[{patient_id}] 0 erreurs, pas de fichier généré.")
        return pd.DataFrame(bons)
