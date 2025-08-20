# compare_mat.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

def _as_1d_float_array(x) -> np.ndarray:
    try:
        a = np.asarray(x, dtype=float).ravel()
        return a if a.size else np.array([], dtype=float)
    except Exception:
        return np.array([], dtype=float)

def _norm_label(x) -> str:
    if isinstance(x, np.ndarray):
        try:
            x = x.item()
        except Exception:
            x = str(x)
    return str(x).strip().lower().replace("_", " ")

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
    s = float(start)
    e = float(end)
    for a, b in intervals:
        if (a - tol) <= s and e <= (b + tol):
            return True
    return False

def _load_rem_phasic_intervals(mat_path: Path, pad_sec: float = 2.0) -> List[Tuple[float, float]]:
    try:
        from scipy.io import loadmat
    except ImportError as e:
        raise RuntimeError("scipy n'est pas installé : pip install scipy") from e

    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    events = np.atleast_1d(mat.get("events", []))

    rem_phasic: List[Tuple[float, float]] = []
    for ev in events:
        label = _norm_label(getattr(ev, "label", ""))
        if label == "rem phasique":
            t = _as_1d_float_array(getattr(ev, "times", []))
            if t.size == 0:
                continue
            if t.size >= 2:
                start, end = float(t[0]), float(t[-1])
            else:
                center = float(t[0])
                start, end = center - pad_sec, center + pad_sec
            if end < start:
                start, end = end, start
            rem_phasic.append((start, end))

    return _merge_intervals(rem_phasic, tol=0.0)

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

    Retourne le DataFrame des erreurs (ou None si pas de .mat ou aucune erreur).
    """
    mat_folder = raw_dir / patient_id
    # IMPORTANT : utiliser glob (wildcard) plutôt que Path.exists()
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
        start, end, pred = float(row["tmin"]), float(row["tmax"]), str(row["label"])
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
        return pd.DataFrame(bons)  # utile si tu veux exploiter les “bons” ensuite

