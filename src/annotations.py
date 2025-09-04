# src/annotations.py
from __future__ import annotations
import pandas as pd
import numpy as np
import mne
from pathlib import Path

__all__ = [
    "load_annotation_file",
    "get_rem_annotations",
    "get_stage_annotations",
]

# --------- Helpers ---------

def _norm_stage(s: str) -> str:
    s = (str(s) or "").strip().upper()
    mapping = {
        "SP": "REM", "R": "REM", "REM": "REM",
        "V": "W", "WAKE": "W", "W": "W",
        "S1": "N1", "N1": "N1",
        "S2": "N2", "STAGE2": "N2", "NREM2": "N2", "N2": "N2",
        "S3": "N3", "STAGE3": "N3", "NREM3": "N3", "N3": "N3",
    }
    return mapping.get(s, s)

def _read_hypnogram_any(path: Path) -> pd.DataFrame:
    """
    Retourne un DataFrame trié: colonnes ['start','duration','stage'] en secondes.
    Gère :
      - TXT tabulé (start, hh:mm:ss, stage, index),
      - CSV (délimiteur auto) avec:
          * 'Position (epoch)' ou 'Epoch'/'epoch' (start = (epoch-1)*30),
          * ou 'Absolute position (hh:mm:ss.ms)' (start = delta depuis 1re ligne),
          * et une colonne de stade: 'Default staging set ("Stage")' / 'Stage' / 'Stade'.
    """
    path = Path(path)
    suf = path.suffix.lower()

    if suf == ".txt":
        # Format tabulé: start (s), time 'hh:mm:ss', stage, index
        df = pd.read_csv(
            path, sep="\t",
            names=["start", "time", "stage", "index"],
            engine="python"
        )
        df = df.dropna(subset=["start", "stage"]).copy()
        df["start"] = df["start"].astype(float)
        stage = df["stage"].map(_norm_stage)

    elif suf == ".csv":
        # Délimiteur auto + colonnes flexibles
        df = pd.read_csv(path, sep=None, engine="python")
        cols = {c.lower().strip(): c for c in df.columns}

        # --- start ---
        if "position (epoch)" in cols:
            df["start"] = (pd.to_numeric(df[cols["position (epoch)"]], errors="coerce").astype("Int64") - 1) * 30.0
        elif "epoch" in cols:
            df["start"] = (pd.to_numeric(df[cols["epoch"]], errors="coerce").astype("Int64") - 1) * 30.0
        elif "absolute position (hh:mm:ss.ms)" in cols:
            # calcule le delta par rapport à la 1re valeur
            t0 = pd.to_timedelta(df[cols["absolute position (hh:mm:ss.ms)"]].iloc[0])
            df["start"] = (pd.to_timedelta(df[cols["absolute position (hh:mm:ss.ms)"]]) - t0).dt.total_seconds()
        else:
            # Fallback si déjà une colonne 'start'
            if "start" in cols:
                df["start"] = pd.to_numeric(df[cols["start"]], errors="coerce")
            else:
                raise ValueError("CSV hypnogram: colonne pour le temps de début introuvable "
                                 "(Position (epoch) / Epoch / Absolute position / start).")

        # --- stage ---
        stage_col = (cols.get('default staging set ("stage")')
                     or cols.get("stage")
                     or cols.get("stade"))
        if not stage_col:
            raise ValueError("CSV hypnogram: colonne de stade introuvable (Stage/Stade).")
        stage = df[stage_col].map(_norm_stage)

    else:
        raise ValueError(f"Extension non gérée: {suf}")

    # Construction finale
    out = pd.DataFrame({
        "start": pd.to_numeric(df["start"], errors="coerce"),
        "stage": stage,
    }).dropna(subset=["start", "stage"]).sort_values("start")

    # Durée = diff des starts (on ignore la dernière ligne si durée inconnue)
    out["duration"] = out["start"].shift(-1) - out["start"]
    out = out.iloc[:-1].copy()  # retire la dernière (durée NaN)

    # Nettoyage: garde durées > 0
    out = out[(out["duration"] > 0)].reset_index(drop=True)
    return out[["start", "duration", "stage"]]

# --------- API publique ---------

def load_annotation_file(txt_or_csv_path) -> np.ndarray:
    """
    Charge un .txt/.csv d’hypnogramme et retourne les segments REM:
      np.array([[start, duration], ...])
    """
    df = _read_hypnogram_any(Path(txt_or_csv_path))
    rem = df[df["stage"] == "REM"]
    return rem[["start", "duration"]].to_numpy(dtype=float)

def _find_annot_for_base(base_name: str, annot_dir: str | Path) -> list[Path]:
    """
    Cherche des fichiers d’annotation dans annot_dir/{patient_code}/*.(txt|csv).
    """
    patient_code = str(base_name).split("_")[0]
    pdir = Path(annot_dir) / patient_code
    return list(pdir.glob("*.txt")) + list(pdir.glob("*.csv"))

def get_rem_annotations(base_name: str, annot_dir: str | Path) -> mne.Annotations | None:
    """
    Retourne des annotations MNE (REM) pour un sujet, sinon None.
    """
    for p in _find_annot_for_base(base_name, annot_dir):
        try:
            df = _read_hypnogram_any(p)
            rem = df[df["stage"] == "REM"]
            if not rem.empty:
                return mne.Annotations(
                    onset=rem["start"].astype(float).tolist(),
                    duration=rem["duration"].astype(float).tolist(),
                    description=["REM"] * len(rem),
                )
        except Exception:
            continue
    return None

def get_stage_annotations(base_name: str, annot_dir: str | Path, stages=("N2", "N3")) -> mne.Annotations | None:
    """
    Retourne des annotations MNE pour un sous-ensemble de stades (par défaut N2/N3).
    """
    want = {s.upper() for s in stages}
    for p in _find_annot_for_base(base_name, annot_dir):
        try:
            df = _read_hypnogram_any(p)
            keep = df[df["stage"].isin(want)]
            if not keep.empty:
                return mne.Annotations(
                    onset=keep["start"].astype(float).tolist(),
                    duration=keep["duration"].astype(float).tolist(),
                    description=keep["stage"].tolist(),
                )
        except Exception:
            continue
    return None
