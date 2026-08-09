"""
human_data_adapter.py
----------------------
Pont entre les enregistrements patients reels (EDF/FIF) et le pipeline
d'echelle de severite RSWA (severity_metrics.py / main_pipeline.py).

Reprend explicitement la methodologie de l'ancien pipeline
(scripts_fonctionnels/01_preprocess.py et 03_segment_rswa.py) :
  - lecture EDF/FIF (mne)
  - filtrage EMG 10-100 Hz + notch secteur (memes parametres que
    preprocessing.py, coherents avec Ferri/Frauscher/Khalil/McCarter)
  - hypnogramme -> intervalles REM (formats {patient}_hypnoEXP.txt/.csv,
    identiques a 01_preprocess.py ; fallback sur les annotations "REM"
    deja presentes dans un .fif deja pretraite)
  - detection d'artefacts (YASA, meme algorithme que 01_preprocess.py)

DECISION EXPLICITE (different de l'ancien pipeline) :
------------------------------------------------------
01_preprocess.py annote les artefacts ("ARTEFACT") ET 02_segment_rem.py
les SOUSTRAIT des intervalles REM avant decoupage en epoques (fonction
_subtract_intervals). Ici, les artefacts sont UNIQUEMENT ANNOTES :
aucune mini-epoque n'est retiree du signal REM. En sortie,
`extract_rem_amplitudes_with_artifacts` renvoie, en plus des
amplitudes, un masque booleen `artifact_flags_1s` aligne mini-epoque
par mini-epoque. Ce masque est ensuite uniquement utilise pour
ANNOTER (jamais filtrer) :
    - chaque activation detectee (activation_detection.detect_activations
      -> champ 'artifact_overlap_fraction') ;
    - une metrique globale (severity_metrics.compute_all_metrics ->
      'pct_epochs_artifact').
Cela preserve integralement le denominateur temporel REM (densites/min,
% de temps) et laisse une exclusion eventuelle des epoques artefactees
explicite et documentee en aval, plutot que silencieuse en amont.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import re

import mne
import numpy as np

from preprocessing import preprocess_emg, mini_epoch_amplitude

# ============================================================
# Hypnogramme -> intervalles REM
# (logique identique a 01_preprocess.py, pour rester coherent avec les
#  fichiers hypno deja utilises dans l'ancien pipeline)
# ============================================================

YASA_CODE = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
EXP_NUM_TO_YASA = {1: 0, 2: 4, 3: 1, 4: 2, 5: 3}


def _infer_epoch_len(seconds_col: List[float]) -> float:
    if len(seconds_col) < 2:
        return 30.0
    diffs = np.diff(seconds_col)
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if len(diffs) else 30.0


def read_hypno_txt(path) -> Tuple[np.ndarray, float]:
    """Lit un hypnogramme {patient}_hypnoEXP.txt (identique a
    01_preprocess.py::_read_hypno_txt)."""
    seconds, labels, codes_num = [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 3:
                continue
            try:
                sec = float(parts[0])
            except Exception:
                continue
            seconds.append(sec)
            labels.append(parts[2].upper())
            try:
                codes_num.append(int(parts[3]) if len(parts) >= 4 else None)
            except Exception:
                codes_num.append(None)

    if not seconds:
        raise ValueError(f"Hypnogramme vide ou non lisible : {path}")

    epoch_len = _infer_epoch_len(seconds)
    out = []
    for lab, cnum in zip(labels, codes_num):
        if lab in YASA_CODE:
            out.append(YASA_CODE[lab])
        elif cnum is not None and cnum in EXP_NUM_TO_YASA:
            out.append(EXP_NUM_TO_YASA[cnum])
        else:
            out.append(-2)
    return np.asarray(out, dtype=int), epoch_len


def read_hypno_csv(path) -> Tuple[np.ndarray, float]:
    """Lit un hypnogramme .csv generique (identique a
    01_preprocess.py::_read_hypno_csv)."""
    import pandas as pd

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    epoch_len = 30.0
    if "seconds" in cols:
        sec_col = df[cols["seconds"]].values
        if len(sec_col) >= 2:
            epoch_len = _infer_epoch_len(list(sec_col))

    labels = None
    if "stage" in cols:
        labels = df[cols["stage"]].astype(str).str.upper().tolist()
    elif "label" in cols:
        labels = df[cols["label"]].astype(str).str.upper().tolist()

    codes = None
    if "code" in cols:
        try:
            codes = df[cols["code"]].astype(float).astype(int).tolist()
        except Exception:
            pass

    out = []
    if labels is not None:
        for i, lab in enumerate(labels):
            if lab in YASA_CODE:
                out.append(YASA_CODE[lab])
            else:
                cnum = None if codes is None else codes[i]
                out.append(EXP_NUM_TO_YASA.get(cnum, -2) if cnum is not None else -2)
    elif codes is not None:
        out = [EXP_NUM_TO_YASA.get(c, -2) for c in codes]
    else:
        raise ValueError(f"Colonnes de stade introuvables dans {path}.")

    return np.asarray(out, dtype=int), float(epoch_len)


def load_hypnogram(hypno_path) -> Tuple[np.ndarray, float]:
    hypno_path = Path(hypno_path)
    if hypno_path.suffix.lower() == ".txt":
        return read_hypno_txt(hypno_path)
    return read_hypno_csv(hypno_path)


def hypno_to_rem_intervals(hypno_epochs: np.ndarray, epoch_len_sec: float):
    """Convertit les codes de stade en intervalles REM (start_s, end_s).
    Identique a 01_preprocess.py::_hypno_to_rem_intervals."""
    intervals, current_start = [], None
    for i, code in enumerate(hypno_epochs):
        if code == 4:  # REM
            if current_start is None:
                current_start = i * epoch_len_sec
        else:
            if current_start is not None:
                end_t = i * epoch_len_sec
                if end_t > current_start:
                    intervals.append((current_start, end_t))
                current_start = None
    if current_start is not None:
        end_t = len(hypno_epochs) * epoch_len_sec
        if end_t > current_start:
            intervals.append((current_start, end_t))
    return intervals


# ============================================================
# Detection d'artefacts (YASA) -- ANNOTATION SEULE, JAMAIS DE SUPPRESSION
# ============================================================

def _merge_intervals(intervals):
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


def _upsample_hypno(hypno_epochs, epoch_len_sec, n_samples, sf):
    import yasa

    hypno_up = yasa.hypno_upsample_to_data(
        hypno=hypno_epochs,
        sf_hypno=1.0 / float(epoch_len_sec),
        data=np.zeros((1, n_samples), dtype=float),
        sf_data=sf,
    )
    return hypno_up.astype(int)


def detect_artifact_windows(
    raw,
    channel_name,
    win_sec=4.0,
    method="covar",
    threshold=3.0,
    hypno_samples=None,
    include_stages="rem",
):
    """
    Detecte les fenetres artefactees sur le canal EMG cible avec YASA.
    Reprend la logique de 01_preprocess.py::_detect_artifacts, appliquee
    au canal EMG analyse (au lieu des canaux EEG de l'ancien script,
    puisque c'est ce canal qui alimente directement le score de severite
    RSWA ici).

    Retourne une liste d'intervalles (start_s, end_s), PUREMENT
    INFORMATIVE (cf. docstring du module : jamais soustraite du signal).
    """
    import yasa

    raw_ch = raw.copy().pick([channel_name])
    sf = float(raw_ch.info["sfreq"])
    data = raw_ch.get_data() * 1e6  # µV
    n_samples = data.shape[1]

    include_map = {"sleep": (1, 2, 3, 4), "rem": (4,), "all": (0, 1, 2, 3, 4)}
    include = include_map.get(include_stages, (4,))

    hypno_vec = hypno_samples.astype(int) if hypno_samples is not None else None
    if hypno_vec is not None and len(hypno_vec) != n_samples:
        raise ValueError(
            f"Hypno upsample ({len(hypno_vec)}) != n_samples ({n_samples})."
        )

    art, _ = yasa.art_detect(
        data=data, sf=sf, window=win_sec,
        hypno=hypno_vec, include=include,
        method=method, threshold=threshold, verbose=False,
    )

    t_end = n_samples / sf
    windows_s = []
    for i, is_art in enumerate(art):
        if is_art:
            s = i * win_sec
            e = min((i + 1) * win_sec, t_end)
            if e > s:
                windows_s.append((float(s), float(e)))
    return _merge_intervals(windows_s)


# ============================================================
# Extraction des amplitudes par mini-epoque + masque d'artefacts
# ============================================================

def extract_rem_amplitudes_with_artifacts(
    fif_or_edf_path,
    channel_name,
    hypno_path=None,
    epoch_duration_s=1.0,
    mains_freq=50.0,
    bandpass=(10.0, 100.0),
    detect_artifacts=True,
    artifact_win_sec=4.0,
    artifact_method="covar",
    artifact_threshold=3.0,
):
    """
    Point d'entree principal : lit un enregistrement reel, filtre le
    canal EMG cible (preprocessing.preprocess_emg, 10-100 Hz + notch
    secteur -- identique a toutes les references du domaine), restreint
    au sommeil REM (hypnogramme reel, meme format que l'ancien pipeline),
    et calcule l'amplitude redressee par mini-epoque.

    Contrairement a 02_segment_rem.py (ancien pipeline), les fenetres
    artefactees ne sont PAS retirees des intervalles REM : elles sont
    seulement detectees, annotees, et renvoyees sous forme d'un masque
    booleen aligne sur `amplitudes_1s` (choix explicite, cf. docstring
    du module).

    Parametres
    ----------
    fif_or_edf_path : chemin vers le fichier .fif ou .edf du patient
    channel_name : nom du canal EMG a analyser (ex. "Menton", "JAMBG",
        "JAMBD", "EMG1", "EMG2" -- canaux reels de l'ancien pipeline)
    hypno_path : chemin vers l'hypnogramme ({patient}_hypnoEXP.txt/.csv).
        None -> fallback sur les annotations "REM" deja presentes dans
        le fichier (cas d'un .fif deja pretraite par 01_preprocess.py).
    detect_artifacts : active la detection YASA (desactivable si aucun
        canal EEG/EMG de reference suffisant, ou pour accelerer les
        tests).

    Retourne
    --------
    amplitudes_1s : np.ndarray
        Amplitude redressee par mini-epoque, sur tout le temps REM
        (artefacts INCLUS).
    artifact_flags_1s : np.ndarray (bool) ou None
        True si la mini-epoque chevauche une fenetre artefactee.
        None si detect_artifacts=False.
    epoch_duration_s : float
        Rappel de la duree de mini-epoque utilisee (pour tracabilite).
    rem_intervals : list[(start_s, end_s)]
        Intervalles REM utilises (NON amputes des artefacts).
    """
    path = Path(fif_or_edf_path)
    reader = mne.io.read_raw_fif if path.suffix.lower() == ".fif" else mne.io.read_raw_edf
    raw = reader(path, preload=True, verbose=False)

    if channel_name not in raw.ch_names:
        raise ValueError(
            f"Canal '{channel_name}' absent de {path.name} "
            f"(canaux disponibles : {raw.ch_names})"
        )

    sf = float(raw.info["sfreq"])

    # ---- 1) Intervalles REM -------------------------------------------
    hypno_epochs = epoch_len = None
    if hypno_path is not None:
        hypno_epochs, epoch_len = load_hypnogram(hypno_path)
        rem_intervals = hypno_to_rem_intervals(hypno_epochs, epoch_len)
    else:
        rem_intervals = [
            (float(a["onset"]), float(a["onset"] + a["duration"]))
            for a in raw.annotations
            if a["description"] == "REM"
        ]

    if not rem_intervals:
        raise RuntimeError(f"Aucun segment REM trouve pour {path.name}.")

    # ---- 2) Detection d'artefacts (annotation seule) -------------------
    artifact_windows = []
    if detect_artifacts:
        hypno_samples = None
        if hypno_epochs is not None:
            hypno_samples = _upsample_hypno(hypno_epochs, epoch_len, raw.n_times, sf)
        try:
            artifact_windows = detect_artifact_windows(
                raw, channel_name, win_sec=artifact_win_sec,
                method=artifact_method, threshold=artifact_threshold,
                hypno_samples=hypno_samples,
                include_stages="rem" if hypno_samples is not None else "all",
            )
        except Exception as e:
            print(
                f"[human_data_adapter] Detection d'artefacts ignoree "
                f"({path.name}, {channel_name}) : {e}"
            )
            artifact_windows = []

    # ---- 3) Signal EMG brut du canal cible ------------------------------
    raw_ch = raw.copy().pick([channel_name])
    signal_uv = raw_ch.get_data()[0] * 1e6

    # ---- 4) Filtrage + amplitude par mini-epoque, segment REM par
    #        segment REM (evite les artefacts de bord entre segments
    #        non contigus lors du filtrage filtfilt) -----------------------
    amplitudes_parts, artifact_parts = [], []
    for start_s, end_s in rem_intervals:
        i0, i1 = int(round(start_s * sf)), int(round(end_s * sf))
        if i1 <= i0:
            continue
        seg = signal_uv[i0:i1]
        if len(seg) < int(round(sf * 0.5)):  # trop court pour filtfilt
            continue
        rectified = preprocess_emg(
            seg, sf, mains_freq=mains_freq, low=bandpass[0], high=bandpass[1]
        )
        amps = mini_epoch_amplitude(rectified, sf, epoch_duration=epoch_duration_s)
        amplitudes_parts.append(amps)

        n_epochs_seg = len(amps)
        flags = np.zeros(n_epochs_seg, dtype=bool)
        for a_start, a_end in artifact_windows:
            ov_start = max(start_s, a_start)
            ov_end = min(end_s, a_end)
            if ov_end <= ov_start:
                continue
            e0 = int(np.floor((ov_start - start_s) / epoch_duration_s))
            e1 = int(np.ceil((ov_end - start_s) / epoch_duration_s))
            e0, e1 = max(0, e0), min(n_epochs_seg, e1)
            if e1 > e0:
                flags[e0:e1] = True
        artifact_parts.append(flags)

    if not amplitudes_parts:
        raise RuntimeError(f"Aucune mini-epoque REM extraite pour {path.name}.")

    amplitudes_1s = np.concatenate(amplitudes_parts)
    artifact_flags_1s = (
        np.concatenate(artifact_parts) if detect_artifacts else None
    )

    return amplitudes_1s, artifact_flags_1s, epoch_duration_s, rem_intervals