"""
main_pipeline.py
------------------
Pipeline complet RSWA pour etude exploratoire multi-etiologique
(iRBD, narcolepsie, synucleinopathies, encephalites auto-immunes vs
controles).

Usage typique (par sujet)
--------------------------
    from main_pipeline import run_subject_pipeline_from_raw

    metrics = run_subject_pipeline_from_raw(
        raw_signal=submentalis_array,   # signal EMG brut, REM sleep only
        fs=256,                         # frequence d'echantillonnage (Hz)
        diagnosis='iRBD',
        subject_id='sub-001',
        fold=4.0,                       # convention Khalil/McCarter
    )

Puis agreger tous les dicts sujets en DataFrame et passer a
statistics_pipeline.full_group_comparison().

NOTE : le chargement EDF necessite `mne` (pip install mne
--break-system-packages). Si vous disposez deja d'un tableau d'amplitudes
par mini-epoque REM (extrait via votre propre pipeline de staging), utilisez
directement run_subject_pipeline_from_amplitudes().
"""

import numpy as np
import pandas as pd

from preprocessing import preprocess_emg, mini_epoch_amplitude
from activation_detection import detect_activations, compute_intervals, local_background
from severity_metrics import compute_all_metrics, tonic_epoch_flags
from threshold_estimation import estimate_duration_categories


def load_edf_channel(edf_path, channel_name, rem_mask=None):
    """
    Charge un canal EMG depuis un fichier EDF, restreint au sommeil REM si
    un masque booleen (echantillon par echantillon) est fourni.
    Necessite `mne`.
    """
    import mne

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    fs = raw.info["sfreq"]
    signal = raw.get_data(picks=[channel_name])[0]
    if rem_mask is not None:
        signal = signal[rem_mask]
    return signal, fs


def _finalize_metrics(metrics, thresholds, total_rem_time_s, n_tonic_epochs,
                       diagnosis, subject_id):
    metrics["subject_id"] = subject_id
    metrics["diagnosis"] = diagnosis
    metrics["duration_threshold_method"] = thresholds["method_used"]
    metrics["duration_threshold_n"] = thresholds["n"]
    metrics["duration_threshold_k"] = thresholds["k"]
    metrics["total_rem_time_s"] = total_rem_time_s
    metrics["n_tonic_epochs_scored"] = n_tonic_epochs
    return metrics


def run_subject_pipeline_from_raw(
    raw_signal,
    fs,
    diagnosis,
    subject_id,
    fold=4.0,
    epoch_duration_s=1.0,
    duration_method="gmm",
):
    """Pipeline complet a partir d'un signal EMG brut (REM sleep uniquement,
    non filtre)."""
    rectified = preprocess_emg(raw_signal, fs)
    amplitudes = mini_epoch_amplitude(rectified, fs, epoch_duration=epoch_duration_s)
    return run_subject_pipeline_from_amplitudes(
        amplitudes, diagnosis, subject_id, fold=fold,
        epoch_duration_s=epoch_duration_s, duration_method=duration_method,
    )


def run_subject_pipeline_from_amplitudes(
    amplitudes_1s,
    diagnosis,
    subject_id,
    fold=4.0,
    epoch_duration_s=1.0,
    duration_method="gmm",
    tonic_epoch_s=30,
    artifact_flags_1s=None,
):
    """
    Pipeline a partir d'un tableau d'amplitudes par mini-epoque deja
    calcule (filtrage/redressement/decoupage deja faits en amont).

    tonic_epoch_s : duree des epoques toniques (s). 30 s = convention
        Frauscher/McCarter (defaut historique de ce pipeline) ; 20 s =
        convention Khalil et al. 2013 (JCSM), a utiliser si l'on
        souhaite comparer directement aux seuils publies par Khalil
        (tonique >=1.28-3.17% selon le groupe, cf. methodo).
        Auparavant fige a 30 dans tonic_epoch_flags() sans etre expose
        au niveau pipeline ; desormais parametrable ici.
    artifact_flags_1s : array booleen optionnel, meme longueur que
        amplitudes_1s, produit par human_data_adapter.py. N'exclut
        rien : propage jusqu'a detect_activations() (annotation par
        activation) et compute_all_metrics() (metrique globale
        pct_epochs_artifact), decision explicite de conserver plutot
        que de supprimer les epoques artefactees.
    """
    amplitudes_1s = np.asarray(amplitudes_1s, dtype=float)
    total_rem_time_s = len(amplitudes_1s) * epoch_duration_s

    activations = detect_activations(
        amplitudes_1s, epoch_duration_s, fold=fold, artifact_flags=artifact_flags_1s
    )
    intervals = compute_intervals(activations)

    durations = [a["duration_s"] for a in activations]
    thresholds = estimate_duration_categories(durations, method=duration_method)

    background = local_background(amplitudes_1s)
    background_mean = float(np.mean(background))

    tonic_flags, epochs_per_tonic = tonic_epoch_flags(
        amplitudes_1s, epoch_duration_s=epoch_duration_s, tonic_epoch_s=tonic_epoch_s
    )
    n_tonic_epochs = len(tonic_flags)
    tonic_pct = 100 * np.mean(tonic_flags) if n_tonic_epochs > 0 else np.nan

    metrics = compute_all_metrics(
        amplitudes_1s=amplitudes_1s,
        activations=activations,
        intervals=intervals,
        total_rem_time_s=total_rem_time_s,
        n_bound=thresholds["n"],
        k_bound=thresholds["k"],
        background_mean=background_mean,
        tonic_pct=tonic_pct,
        tonic_flags=tonic_flags,
        epochs_per_tonic=epochs_per_tonic,
        artifact_flags_1s=artifact_flags_1s,
    )
    return _finalize_metrics(
        metrics, thresholds, total_rem_time_s, n_tonic_epochs, diagnosis, subject_id
    )


def run_subject_pipeline_bilateral(
    amplitudes_left,
    amplitudes_right,
    diagnosis,
    subject_id,
    fold=4.0,
    epoch_duration_s=1.0,
    duration_method="gmm",
    tonic_epoch_s=30,
    artifact_flags_left=None,
    artifact_flags_right=None,
):
    """
    Variante pour canaux bilateraux (ex. FDS ou AT gauche/droit), ajoutant
    l'indice d'asymetrie bilaterale. Les metriques d'amplitude/duree/
    intervalle/AI/tonique sont calculees sur le signal combine (max des deux
    canaux par mini-epoque), l'asymetrie sur les activations detectees
    separement par cote.

    artifact_flags_left/right : masques booleens optionnels (memes
        conventions que run_subject_pipeline_from_amplitudes). Combines
        par OU logique pour le signal combine (une mini-epoque du signal
        combine est consideree artefactee si elle l'est sur au moins un
        des deux cotes) -- annotation uniquement, aucune exclusion.
    """
    amplitudes_left = np.asarray(amplitudes_left, dtype=float)
    amplitudes_right = np.asarray(amplitudes_right, dtype=float)
    n = min(len(amplitudes_left), len(amplitudes_right))
    amplitudes_left, amplitudes_right = amplitudes_left[:n], amplitudes_right[:n]
    combined = np.maximum(amplitudes_left, amplitudes_right)

    artifact_flags_combined = None
    if artifact_flags_left is not None or artifact_flags_right is not None:
        fl = (np.zeros(n, dtype=bool) if artifact_flags_left is None
              else np.asarray(artifact_flags_left, dtype=bool)[:n])
        fr = (np.zeros(n, dtype=bool) if artifact_flags_right is None
              else np.asarray(artifact_flags_right, dtype=bool)[:n])
        artifact_flags_combined = fl | fr

    total_rem_time_s = n * epoch_duration_s

    activations_combined = detect_activations(
        combined, epoch_duration_s, fold=fold, artifact_flags=artifact_flags_combined
    )
    activations_left = detect_activations(
        amplitudes_left, epoch_duration_s, fold=fold, artifact_flags=artifact_flags_left
    )
    activations_right = detect_activations(
        amplitudes_right, epoch_duration_s, fold=fold, artifact_flags=artifact_flags_right
    )

    intervals = compute_intervals(activations_combined)
    durations = [a["duration_s"] for a in activations_combined]
    thresholds = estimate_duration_categories(durations, method=duration_method)

    background = local_background(combined)
    background_mean = float(np.mean(background))
    tonic_flags, epochs_per_tonic = tonic_epoch_flags(
        combined, epoch_duration_s=epoch_duration_s, tonic_epoch_s=tonic_epoch_s
    )
    n_tonic_epochs = len(tonic_flags)
    tonic_pct = 100 * np.mean(tonic_flags) if n_tonic_epochs > 0 else np.nan

    metrics = compute_all_metrics(
        amplitudes_1s=combined,
        activations=activations_combined,
        intervals=intervals,
        total_rem_time_s=total_rem_time_s,
        n_bound=thresholds["n"],
        k_bound=thresholds["k"],
        background_mean=background_mean,
        tonic_pct=tonic_pct,
        tonic_flags=tonic_flags,
        epochs_per_tonic=epochs_per_tonic,
        activations_left=activations_left,
        activations_right=activations_right,
        artifact_flags_1s=artifact_flags_combined,
    )
    return _finalize_metrics(
        metrics, thresholds, total_rem_time_s, n_tonic_epochs, diagnosis, subject_id
    )


def run_subject_pipeline_from_human_recording(
    fif_or_edf_path,
    channel_name,
    diagnosis,
    subject_id,
    hypno_path=None,
    fold=4.0,
    epoch_duration_s=1.0,
    duration_method="gmm",
    tonic_epoch_s=30,
    detect_artifacts=True,
):
    """
    Pipeline a partir d'un enregistrement patient reel (.fif/.edf),
    reprenant la methodologie de l'ancien pipeline 01_preprocess.py /
    03_segment_rswa.py (montage/filtrage/hypnogramme reel) via
    human_data_adapter.py, MAIS sans soustraire les fenetres
    artefactees des intervalles REM (elles sont annotees, pas
    supprimees -- cf. human_data_adapter.py et
    severity_metrics.compute_all_metrics).

    channel_name : canal EMG reel a analyser (ex. "Menton", "JAMBG",
        "JAMBD", "EMG1", "EMG2"). Pour une analyse bilaterale
        (membres), utiliser deux appels a
        human_data_adapter.extract_rem_amplitudes_with_artifacts puis
        run_subject_pipeline_bilateral() directement.
    hypno_path : None -> fallback sur les annotations "REM" deja
        presentes dans le fichier (cas d'un .fif issu de
        01_preprocess.py / 02_segment_rem.py).
    """
    from human_data_adapter import extract_rem_amplitudes_with_artifacts

    amplitudes_1s, artifact_flags_1s, _, _ = extract_rem_amplitudes_with_artifacts(
        fif_or_edf_path,
        channel_name,
        hypno_path=hypno_path,
        epoch_duration_s=epoch_duration_s,
        detect_artifacts=detect_artifacts,
    )

    return run_subject_pipeline_from_amplitudes(
        amplitudes_1s,
        diagnosis,
        subject_id,
        fold=fold,
        epoch_duration_s=epoch_duration_s,
        duration_method=duration_method,
        tonic_epoch_s=tonic_epoch_s,
        artifact_flags_1s=artifact_flags_1s,
    )


def build_cohort_dataframe(subject_metrics_list):
    return pd.DataFrame(subject_metrics_list)


def save_cohort_dataframe(df, filename="cohort_metrics.csv", subdir="cohort"):
    """
    Sauvegarde le DataFrame cohorte sous results/<subdir>/<filename>
    (cf. results_io.py -- convention unique de sortie du pipeline).
    """
    from results_io import save_dataframe

    return save_dataframe(df, subdir, filename)


if __name__ == "__main__":
    # ---- test de fumee sur donnees synthetiques -------------------------
    np.random.seed(0)
    fs = 256
    duration_s = 600  # 10 min de REM synthetique
    t = np.arange(duration_s * fs) / fs
    baseline = 0.3 * np.random.randn(len(t))
    bursts = np.zeros_like(t)
    for onset in np.random.choice(np.arange(0, duration_s - 2, 5), size=40, replace=False):
        dur = np.random.uniform(0.2, 3.0)
        idx = (t >= onset) & (t < onset + dur)
        bursts[idx] += np.random.uniform(3, 8)
    raw_signal = baseline + bursts

    result = run_subject_pipeline_from_raw(
        raw_signal, fs, diagnosis="iRBD", subject_id="demo-001"
    )
    print("=== Metriques sujet demo (donnees synthetiques) ===")
    for k, v in result.items():
        print(f"{k}: {v}")