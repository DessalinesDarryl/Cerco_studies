"""
activation_detection.py
------------------------
Detection des activations ("bursts") EMG phasiques en sommeil REM, fond de
bruit local et intervalles onset-to-onset.

References
----------
- Ferri et al. 2008/2010 : correction de bruit par minimum glissant (fenetre
  de 60 mini-epoques), distributions de duree et d'intervalle.
- Frauscher et al. 2012 / Iranzo et al. 2011 : seuil = 2x fond, fin de burst
  si silence >= 200-250 ms, duree 0.1-5.0 s.
- Khalil et al. 2013 / McCarter et al. 2017 : seuil = 4x fond, duree
  0.1-10/14.9 s (fenetre elargie recommandee par McCarter).

Le parametre `fold` permet de choisir la convention (2.0 ou 4.0). Pour une
etude exploratoire multi-etiologies, la convention 4x (plus conservatrice,
meilleure specificite) est recommandee par defaut.
"""

import numpy as np


def local_background(amplitudes, window_epochs=60):
    """
    Estimation du bruit de fond local par minimum glissant centre.
    Ferri et al. 2010, Sleep Med 11:947-949.
    """
    n = len(amplitudes)
    half = window_epochs // 2
    background = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        background[i] = np.min(amplitudes[lo:hi])
    return background


def detect_activations(
    amplitudes,
    epoch_duration_s,
    fold=4.0,
    window_epochs=60,
    min_gap_s=0.25,
    min_duration_s=0.1,
    max_duration_s=14.9,
    artifact_flags=None,
):
    """
    Detecte les sequences de mini-epoques consecutives depassant
    fold x fond_local, fusionne les runs separes par un silence court
    (<= min_gap_s), puis filtre par duree.

    Parametres
    ----------
    amplitudes : array, amplitude moyenne redressee par mini-epoque
    epoch_duration_s : duree d'une mini-epoque (s)
    fold : multiplicateur du fond de bruit (2.0 = Frauscher/Iranzo,
           4.0 = Khalil/McCarter)
    min_gap_s : silence minimal separant deux activations distinctes
    min_duration_s, max_duration_s : bornes de duree acceptees
                    (0.1-5.0 s = Lapierre/Montplaisir classique,
                     0.1-14.9 s = McCarter, recommande en exploratoire)
    artifact_flags : array booleen optionnel, meme longueur que
        amplitudes (une valeur par mini-epoque), True si la mini-epoque
        chevauche une fenetre artefactee (cf. human_data_adapter.py).
        N'exclut RIEN de la detection : sert uniquement a annoter
        chaque activation detectee avec son taux de recouvrement avec
        des artefacts (champ 'artifact_overlap_fraction'), pour permettre
        un tri/filtre explicite et documente en aval plutot qu'une
        exclusion silencieuse en amont (choix explicite, cf. discussion
        avec l'utilisateur : conserver, ne pas supprimer).

    Retourne une liste de dict :
        onset_epoch, offset_epoch, onset_time_s, duration_s,
        peak_amplitude, mean_amplitude, artifact_overlap_fraction
    """
    amplitudes = np.asarray(amplitudes, dtype=float)
    background = local_background(amplitudes, window_epochs)
    above = amplitudes > (fold * background)
    n = len(above)

    # 1) runs bruts de True
    runs = []
    i = 0
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            runs.append([i, j])  # [start, end) en indices d'epoques
            i = j
        else:
            i += 1

    # 2) fusion des runs separes par un silence <= min_gap_epochs
    min_gap_epochs = max(1, int(round(min_gap_s / epoch_duration_s)))
    merged = []
    for run in runs:
        if merged and (run[0] - merged[-1][1]) <= min_gap_epochs:
            merged[-1][1] = run[1]
        else:
            merged.append(run)

    # 3) construction des activations, filtrage par duree
    artifact_flags = (
        np.asarray(artifact_flags, dtype=bool) if artifact_flags is not None else None
    )

    activations = []
    for start, end in merged:
        duration = (end - start) * epoch_duration_s
        if min_duration_s <= duration <= max_duration_s:
            seg = amplitudes[start:end]
            overlap_frac = np.nan
            if artifact_flags is not None:
                seg_flags = artifact_flags[start:end]
                overlap_frac = float(np.mean(seg_flags)) if seg_flags.size else np.nan
            activations.append(
                {
                    "onset_epoch": start,
                    "offset_epoch": end,
                    "onset_time_s": start * epoch_duration_s,
                    "duration_s": duration,
                    "peak_amplitude": float(np.max(seg)),
                    "mean_amplitude": float(np.mean(seg)),
                    "artifact_overlap_fraction": overlap_frac,
                }
            )
    return activations


def compute_intervals(activations):
    """Intervalles onset-to-onset (s) entre activations consecutives
    (Ferri et al. 2008, Fig. 5)."""
    onsets = [a["onset_time_s"] for a in activations]
    if len(onsets) < 2:
        return []
    return list(np.diff(onsets))