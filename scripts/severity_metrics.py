"""
severity_metrics.py
--------------------
Calcul des metriques de severite RSWA pour un enregistrement REM.
Combine les variables classiques (Ferri 2008/2010, Frauscher 2012,
McCarter 2017, Khalil 2013) et les variables proposees pour une echelle de
severite multi-etiologique (iRBD, narcolepsie, synucleinopathies,
encephalites auto-immunes).

Chaque fonction indique sa reference bibliographique en docstring.
"""

import numpy as np

from activation_detection import local_background


# ---------------------------------------------------------------
# 1. Atonia Index
# Ferri R, Manconi M, Plazzi G, et al. A quantitative statistical analysis
# of the submentalis muscle EMG amplitude during sleep in normal controls
# and patients with REM sleep behavior disorder. J Sleep Res 2008;17:89-100.
# Ferri R, Rundo F, Manconi M, et al. Improved computation of the atonia
# index in normal controls and patients with REM sleep behavior disorder.
# Sleep Med 2010;11:947-949. (correction du bruit)
# ---------------------------------------------------------------

def local_min_correction(amplitudes_1s, window_epochs=60):
    bg = local_background(amplitudes_1s, window_epochs)
    corrected = amplitudes_1s - bg
    corrected[corrected < 0] = 0
    return corrected


def atonia_index(amplitudes_1s, corrected=True, window_epochs=60):
    """AI = %(amp<=1uV) / [100 - %(1<amp<=2uV)]. Ferri et al. 2008/2010."""
    a = (
        local_min_correction(amplitudes_1s, window_epochs)
        if corrected
        else amplitudes_1s
    )
    n = len(a)
    if n == 0:
        return np.nan
    pct_le1 = 100 * np.sum(a <= 1) / n
    pct_1to2 = 100 * np.sum((a > 1) & (a <= 2)) / n
    denom = 100 - pct_1to2
    if denom <= 0:
        return np.nan
    return pct_le1 / denom


def atonia_index_category(ai):
    """
    Categorisation clinique de l'AI, Ferri et al. 2010 (Sleep Med 11:947-949),
    seuils repris sans modification par Van Gorp et al. 2026 (J Parkinsons
    Dis, validation sur 485 enregistrements) :
        AI < 0.8       -> "abnormal"   (atonie REM significativement alteree)
        0.8 <= AI < 0.9 -> "ambiguous" (implication moins nette de l'atonie)
        AI >= 0.9      -> "normal"     (caracterise la majorite des
                                        enregistrements normaux)
    Ces seuils sont les seuls seuils cliniques de l'AI publies et valides
    independamment sur deux cohortes (Ferri 2010 : n=89 ; Van Gorp 2026 :
    n=485) ; ils ne remplacent pas une interpretation clinique individuelle.
    """
    if ai is None or (isinstance(ai, float) and np.isnan(ai)):
        return np.nan
    if ai < 0.8:
        return "abnormal"
    if ai < 0.9:
        return "ambiguous"
    return "normal"


# ---------------------------------------------------------------
# 2. Activite tonique (epoques de 30 s)
# Frauscher B, Iranzo A, Gaig C, et al. Normative EMG values during REM
# sleep for the diagnosis of REM sleep behavior disorder. Sleep
# 2012;35:835-847.
# Khalil A, Wright MA, Walker MC, Eriksson SH. Loss of rapid eye movement
# sleep atonia in patients with REM sleep behavioral disorder, narcolepsy,
# and isolated loss of REM atonia. J Clin Sleep Med 2013;9(10):1039-1048.
#   (utilise des epoques de 20s ; 30s repris ici par convention
#    Frauscher/McCarter, plus frequente dans la litterature)
# McCarter SJ, St Louis EK, Sandness DJ, et al. Diagnostic REM sleep muscle
# activity thresholds in patients with idiopathic REM sleep behavior
# disorder with and without obstructive sleep apnea. Sleep Med
# 2017;33:23-29.
# ---------------------------------------------------------------

def tonic_epoch_flags(
    amplitudes_1s,
    epoch_duration_s=1.0,
    tonic_epoch_s=30,
    fold=2.0,
    min_fraction=0.5,
    window_epochs=60,
):
    """
    Retourne un tableau booleen (un flag par epoque de 30s) indiquant si
    l'epoque est 'tonique' (>= min_fraction de sa duree a une amplitude
    > fold x fond local), ainsi que le nombre de mini-epoques par
    epoque tonique (necessaire pour projeter ce flag au niveau
    mini-epoque, cf. pct_any_activity).
    """
    bg = local_background(amplitudes_1s, window_epochs)
    above = amplitudes_1s > (fold * bg)
    epochs_per_tonic = int(round(tonic_epoch_s / epoch_duration_s))
    n_tonic_epochs = len(above) // epochs_per_tonic
    flags = np.zeros(n_tonic_epochs, dtype=bool)
    for e in range(n_tonic_epochs):
        seg = above[e * epochs_per_tonic : (e + 1) * epochs_per_tonic]
        flags[e] = np.mean(seg) >= min_fraction
    return flags, epochs_per_tonic


def tonic_activity_pct(
    amplitudes_1s,
    epoch_duration_s=1.0,
    tonic_epoch_s=30,
    fold=2.0,
    min_fraction=0.5,
    window_epochs=60,
):
    """% d'epoques de 30s toniques parmi le total d'epoques REM analysables."""
    flags, epochs_per_tonic = tonic_epoch_flags(
        amplitudes_1s, epoch_duration_s, tonic_epoch_s, fold, min_fraction,
        window_epochs,
    )
    if len(flags) == 0:
        return np.nan, flags, epochs_per_tonic
    return 100 * np.mean(flags), flags, epochs_per_tonic


# ---------------------------------------------------------------
# 3. % RSWA phasique, "any" (phasique OU tonique), temps-pondere
# Frauscher B, et al. Sleep 2012;35:835-847.
#   -> "any" EMG activity : AUC 0.990 (mentalis seul), 0.998 (mentalis +
#      FDS bilateral). Definie au niveau mini-epoque (3s dans l'etude
#      source ; ici projetee sur la resolution choisie, 1s par defaut).
# ---------------------------------------------------------------

def pct_epochs_with_activation(n_total_epochs, activations, epoch_duration_s=1.0):
    """% de mini-epoques REM touchees par une activation phasique
    (Frauscher/Iranzo/McCarter)."""
    if n_total_epochs == 0:
        return np.nan
    covered = set()
    for a in activations:
        covered.update(range(a["onset_epoch"], a["offset_epoch"]))
    return 100 * len(covered) / n_total_epochs


def pct_any_activity(n_total_epochs, activations, tonic_flags, epochs_per_tonic):
    """
    % RSWA global ("any") : mini-epoques avec activite phasique OU
    appartenant a une epoque de 30s classee tonique.
    Frauscher B, et al. Sleep 2012;35:835-847 (meilleur AUC discriminant
    de l'etude : 0.990 mentalis seul, 0.998 combine mentalis+FDS).
    """
    if n_total_epochs == 0:
        return np.nan
    covered = set()
    for a in activations:
        covered.update(range(a["onset_epoch"], a["offset_epoch"]))
    for e, is_tonic in enumerate(tonic_flags):
        if is_tonic:
            start = e * epochs_per_tonic
            end = min(start + epochs_per_tonic, n_total_epochs)
            covered.update(range(start, end))
    return 100 * len(covered) / n_total_epochs


def pct_rem_time_with_activation(total_rem_time_s, activations):
    """
    % du TEMPS de sommeil REM occupe par des activations phasiques
    (pondere par la duree). Variable proposee : adaptation temps-ponderee
    de la metrique par comptage de mini-epoques de Frauscher et al. 2012
    et McCarter et al. 2017.
    """
    if total_rem_time_s == 0:
        return np.nan
    total_activation_time = sum(a["duration_s"] for a in activations)
    return 100 * total_activation_time / total_rem_time_s


# ---------------------------------------------------------------
# 4. Metriques episode-level
# Densite d'activations : Ferri R, et al. J Sleep Res 2008;17:89-100
#   (Fig. 4, histogrammes "Number h-1" par duree) ; McCarter SJ, et al.
#   Sleep Med 2017;33:23-29 (comptage de bursts phasiques).
# ---------------------------------------------------------------

def episode_count(activations):
    """Nombre d'episodes RSWA. Ferri et al. 2008 ; McCarter et al. 2017."""
    return len(activations)


def activation_density_per_hour(activations, total_rem_time_s):
    """
    Densite d'activations, evenements/HEURE de REM -- formule telle que
    rapportee par Ferri et al. 2008 (J Sleep Res, Fig. 4, axe "Number h-1")
    et utilisee par McCarter et al. 2017 (Sleep Med).
    """
    rem_hours = total_rem_time_s / 3600.0
    return len(activations) / rem_hours if rem_hours > 0 else np.nan


def phasic_density_per_min(activations, total_rem_time_s):
    """Densite d'evenements phasiques, evenements/min de REM (equivalent
    en minutes de la metrique horaire ci-dessus, plus lisible sur des
    enregistrements courts)."""
    rem_min = total_rem_time_s / 60.0
    return len(activations) / rem_min if rem_min > 0 else np.nan


def duration_stats(activations):
    """Statistiques de duree des activations. Extension de McCarter et al.
    2017 (variable de duree de burst) et Ferri et al. 2008 (Fig. 4)."""
    d = np.array([a["duration_s"] for a in activations])
    if len(d) == 0:
        return dict(mean=np.nan, median=np.nan, sd=np.nan, total=0.0, max=np.nan)
    return dict(
        mean=float(np.mean(d)),
        median=float(np.median(d)),
        sd=float(np.std(d)),
        total=float(np.sum(d)),
        max=float(np.max(d)),
    )


def interval_stats(intervals):
    """
    Distribution des intervalles onset-to-onset : moyenne, mediane,
    variance (Ferri et al. 2008, J Sleep Res, Fig. 5) et coefficient de
    variation (CV, variable proposee fondee sur l'observation par Ferri
    et al. 2008 d'une structure de bruit 1/f monomodale -- un CV
    anormalement bas signalerait un ecart a cette structure, ex.
    periodicite pathologique).
    """
    iv = np.array(intervals)
    if len(iv) < 2:
        return dict(mean=np.nan, median=np.nan, variance=np.nan, cv=np.nan)
    mean_iv = float(np.mean(iv))
    var_iv = float(np.var(iv))
    sd_iv = float(np.std(iv))
    cv = sd_iv / mean_iv if mean_iv > 0 else np.nan
    return dict(mean=mean_iv, median=float(np.median(iv)), variance=var_iv, cv=cv)


def amplitude_intensity_stats(activations, background_mean):
    """
    Intensite des bursts relative au fond. Variable proposee : transforme
    en variable continue le seuil binaire "amplitude > N x fond" de
    Khalil et al. 2013 (JCSM) et McCarter et al. 2017 (Sleep Med).
    """
    if len(activations) == 0 or background_mean in (0, None) or np.isnan(background_mean):
        return dict(mean_relative_amplitude=np.nan, max_relative_amplitude=np.nan)
    rel = [a["peak_amplitude"] / background_mean for a in activations]
    return dict(
        mean_relative_amplitude=float(np.mean(rel)),
        max_relative_amplitude=float(np.max(rel)),
    )


# ---------------------------------------------------------------
# 5. Densites par sous-categorie de duree (bornes data-driven n / k)
# Variable proposee : Ferri et al. 2008 (J Sleep Res) demontrent l'absence
# de bimodalite dans leurs distributions de duree, justifiant le rejet
# d'un seuil fixe a priori au profit de bornes estimees sur les donnees
# (cf. threshold_estimation.py).
# ---------------------------------------------------------------

def density_by_duration_category(activations, total_rem_time_s, n_bound, k_bound):
    rem_min = total_rem_time_s / 60.0
    cats = {"short": [], "medium": [], "long": []}
    for a in activations:
        d = a["duration_s"]
        if d < n_bound:
            cats["short"].append(a)
        elif d <= k_bound:
            cats["medium"].append(a)
        else:
            cats["long"].append(a)

    out = {}
    total_n = len(activations)
    for cat, acts in cats.items():
        out[f"{cat}_count"] = len(acts)
        out[f"{cat}_pct_of_activations"] = (
            100 * len(acts) / total_n if total_n else np.nan
        )
        out[f"{cat}_density_per_min"] = (
            len(acts) / rem_min if rem_min > 0 else np.nan
        )
        out[f"{cat}_total_duration_s"] = float(sum(x["duration_s"] for x in acts))
    return out


# ---------------------------------------------------------------
# 6. Balance phasique/tonique et asymetrie bilaterale
# Ratio phasique/tonique : variable proposee combinant 2 metriques
# rapportees separement par McCarter et al. 2017 et Frauscher et al. 2012.
# Asymetrie bilaterale : variable proposee, s'appuyant sur le montage
# bilateral rapporte separement (gauche/droite) par Iranzo et al. 2011
# (Sleep Med) et Frauscher et al. 2012 (Sleep), sans indice d'asymetrie
# explicite dans ces etudes.
# ---------------------------------------------------------------

def phasic_tonic_ratio(pct_phasic, pct_tonic):
    if pct_tonic in (0, None) or (isinstance(pct_tonic, float) and np.isnan(pct_tonic)):
        return np.inf if (pct_phasic and pct_phasic > 0) else np.nan
    return pct_phasic / pct_tonic


def bilateral_asymmetry_index(activations_left, activations_right, total_rem_time_s):
    dl = phasic_density_per_min(activations_left, total_rem_time_s)
    dr = phasic_density_per_min(activations_right, total_rem_time_s)
    dl = 0.0 if (dl is None or np.isnan(dl)) else dl
    dr = 0.0 if (dr is None or np.isnan(dr)) else dr
    if (dl + dr) == 0:
        return 0.0
    return (dr - dl) / (dr + dl)


# ---------------------------------------------------------------
# 7. Progression intra-nuit
# Variable proposee, inspiree des travaux sur la variabilite nuit-a-nuit
# cites par Khalil et al. 2013 (JCSM) :
#   Ferri R, Marelli S, Cosentino FI, et al. Night-to-night variability of
#   automatic quantitative parameters of the chin EMG amplitude (Atonia
#   Index) in REM sleep behavior disorder. J Clin Sleep Med 2013;9:253-8.
#   Cygan F, Oudiette D, Leclair-Visonneau L, et al. Night-to-night
#   variability of muscle tone, movements, and vocalizations in patients
#   with REM sleep behavior disorder. J Clin Sleep Med 2010;6:551-555.
# ---------------------------------------------------------------

def progression_index(activations, total_rem_time_s, n_thirds=3):
    if total_rem_time_s <= 0 or len(activations) == 0:
        return np.nan
    third = total_rem_time_s / n_thirds
    first = [a for a in activations if a["onset_time_s"] < third]
    last = [a for a in activations if a["onset_time_s"] >= 2 * third]
    third_min = third / 60.0
    d_first = len(first) / third_min if third_min > 0 else np.nan
    d_last = len(last) / third_min if third_min > 0 else np.nan
    if d_first == 0:
        return np.inf if d_last > 0 else np.nan
    return d_last / d_first


# ---------------------------------------------------------------
# 8. Bundle complet pour un sujet
# ---------------------------------------------------------------

def compute_all_metrics(
    amplitudes_1s,
    activations,
    intervals,
    total_rem_time_s,
    n_bound,
    k_bound,
    background_mean,
    tonic_pct=None,
    tonic_flags=None,
    epochs_per_tonic=None,
    activations_left=None,
    activations_right=None,
    artifact_flags_1s=None,
):
    """
    artifact_flags_1s : array booleen optionnel, meme longueur que
        amplitudes_1s, True si la mini-epoque chevauche une fenetre
        artefactee (cf. human_data_adapter.py). Les artefacts ne sont
        JAMAIS utilises ici pour exclure des mini-epoques ou des
        activations du calcul -- conformement au choix explicite de ne
        pas amputer les intervalles REM (contrairement a l'ancien
        02_segment_rem.py) : ils sont uniquement reportes en sortie
        (pct_epochs_artifact) pour permettre une exclusion a posteriori,
        documentee, plutot que silencieuse en amont du pipeline.
    """
    metrics = {}
    n_total_epochs = len(amplitudes_1s)

    metrics["AI_atonia_index"] = atonia_index(amplitudes_1s)
    metrics["AI_atonia_index_category"] = atonia_index_category(
        metrics["AI_atonia_index"]
    )
    metrics["pct_epochs_phasic"] = pct_epochs_with_activation(
        n_total_epochs, activations, epoch_duration_s=1.0
    )
    # NOTE : pct_rem_time_phasic (variable #9, redondante avec
    # pct_epochs_phasic) et activation_density_per_hour (variable #4,
    # redondante avec phasic_density_per_min / episode_count) ont ete
    # retirees du bundle par defaut (cf. discussion redondance /
    # reduction de dimension). Les fonctions restent disponibles dans ce
    # module si besoin ponctuel.
    metrics["episode_count"] = episode_count(activations)
    metrics["phasic_density_per_min"] = phasic_density_per_min(
        activations, total_rem_time_s
    )
    metrics.update({f"duration_{k}": v for k, v in duration_stats(activations).items()})
    metrics.update({f"interval_{k}": v for k, v in interval_stats(intervals).items()})
    metrics.update(amplitude_intensity_stats(activations, background_mean))
    metrics.update(
        density_by_duration_category(activations, total_rem_time_s, n_bound, k_bound)
    )

    if tonic_pct is not None:
        metrics["pct_tonic"] = tonic_pct
        metrics["phasic_tonic_ratio"] = phasic_tonic_ratio(
            metrics["pct_epochs_phasic"], tonic_pct
        )

    if tonic_flags is not None and epochs_per_tonic is not None:
        metrics["pct_any_RSWA"] = pct_any_activity(
            n_total_epochs, activations, tonic_flags, epochs_per_tonic
        )

    if activations_left is not None and activations_right is not None:
        metrics["bilateral_asymmetry_index"] = bilateral_asymmetry_index(
            activations_left, activations_right, total_rem_time_s
        )

    metrics["progression_index"] = progression_index(activations, total_rem_time_s)

    if artifact_flags_1s is not None:
        artifact_flags_1s = np.asarray(artifact_flags_1s, dtype=bool)
        metrics["pct_epochs_artifact"] = (
            100 * np.mean(artifact_flags_1s) if len(artifact_flags_1s) else np.nan
        )
    else:
        metrics["pct_epochs_artifact"] = np.nan

    return metrics