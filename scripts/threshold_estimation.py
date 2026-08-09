"""
threshold_estimation.py
------------------------
Determination data-driven des bornes de sous-categories de duree des
activations phasiques (short / medium / long), pour une echelle de severite
sans a priori sur une dichotomie phasique/tonique fixe.

Ferri et al. 2008 (J Sleep Res) montrent explicitement l'absence de
bimodalite dans leurs distributions de duree ("This was not the case and
the small peak visible in the last category is only because of the fact
that it represents the cumulative count of all activations lasting >19 s").
Il n'existe donc pas de seuil canonique n/x/y/k dans la litterature : ces
bornes doivent etre estimees empiriquement sur VOS donnees (idealement par
groupe/etiologie, ou sur l'ensemble de la cohorte pour une echelle commune).

Deux methodes sont proposees :
1. Gaussian Mixture Model (GMM) sur log(duree) -> bornes naturelles entre
   composantes, avec selection du nombre de composantes par BIC.
2. Repli sur des tertiles empiriques si l'echantillon est trop petit ou le
   GMM instable.
"""

import numpy as np
from sklearn.mixture import GaussianMixture


def _log_durations(durations):
    d = np.asarray(durations, dtype=float)
    d = d[d > 0]
    return np.log(d)


def gmm_thresholds(durations, n_components=3, random_state=0):
    """
    Ajuste un GMM a n_components sur log(duree). Retourne les bornes
    (en secondes, espace original) entre composantes triees par moyenne
    croissante, ainsi que le BIC du modele.
    """
    logd = _log_durations(durations).reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=n_components, random_state=random_state, n_init=10
    ).fit(logd)
    means = gmm.means_.flatten()
    order = np.argsort(means)
    sorted_means = means[order]
    sorted_stds = np.sqrt(gmm.covariances_.flatten())[order]

    boundaries = []
    for a in range(len(sorted_means) - 1):
        m1, s1 = sorted_means[a], max(sorted_stds[a], 1e-6)
        m2, s2 = sorted_means[a + 1], max(sorted_stds[a + 1], 1e-6)
        w1, w2 = 1.0 / s1, 1.0 / s2
        boundary_log = (m1 * w1 + m2 * w2) / (w1 + w2)
        boundaries.append(float(np.exp(boundary_log)))
    return boundaries, gmm.bic(logd)


def select_best_k(durations, k_range=(2, 3, 4)):
    """Choisit le nombre de modes de duree par BIC minimal (GMM)."""
    logd = _log_durations(durations).reshape(-1, 1)
    results = {}
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=0, n_init=10).fit(logd)
        results[k] = gmm.bic(logd)
    best_k = min(results, key=results.get)
    return best_k, results


def percentile_thresholds(durations, percentiles=(33.3, 66.7)):
    """Repli : bornes de tertiles empiriques (data-driven, sans hypothese
    de modele parametrique)."""
    d = np.asarray(durations, dtype=float)
    return list(np.percentile(d, percentiles))


def estimate_duration_categories(durations, method="gmm", min_n=30):
    """
    Point d'entree principal.

    Retourne un dict :
        'n' : borne short/medium (s)
        'k' : borne medium/long (s)
        'method_used'
        'n_activations'

    Repli automatique sur les tertiles si :
        - moins de min_n activations disponibles
        - le GMM ne converge pas vers 2 bornes croissantes et finies
    """
    durations = np.asarray(durations, dtype=float)
    durations = durations[durations > 0]

    if len(durations) < min_n or method == "percentile":
        b = percentile_thresholds(durations) if len(durations) > 0 else [1.0, 5.0]
        return {
            "n": b[0],
            "k": b[1],
            "method_used": "percentile_tertiles",
            "n_activations": len(durations),
        }

    try:
        boundaries, bic = gmm_thresholds(durations, n_components=3)
        valid = (
            len(boundaries) == 2
            and boundaries[0] < boundaries[1]
            and np.all(np.isfinite(boundaries))
        )
        if valid:
            return {
                "n": boundaries[0],
                "k": boundaries[1],
                "method_used": "gmm_3component",
                "bic": bic,
                "n_activations": len(durations),
            }
        raise ValueError("bornes GMM instables")
    except Exception:
        b = percentile_thresholds(durations)
        return {
            "n": b[0],
            "k": b[1],
            "method_used": "percentile_fallback",
            "n_activations": len(durations),
        }
