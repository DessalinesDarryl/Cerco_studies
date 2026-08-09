"""
multicollinearity_pca.py
--------------------------
Methodologie de reduction de dimension pour l'echelle composite de
severite RSWA : correlations de Spearman (avec correction FDR), facteur
d'inflation de la variance (VIF), tests de faisabilite factorielle (KMO,
sphericite de Bartlett), et analyse en composantes principales (ACP).

Ces methodes sont des standards de statistique multivariee, independants
de la litterature RBD :
    - Spearman C. The proof and measurement of association between two
      things. Am J Psychol 1904;15:72-101.
    - Bartlett MS. Tests of significance in factor analysis. Br J Psychol
      1950;3:77-85. (test de sphericite)
    - Kaiser HF. An index of factorial simplicity. Psychometrika
      1974;39:31-36. (KMO ; regle des valeurs propres > 1)
    - Benjamini Y, Hochberg Y. Controlling the false discovery rate: a
      practical and powerful approach to multiple testing. J R Stat Soc
      Series B 1995;57:289-300. (correction FDR)
    - O'Brien RM. A caution regarding rules of thumb for variance
      inflation factors. Qual Quant 2007;41:673-690. (seuils VIF 5/10)

Usage recommande, en amont de statistics_pipeline.composite_severity_score :

    from multicollinearity_pca import full_dimensionality_reduction_report
    report = full_dimensionality_reduction_report(df, feature_cols)
    retained_features = report['retained_features']   # -> a utiliser pour
                                                        #    le score composite
    pca_scores = report['pca']['scores']               # -> alternative : score
                                                        #    = PC1 directement
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------
# 1. Matrice de correlations de Spearman (+ correction FDR)
# ---------------------------------------------------------------

def spearman_matrix(df, feature_cols, fdr_correction=True, alpha=0.05):
    """
    Matrice de correlations de Spearman (rho) et matrice de p-values
    associee entre toutes les paires de variables.

    Spearman (plutot que Pearson) est coherent avec le reste du pipeline
    (tests non-parametriques Kruskal-Wallis / Mann-Whitney deja utilises
    dans statistics_pipeline.py) : les variables RSWA (densites, ratios,
    indices) ne suivent generalement pas une distribution normale.

    fdr_correction : correction de Benjamini-Hochberg (FDR) sur les
    p-values des paires uniques -- le nombre de tests croit en p(p-1)/2
    avec le nombre de variables, une correction de Bonferroni serait ici
    trop conservatrice (contrairement aux comparaisons de groupes, ou
    Bonferroni est conserve pour rester coherent avec Khalil et al. 2013).
    """
    data = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    p = len(feature_cols)
    rho_mat = pd.DataFrame(np.eye(p), index=feature_cols, columns=feature_cols)
    p_mat = pd.DataFrame(np.zeros((p, p)), index=feature_cols, columns=feature_cols)

    pvals_flat, pairs = [], []
    for i in range(p):
        for j in range(i + 1, p):
            x, y = data.iloc[:, i], data.iloc[:, j]
            mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3:
                rho, pval = np.nan, np.nan
            else:
                rho, pval = stats.spearmanr(x[mask], y[mask])
            rho_mat.iloc[i, j] = rho_mat.iloc[j, i] = rho
            p_mat.iloc[i, j] = p_mat.iloc[j, i] = pval
            pvals_flat.append(pval)
            pairs.append((i, j))

    if fdr_correction and len(pvals_flat) > 0:
        valid = [not np.isnan(pv) for pv in pvals_flat]
        pvals_valid = [pv for pv, v in zip(pvals_flat, valid) if v]
        if len(pvals_valid) > 0:
            _, pvals_corr, _, _ = multipletests(pvals_valid, alpha=alpha, method="fdr_bh")
            it = iter(pvals_corr)
            for (i, j), v in zip(pairs, valid):
                if v:
                    corrected = next(it)
                    p_mat.iloc[i, j] = p_mat.iloc[j, i] = corrected

    return rho_mat, p_mat


def flag_redundant_pairs(rho_matrix, threshold=0.8):
    """
    Paires de variables dont |rho| >= threshold, triees par force
    decroissante. Seuil 0.8 = convention usuelle pour signaler une
    redondance forte en construction d'echelle composite.
    """
    pairs = []
    cols = rho_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rho = rho_matrix.iloc[i, j]
            if pd.notna(rho) and abs(rho) >= threshold:
                pairs.append({"var1": cols[i], "var2": cols[j], "rho": rho})
    if not pairs:
        return pd.DataFrame(columns=["var1", "var2", "rho"])
    return (
        pd.DataFrame(pairs)
        .sort_values("rho", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------
# 2. Facteur d'inflation de la variance (VIF)
# ---------------------------------------------------------------

def compute_vif(df, feature_cols, impute_missing=False, impute_strategy="median"):
    """
    VIF de chaque variable (regression de chaque variable sur toutes les
    autres). VIF > 5 = multicollinearite moderee, VIF > 10 = severe
    (O'Brien 2007, Qual Quant 41:673-690).

    Le VIF exige des cas complets (listwise deletion). En pratique, les
    sujets avec tres peu d'activations (typiquement les controles) ont des
    statistiques d'intervalle (variance, CV) non definies (< 2 intervalles
    disponibles), ce qui peut faire chuter drastiquement le nombre de cas
    complets. impute_missing=True active une imputation simple par
    mediane (ou moyenne) pour eviter cette perte de puissance -- a
    documenter comme limite methodologique si utilisee (l'imputation par
    mediane est un choix pragmatique standard pour le calcul du VIF, pas
    une recommandation pour l'analyse statistique finale elle-meme).
    """
    data = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)

    if impute_missing:
        if impute_strategy == "median":
            data = data.fillna(data.median())
        elif impute_strategy == "mean":
            data = data.fillna(data.mean())
        else:
            raise ValueError('impute_strategy doit etre "median" ou "mean"')
    else:
        data = data.dropna()

    if data.shape[0] < len(feature_cols) + 2:
        raise ValueError(
            "Echantillon trop petit pour un VIF stable "
            f"({data.shape[0]} observations completes pour {len(feature_cols)} variables). "
            "Envisagez impute_missing=True, ou reduisez le nombre de variables candidates."
        )
    X = np.column_stack([np.ones(data.shape[0]), data.values])  # constante requise
    vifs = [variance_inflation_factor(X, i) for i in range(1, X.shape[1])]
    return (
        pd.DataFrame({"feature": feature_cols, "VIF": vifs})
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )


def iterative_vif_reduction(
    df, feature_cols, vif_threshold=5.0, verbose=True,
    impute_missing=False, impute_strategy="median",
):
    """
    Retire iterativement la variable au VIF le plus eleve tant qu'au moins
    une variable depasse vif_threshold.

    Retourne : retained_features, removed_features (avec VIF au retrait),
    history (VIF a chaque iteration).
    """
    remaining = list(feature_cols)
    removed = []
    history = []

    while len(remaining) > 1:
        vif_df = compute_vif(
            df, remaining, impute_missing=impute_missing, impute_strategy=impute_strategy
        )
        history.append(vif_df.copy())
        max_vif = vif_df["VIF"].max()
        if max_vif <= vif_threshold:
            break
        worst = vif_df.iloc[0]
        removed.append({"feature": worst["feature"], "VIF_at_removal": worst["VIF"]})
        if verbose:
            print(f"Retrait de '{worst['feature']}' (VIF={worst['VIF']:.2f} > {vif_threshold})")
        remaining.remove(worst["feature"])

    return {
        "retained_features": remaining,
        "removed_features": pd.DataFrame(removed),
        "history": history,
    }


# ---------------------------------------------------------------
# 3. Faisabilite factorielle : KMO et test de sphericite de Bartlett
# ---------------------------------------------------------------

def kmo_test(df, feature_cols):
    """
    Indice de Kaiser-Meyer-Olkin (KMO) : adequation de l'echantillonnage
    pour l'ACP (Kaiser 1974, Psychometrika 39:31-36).
    KMO global : <0.5 inacceptable, 0.5-0.7 mediocre, 0.7-0.8 correct,
    0.8-0.9 tres bon, >0.9 excellent.
    """
    data = df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    corr = data.corr(method="spearman").values
    try:
        inv_corr = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        inv_corr = np.linalg.pinv(corr)

    p = corr.shape[0]
    partial_corr = np.full((p, p), np.nan)
    diag = np.diag(inv_corr)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            product = diag[i] * diag[j]
            if product <= 0:
                continue  # matrice quasi-singuliere : cette paire est ininterpretable
            partial_corr[i, j] = -inv_corr[i, j] / np.sqrt(product)

    corr_sq_sum = np.nansum(corr ** 2) - np.nansum(np.diag(corr) ** 2)
    partial_sq_sum = np.nansum(partial_corr ** 2)
    denom_total = corr_sq_sum + partial_sq_sum
    kmo_overall = corr_sq_sum / denom_total if denom_total > 0 else np.nan

    kmo_per_var = []
    for i in range(p):
        num = np.nansum(corr[i, :] ** 2) - corr[i, i] ** 2
        denom = num + np.nansum(partial_corr[i, :] ** 2)
        kmo_per_var.append(num / denom if denom > 0 else np.nan)

    if np.isnan(kmo_overall):
        print(
            "AVERTISSEMENT (kmo_test) : matrice de correlation quasi-singuliere "
            "(souvent du a des valeurs imputees identiques sur plusieurs sujets, "
            "ou trop de variables pour le nombre de sujets). KMO non interpretable "
            "en l'etat -- reduire le nombre de variables candidates ou revoir "
            "l'imputation avant de conclure sur la faisabilite de l'ACP."
        )

    return {
        "kmo_overall": float(kmo_overall) if not np.isnan(kmo_overall) else np.nan,
        "kmo_per_variable": pd.DataFrame(
            {"feature": feature_cols, "KMO": kmo_per_var}
        ).sort_values("KMO"),
    }


def bartlett_sphericity_test(df, feature_cols):
    """
    Test de sphericite de Bartlett : H0 = matrice de correlation = matrice
    identite (variables non correlees -> ACP non pertinente).
    Bartlett 1950, Br J Psychol 3:77-85.
    """
    data = df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n, p = data.shape
    corr = data.corr(method="spearman").values
    det_corr = np.linalg.det(corr)
    if det_corr <= 0:
        det_corr = 1e-12  # evite log(0)/log(negatif) en cas de quasi-singularite

    chi2_stat = -((n - 1) - (2 * p + 5) / 6) * np.log(det_corr)
    dof = p * (p - 1) / 2
    p_value = 1 - chi2.cdf(chi2_stat, dof)
    return {"chi2": float(chi2_stat), "dof": int(dof), "p_value": float(p_value), "n": n, "p": p}


# ---------------------------------------------------------------
# 4. Analyse en composantes principales (ACP)
# ---------------------------------------------------------------

def run_pca_pipeline(df, feature_cols, variance_threshold=0.80, standardize=True,
                      impute_missing=False, impute_strategy="median"):
    """
    ACP complete sur les variables retenues (post-reduction VIF).

    impute_missing : cf. compute_vif -- meme logique d'imputation par
    mediane/moyenne si des sujets ont des valeurs manquantes (statistiques
    d'intervalle non definies pour les sujets a tres peu d'activations).

    Retourne : eigenvalues, explained_variance_ratio, cumulative_variance,
    n_components_kaiser (valeurs propres > 1), n_components_variance_threshold
    (nb composantes pour atteindre variance_threshold cumule), loadings
    (variables x composantes, pour interpretation), scores (sujets x
    composantes), pca_object (sklearn.decomposition.PCA ajuste).
    """
    data = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)

    if impute_missing:
        if impute_strategy == "median":
            data = data.fillna(data.median())
        elif impute_strategy == "mean":
            data = data.fillna(data.mean())
        else:
            raise ValueError('impute_strategy doit etre "median" ou "mean"')
        valid_idx = data.index
    else:
        valid_idx = data.dropna().index
        data = data.loc[valid_idx]

    X = StandardScaler().fit_transform(data.values) if standardize else data.values

    pca = PCA(n_components=min(X.shape))
    scores_all = pca.fit_transform(X)

    eigenvalues = pca.explained_variance_
    explained_ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained_ratio)

    n_kaiser = int(np.sum(eigenvalues > 1))
    n_variance = int(np.searchsorted(cumulative, variance_threshold) + 1)
    n_components = max(1, min(n_kaiser if n_kaiser > 0 else n_variance, len(feature_cols)))

    loadings = pd.DataFrame(
        pca.components_[:n_components].T,
        index=feature_cols,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    scores = pd.DataFrame(
        scores_all[:, :n_components],
        index=valid_idx,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    return {
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": explained_ratio,
        "cumulative_variance": cumulative,
        "n_components_kaiser": n_kaiser,
        "n_components_variance_threshold": n_variance,
        "n_components_selected": n_components,
        "loadings": loadings,
        "scores": scores,
        "pca_object": pca,
    }


# ---------------------------------------------------------------
# 5. Rapport complet (orchestration)
# ---------------------------------------------------------------

def full_dimensionality_reduction_report(
    df,
    feature_cols,
    vif_threshold=5.0,
    corr_threshold=0.8,
    variance_threshold=0.80,
    impute_missing=False,
    impute_strategy="median",
    verbose=True,
):
    """
    Pipeline complet recommande en amont du score composite de severite :
        1) matrice de Spearman + paires redondantes (|rho| >= corr_threshold)
        2) reduction iterative par VIF (seuil vif_threshold)
        3) KMO + Bartlett sur les variables retenues
        4) ACP sur les variables retenues

    impute_missing : si True, impute par mediane/moyenne les valeurs
    manquantes (utile si des sujets ont trop peu d'activations pour des
    statistiques d'intervalle definies -- typiquement les groupes
    controle) avant le calcul du VIF et de l'ACP. A documenter comme
    limite methodologique.

    Retourne un dict avec toutes les etapes intermediaires et les
    variables finalement retenues.
    """
    rho_mat, p_mat = spearman_matrix(df, feature_cols)
    redundant_pairs = flag_redundant_pairs(rho_mat, threshold=corr_threshold)

    if verbose and not redundant_pairs.empty:
        print(f"Paires fortement correlees (|rho| >= {corr_threshold:.2f}) :")
        print(redundant_pairs.to_string(index=False))
    elif verbose:
        print(f"Aucune paire avec |rho| >= {corr_threshold:.2f}.")

    vif_result = iterative_vif_reduction(
        df, feature_cols, vif_threshold=vif_threshold, verbose=verbose,
        impute_missing=impute_missing, impute_strategy=impute_strategy,
    )
    retained = vif_result["retained_features"]

    kmo_result = kmo_test(df, retained)
    bartlett_result = bartlett_sphericity_test(df, retained)

    if verbose:
        print(f"\nKMO global (variables retenues) : {kmo_result['kmo_overall']:.3f}")
        print(
            f"Test de Bartlett : chi2={bartlett_result['chi2']:.2f}, "
            f"dof={bartlett_result['dof']}, p={bartlett_result['p_value']:.4g}"
        )
        if not np.isnan(kmo_result["kmo_overall"]) and kmo_result["kmo_overall"] < 0.5:
            print("ATTENTION : KMO < 0.5, l'ACP est deconseillee sur ce jeu de variables.")
        if bartlett_result["p_value"] >= 0.05:
            print(
                "ATTENTION : Bartlett non significatif, les variables pourraient "
                "etre trop peu correlees pour une ACP informative."
            )

    pca_result = run_pca_pipeline(
        df, retained, variance_threshold=variance_threshold,
        impute_missing=impute_missing, impute_strategy=impute_strategy,
    )

    return {
        "spearman_rho": rho_mat,
        "spearman_p": p_mat,
        "redundant_pairs": redundant_pairs,
        "vif_reduction": vif_result,
        "retained_features": retained,
        "kmo": kmo_result,
        "bartlett": bartlett_result,
        "pca": pca_result,
    }


def save_dimensionality_reduction_report(report, subdir="dimensionality_reduction"):
    """
    Sauvegarde les elements du rapport full_dimensionality_reduction_report()
    sous results/<subdir>/ (cf. results_io.py -- convention unique de
    sortie du pipeline) :
        spearman_rho.csv, redundant_pairs.csv, vif_history_<n>.csv,
        vif_retained.txt, kmo_per_variable.csv, kmo_bartlett_summary.csv,
        pca_loadings.csv, pca_scores.csv, pca_variance.csv
    """
    from results_io import save_dataframe, save_text
    import pandas as pd

    paths = {}
    paths["spearman_rho"] = save_dataframe(
        report["spearman_rho"], subdir, "spearman_rho.csv", index=True
    )
    paths["redundant_pairs"] = save_dataframe(
        report["redundant_pairs"], subdir, "redundant_pairs.csv"
    )

    vif_result = report["vif_reduction"]
    for i, vif_df in enumerate(vif_result["history"]):
        save_dataframe(vif_df, subdir, f"vif_history_step{i:02d}.csv")
    paths["vif_removed"] = save_dataframe(
        vif_result["removed_features"], subdir, "vif_removed_features.csv"
    )
    paths["vif_retained"] = save_text(
        "\n".join(report["retained_features"]), subdir, "vif_retained_features.txt"
    )

    paths["kmo_per_variable"] = save_dataframe(
        report["kmo"]["kmo_per_variable"], subdir, "kmo_per_variable.csv"
    )
    summary = pd.DataFrame([{
        "kmo_overall": report["kmo"]["kmo_overall"],
        "bartlett_chi2": report["bartlett"]["chi2"],
        "bartlett_dof": report["bartlett"]["dof"],
        "bartlett_p_value": report["bartlett"]["p_value"],
        "bartlett_n": report["bartlett"]["n"],
        "bartlett_p_vars": report["bartlett"]["p"],
    }])
    paths["kmo_bartlett_summary"] = save_dataframe(
        summary, subdir, "kmo_bartlett_summary.csv"
    )

    pca = report["pca"]
    paths["pca_loadings"] = save_dataframe(
        pca["loadings"], subdir, "pca_loadings.csv", index=True
    )
    paths["pca_scores"] = save_dataframe(
        pca["scores"], subdir, "pca_scores.csv", index=True
    )
    variance_df = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(pca["explained_variance_ratio"]))],
        "eigenvalue": pca["eigenvalues"],
        "explained_variance_ratio": pca["explained_variance_ratio"],
        "cumulative_variance": pca["cumulative_variance"],
    })
    paths["pca_variance"] = save_dataframe(variance_df, subdir, "pca_variance.csv")

    return paths