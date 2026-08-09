"""
statistics_pipeline.py
------------------------
Comparaison statistique multi-groupes et construction du score composite de
severite, suivant la logique de design de Khalil et al. 2013 (JCSM, 4
groupes) et la methodologie bootstrap ROC de Frauscher et al. 2012 (Sleep).
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve


def zscore_against_controls(df, control_label, group_col="diagnosis", feature_cols=None):
    """
    Standardise chaque variable par rapport a la moyenne/SD du groupe
    controle (0 = niveau controle typique, positif = plus severe).
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    controls = df[df[group_col] == control_label]
    z = df.copy()
    for c in feature_cols:
        mu, sd = controls[c].mean(), controls[c].std()
        if sd == 0 or np.isnan(sd):
            z[c] = 0.0
        else:
            z[c] = (df[c] - mu) / sd
    return z, feature_cols


def composite_severity_score(z_df, feature_cols, method="mean"):
    """
    method='mean' : moyenne simple des z-scores (interpretable, style
                     echelle clinique additive)
    method='pca'  : score de la 1ere composante principale (ponderation
                     data-driven, recommande si forte redondance entre
                     variables)
    """
    X = z_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    if method == "mean":
        return X.mean(axis=1)
    elif method == "pca":
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(X).flatten()
        if pca.components_[0].sum() < 0:
            pc1 = -pc1
        return pc1, pca.explained_variance_ratio_[0]
    else:
        raise ValueError('method doit etre "mean" ou "pca"')


def kruskal_wallis_by_feature(df, feature, group_col="diagnosis"):
    groups = [g[feature].dropna().values for _, g in df.groupby(group_col)]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return np.nan, np.nan
    stat, p = stats.kruskal(*groups)
    return stat, p


def pairwise_mannwhitney_bonferroni(df, feature, group_col="diagnosis"):
    """
    Comparaisons deux-a-deux avec correction de Bonferroni (convention
    Khalil et al. 2013 : alpha=0.05/6 pour 4 groupes -> seuil 0.008).
    """
    labels = df[group_col].unique()
    pairs = list(combinations(labels, 2))
    if not pairs:
        return pd.DataFrame(), 0.05
    alpha_corrected = 0.05 / len(pairs)
    results = []
    for g1, g2 in pairs:
        x = df[df[group_col] == g1][feature].dropna()
        y = df[df[group_col] == g2][feature].dropna()
        if len(x) == 0 or len(y) == 0:
            continue
        stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        results.append(
            {
                "group1": g1,
                "group2": g2,
                "U": stat,
                "p": p,
                "significant_bonferroni": p < alpha_corrected,
            }
        )
    return pd.DataFrame(results), alpha_corrected


def roc_vs_controls(
    df, feature, group_col, group_label, control_label, n_bootstrap=2000, random_state=0
):
    """
    ROC + AUC + IC95% bootstrap pour un groupe vs controles (2000 replicats
    stratifies, convention Frauscher et al. 2012), avec seuil optimal de
    Youden (convention Khalil et al. 2013).
    """
    sub = df[df[group_col].isin([group_label, control_label])].copy()
    y = (sub[group_col] == group_label).astype(int).values
    x = sub[feature].values.astype(float)
    valid = np.isfinite(x)
    y, x = y[valid], x[valid]

    if len(np.unique(y)) < 2 or len(x) < 4:
        return None

    auc = roc_auc_score(y, x)
    fpr, tpr, thresholds = roc_curve(y, x)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    best_cutoff = thresholds[best_idx]
    sens = tpr[best_idx]
    spec = 1 - fpr[best_idx]

    rng = np.random.RandomState(random_state)
    boot_aucs = []
    n = len(y)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        yb, xb = y[idx], x[idx]
        if len(np.unique(yb)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yb, xb))

    if len(boot_aucs) > 0:
        ci_lower, ci_upper = np.percentile(boot_aucs, [2.5, 97.5])
    else:
        ci_lower, ci_upper = np.nan, np.nan

    return {
        "feature": feature,
        "group": group_label,
        "auc": auc,
        "auc_ci_lower": ci_lower,
        "auc_ci_upper": ci_upper,
        "cutoff": best_cutoff,
        "sensitivity": sens,
        "specificity": spec,
    }


def full_group_comparison(df, feature_cols, group_col="diagnosis", control_label="control"):
    """
    Lance Kruskal-Wallis + Mann-Whitney pairwise (Bonferroni) + ROC-vs-
    controles pour chaque variable et chaque groupe non-controle.
    Retourne 3 DataFrames : kw_df, mw_all, roc_df.
    """
    kw_rows, roc_rows, mw_frames = [], [], []
    other_groups = [g for g in df[group_col].unique() if g != control_label]

    for feat in feature_cols:
        stat, p = kruskal_wallis_by_feature(df, feat, group_col)
        kw_rows.append({"feature": feat, "H": stat, "p": p})

        mw_df, alpha_c = pairwise_mannwhitney_bonferroni(df, feat, group_col)
        if not mw_df.empty:
            mw_df["feature"] = feat
            mw_frames.append(mw_df)

        for g in other_groups:
            res = roc_vs_controls(df, feat, group_col, g, control_label)
            if res is not None:
                roc_rows.append(res)

    kw_df = pd.DataFrame(kw_rows)
    mw_all = pd.concat(mw_frames, ignore_index=True) if mw_frames else pd.DataFrame()
    roc_df = pd.DataFrame(roc_rows)
    return kw_df, mw_all, roc_df


def save_group_comparison(kw_df, mw_df, roc_df, subdir="statistics"):
    """
    Sauvegarde les 3 tables issues de full_group_comparison() sous
    results/<subdir>/ (cf. results_io.py -- convention unique de
    sortie du pipeline).
    """
    from results_io import save_dataframe

    paths = {}
    paths["kruskal_wallis"] = save_dataframe(kw_df, subdir, "kruskal_wallis.csv")
    paths["mann_whitney"] = save_dataframe(mw_df, subdir, "mann_whitney_bonferroni.csv")
    paths["roc_vs_controls"] = save_dataframe(roc_df, subdir, "roc_vs_controls.csv")
    return paths