# src/viz/plots.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _topk_order(values: np.ndarray, names, k: int):
    """Retourne indices + noms triés par importance décroissante (top-k)."""
    values = np.asarray(values)
    k = min(k, values.shape[0])
    order = np.argsort(values)[::-1][:k]
    return order, [names[i] for i in order], values[order]


# ======================================================================
# 1) Importance globale des features
# ======================================================================

def plot_global_feature_importance(
    importances: np.ndarray,
    feature_names,
    top_k: int = 25,
    title: str = "Importances globales des features (RandomForest)",
    out_path=None,
    figsize=(8, 10),
):
    """
    Barplot vertical des top-k features les plus importantes.
    """
    importances = np.asarray(importances)
    order, names_top, vals_top = _topk_order(importances, feature_names, top_k)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(names_top)), vals_top[::-1])
    ax.set_yticks(range(len(names_top)))
    ax.set_yticklabels(names_top[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
    else:
        return fig, ax


# ======================================================================
# 2) Importance par groupe de patients (heatmap)
# ======================================================================

def compute_group_feature_contributions(
    df: pd.DataFrame,
    feature_cols,
    label_col: str,
    importances: np.ndarray,
):
    """
    Approche simple :
      contrib_g,j = |mean(X_j | groupe=g) - mean(X_j global)| * importance_j

    Retourne un DataFrame (group x feature) utilisable pour une heatmap.
    """
    X = df[feature_cols].values.astype(float)
    y = df[label_col].values

    importances = np.asarray(importances)
    if importances.shape[0] != X.shape[1]:
        raise ValueError("importances et feature_cols de tailles incompatibles.")

    global_mean = X.mean(axis=0)

    groups = np.unique(y)
    data = []
    for g in groups:
        mask = (y == g)
        if mask.sum() == 0:
            continue
        mean_g = X[mask].mean(axis=0)
        diff = np.abs(mean_g - global_mean)
        contrib = diff * importances
        data.append(contrib)

    if not data:
        return pd.DataFrame(columns=feature_cols)

    mat = np.vstack(data)  # shape (n_groups, n_features)
    df_contrib = pd.DataFrame(mat, index=groups, columns=feature_cols)
    return df_contrib


def plot_group_feature_importance_heatmap(
    df_contrib: pd.DataFrame,
    top_k: int = 25,
    title: str = "Importances par groupe (contributions moyennes)",
    out_path=None,
    figsize=(10, 6),
):
    """
    Heatmap des top-k features les plus discriminantes entre groupes.
    df_contrib : DataFrame index = groupes, colonnes = features, valeurs = contrib.
    """
    # On choisit les features avec plus grande importance moyenne
    mean_contrib = df_contrib.abs().mean(axis=0)
    order, feat_top, _ = _topk_order(mean_contrib.values, df_contrib.columns.tolist(), top_k)

    df_top = df_contrib[feat_top]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        df_top,
        cmap="mako",
        annot=False,
        cbar_kws={"label": "Contribution (importance x différence de moyenne)"},
        ax=ax,
    )
    ax.set_xlabel("Features")
    ax.set_ylabel("Groupe")
    ax.set_title(title)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
    else:
        return fig, ax


# ======================================================================
# 3) Contributions locales par patient
# ======================================================================

def compute_patient_contributions(
    x_row: np.ndarray,
    baseline: np.ndarray,
    importances: np.ndarray,
    feature_names,
):
    """
    Approx local type "attribution" (très simple) :

      contrib_j = (x_j - baseline_j) * importance_j

    - baseline = moyenne globale (ou moyenne du groupe contrôle)
    - signe : direction (valeur > baseline et importance positive => pousse la classe)
    - magnitude : impact relatif

    Retourne DataFrame avec colonnes [feature, contribution, abs_contribution].
    """
    x_row = np.asarray(x_row).astype(float)
    baseline = np.asarray(baseline).astype(float)
    importances = np.asarray(importances).astype(float)

    if x_row.shape[0] != baseline.shape[0] or x_row.shape[0] != importances.shape[0]:
        raise ValueError("x_row, baseline et importances doivent avoir la même taille.")

    diff = x_row - baseline
    contrib = diff * importances
    abs_contrib = np.abs(contrib)

    df = pd.DataFrame({
        "feature": feature_names,
        "contribution": contrib,
        "abs_contribution": abs_contrib,
    })
    df = df.sort_values("abs_contribution", ascending=False)
    return df


def plot_patient_contributions(
    df_contrib: pd.DataFrame,
    patient_id: str,
    top_k: int = 20,
    title_prefix: str = "Attributions locales (RF, proxy importance x écart)",
    out_path=None,
    figsize=(8, 6),
):
    """
    Barplot des top-k contributions (signées) pour un patient.
    """
    df_top = df_contrib.head(top_k).copy()
    df_top = df_top.iloc[::-1]  # pour que la plus grande soit en haut

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(
        y=df_top["feature"],
        width=df_top["contribution"],
    )
    ax.axvline(0.0, color="k", linewidth=1)
    ax.set_xlabel("Contribution")
    ax.set_title(f"{title_prefix} – patient {patient_id}")
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
    else:
        return fig, ax
