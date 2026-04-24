"""Fonctions de visualisation des métriques de classification.

Le module regroupe les tracés standards (matrice de confusion, courbes ROC)
utilisés dans les rapports d'évaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)


# ==========================================================
# 1) Matrice de confusion
# ==========================================================
def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    normalize="true",
    title="Matrice de confusion",
    out_path=None,
    figsize=(6, 6),
):
    """
    Trace une matrice de confusion, éventuellement normalisée.

    `normalize` peut valoir `None`, `"true"`, `"pred"` ou `"all"`.
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="mako",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(title)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
    else:
        return fig, ax


# ==========================================================
# 2) Courbes ROC multiclasses (one-vs-rest)
# ==========================================================
def plot_multiclass_roc(
    y_true,
    y_proba,
    class_names,
    out_path=None,
    title="ROC multiclasses (ovr)",
    figsize=(8, 6),
):
    """
    Trace les courbes ROC one-vs-rest pour un problème multiclasses.

    `y_true` peut contenir des labels texte ou entiers.
    `y_proba` doit être de forme `(n_samples, n_classes)`.
    """
    # Encodage local des classes en indices entiers pour le calcul ROC.
    classes = np.unique(y_true)
    name_to_int = {c: i for i, c in enumerate(classes)}
    y_int = np.array([name_to_int[v] for v in y_true])

    fig, ax = plt.subplots(figsize=figsize)

    for idx, c in enumerate(classes):
        # Construction de la cible binaire one-vs-rest pour la classe courante.
        y_bin = (y_int == idx).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, idx])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{c} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
    else:
        return fig, ax
