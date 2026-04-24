"""Calcul d'attributions Integrated Gradients pour modèles PyTorch.

Le module encapsule Captum Integrated Gradients avec paramétrage du
nombre de pas d'intégration pour l'explicabilité des prédictions.
"""

import torch
from captum.attr import IntegratedGradients


def compute_ig(model, inputs, target_idx, steps=50):
    """Calcule les attributions Integrated Gradients pour un modèle PyTorch.

    Paramètres
    ----------
    model : nn.Module
        Modèle PyTorch en mode eval.
    inputs : torch.Tensor
        Entrées du modèle (batch de features).
    target_idx : int
        Index de classe cible pour l'attribution.
    steps : int
        Nombre de pas d'intégration (par défaut 50).

    Retours
    -------
    attributions : torch.Tensor
        Scores d'attribution, même shape que `inputs`.
    """
    model.eval()
    ig = IntegratedGradients(model)
    return ig.attribute(inputs, target=target_idx, n_steps=steps)
