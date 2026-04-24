"""Calcul d'attributions DeepLIFT pour les modèles PyTorch.

Ce module encapsule Captum DeepLIFT pour produire des scores
d'importance de features ciblés par classe.
"""

import torch
from captum.attr import DeepLift


def compute_deeplift(model, inputs, target_idx):
    """Calcule les attributions DeepLIFT pour un modèle PyTorch.

    Paramètres
    ----------
    model : nn.Module
        Modèle PyTorch en mode eval.
    inputs : torch.Tensor
        Entrées du modèle (batch de features).
    target_idx : int
        Index de classe cible pour l'attribution.

    Retours
    -------
    attributions : torch.Tensor
        Scores d'attribution, même shape que `inputs`.
    """
    model.eval()
    dl = DeepLift(model)
    attributions = dl.attribute(inputs, target=target_idx)
    return attributions
