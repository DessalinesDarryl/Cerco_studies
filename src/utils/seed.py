"""Initialisation déterministe des graines aléatoires.

Le module fixe les seeds Python/NumPy/PyTorch pour améliorer la
reproductibilité des résultats expérimentaux.
"""

import os
import random

import numpy as np

try:
    import torch
except ImportError:  # Optional for pre-ML workflows.
    torch = None


def set_seed(seed: int = 42):
    """Fixe les graines Python/NumPy et PyTorch si disponible."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
