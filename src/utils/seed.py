"""Initialisation déterministe des graines aléatoires.

Le module fixe les seeds Python/NumPy/PyTorch pour améliorer la
reproductibilité des résultats expérimentaux.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Fixe les graines Python, NumPy et PyTorch pour améliorer la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
