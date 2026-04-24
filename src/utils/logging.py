"""Fabrique de loggers standardisés pour les scripts du pipeline.

Le module crée un logger console/fichier avec format homogène afin de
faciliter le suivi des exécutions et le débogage.
"""

import logging
import os


def get_logger(name="app", logfile=None):
    """Construit un logger console, avec sortie fichier optionnelle."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.handlers = [ch]
    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
