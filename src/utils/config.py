"""Chargement et gestion centralisée des configurations YAML.

Ce module fournit des helpers légers pour lire les fichiers de config
et injecter les arguments communs des scripts CLI.
"""

import argparse
import os

import yaml


def load_yaml(path: str) -> dict:
    """Charge un fichier YAML et retourne son contenu sous forme de dictionnaire."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def env_or_default(v, default): return os.getenv(v, default)


def add_common_args(p: argparse.ArgumentParser):
    """Ajoute l'argument CLI `--config` partagé par les scripts du pipeline."""
    p.add_argument("--config", type=str, required=True)
    return p
