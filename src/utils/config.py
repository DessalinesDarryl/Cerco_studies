# src/utils/config.py
import argparse, yaml, os
def load_yaml(path: str) -> dict:
    with open(path, "r") as f: return yaml.safe_load(f)
def env_or_default(v, default): return os.getenv(v, default)
def add_common_args(p: argparse.ArgumentParser):
    p.add_argument("--config", type=str, required=True); return p
