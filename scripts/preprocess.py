# scripts/preprocess.py
import os, argparse, glob, mne
from src.utils.config import load_yaml, add_common_args
from src.utils.logging import get_logger
from src.data.preprocessing import basic_preprocess
from src.data.io import read_raw, save_fif
def main(cfg):
    log = get_logger("preprocess")
    raw_root, out_root = cfg["raw_root"], cfg["out_root"]
    edfs = glob.glob(os.path.join(raw_root, "**", "*.edf"), recursive=True)
    for p in edfs:
        raw = read_raw(p, preload=True)
        raw = basic_preprocess(raw, **cfg["eeg"])
        rel = os.path.relpath(p, raw_root).replace(".edf",".fif")
        out = os.path.join(out_root, rel)
        save_fif(raw, out); log.info(f"Saved {out}")
if __name__=="__main__":
    ap = add_common_args(argparse.ArgumentParser()); args=ap.parse_args(); cfg=load_yaml(args.config); main(cfg)
