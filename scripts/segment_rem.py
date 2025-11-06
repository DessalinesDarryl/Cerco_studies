# scripts/segment_rem.py
import os, argparse, glob
from src.utils.config import load_yaml, add_common_args
from src.segmentation.rem import segment_rem
def main(cfg):
    fif_list = glob.glob(os.path.join(cfg["in_root"], "**", "*.fif"), recursive=True)
    for fif in fif_list:
        base = os.path.splitext(os.path.basename(fif))[0]
        annot = os.path.join(os.path.dirname(fif), base + cfg.get("label_file_ext",".txt"))
        if not os.path.exists(annot): continue
        out = os.path.join(cfg["out_root"], base+"_REM-epo.fif")
        segment_rem(fif, annot, cfg["tonic_phasic"].get("burst_max_ms",500)/1000.0, out)
if __name__=="__main__":
    ap = add_common_args(argparse.ArgumentParser()); args=ap.parse_args(); cfg=load_yaml(args.config); main(cfg)
