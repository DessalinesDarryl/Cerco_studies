# scripts/build_reports.py
import os, argparse
from src.utils.config import add_common_args, load_yaml
from src.viz.reports import build_report
def main(cfg):
    build_report(cfg["title"], cfg["xai_root"], cfg["stats_root"], cfg["fig_root"], cfg["out_dir"])
if __name__=="__main__":
    ap = add_common_args(argparse.ArgumentParser()); args=ap.parse_args(); cfg=load_yaml(args.config); main(cfg)
