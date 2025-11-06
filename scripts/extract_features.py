# scripts/extract_features.py
import os, argparse, glob, pandas as pd
from src.utils.config import load_yaml, add_common_args
from src.features.spectral import extract_epoch_features
def main(cfg):
    os.makedirs(cfg["out_root"], exist_ok=True)
    epochs = glob.glob(os.path.join(cfg["in_root"], "**", "*REM-epo.fif"), recursive=True)
    all_df=[]
    for fif in epochs:
        df,_ = extract_epoch_features(fif, cfg["bands"])
        df["file"]=os.path.basename(fif)
        all_df.append(df)
    out = pd.concat(all_df, ignore_index=True) if all_df else pd.DataFrame()
    out.to_csv(os.path.join(cfg["out_root"], "features.csv"), index=False)
    # labels stub (à remplacer par tes vraies étiquettes)
    if not os.path.exists(os.path.join(cfg["out_root"], "labels.csv")):
        pd.DataFrame({"file": out.get("file", pd.Series(dtype=str)), "label": 0}).to_csv(os.path.join(cfg["out_root"], "labels.csv"), index=False)
if __name__=="__main__":
    ap = add_common_args(argparse.ArgumentParser()); args=ap.parse_args(); cfg=load_yaml(args.config); main(cfg)
