# scripts/xai_attributions.py
import os, argparse, numpy as np, torch, pandas as pd
from src.utils.config import load_yaml, add_common_args
from src.models.xaimodel import SimpleTransformer
from src.xai.deeplift import compute_deeplift
from src.xai.integrated_gradients import compute_ig
from src.xai.aggregation import aggregate_attributions
def main(cfg):
    feats = pd.read_csv(os.path.join(cfg["features_root"],"features.csv")).fillna(0.0)
    labels = pd.read_csv(os.path.join(cfg["features_root"],"labels.csv"))["label"].values
    feature_cols = [c for c in feats.columns if c not in ("file","epoch_idx")]
    X = feats[feature_cols].values.astype("float32")
    input_dim = X.shape[1]; X = np.pad(X, ((0,0),(0,max(0,256-input_dim))), mode="constant")[:,:256]
    X = torch.tensor(X[:,None,:], requires_grad=True)
    num_classes = len(np.unique(labels))
    model = SimpleTransformer(input_dim=256, hidden_dim=128, heads=4, depth=2, num_classes=num_classes)
    model.load_state_dict(torch.load(cfg["ckpt"], map_location="cpu")); model.eval()
    os.makedirs(cfg["out_dir"], exist_ok=True)
    all_local=[]
    for i in range(min(64, X.shape[0])):  # MVP: 64 premiers
        target = int(labels[i])
        xin = X[i:i+1]
        attr = compute_deeplift(model, xin, target) if cfg["method"]=="deeplift" else compute_ig(model, xin, target)
        vec = attr.detach().cpu().numpy().reshape(-1)[:len(feature_cols)]
        df,aggs = aggregate_attributions(vec, feature_cols)
        df["idx"]=i; df["target"]=target; all_local.append(df)
    local_df = pd.concat(all_local, ignore_index=True)
    local_df.to_csv(os.path.join(cfg["out_dir"], "local_attributions.csv"), index=False)
    # agrégations globales
    g_band = local_df.groupby("band")["value"].mean().sort_values(ascending=False)
    g_band.to_csv(os.path.join(cfg["out_dir"], "global_by_band.csv"))
    g_chan = local_df.groupby("channel")["value"].mean().sort_values(ascending=False)
    g_chan.to_csv(os.path.join(cfg["out_dir"], "global_by_channel.csv"))
    if "modality" in local_df.columns:
        g_mod = local_df.groupby("modality")["value"].mean().sort_values(ascending=False)
        g_mod.to_csv(os.path.join(cfg["out_dir"], "global_by_modality.csv"))
if __name__=="__main__":
    ap = add_common_args(argparse.ArgumentParser()); args=ap.parse_args(); cfg=load_yaml(args.config); main(cfg)
