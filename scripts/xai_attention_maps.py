# scripts/xai_attention_maps.py
import os, argparse, numpy as np, torch, pandas as pd
from src.utils.config import load_yaml, add_common_args
from src.models.xaimodel import SimpleTransformer
from src.xai.attention import extract_dummy_attention
def main(cfg):
    feats = pd.read_csv(os.path.join(cfg["features_root"],"features.csv")).fillna(0.0)
    feature_cols = [c for c in feats.columns if c not in ("file","epoch_idx")]
    X = feats[feature_cols].values.astype("float32")
    input_dim = X.shape[1]; X = np.pad(X, ((0,0),(0,max(0,256-input_dim))), mode="constant")[:,:256]
    X = torch.tensor(X[:,None,:])
    model = SimpleTransformer(input_dim=256, hidden_dim=128, heads=4, depth=2, num_classes=4)
    model.load_state_dict(torch.load(cfg["ckpt"], map_location="cpu")); model.eval()
    A = extract_dummy_attention(model, X[:32])  # (B,F,F)
    A_mean = A.mean(0).numpy()
    os.makedirs(cfg["out_dir"], exist_ok=True)
    np.save(os.path.join(cfg["out_dir"], "attention_mean.npy"), A_mean)
if __name__=="__main__":
    ap = add_common_args(argparse.ArgumentParser()); args=ap.parse_args(); cfg=load_yaml(args.config); main(cfg)
