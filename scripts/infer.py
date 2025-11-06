# scripts/infer.py
import os, argparse, torch, pandas as pd, numpy as np
from src.utils.config import load_yaml, add_common_args
from src.models.xaimodel import SimpleTransformer
from src.training.lit_module import make_dataloaders
def main(cfg):
    feats = pd.read_csv(os.path.join(cfg["features_root"],"features.csv")).fillna(0.0)
    y = pd.read_csv(os.path.join(cfg["features_root"],"labels.csv"))["label"].values
    X = feats.drop(columns=[c for c in ["file","epoch_idx"] if c in feats.columns]).values.astype("float32")
    input_dim = X.shape[1]; X = np.pad(X, ((0,0),(0,max(0,256-input_dim))), mode="constant")[:,:256]
    X = torch.tensor(X[:,None,:]); y = torch.tensor(y)
    dl = make_dataloaders(X,y,X,y,batch=128)[1]
    m = SimpleTransformer(input_dim=256, hidden_dim=128, heads=4, depth=2, num_classes=len(np.unique(y)))
    m.load_state_dict(torch.load(cfg["ckpt"], map_location="cpu")); m.eval()
    import torch as T
    with T.no_grad():
        probs=[]; tru=[]
        for xb,yb in dl: probs.append(m(xb).softmax(1)); tru.append(yb)
    probs = T.cat(probs).numpy(); tru = T.cat(tru).numpy()
    out = pd.DataFrame(probs, columns=[f"p{i}" for i in range(probs.shape[1])]); out["y"]=tru
    os.makedirs(os.path.dirname(cfg["out_csv"]), exist_ok=True); out.to_csv(cfg["out_csv"], index=False)
if __name__=="__main__":
    ap = add_common_args(argparse.ArgumentParser()); args=ap.parse_args(); cfg=load_yaml(args.config); main(cfg)
