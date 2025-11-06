# scripts/train.py
import os, argparse, torch, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from src.utils.config import load_yaml, add_common_args
from src.utils.seed import set_seed
from src.models.xaimodel import SimpleTransformer
from src.training.lit_module import SimpleModule, make_dataloaders
from src.utils.logging import get_logger
def load_features_labels(root, input_dim):
    X = pd.read_csv(os.path.join(root,"features.csv"))
    y = pd.read_csv(os.path.join(root,"labels.csv"))["label"].values
    feats = [c for c in X.columns if c not in ("file","epoch_idx")]
    X = X[feats].fillna(0.0).values.astype("float32")
    if X.shape[1]<input_dim:  # pad
        pad = np.zeros((X.shape[0], input_dim-X.shape[1]), dtype=np.float32); X=np.hstack([X,pad])
    elif X.shape[1]>input_dim:
        X= X[:,:input_dim]
    X = X[:,None,:]  # (B, T=1, F)
    return torch.tensor(X), torch.tensor(y, dtype=torch.long)
def main(cfg):
    set_seed(cfg["seed"]); log=get_logger("train")
    X,y = load_features_labels(cfg["features_root"], cfg["model"]["input_dim"])
    Xtr,Xv, ytr,yv = train_test_split(X,y, test_size=0.2, stratify=y, random_state=cfg["seed"])
    train_dl, val_dl = make_dataloaders(Xtr,ytr,Xv,yv,batch=cfg["batch_size"])
    model = SimpleTransformer(**cfg["model"])
    module = SimpleModule(model, lr=cfg["lr"])
    device = "cuda" if torch.cuda.is_available() else "cpu"; module.to(device)
    module.fit(train_dl, val_dl, epochs=cfg["max_epochs"], device=device)
    os.makedirs(os.path.dirname(cfg["out_ckpt"]), exist_ok=True)
    torch.save(model.state_dict(), cfg["out_ckpt"]); log.info(f"Saved {cfg['out_ckpt']}")
if __name__=="__main__":
    ap = add_common_args(argparse.ArgumentParser()); args=ap.parse_args(); cfg=load_yaml(args.config); main(cfg)
