# scripts/plot_roc.py
import os, argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
def main(pred_csv, out_csv, out_png):
    df = pd.read_csv(pred_csv); y = df["y"].values
    proba = df[[c for c in df.columns if c.startswith("p")]].values
    # micro-avg ROC
    y_onehot = np.eye(proba.shape[1])[y]
    fpr, tpr, thr = roc_curve(y_onehot.ravel(), proba.ravel())
    roc_auc = auc(fpr, tpr)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame({"fpr":fpr,"tpr":tpr,"threshold":thr}).to_csv(out_csv, index=False)
    plt.figure(); plt.plot(fpr,tpr,label=f"micro-avg AUC={roc_auc:.3f}"); plt.plot([0,1],[0,1],"--"); plt.legend(); plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True); plt.savefig(out_png, dpi=150); plt.close()
if __name__=="__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--pred", required=True); ap.add_argument("--out_csv", default="outputs/figures/roc_microavg.csv"); ap.add_argument("--out_png", default="outputs/figures/roc.png"); a=ap.parse_args()
    main(a.pred, a.out_csv, a.out_png)
