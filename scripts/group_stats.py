# scripts/group_stats.py
import os, argparse, pandas as pd, numpy as np
from src.utils.config import add_common_args, load_yaml
from src.stats.anova_kruskal import compare_groups
from src.stats.cbpt import dummy_cbpt
from src.stats.graphs_metrics import metrics_from_attention
def main(cfg):
    out = cfg["out_dir"]; os.makedirs(out, exist_ok=True)
    local = pd.read_csv(os.path.join(cfg["xai_root"], "attributions", "local_attributions.csv"))
    # ANOVA/Kruskal par bande
    local["group"]=local["target"]
    anova = compare_groups(local.rename(columns={"value":"score"}), value_col="score", group_col="group", by="band")
    anova.to_csv(os.path.join(out, "anova_by_band.csv"), index=False)
    # CBPT proxy sur features top-k (ex: agrégé par groupe)
    tensor_by_class={g: v["value"].values.reshape(-1,1) for g,v in local.groupby("group")}
    cbpt = dummy_cbpt(tensor_by_class, n_perm=cfg["cbpt"]["n_perm"], p_cluster=cfg["cbpt"]["p_cluster"])
    cbpt.to_csv(os.path.join(out, "cbpt_results.csv"), index=False)
    # Graph metrics si attention dispo
    attn_path = os.path.join("outputs","xai","attention","attention_mean.npy")
    if os.path.exists(attn_path):
        import numpy as np
        A = np.load(attn_path); gm = metrics_from_attention(A)
        gm.to_csv(os.path.join(out,"graph_metrics.csv"), index=False)
if __name__=="__main__":
    ap = add_common_args(argparse.ArgumentParser()); args=ap.parse_args(); cfg=load_yaml(args.config); main(cfg)
