# src/xai/aggregation.py
import numpy as np, pandas as pd
def aggregate_attributions(attr, feature_names, groupby=("band","channel","modality")):
    # feature_names ex: CH_BAND_MODALITY
    rows=[]
    for a,n in zip(attr, feature_names):
        parts = n.split("|") if "|" in n else n.split("_")
        d = {"name":n, "value":float(a)}
        if len(parts)>=3: d.update({"channel":parts[0], "band":parts[1], "modality":parts[2]})
        rows.append(d)
    df = pd.DataFrame(rows)
    aggs = {}
    for g in ["band","channel","modality"]:
        if g in df.columns: aggs[g]=df.groupby(g)["value"].mean().sort_values(ascending=False)
    return df, aggs
