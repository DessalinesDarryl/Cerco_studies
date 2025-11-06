# src/stats/cbpt.py
# MVP: proxy très simplifié (pas un CBPT complet) pour illustrer l'API.
import numpy as np, pandas as pd
def dummy_cbpt(tensor_by_class, n_perm=1000, p_cluster=0.05):
    # tensor_by_class: dict{class: np.array [N, F]} → retourne “clusters” sur dimension F
    results=[]
    keys = list(tensor_by_class.keys())
    if len(keys)<2: return pd.DataFrame()
    A = tensor_by_class[keys[0]].mean(0); B = tensor_by_class[keys[1]].mean(0)
    diff = A-B
    thr = np.percentile(np.abs(diff), 95)  # faux seuil
    clusters = np.where(np.abs(diff)>=thr)[0]
    for idx in clusters:
        results.append({"feature_idx": int(idx), "pcluster": 0.04, "effect": float(diff[idx])})
    return pd.DataFrame(results)
