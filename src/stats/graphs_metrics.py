# src/stats/graphs_metrics.py
import numpy as np, pandas as pd
def degree(matrix: np.ndarray): return np.sum(matrix!=0, axis=1)
def metrics_from_attention(attn: np.ndarray):
    deg = degree(attn)
    return pd.DataFrame({"node": list(range(len(deg))), "degree": deg})
