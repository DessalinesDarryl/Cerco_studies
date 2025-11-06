# src/viz/plots.py
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
def save_heatmap(df, index, columns, values, out_png):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    pvt = df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    plt.figure(); sns.heatmap(pvt, annot=False); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
def save_bar(series, out_png, title=""):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(); series.head(20).plot(kind="bar"); plt.title(title); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
