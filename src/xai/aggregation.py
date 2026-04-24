"""Agrégation des attributions XAI par niveaux sémantiques.

Le module regroupe les contributions par bande/canal/modalité afin de
faciliter l'analyse globale des facteurs explicatifs.
"""

import numpy as np
import pandas as pd


def aggregate_attributions(attr, feature_names, groupby=("band","channel","modality")):
    """Agrège des attributions selon les dimensions sémantiques présentes dans le nom des features."""
    # Les noms de features peuvent suivre deux conventions : `CH|BAND|MODALITY` ou `CH_BAND_MODALITY`.
    rows = []
    for a, n in zip(attr, feature_names):
        parts = n.split("|") if "|" in n else n.split("_")
        d = {"name": n, "value": float(a)}
        if len(parts) >= 3:
            d.update({"channel": parts[0], "band": parts[1], "modality": parts[2]})
        rows.append(d)
    df = pd.DataFrame(rows)
    aggs = {}
    for g in ["band", "channel", "modality"]:
        if g in df.columns:
            aggs[g] = df.groupby(g)["value"].mean().sort_values(ascending=False)
    return df, aggs
