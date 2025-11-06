# src/stats/anova_kruskal.py
import pandas as pd
from scipy.stats import f_oneway, kruskal
def compare_groups(df_long: pd.DataFrame, value_col: str, group_col: str, by: str="band"):
    res={}
    for k,g in df_long.groupby(by):
        vals=[v[value_col].values for _,v in g.groupby(group_col)]
        if len(vals)>=2 and all(len(v)>1 for v in vals):
            try: res[k]={"anova_p":f_oneway(*vals).pvalue, "kruskal_p":kruskal(*vals).pvalue}
            except Exception: res[k]={"anova_p":None,"kruskal_p":None}
    return pd.DataFrame.from_dict(res, orient="index").reset_index().rename(columns={"index":by})
