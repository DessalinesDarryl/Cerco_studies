# tests/test_stats.py
def test_anova():
    import pandas as pd
    from src.stats.anova_kruskal import compare_groups
    df = pd.DataFrame({"band":["a","a","a","b","b","b"],"group":[0,1,2,0,1,2],"score":[1,2,3,1,2,3]})
    res = compare_groups(df, "score", "group", by="band")
    assert "anova_p" in res.columns
