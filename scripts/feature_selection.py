import pandas as pd
import numpy as np

def correlation(featureDf: pd.DataFrame, threshold: float = 0.5):
    corrMatrix = featureDf.corr('pearson').abs()
    mask = np.triu(np.ones_like(corrMatrix, dtype=bool), k=1)
    upper = corrMatrix.where(mask)

    correlated_pairs = (
        upper.stack() # transforms DF to a table with with levels and values as the columns
        .reset_index() # 
        .rename(columns={"level_0": "feature_1",
                         "level_1": "feature_2", 
                         0: "correlation"})
        .query("correlation > @threshold")
        .sort_values(by="correlation", ascending=False)
        .reset_index(drop=True) # drop extra index column
    )

    return correlated_pairs