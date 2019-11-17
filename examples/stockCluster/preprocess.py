import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale(path):
    df = pd.read_csv(path, index_col = 0)
    scaled_feats = StandardScaler().fit_transform(df.values)
    scaled_features_df = pd.DataFrame(scaled_feats,
                                      columns=df.columns)
    return(scaled_features_df)

