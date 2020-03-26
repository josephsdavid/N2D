# Third party modules
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline  # for warping
from transforms3d.axangles import axangle2mat  # for rotation

# augmentation of data


def Jitter(X, sigma=0.5):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + myNoise


df = pd.read_csv("Data/stock_close.csv")

df.apply(Jitter, axis=1)


def augment(df, n):
    res = []
    for i in range(0, n):
        x = df.apply(Jitter, axis=1)
        res.append(np.asarray(x))
    return np.hstack(res)
