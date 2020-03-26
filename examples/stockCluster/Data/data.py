# Third party modules
import numpy as np
import pandas as pd

rawdata = pd.read_csv("sp500.csv")
ticks = [r for r in rawdata.Name.unique()]


def getSeries(df, value):
    df2 = df[df.Name == value]
    return np.asarray(df2["close"])


tickDict = {t: getSeries(rawdata, t) for t in ticks}

tempDict = {}
for t in tickDict.keys():
    if len(tickDict[t]) >= 1225:
        tempDict[t] = tickDict[t]

print(len(tempDict.keys()))

for t in tempDict.keys():
    tempDict[t] = tempDict[t][:1225]

dat = pd.DataFrame.from_dict(tempDict)

dat.to_csv("stock_close.csv", sep=",")
