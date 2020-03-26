# Third party modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from augment import augment
from keras.layers import LSTM, Input, RepeatVector
from keras.models import Model
from preprocess import scale
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from statsmodels.tsa.stattools import coint

# First party modules
import n2d

# real data for clustering
test_x = scale("Data/stock_close.csv")

# fake data for training
train_x = augment(test_x, 100)
# transpose for our autoencoder
train_x = train_x.T

# x_test = np.asarray(test_x.values)
# x_test = x_test.reshape(476, 1225, 1)
#
# train_x = train_x.reshape(47600, 1225, 1)
#
# x.shape[0]
# x.shape[1]

# not used, an experiment
# it works but i lack any of understanding of LSTMs and how to input the data so
# this will be for another example
class LSTMAE:
    def __init__(self, data, ndim, architecture, act="relu"):
        # in this simple case, architecture is unused, but we keep for compat
        # with n2d
        dims = [data.shape[0]] + [data.shape[1]] + [ndim]
        self.dims = dims
        self.act = act
        self.x = Input(shape=(self.dims[1], 1), name="input")
        self.h = self.x
        # if manually naming, just start at one. The naming convention is input
        # layer gets named input, encoders get named encoder_NUM where number
        # goes up from 1, and decoders do the same!
        self.h = LSTM(self.dims[2], activation=self.act, name="encoder_1")(self.h)
        self.h = RepeatVector(self.dims[1])(self.h)
        self.h = LSTM(1, return_sequences=True)(self.h)
        self.Model = Model(inputs=self.x, outputs=self.h)

    def fit(
        self,
        dataset,
        batch_size=256,
        pretrain_epochs=1000,
        loss="mse",
        optimizer="adam",
        weights=None,
        verbose=0,
        weightname="fashion",
    ):
        if weights == None:
            self.Model.compile(loss=loss, optimizer=optimizer)
            self.Model.fit(
                dataset, dataset, batch_size=batch_size, epochs=pretrain_epochs
            )

            self.Model.save_weights(
                "weights/" + weightname + "-" + str(pretrain_epochs) + "-ae_weights.h5"
            )
        else:
            self.Model.load_weights(weights)


# proof of concept clustering
n_clusters = 12


model = n2d.n2d(
    train_x,
    manifoldLearner=n2d.UmapGMM(n_clusters, umapdim=12, umapN=20),
    ndim=20,
    ae_args={"act": "relu"},
)
model.fit(weights="sp_500-20", pretrain_epochs=50)
model.predict(np.asarray(test_x.values).T)

resD = {"name": test_x.columns, "cluster": model.preds}
pivot = pd.DataFrame.from_dict(resD)

# make some pretty plots
def plot_cluster(x, piv, n, r=None, c=None):
    cl = list(piv[piv.cluster == n].name.values)
    x[cl].plot(legend=False)


plot_cluster(test_x, pivot, 10)
plt.show()


for i in range(0, 11):
    plot_cluster(test_x, pivot, i)

plt.show()

# make too many pretty plots
# this is horrific code
fig, axes = plt.subplots(3, 4, figsize=(40, 20))
cl = list(pivot[pivot.cluster == 0].name.values)
test_x[cl].plot(ax=axes[0, 0], legend=False)
cl = list(pivot[pivot.cluster == 1].name.values)
test_x[cl].plot(ax=axes[0, 1], legend=False)
cl = list(pivot[pivot.cluster == 2].name.values)
test_x[cl].plot(ax=axes[0, 2], legend=False)
cl = list(pivot[pivot.cluster == 3].name.values)
test_x[cl].plot(ax=axes[1, 0], legend=False)
cl = list(pivot[pivot.cluster == 4].name.values)
test_x[cl].plot(ax=axes[1, 1], legend=False)
cl = list(pivot[pivot.cluster == 5].name.values)
test_x[cl].plot(ax=axes[1, 2], legend=False)
cl = list(pivot[pivot.cluster == 6].name.values)
test_x[cl].plot(ax=axes[2, 0], legend=False)
cl = list(pivot[pivot.cluster == 7].name.values)
test_x[cl].plot(ax=axes[2, 1], legend=False)
cl = list(pivot[pivot.cluster == 8].name.values)
test_x[cl].plot(ax=axes[2, 2], legend=False)
cl = list(pivot[pivot.cluster == 9].name.values)
test_x[cl].plot(ax=axes[0, 3], legend=False)
cl = list(pivot[pivot.cluster == 10].name.values)
test_x[cl].plot(ax=axes[1, 3], legend=False)
cl = list(pivot[pivot.cluster == 11].name.values)
test_x[cl].plot(ax=axes[2, 3], legend=False)
plt.savefig("12clust12dim20NN.png")
plt.show()

model.visualize(model.preds, None, savePath="stockcl12", nclust=n_clusters)


sils = []
for nclust in range(2, 31):
    print("calc score for: %d clusters" % nclust)
    mod = n2d.n2d(
        train_x,
        manifoldLearner=n2d.UmapGMM(nclust, umapdim=nclust, umapN=20),
        ndim=nclust,
        ae_args={"act": "relu"},
    )
    mod.fit(weight_id="weights/sp_%d_clust.h5" % nclust, pretrain_epochs=25)
    mod.predict(np.asarray(test_x.values).T)

    sil_score = silhouette_score(mod.hle, mod.preds, metric="euclidean")
    sils.append(sil_score)
    # x = mod.manifoldLearner.clusterManifold.bic(mod.hle)
    # x2 = mod.manifoldLearner.clusterManifold.aic(mod.hle)
    # BIC_scores.append(x)
    # AIC_scores.append(x2)

plt.plot(list(range(2, 26)), sils)
plt.show()

# 4 is the best!?

# load weights of our best silhouette clustering
clust4 = n2d.n2d(
    train_x, manifoldLearner=n2d.UmapGMM(4, umapdim=4), ndim=4, ae_args={"act": "relu"}
)

clust4.fit(weights="weights/sp_4_clust.h5")

clust4.predict(np.asarray(test_x.values).T)


def get_cluster(x, piv, n):
    cl = list(piv[piv.cluster == n].name.values)
    return x[cl]


def grouper(n2d_obj, df):
    resD = {"name": df.columns, "cluster": n2d_obj.preds}
    pivot = pd.DataFrame.from_dict(resD)
    return (get_cluster(df, pivot, i) for i in np.unique(n2d_obj.preds))


# example frame, just to pick which cluster is most interesting
ef_cl1, ef_cl2, ef_cl3, ef_cl4 = grouper(clust4, test_x)

ef_cl1.plot()
plt.show()


df_cl1, df_cl2, df_cl3, df_cl4 = grouper(
    clust4, pd.read_csv("Data/stock_close.csv", index_col=0)
)

# find cointegrated series

# find smallest values in a distance matrix short form


def argsmallest_n(a, n):
    ret = np.argpartition(a, n)[:n]
    b = np.take(a, ret)
    return np.take(ret, np.argsort(b))


def p_value(x, y):
    _, pv, _ = coint(x, y)
    return pv


res = pdist(df_cl1.transpose(), p_value)


# sanity check
squares = squareform(res)
pd.DataFrame(squares, index=df_cl1.columns, columns=df_cl1.columns)

closest = argsmallest_n(res, 20)
idx = np.triu_indices(104, 1)
pairs = np.column_stack((np.take(idx[0], closest), np.take(idx[1], closest))) + 1

pair_ticks = []
for i in range(0, pairs.shape[0]):
    res = df_cl1.iloc[:, pairs[i, :]].columns
    pair_ticks.append(list(res))

fig = plt.figure()
for i in range(0, len(pair_ticks)):
    df_p = df_cl1[pair_ticks[i]]
    ax = fig.add_subplot(5, 5, i + 1)
    df_p.plot(ax=ax)
plt.show()
