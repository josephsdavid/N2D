import os
import random as rn

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import sys
import tensorflow as tf

from . import linear_assignment as la

import umap
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from sklearn import metrics
from sklearn import mixture
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding

import pandas as pd

class AutoEncoder:
    def __init__(self, data, ndim, architecture, act = 'relu'):
        shape = [data.shape[-1]] + architecture + [ndim]
        self.dims = shape
        self.act = act
        self.x = Input(shape = (self.dims[0],), name = 'input')
        self.h = self.x
        n_stacks = len(self.dims) - 1
        for i in range(n_stacks - 1):
            self.h = Dense(self.dims[i + 1], activation = self.act, name = 'encoder_%d' %i)(self.h)
        self.h = Dense(self.dims[-1], name = 'encoder_%d' % (n_stacks -1))(self.h)
        for i in range(n_stacks - 1, 0, -1):
            self.h = Dense(self.dims[i], activation = self.act, name = 'decoder_%d' % i )(self.h)
        self.h = Dense(self.dims[0], name = 'decoder_0')(self.h)

        self.Model = Model(inputs = self.x, outputs = self.h)

    def fit(self, dataset, batch_size = 256, pretrain_epochs = 1000,
                     loss = 'mse', optimizer = 'adam',weights = None,
                     verbose = 0, weightname = 'fashion'):
        if weights == None:
            self.Model.compile(
                loss = loss, optimizer = optimizer
            )
            self.Model.fit(
                dataset, dataset,
                batch_size = batch_size,
                epochs = pretrain_epochs
            )
            # make this less stupid
            self.Model.save_weights(weightname)
        else:
            self.Model.load_weights(weights)



class UmapGMM:
    def __init__(self, nclust,
                 umapdim = 2,
                 umapN = 10,
                 umapMd = float(0),
                 umapMetric = 'euclidean',
                 random_state = 0
                 ):
        self.nclust = nclust
        self.manifoldInEmbedding = umap.UMAP(
            random_state = random_state,
            metric = umapMetric,
            n_components = umapdim,
            n_neighbors = umapN,
            min_dist = umapMd
        )

        self.clusterManifold = mixture.GaussianMixture(
            covariance_type = 'full',
            n_components = nclust, random_state = random_state
        )
        self.hle = None

    def fit(self, hl):
        self.hle = self.manifoldInEmbedding.fit_transform(hl)
        self.clusterManifold.fit(self.hle)

    def predict(self):
        y_prob = self.clusterManifold.predict_proba(self.hle)
        y_pred = y_prob.argmax(1)
        return(np.asarray(y_pred))




def best_cluster_fit(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = la.linear_assignment(w.max() - w)
    best_fit = []
    for i in range(y_pred.size):
        for j in range(len(ind)):
            if ind[j][0] == y_pred[i]:
                best_fit.append(ind[j][1])
    return best_fit, ind, w

def cluster_acc(y_true, y_pred):
    _, ind, w = best_cluster_fit(y_true, y_pred)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def plot(x, y, plot_id, names=None,  savepath = "Generic_figure", n_clusters = 10):
    viz_df = pd.DataFrame(data=x[:5000])
    viz_df['Label'] = y[:5000]
    if names is not None:
        viz_df['Label'] = viz_df['Label'].map(names)
    plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=0, y=1, hue='Label', legend='full', hue_order=sorted(viz_df['Label'].unique()), palette=sns.color_palette("hls", n_colors=n_clusters),
                    alpha=.5,
                    data=viz_df)
    l = plt.legend(bbox_to_anchor=(-.1, 1.00, 1.1, .5), loc="lower left", markerfirst=True,
                   mode="expand", borderaxespad=0, ncol=n_clusters + 1, handletextpad=0.01, )

    # l = plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    l.texts[0].set_text("")
    plt.ylabel("")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig( savepath +
                '-' + plot_id + '.png', dpi=300)
    plt.clf()

class n2d:
    def __init__(self,
                 x,
                 manifoldLearner,
                 autoencoder = AutoEncoder,
                 architecture = [500,500,2000],
                 ndim = 10,
                 ae_args = {"act":"relu"},
                 ):
        shape = [x.shape[-1]] + architecture + [ndim]

        self.autoencoder = autoencoder(data = x,
                                       ndim = ndim,
                                       architecture = architecture,
                                       **ae_args)
        self.manifoldLearner = manifoldLearner
        self.hidden = self.autoencoder.Model.get_layer(name='encoder_%d' % (len(shape) - 2)).output
        self.encoder = Model(inputs = self.autoencoder.Model.input, outputs = self.hidden)
        self.x = x
        self.ndim = ndim
        self.preds = None
        self.hle = None



    def fit(self,batch_size = 256, pretrain_epochs = 1000,
                     loss = 'mse', optimizer = 'adam',weights = None,
                     verbose = 0, weight_id = 'generic_autoencoder'):

        self.autoencoder.fit(dataset = self.x,
                                  batch_size = batch_size,
                                  pretrain_epochs = pretrain_epochs,
                                  loss = loss,
                                  optimizer =optimizer, weights = weights,
                                  verbose = verbose, weightname = weight_id)


    def predict(self, x = None):
        if (x is None):
            x_test = self.x
        else:
            x_test = x
        hl = self.encoder.predict(x_test)
        self.manifoldLearner.fit(hl)
        self.preds = self.manifoldLearner.predict()
        self.hle = self.manifoldLearner.hle

    def assess(self, y):
        y = np.asarray(y)
        acc = np.round(cluster_acc(y, self.preds), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(y, self.preds), 5)
        ari = np.round(metrics.adjusted_rand_score(y, self.preds), 5)

        return(acc, nmi, ari)


    def visualize(self, y, names, savePath = "Generic_Dataset", nclust = 10):
        y = np.asarray(y)
        y_pred = np.asarray(self.preds)
        hle = self.hle
        plot(hle, y, 'n2d', names, savepath = savePath, n_clusters = nclust)
        y_pred_viz, _, _ = best_cluster_fit(y, y_pred)
        plot(hle, y_pred_viz, 'n2d-predicted', names, savepath = savePath, n_clusters = nclust)
