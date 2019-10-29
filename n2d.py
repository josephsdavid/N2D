import os
import random as rn

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use(['seaborn-white', 'seaborn-paper'])
sns.set_context("paper", font_scale=1.3)
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
import umap
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from sklearn import metrics
from sklearn import mixture
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.utils.linear_assignment_ import linear_assignment

import pandas as pd
from keras.datasets import fashion_mnist

def load_har():
    x_train = pd.read_csv(
        'data/har/train/X_train.txt',
        sep=r'\s+',
        header=None)
    y_train = pd.read_csv('data/har/train/y_train.txt', header=None)
    x_test = pd.read_csv('data/har/test/X_test.txt', sep=r'\s+', header=None)
    y_test = pd.read_csv('data/har/test/y_test.txt', header=None)
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    # labels start at 1 so..
    y = y - 1
    y = y.reshape((y.size,))
    y_names = {0: 'Walking', 1: 'Upstairs', 2: 'Downstairs', 3: 'Sitting', 4: 'Standing', 5: 'Laying', }
    return x, y, y_names
def load_fashion():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    y_names = {0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
               5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}
    return x, y, y_names
from time import time


class AutoEncoder:
    def __init__(self, dims, act = 'relu'):
        self.dims = dims
        self.act = act
        self.x = Input(shape = (dims[0],), name = 'input')
        self.h = self.x
        n_stacks = len(self.dims) - 1
        for i in range(n_stacks - 1):
            self.h = Dense(self.dims[i + 1], activation = self.act, name = 'encoder_%d' %i)(self.h)
        self.h = Dense(self.dims[-1], name = 'encoder_%d' % (n_stacks -1))(self.h)
        for i in range(n_stacks - 1, 0, -1):
            self.h = Dense(self.dims[i], activation = self.act, name = 'decoder_%d' % i )(self.h)
        self.h = Dense(dims[0], name = 'decoder_0')(self.h)

        self.Model = Model(inputs = self.x, outputs = self.h)

    def pretrain(self, dataset, batch_size = 256, pretrain_epochs = 1000,
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
            self.Model.save_weights('weights/' + weightname + "-" +
                                    str(pretrain_epochs) +
                                    "-ae_weights.h5")
        else:
            self.Model.load_weights('weights/' + weights)



class UmapClust:
    def __init__(self, nclust,
                 umapdim = 2,
                 umapN = 10,
                 umapMd = float(0),
                 umapMetric = 'euclidean',
                 clusterMethod = 'GMM'
                 ):
        self.nclust = nclust
        self.manifoldInEmbedding = umap.UMAP(
            random_state = 0,
            metric = umapMetric,
            n_components = umapdim,
            n_neighbors = umapN,
            min_dist = umapMd
        )

        self.clusterManifold = mixture.GaussianMixture(
            covariance_type = 'full',
            n_components = nclust, random_state = 0
        )

    def run(self, hl):
        hle = self.manifoldInEmbedding.fit_transform(hl)
        self.clusterManifold.fit(hle)
        y_prob = self.clusterManifold.predict_proba(hle)
        y_pred = y_prob.argmax(1)
        return(np.asarray(y_pred))




def best_cluster_fit(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    best_fit = []
    for i in range(y_pred.size):
        for j in range(len(ind)):
            if ind[j][0] == y_pred[i]:
                best_fit.append(ind[j][1])
    return best_fit, ind, w

def cluster_acc(y_true, y_pred):
    _, ind, w = best_cluster_fit(y_true, y_pred)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def plot(x, y, plot_id, names=None, save_dir = "viz", dataset = "fashion", n_clusters = 10):
    viz_df = pd.DataFrame(data=x[:5000])
    viz_df['Label'] = y[:5000]
    if names is not None:
        viz_df['Label'] = viz_df['Label'].map(names)
    plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=0, y=1, hue='Label', legend='full', palette=sns.color_palette("hls", n_colors=n_clusters),
                    alpha=.5,
                    data=viz_df)
    # Look into ordering and why its not consistent - should use debug mode..
    l = plt.legend(bbox_to_anchor=(-.1, 1.00, 1.1, .5), loc="lower left", markerfirst=True,
                   mode="expand", borderaxespad=0, ncol=n_clusters + 1, handletextpad=0.01, )

    # l = plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    l.texts[0].set_text("")
    plt.ylabel("")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(save_dir + '/' + dataset +
                '-' + plot_id + '.png', dpi=300)
    plt.clf()

class n2d:
    def __init__(self,
                 x, nclust = 10, act = 'relu'
                 ):

        shape = [x.shape[-1], 500, 500, 2000, nclust]
        self.autoencoder = AutoEncoder(shape, act)

        self.hidden = self.autoencoder.Model.get_layer(name='encoder_%d' % (len(shape) - 2)).output
        self.encoder = Model(inputs = self.autoencoder.Model.input, outputs = self.hidden)
        self.x = x
        self.nclust = nclust


    def preTrainEncoder(self,batch_size = 256, pretrain_epochs = 1000,
                     loss = 'mse', optimizer = 'adam',weights = None,
                     verbose = 0, weightname = 'fashion'):

        self.autoencoder.pretrain(dataset = self.x,
                                  batch_size = batch_size,
                                  pretrain_epochs = pretrain_epochs,
                                  loss = loss,
                                  optimizer =optimizer, weights = weights,
                                  verbose = verbose, weightname = weightname)




    def run(self, umapdim = 2,
                 umapN = 10,
                 umapMd = float(0),
                 umapMetric = 'euclidean',
                 clusterMethod = 'GMM'):

        hl = self.encoder.predict(self.x)
        self.hle = hl
        umapclust  = UmapClust(nclust = self.nclust, umapdim = umapdim,
                               umapN = umapN, umapMd = umapMd,
                               umapMetric = umapMetric,
                               clusterMethod = clusterMethod)

        preds = umapclust.run(hl)
        self.preds = preds

    def assess(self, y):
        y = np.asarray(y)
        acc = np.round(cluster_acc(y, self.preds), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(y, self.preds), 5)
        ari = np.round(metrics.adjusted_rand_score(y, self.preds))

        return(acc, nmi, ari)

    def visualize(self, y, names, dataset = "fashion", n_clusters = 10):
        y = np.asarray(y)
        y_pred = np.asarray(self.preds)
        hle = self.hle
        plot(hle, y, 'n2d', names, dataset = dataset, n_clusters = n_clusters)
        y_pred_viz, _, _ = best_cluster_fit(y, y_pred)
        plot(hle, y_pred_viz, 'n2d-predicted', names, dataset = dataset, n_clusters = n_clusters)
#x, y, y_names =  load_fashion()




#os.environ['PYTHONHASHSEED'] = '0'
#
#
#rn.seed(0)
#tf.set_random_seed(0)
#np.random.seed(0)
#
#if len(K.tensorflow_backend._get_available_gpus()) > 0:
#    print("Using GPU")
#    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
#                                  inter_op_parallelism_threads=1,
#                                  )
#    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#    K.set_session(sess)
#
#
#
#x,y, y_names = nd.load_har()
#
#
#harcluster = nd.n2d(x, nclust = 6)
#
#harcluster.preTrainEncoder(weights = "har-1000-ae_weights.h5")
#
#harcluster.run()
#
#harcluster.visualize(y, y_names, dataset = "har", n_clusters = 6)
#print(harcluster.assess(y))
